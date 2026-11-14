"""Auditable CSV preparation; no bundled or implicitly downloaded market data.

Input timestamps must be ISO-8601 with an explicit UTC suffix (``Z`` or
``+00:00``), e.g. ``2020-01-02T09:30:00Z``. A UTC calendar date is only a
conservative session proxy: exchange calendars must be supplied separately
before using this module on markets whose sessions cross UTC midnight.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Union

import numpy as np
import pandas as pd


FrameSource = Union[str, Path, pd.DataFrame]
_UTC_ISO = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|\+00:00)$"
)


@dataclass
class PreparedData:
    """Prepared observations plus explicit, serializable audit information."""

    frame: pd.DataFrame
    diagnostics: dict
    metadata: dict


def _load(source: FrameSource, required: tuple[str, ...]) -> pd.DataFrame:
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    frame = frame.loc[:, list(required)].copy()
    # Retain file order for a deterministic duplicate-timestamp policy.
    frame["_original_order"] = np.arange(len(frame))
    return frame


def _parse(frame: pd.DataFrame, numerical: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    audit = {"input_rows": len(frame)}
    iso_ok = frame["timestamp"].map(lambda value: isinstance(value, str) and bool(_UTC_ISO.fullmatch(value)))
    # Coerce each accepted ISO representation independently, so mixed Z/+00:00
    # and fractional/nonfractional strings work across supported pandas versions.
    parsed = frame["timestamp"].where(iso_ok).map(
        lambda value: pd.to_datetime(value, utc=True, errors="coerce") if pd.notna(value) else pd.NaT
    )
    frame["timestamp"] = pd.to_datetime(parsed, utc=True)
    audit["invalid_timestamp_rows"] = int(frame["timestamp"].isna().sum())
    finite = np.ones(len(frame), dtype=bool)
    for column in numerical:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        finite &= np.isfinite(frame[column].to_numpy(dtype=float))
    audit["nonfinite_numeric_rows"] = int((~finite).sum())
    frame = frame.loc[frame["timestamp"].notna() & finite].copy()
    audit["out_of_order_adjacent_pairs"] = int((frame["timestamp"].diff().dt.total_seconds() < 0).sum())
    return frame, audit


def _sort_deduplicate(frame: pd.DataFrame, audit: dict) -> pd.DataFrame:
    # Filtering precedes deduplication: the last VALID row in original file
    # order wins. A later crossed/invalid quote does not erase an earlier valid one.
    audit["duplicate_timestamp_rows_removed"] = int(frame.duplicated("timestamp", keep="last").sum())
    frame = frame.drop_duplicates("timestamp", keep="last")
    return frame.sort_values("timestamp", kind="stable").drop(columns="_original_order")


def _transitions(frame: pd.DataFrame, max_gap: str) -> pd.DataFrame:
    threshold = pd.Timedelta(max_gap)
    if threshold <= pd.Timedelta(0):
        raise ValueError("max_gap must be positive")
    frame = frame.reset_index(drop=True)
    # Right-edge labels can lie on the next UTC date. Session membership and
    # elapsed time therefore follow the underlying quote, never the bin label.
    observed = frame["observed_at"] if "observed_at" in frame else frame["timestamp"]
    if "source_session" not in frame:
        frame["source_session"] = observed.dt.strftime("%Y-%m-%d")
    delta = observed.diff()
    same_session = frame["source_session"].eq(frame["source_session"].shift())
    frame["gap_seconds"] = delta.dt.total_seconds()
    frame["transition_eligible"] = same_session & delta.gt(pd.Timedelta(0)) & delta.le(threshold)
    # Never silently pass an overnight/long gap as a normal one-step transition.
    frame["eligible_dt_seconds"] = frame["gap_seconds"].where(frame["transition_eligible"])
    return frame


def _result(frame: pd.DataFrame, audit: dict, kind: str, max_gap: str, **metadata) -> PreparedData:
    if frame.empty:
        raise ValueError(f"No valid {kind} observations after filtering; diagnostics={audit}")
    frame = _transitions(frame, max_gap)
    audit["output_rows"] = len(frame)
    audit["eligible_transitions"] = int(frame["transition_eligible"].sum())
    audit["excluded_transitions"] = max(0, len(frame) - 1 - audit["eligible_transitions"])
    return PreparedData(frame, audit, {
        "source_kind": kind,
        "timezone": "UTC",
        "duplicate_policy": "last valid row in original file order",
        "forward_fill": False,
        "max_gap": max_gap,
        "session_policy": "UTC calendar date proxy; no cross-date transitions",
        "gap_clock": "observed_at for quotes; timestamp for direct prices",
        "corporate_actions": "not supplied; split/dividend adjustment NOT performed",
        "evidence_status": "user-supplied input only; no claim of original-paper real-data replication",
        **metadata,
    })


def prepare_quotes(source: FrameSource, frequency: str = "30min", max_gap: str = "90min") -> PreparedData:
    """Filter UTC quotes and return last valid mid-price in each sampling bin.

    Required columns: timestamp,bid,ask. Reject nonfinite/nonpositive quotes,
    crossed quotes (bid > ask), and spreads >= 5 price units. Zero spread is
    allowed. Empty bins are dropped; no forward filling occurs. Bin timestamps
    identify the right bin edge; ``observed_at`` retains the actual quote time.
    ``source_session`` retains that quote's UTC date. If separate source
    sessions yield the same right-edge label, ALL colliding rows are dropped
    and audited rather than silently mixing sessions or emitting duplicates.
    """
    frame, audit = _parse(_load(source, ("timestamp", "bid", "ask")), ("bid", "ask"))
    positive = frame["bid"].gt(0) & frame["ask"].gt(0)
    crossed = frame["bid"].gt(frame["ask"])
    wide = (frame["ask"] - frame["bid"]).ge(5)
    audit["nonpositive_quote_rows"] = int((~positive).sum())
    audit["crossed_quote_rows"] = int(crossed.sum())
    audit["spread_ge_5_rows"] = int(wide.sum())
    frame = _sort_deduplicate(frame.loc[positive & ~crossed & ~wide].copy(), audit)
    if frame.empty:
        return _result(frame.assign(price=[]), audit, "quotes", max_gap, frequency=frequency)
    frame["price"] = (frame["bid"] + frame["ask"]) / 2
    frame["observed_at"] = frame["timestamp"]
    frame["source_session"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    # Group separately by UTC date, preventing an empty overnight bin from
    # inheriting a quote. .last() uses only fully valid rows prepared above.
    groups = []
    for _, session in frame.groupby(frame["timestamp"].dt.date, sort=True):
        sampled = session.set_index("timestamp").resample(frequency, closed="right", label="right").last()
        groups.append(sampled.dropna(subset=["price"]).reset_index())
    sampled = pd.concat(groups, ignore_index=True)
    sampled = sampled.sort_values("timestamp", kind="stable")
    audit["valid_quote_rows_before_resampling"] = len(frame)
    return _result(sampled, audit, "quotes", max_gap, frequency=frequency, spread_cutoff=5.0)


def prepare_prices(source: FrameSource, max_gap: str = "90min") -> PreparedData:
    """Validate a UTC timestamp,price CSV without inferring corporate actions.

    Returns sorted positive finite prices. Consumers must filter by
    transition_eligible and choose an explicit time unit before SDE inference.
    Daily data need an exchange-calendar treatment; this conservative default
    deliberately excludes every cross-date transition.
    """
    frame, audit = _parse(_load(source, ("timestamp", "price")), ("price",))
    positive = frame["price"].gt(0)
    audit["nonpositive_price_rows"] = int((~positive).sum())
    frame = _sort_deduplicate(frame.loc[positive].copy(), audit)
    return _result(frame, audit, "prices", max_gap)
