"""Synthetic fixtures only: these tests make no real-market-data claim."""
import numpy as np
import pandas as pd
import unittest

from frontier_econophysics.data import prepare_prices, prepare_quotes


def test_quote_filter_duplicates_resample_and_no_overnight_fill():
    quotes = pd.DataFrame({
        "timestamp": [
            "2020-01-02T09:00:00Z", "2020-01-02T09:15:00Z",
            "2020-01-02T09:15:00Z", "2020-01-02T09:20:00Z",
            "2020-01-02T09:25:00Z", "2020-01-02T09:30:00Z",
            "2020-01-02T10:00:00Z", "2020-01-03T09:00:00Z",
            "2020-01-03T12:00:00Z", "not-a-date",
        ],
        "bid": [99, 100, 101, 110, 90, np.nan, 103, 104, 105, 1],
        "ask": [101, 102, 103, 109, 95, 102, 105, 106, 107, 2],
    })
    result = prepare_quotes(quotes)
    assert result.frame["price"].tolist() == [100, 102, 104, 105, 106]
    assert result.frame["transition_eligible"].tolist() == [False, True, True, False, False]
    assert np.isnan(result.frame.loc[3, "eligible_dt_seconds"])
    assert result.diagnostics["crossed_quote_rows"] == 1
    assert result.diagnostics["spread_ge_5_rows"] == 1
    assert result.diagnostics["duplicate_timestamp_rows_removed"] == 1
    assert result.diagnostics["nonfinite_numeric_rows"] == 1
    assert result.diagnostics["invalid_timestamp_rows"] == 1
    assert result.metadata["forward_fill"] is False


def test_prices_preserve_last_original_valid_row_and_reject_unzoned():
    result = prepare_prices(pd.DataFrame({
        "timestamp": ["2020-01-02T10:00:00Z", "2020-01-02T09:30:00Z",
                      "2020-01-02T10:00:00+00:00", "2020-01-02T10:30:00",
                      "2020-01-02T11:00:00Z", "2020-01-02T11:30:00Z"],
        "price": [2, 1, 3, 4, 0, np.inf],
    }))
    assert result.frame["price"].tolist() == [1, 3]
    assert result.diagnostics["out_of_order_adjacent_pairs"] == 1
    assert result.diagnostics["invalid_timestamp_rows"] == 1
    assert result.diagnostics["nonpositive_price_rows"] == 1
    assert result.diagnostics["nonfinite_numeric_rows"] == 1
    assert "NOT performed" in result.metadata["corporate_actions"]


def test_missing_and_empty_inputs_raise():
    case = unittest.TestCase()
    with case.assertRaisesRegex(ValueError, "Missing required"):
        prepare_prices(pd.DataFrame({"price": [1]}))
    with case.assertRaisesRegex(ValueError, "No valid"):
        prepare_quotes(pd.DataFrame({"timestamp": ["2020-01-02T09:00:00Z"], "bid": [2], "ask": [1]}))


def test_last_invalid_duplicate_does_not_erase_valid_quote():
    result = prepare_quotes(pd.DataFrame({
        "timestamp": ["2020-01-02T09:00:00Z"] * 2,
        "bid": [99, 102], "ask": [101, 100],
    }))
    assert result.frame["price"].tolist() == [100]


def test_midnight_overlapping_bins_are_dropped_and_audited():
    result = prepare_quotes(pd.DataFrame({
        "timestamp": ["2020-01-01T23:30:00Z", "2020-01-01T23:59:00Z",
                      "2020-01-02T00:00:00Z", "2020-01-02T00:01:00Z"],
        "bid": [99, 100, 101, 102], "ask": [101, 102, 103, 104],
    }))
    frame = result.frame
    assert frame["timestamp"].is_unique
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["price"].tolist() == [100, 103]
    assert frame["source_session"].tolist() == ["2020-01-01", "2020-01-02"]
    assert frame["transition_eligible"].tolist() == [False, False]
    assert result.diagnostics["resampling_collision_bins"] == 1
    assert result.diagnostics["resampling_collision_rows_removed"] == 2


def test_source_session_and_real_quote_gap_override_bin_labels():
    # Both labels are on Jan 2, but the first observation belongs to Jan 1.
    # No collision is required to trigger this important cross-session case.
    result = prepare_quotes(pd.DataFrame({
        "timestamp": ["2020-01-01T23:59:00Z", "2020-01-02T00:01:00Z",
                      "2020-01-02T00:31:00Z"],
        "bid": [99, 100, 101], "ask": [101, 102, 103],
    }))
    assert result.frame["timestamp"].is_unique
    assert result.frame["timestamp"].is_monotonic_increasing
    assert result.frame["transition_eligible"].tolist() == [False, False, True]
    assert result.frame["gap_seconds"].iloc[1:].tolist() == [120, 1800]
    assert result.frame["eligible_dt_seconds"].iloc[2] == 1800
    assert result.diagnostics["resampling_collision_rows_removed"] == 0


class TestData(unittest.TestCase):
    test_quotes = staticmethod(test_quote_filter_duplicates_resample_and_no_overnight_fill)
    test_prices = staticmethod(test_prices_preserve_last_original_valid_row_and_reject_unzoned)
    test_invalid = staticmethod(test_missing_and_empty_inputs_raise)
    test_last_valid = staticmethod(test_last_invalid_duplicate_does_not_erase_valid_quote)
    test_midnight_collision = staticmethod(test_midnight_overlapping_bins_are_dropped_and_audited)
    test_source_session = staticmethod(test_source_session_and_real_quote_gap_override_bin_labels)


if __name__ == "__main__":
    unittest.main()
