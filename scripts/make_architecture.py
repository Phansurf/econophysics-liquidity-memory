#!/usr/bin/env python3
"""Generate the code-architecture diagram (results/figures/architecture.pdf)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'results/figures'
FIG.mkdir(parents=True, exist_ok=True)

NAVY, TEAL, GRAY = '#12304a', '#087f8c', '#4e6070'


def main():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set(xlim=(0, 10), ylim=(0, 4.6))
    ax.axis('off')
    boxes = [
        (0.3, 2.7, 'CSV / synthetic input', 'data.py: auditable cleaning'),
        (3.8, 2.7, 'Baseline inference', 'baseline.py + inference.py\nEM / MLE / AIC / Gibbs'),
        (7.2, 2.7, 'Saved evidence', 'CSV / NPZ / JSON'),
        (0.3, 0.7, 'Liquidity extension', 'liquidity.py\nlog-space splitting'),
        (3.8, 0.7, 'Multi-scale memory', 'multiscale.py\nOU superposition'),
        (7.2, 0.7, 'Observables / diagnostics', 'statistics.py\nACF / kurtosis / Wilson'),
    ]
    for x, y, title, detail in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.5, 1.25, boxstyle='round,pad=.08',
                                    fc='#edf6f5', ec=TEAL))
        ax.text(x + 1.25, y + 0.86, title, ha='center', fontsize=9.5, color=NAVY)
        ax.text(x + 1.25, y + 0.34, detail, ha='center', fontsize=6.8, color=GRAY, linespacing=1.35)
    arrows = [((2.85, 3.3), (3.65, 3.3)), ((6.35, 3.3), (7.05, 3.3)),
              ((2.85, 1.3), (3.65, 1.3)), ((6.35, 1.3), (7.05, 1.3)),
              ((8.45, 2.6), (8.45, 2.05))]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.5))
    fig.savefig(FIG / 'architecture.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG / "architecture.pdf"}')


if __name__ == '__main__':
    main()
