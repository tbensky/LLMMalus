"""Draw one graph: measured points and, optionally, the model's curve.

PNG through matplotlib when it is installed; otherwise a plain SVG, so a
run never fails for want of a plotting library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def render(path_stem: Path, *, x_label: str, y_label: str, title: str,
           points: Sequence[tuple[float, float]],
           curve: Sequence[tuple[float, float]] | None = None,
           curve_label: str = "") -> Path:
    """Write path_stem.png (or .svg) and return the file written."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - any import problem means: use the fallback
        return _render_svg(path_stem.with_suffix(".svg"), x_label=x_label, y_label=y_label,
                           title=title, points=points, curve=curve)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=130)
    if curve:
        ax.plot([c[0] for c in curve], [c[1] for c in curve], color="#2a78d6",
                linewidth=2, label=curve_label or "fit", zorder=2)
    ax.scatter([p[0] for p in points], [p[1] for p in points], s=28, color="#111111",
               edgecolor="white", linewidth=0.8, label="data", zorder=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11)
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = path_stem.with_suffix(".png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _render_svg(path: Path, *, x_label: str, y_label: str, title: str,
                points: Sequence[tuple[float, float]],
                curve: Sequence[tuple[float, float]] | None) -> Path:
    width, height, left, right, top, bottom = 720, 460, 70, 20, 40, 55
    every = list(points) + list(curve or [])
    xs, ys = [p[0] for p in every], [p[1] for p in every]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(min(ys), 0.0), max(ys)
    if x1 == x0:
        x0, x1 = x0 - 1, x1 + 1
    if y1 == y0:
        y1 = y0 + 1

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * (width - left - right)

    def sy(y: float) -> float:
        return height - bottom - (y - y0) / (y1 - y0) * (height - top - bottom)

    def esc(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'font-family="sans-serif" font-size="12">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14">{esc(title)}</text>',
             f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#555"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#555"/>',
             f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle">{esc(x_label)}</text>',
             f'<text x="18" y="{height / 2}" text-anchor="middle" '
             f'transform="rotate(-90 18 {height / 2})">{esc(y_label)}</text>']
    for i in range(5):
        xv, yv = x0 + i * (x1 - x0) / 4, y0 + i * (y1 - y0) / 4
        parts.append(f'<text x="{sx(xv):.1f}" y="{height - bottom + 16}" text-anchor="middle">{xv:g}</text>')
        parts.append(f'<text x="{left - 6}" y="{sy(yv) + 4:.1f}" text-anchor="end">{yv:.3g}</text>')
    if curve:
        path_d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                          for i, (x, y) in enumerate(curve))
        parts.append(f'<path d="{path_d}" fill="none" stroke="#2a78d6" stroke-width="2"/>')
    for x, y in points:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#111"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
