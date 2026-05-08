"""
Operation-count scaling comparison for Quest-style page scan vs. HNSW search.

This is an analytic artifact, not a wall-clock benchmark. It estimates the
number of distance/dot-product scalar multiply-adds needed to choose sparse
attention candidates for one query at different context lengths.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def quest_ops(n_tokens: int, page_size: int, d_head: int) -> int:
    """Quest scores each page using min/max metadata in the native head dim."""
    return math.ceil(n_tokens / page_size) * d_head


def hnsw_ops(n_tokens: int, hnsw_m: int, d_search: int) -> int:
    """
    Simple HNSW scoring proxy: M graph neighbors per level/step times log2(N)
    levels/steps times search-space dimension.
    """
    return math.ceil(hnsw_m * math.log2(max(2, n_tokens)) * d_search)


def write_svg(rows: list[dict], out_path: Path):
    width, height = 820, 460
    margin_l, margin_r, margin_t, margin_b = 86, 28, 32, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    xs = [r["n_tokens"] for r in rows]
    ys = [r["quest_ops"] for r in rows] + [r["hnsw_ops"] for r in rows]
    x_min, x_max = math.log10(min(xs)), math.log10(max(xs))
    y_min, y_max = math.log10(min(ys)), math.log10(max(ys))
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def sx(x):
        return margin_l + (math.log10(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return margin_t + (y_max - math.log10(y)) / (y_max - y_min) * plot_h

    def poly(values, color):
        pts = " ".join(f"{sx(r['n_tokens']):.1f},{sy(r[values]):.1f}" for r in rows)
        return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{pts}" />'

    x_ticks = [8192, 32768, 131072, 524288, 1048576]
    y_ticks = [10_000, 100_000, 1_000_000, 10_000_000]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#333"/>',
    ]
    for t in x_ticks:
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{height-margin_b}" x2="{x:.1f}" y2="{height-margin_b+5}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-margin_b+24}" text-anchor="middle" font-size="12">{t//1024}K</text>')
    for t in y_ticks:
        y = sy(t)
        parts.append(f'<line x1="{margin_l-5}" y1="{y:.1f}" x2="{margin_l}" y2="{y:.1f}" stroke="#333"/>')
        label = f"{t//1_000_000}M" if t >= 1_000_000 else f"{t//1000}K"
        parts.append(f'<text x="{margin_l-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{label}</text>')
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#eee"/>')
    parts.extend([
        poly("quest_ops", "#c0392b"),
        poly("hnsw_ops", "#1f77b4"),
        '<text x="410" y="22" text-anchor="middle" font-size="16" font-weight="600">Candidate-scoring operations per query</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="13">Context length (tokens, log scale)</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18,{height/2})" text-anchor="middle" font-size="13">Scalar ops (log scale)</text>',
        '<rect x="545" y="52" width="226" height="58" fill="white" stroke="#ddd"/>',
        '<line x1="562" y1="72" x2="602" y2="72" stroke="#c0392b" stroke-width="3"/>',
        '<text x="612" y="76" font-size="13">Quest page scan</text>',
        '<line x1="562" y1="94" x2="602" y2="94" stroke="#1f77b4" stroke-width="3"/>',
        '<text x="612" y="98" font-size="13">HNSW over learned search</text>',
        '</svg>',
    ])
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--d-head", type=int, default=128)
    parser.add_argument("--d-search", type=int, default=128)
    parser.add_argument("--hnsw-m", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contexts = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

    rows = []
    for n in contexts:
        q = quest_ops(n, args.page_size, args.d_head)
        h = hnsw_ops(n, args.hnsw_m, args.d_search)
        rows.append(
            {
                "n_tokens": n,
                "quest_ops": q,
                "hnsw_ops": h,
                "quest_over_hnsw": q / h,
            }
        )

    csv_path = out_dir / "asymptotic_scoring_ops.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n_tokens", "quest_ops", "hnsw_ops", "quest_over_hnsw"]
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "asymptotic_scoring_ops.md"
    lines = [
        "| Context | Quest ops/query | HNSW ops/query | Quest / HNSW |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_tokens'] // 1024}K | {r['quest_ops']:,} | "
            f"{r['hnsw_ops']:,} | {r['quest_over_hnsw']:.2f}x |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    svg_path = out_dir / "asymptotic_scoring_ops.svg"
    write_svg(rows, svg_path)

    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
