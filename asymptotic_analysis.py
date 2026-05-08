"""
Operation-count scaling comparison for full attention scoring, Quest-style
page scan, and HNSW search over trained projections.

This is an analytic artifact, not a wall-clock benchmark. It estimates the
number of distance/dot-product scalar multiply-adds needed to choose sparse
attention candidates for one query at different context lengths.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def full_ops(n_tokens: int, d_head: int) -> int:
    """Full attention scores every key with the native-head query."""
    return n_tokens * d_head


def quest_ops(n_tokens: int, page_size: int, d_head: int) -> int:
    """Quest scores min/max metadata for each page in the native head dim."""
    return math.ceil(n_tokens / page_size) * 2 * d_head


def hnsw_ops(n_tokens: int, hnsw_m: int, ef_search: int, d_search: int) -> int:
    """
    HNSW scoring proxy: M graph neighbors, ef_search candidate expansion,
    log2(N) search depth, and search-space dimension.
    """
    return math.ceil(hnsw_m * ef_search * math.log2(max(2, n_tokens)) * d_search)


def find_crossover(page_size: int, d_head: int, hnsw_m: int, ef_search: int, d_search: int) -> float:
    """Solve Quest(N) = HNSW(N) by bisection."""
    lo, hi = 2.0, 8_000_000.0
    for _ in range(96):
        mid = (lo + hi) / 2
        diff = quest_ops(int(mid), page_size, d_head) - hnsw_ops(
            int(mid), hnsw_m, ef_search, d_search
        )
        if diff < 0:
            lo = mid
        else:
            hi = mid
    return hi


def write_svg(rows: list[dict], out_path: Path, crossover: float | None = None):
    width, height = 820, 460
    margin_l, margin_r, margin_t, margin_b = 86, 28, 32, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    xs = [r["n_tokens"] for r in rows]
    ys = (
        [r["full_flops"] for r in rows]
        + [r["quest_flops"] for r in rows]
        + [r["learned_flops"] for r in rows]
    )
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

    x_ticks = [4_000, 32_000, 128_000, 512_000, 1_000_000, 4_000_000]
    y_ticks = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#333"/>',
    ]
    for t in x_ticks:
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{height-margin_b}" x2="{x:.1f}" y2="{height-margin_b+5}" stroke="#333"/>')
        label = f"{t//1_000_000}M" if t >= 1_000_000 else f"{t//1000}K"
        parts.append(f'<text x="{x:.1f}" y="{height-margin_b+24}" text-anchor="middle" font-size="12">{label}</text>')
    for t in y_ticks:
        y = sy(t)
        parts.append(f'<line x1="{margin_l-5}" y1="{y:.1f}" x2="{margin_l}" y2="{y:.1f}" stroke="#333"/>')
        label = f"{t//1_000_000}M" if t >= 1_000_000 else f"{t//1000}K"
        parts.append(f'<text x="{margin_l-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{label}</text>')
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#eee"/>')
    parts.extend([
        poly("full_flops", "#555555"),
        poly("quest_flops", "#c0392b"),
        poly("learned_flops", "#1f77b4"),
    ])
    if crossover is not None and min(xs) <= crossover <= max(xs):
        x = sx(crossover)
        y = sy(next(r["learned_flops"] for r in rows if r["n_tokens"] >= crossover))
        parts.extend([
            f'<line x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{height-margin_b}" stroke="#777" stroke-dasharray="5 5"/>',
            f'<text x="{x+8:.1f}" y="{max(margin_t+18, y-14):.1f}" font-size="12" fill="#555">Quest/HNSW crossover ~{int(crossover/1000)}K</text>',
        ])
    parts.extend([
        '<text x="410" y="22" text-anchor="middle" font-size="16" font-weight="600">Candidate-scoring operations per query</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="13">Context length (tokens, log scale)</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18,{height/2})" text-anchor="middle" font-size="13">Scalar ops (log scale)</text>',
        '<rect x="515" y="52" width="258" height="82" fill="white" stroke="#ddd"/>',
        '<line x1="532" y1="72" x2="572" y2="72" stroke="#555555" stroke-width="3"/>',
        '<text x="582" y="76" font-size="13">Full attention scoring</text>',
        '<line x1="532" y1="96" x2="572" y2="96" stroke="#c0392b" stroke-width="3"/>',
        '<text x="582" y="100" font-size="13">Quest page scan</text>',
        '<line x1="532" y1="120" x2="572" y2="120" stroke="#1f77b4" stroke-width="3"/>',
        '<text x="582" y="124" font-size="13">HNSW over learned search</text>',
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
    parser.add_argument("--ef-search", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contexts = [
        4_000,
        8_000,
        16_000,
        32_000,
        64_000,
        128_000,
        256_000,
        512_000,
        1_000_000,
        2_000_000,
        4_000_000,
    ]
    crossover = find_crossover(
        args.page_size, args.d_head, args.hnsw_m, args.ef_search, args.d_search
    )

    rows = []
    for n in contexts:
        full = full_ops(n, args.d_head)
        q = quest_ops(n, args.page_size, args.d_head)
        h = hnsw_ops(n, args.hnsw_m, args.ef_search, args.d_search)
        rows.append(
            {
                "n_tokens": n,
                "full_flops": full,
                "quest_flops": q,
                "learned_flops": h,
                "quest_speedup_vs_full": full / q,
                "learned_speedup_vs_full": full / h,
                "learned_speedup_vs_quest": q / h,
            }
        )

    csv_path = out_dir / "scaling_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "context_length",
                "full_flops",
                "quest_flops",
                "learned_flops",
                "quest_speedup_vs_full",
                "learned_speedup_vs_full",
                "learned_speedup_vs_quest",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "context_length": r["n_tokens"],
                    "full_flops": r["full_flops"],
                    "quest_flops": r["quest_flops"],
                    "learned_flops": r["learned_flops"],
                    "quest_speedup_vs_full": f"{r['quest_speedup_vs_full']:.2f}",
                    "learned_speedup_vs_full": f"{r['learned_speedup_vs_full']:.2f}",
                    "learned_speedup_vs_quest": f"{r['learned_speedup_vs_quest']:.2f}",
                }
            )

    md_path = out_dir / "scaling_analysis.md"
    lines = [
        "# Candidate-Scoring Operation Count",
        "",
        "This is an analytic operation-count proxy, not a wall-clock benchmark.",
        "It counts the per-query work to identify candidate keys before running",
        "the sparse attention softmax and value multiply over the selected keys.",
        "",
        "## Assumptions",
        "",
        f"- Native head dimension: `d_head = {args.d_head}`.",
        f"- Learned search dimension: `d_search = {args.d_search}`.",
        f"- Quest page size: `page_size = {args.page_size}`.",
        f"- HNSW parameters: `M = {args.hnsw_m}`, `ef_search = {args.ef_search}`.",
        "",
        "Per-query scoring formulas:",
        "",
        f"- Full attention: `N * d_head = N * {args.d_head}`.",
        f"- Quest: `(N / page_size) * 2 * d_head = N * {2 * args.d_head / args.page_size:.0f}`.",
        f"- Learned HNSW: `M * ef_search * log2(N) * d_search = {args.hnsw_m * args.ef_search * args.d_search:,} * log2(N)`.",
        "",
        f"Under these constants, the Quest/HNSW operation-count crossover is approximately `{int(crossover):,}` tokens.",
        "Smaller HNSW settings move the crossover earlier; higher-recall settings move it later.",
        "",
        "## Table",
        "",
        "| Context | Full ops/query | Quest ops/query | Learned HNSW ops/query | Quest / learned |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        context = f"{r['n_tokens'] // 1_000_000}M" if r["n_tokens"] >= 1_000_000 else f"{r['n_tokens'] // 1000}K"
        lines.append(
            f"| {context} | {r['full_flops']:,} | {r['quest_flops']:,} | "
            f"{r['learned_flops']:,} | {r['learned_speedup_vs_quest']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Quest is cheaper than this high-recall HNSW proxy below the few-hundred-thousand-token regime.",
            "At 1M context, Quest costs about 16M scalar ops/query while learned HNSW costs about 5.2M,",
            "a roughly 3x operation-count advantage for learned projections.",
            "",
            "This does not establish production wall-clock speedup. That still requires GPU-resident ANN",
            "retrieval and decode/KV-cache integration. Memory bandwidth may further favor learned ANN at",
            "very long context, but that is not included in this FLOP-only proxy.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    svg_path = out_dir / "scaling_plot.svg"
    write_svg(rows, svg_path, crossover=crossover)

    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
