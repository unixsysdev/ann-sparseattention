# Paper Draft

This directory contains a LaTeX-only workshop-paper draft for the current ANN
sparse-attention prototype. It was rebuilt from
`/home/marcel/Downloads/make_attention_sub_quadratic_again.pdf` and expanded
with the May 9 all-layer experiments.

Build locally with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft is intentionally conservative:

- no wall-clock speedup claim;
- no clean PPL advantage over Quest;
- dynamic indexing framed as a retrieval-mass proxy, not a generation result;
- scaling framed as an operation-count proxy, not production latency;
- all-36 substitution reported as feasible but not parity;
- all32 reserved-layer substitution listed as the active next experiment.
