# Paper Draft

This directory contains a first LaTeX-only workshop-paper draft for the current
ANN sparse-attention prototype.

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
- scaling framed as an operation-count proxy, not production latency.
