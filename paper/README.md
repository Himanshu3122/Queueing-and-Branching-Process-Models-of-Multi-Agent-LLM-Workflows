# Paper

Unmodified copies of the manuscript the repository implements:

- `main.tex`: journal version (IEEEtran journal class)
- `references.bib`: bibliography, verified against Crossref, publisher and arXiv records on 2026-10-23

Build with the real IEEEtran class and `IEEEtran.bst` (not bundled):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Equation, table and figure numbers used throughout the repository refer to this manuscript.
