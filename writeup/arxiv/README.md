# arXiv submission package

## Files
- `main.tex` — the paper
- `references.bib` — bibliography

## How to compile

```bash
cd writeup/arxiv
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Produces `main.pdf`. Check it before submitting.

## How to submit to arXiv

1. **Get endorsement first.** First-time submitters in `cs.LG` need endorsement from an existing arXiv author. See `../endorsement_email_draft.md` for a template.

2. Once endorsed:
   - Go to <https://arxiv.org/submit>
   - Create a new submission
   - Category: `cs.LG` (primary), `cs.AI` (secondary), optionally `cs.CL`
   - Upload `main.tex` and `references.bib` as a single archive (or upload separately)
   - arXiv will compile and show you the PDF
   - Review carefully, then submit
   - Goes live in ~24 hours; Google Scholar indexes within 2-4 weeks

3. **License:** the default arXiv license is fine. Alternatively, choose CC-BY 4.0 for maximum reuse.

## Things to double-check before submitting

- [ ] Author name spelled correctly
- [ ] Email address correct
- [ ] GitHub URL works
- [ ] All references resolve (no `[?]` in PDF)
- [ ] Tables fit on the page
- [ ] No leftover TODO/XXX comments in the source
- [ ] Abstract is under 1920 characters (arXiv limit)
- [ ] Title is under 240 characters
