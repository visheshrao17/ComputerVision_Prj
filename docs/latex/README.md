# LaTeX research report

`main.tex` is a standard single-column A4 article with 11-point Times-style text,
numbered sections, equations, booktabs tables, labeled figures and BibTeX references.
It reports verified project results, preserves earlier negative outcomes and
explains the exploratory comparison limits.

## Overleaf

Upload `deliverables/brain_mri_latex_report.zip` as a new Overleaf project.
Set `main.tex` as the main document and use pdfLaTeX. Compile; BibTeX runs
through Overleaf's normal build workflow. No model training is required.

## Local compilation

From this directory, with a TeX installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively run `tectonic main.tex`. The project used Tectonic 0.17.0 for its
verified PDF. Auxiliary files and the compiler are not committed.
The delivered PDF is `deliverables/final_evaluation/Brain_MRI_Research_Report.pdf`.

## Figures and provenance

- The confusion matrix and learning curves are copied unchanged from the
  accepted proposed-model evidence under `results/training_extension/proposed`.
- `example_glioma.jpg` is `Testing/glioma/Te-glTr_0000.jpg` from the pinned dataset.
  Its decoded pixels match the image supplied by the user.
- `example_no_tumor.jpg` is `Testing/notumor/Te-noTr_0000.jpg`. The reported
  meningioma prediction is a retained error, matching saved Colab and checked
  Mac CPU predictions. No anatomical diagnosis is inferred from either image.
- Both are deterministic first-per-class examples already recorded in
  `results/training_extension/inference_verification.json`.
- The full Colab UI screenshot is not reproduced in the research paper.

Numbers are grounded in the accepted prediction CSVs, metrics and verification
records. Illustrative images do not supply an independent accuracy estimate.
