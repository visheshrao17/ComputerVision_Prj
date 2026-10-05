# Disci et al. (2025) Paper Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline implementation, or superpowers:subagent-driven-development only if the user selects that execution method. Steps use checkbox (`- [ ]`) syntax for tracking. Do not build until the user approves this document.

**Goal:** Produce an executed, auditable notebook implementing the six CNN experiments in Disci et al. and a clearly separated evaluation extension grounded in the existing literature-survey report.

**Architecture:** Use a dedicated `src/reproduction/` package with TensorFlow/Keras applications, configuration-driven preprocessing, training, and evaluation. A notebook and one CLI call the same package. Preserve the Phase 1 report and existing PyTorch roadmap; distinguish published numbers, paper-based runs, and evaluation extensions in every results table.

**Tech Stack:** Python 3.11 target, TensorFlow/Keras, OpenCV, NumPy, pandas, scikit-learn, matplotlib, PyYAML, Jupyter, nbformat/nbclient, pytest. Resolve and record compatible dependency versions during approved implementation.

**Spec:** The design brief and experimental contract in this document, below.

**Status:** Approved by the user on 5 October 2026; implementation in progress. The user subsequently prioritized simple code, lower resource use, repository cleanup, and deliverables for the 6 October M2/M3 evaluation.

**Approved execution amendments:** Pin dataset version 1 (version 2 now has 7,200 images). Train MobileNetV2/Xception first, followed by their audited comparison and then the remaining four paper models. Replace unused PyTorch dependencies/stubs and the inactive default configuration with one working reproduction pipeline; retain the original PDF and this plan. Use `requirements.txt` plus optional `requirements-mac.txt` rather than a second independent dependency tree. Keep a Colab bundle as the fallback, and mark unfinished experiments explicitly rather than substituting smoke scores.

## Design brief and source review

The user requests reading the report and README, planning implementation of the cited paper, and obtaining approval before building. A PPT is no longer requested. The target remains a notebook with actual reproduced results, with the report's introduction and research hypothesis supplying project context.

Sources read:

- `README.md`, including methodology, evaluation protocol, dataset instructions, roadmap, and references.
- `docs/report/Brain_Tumor_MRI_Classification_Phase1_Report.pdf`, all four pages via text extraction. This was a content review, not a layout review.
- `configs/default.yaml`, dependencies, script/model/data/training stubs, and `tests/test_splitting.py`.
- Full paper, publisher-supplied XML retrieved through Europe PMC: [Disci, R.; Gurcan, F.; Soylu, A. Cancers 2025, 17(1), 121](https://doi.org/10.3390/cancers17010121); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11719945/); [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11719945/fullTextXML).
- [Keras Applications documentation](https://keras.io/api/applications/) to confirm availability of all six pretrained backbones.

Repository findings:

- Python modules and scripts contain only docstrings; there is no working data or training pipeline.
- `notebooks/` and the data directories contain placeholders only. There are no MRI images, checkpoints, or measured results.
- The report already has an introduction. Its focus is leakage-resistant evaluation, macro-F1, lightweight models, preprocessing ablations, and conditional quantitative explainability.
- The report incorrectly includes EfficientNet variants among the models in Disci's six-model comparison. EfficientNetB0 is a separate proposed project model.
- The README links a different Kaggle collection and includes an incorrect Disci DOI in its dataset section. The author's initials in the repository citation also differ from the published record.

### Scope alternatives and recommendation

| Approach | Advantage | Limitation |
| --- | --- | --- |
| Six-model paper reproduction only | Directly addresses the assignment | Does not examine the report's evaluation concerns |
| Six-model reproduction plus focused evaluation extension **(recommended)** | Addresses the paper and tests a bounded hypothesis from the report | Requires two additional training runs and a duplicate audit |
| Implement the entire Phase 1 roadmap | Covers crops, radiomics, explainability, robustness, and external validation | Much larger project; some experiments need unavailable masks, subject IDs, or external data |

Recommended approval scope: six paper-based runs and two evaluation-extension runs, an executed notebook, saved experiment artifacts, README corrections, and a reproduction addendum in the report folder. Preserve the original PDF. No PPT, deployment, full radiomics pipeline, segmentation model, or full Phase 1 implementation.

## Experimental contract

### 1. Verified paper settings

| Item | Paper specification | Evidence |
| --- | --- | --- |
| Dataset | Masoud Nickparvar Brain Tumor MRI Dataset | Reference 23; Section 2.1 |
| Dataset URL | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset | Reference 23 |
| Official split | 5,712 training images, 1,311 testing images | Section 2.1 |
| Classes | Glioma, meningioma, no tumor, pituitary | Sections 2.1, 3 |
| Training counts | 1,321 / 1,339 / 1,595 / 1,457 in that class order | Section 2.1 |
| Testing counts | 300 / 306 / 405 / 300 in that class order | Section 2.1 |
| Image processing | Grayscale, Gaussian blur, binary threshold, largest-contour crop, resize | Section 2.2 |
| Model input | 128 x 128 x 3 | Sections 2.2, 2.5; Table 1 |
| Augmentation | Brightness and contrast factors each in [0.8, 1.2] | Section 2.3 |
| Normalization | Divide intensity by 255 | Section 2.3 |
| Models | Xception, MobileNetV2, InceptionV3, ResNet50, VGG16, DenseNet121 | Section 2.4 |
| Head | Flatten -> Dropout(0.3) -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(4, Softmax) | Table 1 |
| Optimizer / learning rate | Adam / 0.0001 | Section 2.5 |
| Loss | Sparse categorical cross-entropy | Section 2.5 |
| Batch size / epochs | 20 / 5 | Section 2.5 |

The paper describes augmentation of testing images too. This is different from the report's proposed deterministic evaluation and must be disclosed, rather than silently changed.

### 2. Published comparison targets

These are source values, not newly reproduced results. Fractions are shown below; notebook tables also show percentages.

| Model | Training accuracy | Testing accuracy (Table 8) | Testing macro-F1 (Tables 2-7) | Testing weighted-F1 (Tables 2-7) | Paper weighted accuracy average |
| --- | --- | --- | --- | --- | --- |
| Xception | 0.9952 | 0.9527 | 0.9491 | 0.9529 | 0.9873 |
| MobileNetV2 | 0.9898 | 0.9451 | 0.9411 | 0.9457 | 0.9815 |
| InceptionV3 | 0.9810 | 0.9451 | 0.9409 | 0.9452 | 0.9743 |
| ResNet50 | 0.9897 | 0.9062 | 0.9010 | 0.9045 | 0.9741 |
| VGG16 | 0.9721 | 0.9504 | 0.9470 | 0.9478 | 0.9680 |
| DenseNet121 | 0.9652 | 0.9285 | 0.9220 | 0.9265 | 0.9583 |

Interpretation: the paper's weighted accuracy average is consistent with `(5712 * training_accuracy + 1311 * testing_accuracy) / 7023`. For Xception this yields 0.987266, rounding to 0.9873. This is an inferred reconstruction of the statistic, not an explicit formula supplied in the paper. It includes training performance and is not held-out accuracy. Its macro accuracy average likewise matches the mean of training and testing accuracy, not class-balanced accuracy.

Some tables are internally inconsistent: for example, VGG16's classification-report weighted recall is 0.9476 while Table 8's testing accuracy is 0.9504. Preserve the different source values and flag discrepancies. Never force measured results to match them.

### 3. Disclosed implementation assumptions

The paper does not fully specify the random seed, blur kernel, threshold value, exact crop geometry, backbone trainability, validation policy, or augmentation sampling order. The reviewed text supplies no executable training code resolving these choices. Therefore this is a paper-based methodological reproduction, not a guaranteed exact numerical replication.

Proposed choices fixed before test evaluation:

- Seed 42, deterministic operations where supported, float32 training.
- Gaussian kernel 5 x 5, sigma 0; threshold 45 on the 0-255 blurred grayscale image; largest external contour's bounding rectangle, no extra margin.
- Thresholding supplies a crop mask; retain grayscale intensities inside the rectangular crop. This interpretation is an assumption because the paper does not clearly state whether its final classifier sees binary or grayscale pixels.
- Fall back to the full slice for no valid contour; log every fallback. Reject unreadable files with their paths; do not silently skip them.
- Replicate the resulting grayscale channel three times; resize with bilinear interpolation to 128 x 128.
- Sample independent multiplicative brightness and mean-centered contrast factors in [0.8, 1.2], clip intensities, then divide by 255. This operational definition of contrast is an assumption.
- Use ImageNet weights, `include_top=False`, and all backbone layers trainable from epoch one. No warm-up, class weighting, early stopping, or learning-rate search in the paper-based experiment.
- Use the paper's [0,1] scaling uniformly. Do not silently substitute model-specific ImageNet preprocessing; such a substitution would be a separately labeled experiment.
- Save final-epoch weights after five epochs. The paper-based run uses all official training images and has no validation-driven selection.
- Do not call this crop a validated tumor ROI. The largest bright contour can capture brain foreground and also exists in no-tumor images.

### 4. Two named protocols

**`paper_based`**: preserve the official Training/Testing folders and audit their overlap without deleting images. Train the six models for five epochs each. After freezing each model's weights, evaluate (a) clean deterministic test images and (b) one seeded, fixed brightness/contrast transformation per test image. Use transformed test images for the paper-protocol comparison and show clean test results alongside it. Do not average predictions or call this test-time augmentation ensembling. Compute the combined training/testing statistic only as a secondary paper-comparison column, using inference-mode training-set evaluation with the same fixed transformation policy.

**`leakage_audited`**: run Xception and MobileNetV2 with the same model/training settings but training-only augmentation and clean validation/testing. Keep the official test set fixed. Remove training images whose decoded pixels exactly match a test image; collapse exact duplicate training images; reject conflicting labels for exact duplicates. Generate perceptual-hash near-duplicate candidates (64-bit pHash, Hamming distance <= 4), inspect candidate pairs, and record removal decisions for confirmed duplicates without inferring patient IDs. Keep unresolved candidates explicitly flagged. For confirmed cross-split pairs, remove the training copy. Split the retained training pool into 85% train / 15% validation, stratified at image level when no patient metadata exists; use grouped allocation if reliable supplied subject metadata exists. Save the allocation and report actual fractions when grouping prevents an exact ratio. Use five epochs and final-epoch weights; validation is diagnostic and does not select a checkpoint.

This second protocol changes the training pool and augmentation policy, so any performance difference is the combined protocol effect; it cannot be attributed solely to leakage removal. No claim of patient-independent or clinical validation is permitted without supporting data. Patient-level five-fold CV and external testing remain later work.

### 5. Hypothesis from the report

The report motivates a balance of predictive performance, computational cost, and reliable evaluation, but does not specify a numerical non-inferiority margin. Proposed operational hypothesis for approval:

> Under the same leakage-audited protocol, MobileNetV2 will achieve test macro-F1 no more than 0.02 below Xception while using fewer total model parameters.

Measure total parameters including the shared classifier head. Also report training time and warmed-up batch-1 median/p95 model inference latency on the same hardware, separately from preprocessing and weight download. A single-seed result is preliminary evidence, not a statistical non-inferiority claim. Three-seed replication and patient-level evaluation are optional follow-up experiments, not prerequisites hidden inside the initial scope.

The notebook and report addendum include a brief introduction connecting four-class MRI classification, transfer learning, class-sensitive metrics, leakage, and this hypothesis. They preserve the literature-survey report's central motivation while correcting the interpretation of the paper's headline score.

## Global Constraints

- No PPT. Preserve the existing Phase 1 PDF; retire unused scaffolding/configuration under the user's approved cleanup instruction.
- Implement six paper backbones at 128 x 128 x 3 with the Table 1 head, Adam 0.0001, batch size 20, five epochs, seed 42.
- Canonical class order: `glioma`, `meningioma`, `no_tumor`, `pituitary`; map Kaggle `notumor` and existing `no_tumor` explicitly.
- Separate `paper_based` and `leakage_audited` run directories and results; never merge their scores.
- Never use held-out test outcomes for hyperparameter tuning or checkpoint selection.
- Preserve official raw images; write manifests/derived data separately and record exclusions.
- Never substitute synthetic tests, untrained predictions, or copied paper scores for real MRI results.
- Record package versions, seed, hardware, data fingerprint, model trainability, and complete resolved configuration per run.
- Keep large datasets and checkpoints ignored by Git. Keep notebook outputs and compact results summaries reviewable.
- Use the same package from the notebook and CLI; no second training implementation in notebook cells.

## Review Focus

1. Wrong dataset version or class mapping: count mismatch must stop paper-comparison mode with an actionable error.
2. Empty contours or corrupt images: valid full-slice fallback must be logged; corrupt files must stop preparation.
3. Cross-split duplicates or missing patient IDs: auditing must disclose remaining risks and never invent patient independence.
4. Missing predicted/true classes: metric computation must keep four-class confusion-matrix shape and return undefined AUC explicitly when necessary.
5. Interrupted runs or unavailable GPU/weights: resume must verify data/config identity; reduced smoke runs must never be promoted to full results.

## File responsibilities

| File | Responsibility |
| --- | --- |
| `requirements-reproduction.txt` | Reproduction runtime and notebook/test dependencies, independent of the PyTorch dependency file |
| `configs/disci2025.yaml` | Verified paper settings, assumptions, protocol selection, model list, output paths |
| `src/reproduction/config.py` | Configuration validation and environment/run metadata |
| `src/reproduction/data.py` | Dataset inventory, label mapping, hashing, audit, protocol manifests |
| `src/reproduction/preprocessing.py` | Grayscale/crop/resize, augmentation, `tf.data` datasets |
| `src/reproduction/models.py` | Six Keras backbones and paper classifier head |
| `src/reproduction/training.py` | Sequential model training, final checkpoint, history, resumable run status |
| `src/reproduction/evaluation.py` | Predictions, metrics, paper comparisons, timing, plots |
| `scripts/reproduce_disci2025.py` | Prepare/train/evaluate CLI with explicit protocol and model controls |
| `notebooks/disci2025_reproduction.ipynb` | Introduction, hypothesis, methods, execution, plots, comparisons, limitations |
| `docs/report/Disci2025_Reproduction_Addendum.md` | Introduction, hypothesis, experiment methods, actual results, discrepancies, limitations |
| `README.md` | Correct citation/dataset links and add working reproduction commands |
| `results/disci2025/` | Compact summary CSV/JSON and provenance tracked for review |
| `outputs/disci2025/` | Ignored per-run checkpoints, prediction CSVs, histories, audit details, and figures |

Package `__init__.py` and relevant tests accompany these modules. Existing PyTorch stub files are not rewritten as part of this scoped reproduction.

## Implementation tasks

### Task 1: Establish the experiment configuration and runtime

**Files:** create `requirements-reproduction.txt`, `configs/disci2025.yaml`, `src/reproduction/{__init__,config}.py`, `tests/test_reproduction_config.py`.

**Interfaces:** `load_config(path: Path) -> dict`; `validate_config(config: dict) -> None`; `environment_metadata() -> dict`.

- [ ] Write failing tests that reject unknown model/protocol names, reject a class order different from the canonical order, and verify paper-mode defaults: size 128, batch 20, epochs 5, learning rate 0.0001, seed 42.
- [ ] Run `python -m pytest tests/test_reproduction_config.py -q`; confirm failures identify missing configuration implementation.
- [ ] Implement validated configuration and metadata collection; resolve compatible packages in an isolated local environment and document a GPU notebook runtime option. Keep Keras cache inside the workspace for local execution.
- [ ] Run the configuration tests and import TensorFlow/Keras; record versions and available devices. Commit the task if commits are part of the approved execution workflow.

### Task 2: Prepare dataset manifests and leakage audit

**Files:** create `src/reproduction/data.py`, `tests/test_reproduction_data.py`.

**Interfaces:** `inventory_dataset(root: Path) -> pd.DataFrame`; `audit_duplicates(records: pd.DataFrame) -> dict`; `build_manifests(records: pd.DataFrame, protocol: str, seed: int, patient_metadata: Path | None = None, duplicate_decisions: Path | None = None) -> dict[str, pd.DataFrame]`.

Manifest fields: relative path, class name/index, original partition, decoded-pixel hash, perceptual hash, optional supplied patient ID, assigned partition. Audit decisions and data fingerprints are persisted separately.

- [ ] Write tests for exact paper counts, `notumor` mapping, decoded-pixel duplicates despite different file encodings, conflicting labels, supplied-patient disjointness, and reproducible allocations. Use tiny generated images only for unit tests.
- [ ] Run `python -m pytest tests/test_reproduction_data.py -q`; confirm expected failures before implementation.
- [ ] Implement inventory/audit/manifests and actionable errors for corrupt images, invalid layouts, and count mismatches. Preserve official partitions for `paper_based`; implement the declared exclusions and split for `leakage_audited`.
- [ ] Run tests. Download the exact Kaggle collection after build approval or accept its local path, verify all eight class/partition counts, and save the audit. Resolve near-duplicate candidates before calling the second protocol audited; disclose unresolved candidates if any remain.

### Task 3: Implement preprocessing and six-model construction

**Files:** create `src/reproduction/preprocessing.py`, `src/reproduction/models.py`, `tests/test_reproduction_preprocessing.py`, `tests/test_reproduction_models.py`.

**Interfaces:** `preprocess_image(path: Path, config: dict) -> tuple[np.ndarray, dict]`; `augment_image(image: tf.Tensor, seed: tuple[int, int]) -> tf.Tensor`; `make_dataset(manifest: pd.DataFrame, config: dict, split: str, evaluation_view: str = 'clean') -> tf.data.Dataset`; `build_model(name: str, config: dict, weights: str | None = 'imagenet') -> keras.Model`.

- [ ] Write failing tests for empty-contour fallback, corrupt-file rejection, three equal grayscale channels, 128 x 128 x 3 shape, augmentation bounds, stable seeded evaluation, and unchanged clean evaluation. For each backbone with `weights=None`, assert four softmax outputs and exact head/dropout structure.
- [ ] Run the two test modules; confirm expected failures.
- [ ] Implement the declared preprocessing and augmentation, preserving manifest order during evaluation and separating normalization from cropping. Implement all six backbones with no top and the Table 1 head; expose backbone trainability in metadata.
- [ ] Run tests without network downloads. Then smoke-load ImageNet weights and perform one inference batch per model, checking finite probabilities summing to one. An unsupported input shape must fail explicitly, never silently resize to 224 or 299.

### Task 4: Implement training and resumable CLI execution

**Files:** create `src/reproduction/training.py`, `scripts/reproduce_disci2025.py`, `tests/test_reproduction_training.py`.

**Interfaces:** `train_model(name: str, manifests: dict[str, pd.DataFrame], config: dict, run_dir: Path) -> Path`; CLI subcommands `prepare`, `train`, `evaluate`, with `--config`, `--protocol`, `--models`, and `--resume`.

- [ ] Write failing integration tests using a tiny four-class fixture and a tiny injected test model: training changes weights, reload reproduces predictions, resume rejects mismatched configuration/data fingerprints, and paper-mode checkpoint selection does not consume validation/test metrics.
- [ ] Run `python -m pytest tests/test_reproduction_training.py -q`; confirm expected failures.
- [ ] Implement Adam/sparse-cross-entropy training with a retained final partial batch, final-epoch checkpoints, history JSON/CSV, incremental run status, and sequential model cleanup. Save final status only after checkpoint and metadata validation. Resume skips verified complete model runs and restarts an incomplete model; exact mid-epoch resume is out of scope.
- [ ] Run tests and a one-epoch real-image smoke run. Save smoke results under a distinct smoke directory; do not include them in reproduction tables. Measure runtime to estimate the full eight-run workload before launching it.

### Task 5: Implement metrics, published comparisons, and resource measurements

**Files:** create `src/reproduction/evaluation.py`, `tests/test_reproduction_evaluation.py`.

**Interfaces:** `evaluate_model(checkpoint: Path, manifest: pd.DataFrame, config: dict, evaluation_view: str) -> tuple[pd.DataFrame, dict]`; `classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict`; `compare_with_paper(measured: pd.DataFrame) -> pd.DataFrame`.

- [ ] Write failing tests against hand-calculated four-class confusion matrices: ordinary accuracy, macro/weighted F1, balanced accuracy, per-class recall/specificity, and support. Test that absent true classes produce explicitly undefined AUC and that published and measured columns remain separate. Assert the inferred combined statistic rounds to 0.9873 for the paper's Xception inputs.
- [ ] Run `python -m pytest tests/test_reproduction_evaluation.py -q`; confirm expected failures.
- [ ] Implement stable-order prediction export, 4 x 4 confusion matrices, one-vs-rest ROC-AUC, finite-probability validation, learning curves, and comparison tables. Preserve inconsistent paper values with source labels. Implement parameter counts and same-device warmed-up latency measurement.
- [ ] Run tests. Recompute metrics from saved predictions and assert they match exported summary values; label differences in percentage points and never interpret the combined statistic as test performance.

### Task 6: Author notebook and execute real experiments

**Files:** create `notebooks/disci2025_reproduction.ipynb`; generate `results/disci2025/{summary.csv,provenance.json}` and ignored run artifacts; create `tests/test_reproduction_notebook.py` for structural/import checks.

- [ ] Build notebook sections: introduction; citation; report-derived hypothesis; verified methods and assumptions; environment; dataset/audit; configurations; training; learning curves; clean/transformed test comparisons; leakage-audited pair; efficiency; hypothesis outcome; limitations and conclusions.
- [ ] Validate notebook structure and imports; confirm notebook cells invoke the shared package rather than duplicate its algorithms. Include an explicit smoke/full switch and a configurable dataset path.
- [ ] Run six `paper_based` models for five epochs each, then Xception and MobileNetV2 under `leakage_audited`. Full workload is 40 model-epochs. Record actual hardware/time; no credible runtime estimate is available until Task 4's smoke measurement.
- [ ] Evaluate final frozen checkpoints, export predictions and summaries, and execute the notebook in a fresh kernel. Support analysis of verified saved runs so re-rendering the notebook does not unnecessarily retrain eight models.
- [ ] Check every executed cell for errors, inspect plots, reconcile all test support counts and metrics against saved predictions, and classify the hypothesis as supported/not supported by this single-seed experiment. If data, weights, or compute prevents full runs, state exactly which runs remain incomplete; do not call results reproduced.

### Task 7: Document results and verify the submission

**Files:** modify `README.md`; create `docs/report/Disci2025_Reproduction_Addendum.md`.

- [ ] Correct the Disci citation to R. Disci, F. Gurcan, A. Soylu; DOI `10.3390/cancers17010121`; link the exact Kaggle dataset; clarify the six-model comparison and distinguish it from EfficientNetB0 roadmap work.
- [ ] Write the report addendum with the introduction, operational hypothesis, methods, actual comparison table, evaluation audit, computational measurements, assumptions, and source-table discrepancies. Reference the original PDF and preserve its contents.
- [ ] Document tested setup and CLI/notebook commands, full versus smoke behavior, saved-run analysis, dataset layout, missing-patient-ID limitation, and optional next experiments.
- [ ] Run `python -m pytest tests/test_reproduction_*.py -q`, execute the notebook fresh against saved full runs, run `git diff --check`, and verify only intended files changed. Review implementation against this plan before reporting completion.

## Implementation progress (5 October 2026)

Configuration, pinned dataset download, duplicate/crop auditing, six backbones, training/evaluation, CLI, and presentation notebook are implemented. Tests are consolidated in `tests/test_reproduction.py`, `test_training.py`, `test_download.py`, and `test_artifacts.py`; 31 tests pass. CLI commands are `download`, `prepare`, `train`, and `analyze`; training includes checkpoint evaluation automatically. Compact validated evidence is exported under `results/disci2025/runs` so saved-result analysis works without heavy checkpoints.

### Approved scope correction and completed cloud evidence

The user clarified that baseline formation must precede proposed changes; the four additional paper models are outside the revised scope. Replicate paper-based Xception first, then compare audited Xception with the proposed MobileNetV2 replacement. Existing paper-based MobileNetV2 evidence is supplementary. Future notebook training follows this three-run order and has no all-six-model training switch.

The user exported `disci2025_results.zip` from Colab. All four full runs passed history/manifest/fingerprint/prediction-metric verification: Xception and MobileNetV2 under both protocols. Canonical compact evidence now uses cloud results; the earlier Mac run is separately preserved. Xception baseline clean accuracy is 93.67% (transformed 93.29%), versus the paper's 95.27%. Audited macro-F1 is 0.9176 versus 0.7372, gap 0.1804. MobileNetV2 uses 80.52% fewer parameters; the joint hypothesis is not supported. Both audited runs share settings, ordered data fingerprints and environment metadata. No cloud checkpoints were exported; saved predictions support metrics verification, not repeat inference. No further training is required for the revised scope.

The original six-model acceptance criteria below describe the superseded full-comparison scope. Revised acceptance is verified selected Xception baseline, matched audited comparison, clean notebook/report evidence and explicit numerical-reproduction and single-seed limitations.

## Acceptance criteria

- All six named architectures train and evaluate using the documented paper settings and assumptions.
- The executed notebook includes real MRI results, learning curves, confusion matrices, class-sensitive metrics, published comparisons, and provenance.
- Both evaluation views and both protocols are visibly labeled; source numbers and measured numbers cannot be confused.
- Dataset counts and excluded-image decisions are auditable; no unsupported patient-level claim is made.
- The focused MobileNetV2/Xception hypothesis is evaluated with actual macro-F1 and total parameter counts, with single-seed limitations disclosed.
- README commands work, compact results are reviewable, and the report addendum includes introduction/hypothesis/results without overwriting the original PDF.

## Approval and execution boundary

Approval is requested for the recommended scope, Keras implementation, declared assumptions, focused two-model extension, and operational hypothesis. Once approved, implement inline in this session unless the user explicitly requests delegated execution. No dependency installation, dataset download, code implementation, or training will occur before that approval. Dataset/weight access and the chosen local or GPU-notebook runtime must be working before promising an executed reproduction.

Optional subsequent work, requiring a separately agreed scope: EfficientNetB0/simple-CNN baselines, class-weighting ablations, validated tumor-centered crops, radiomics/fusion, Grad-CAM with annotation-based audit, perturbation robustness, three-seed replication, grouped five-fold CV, and external-dataset validation.
