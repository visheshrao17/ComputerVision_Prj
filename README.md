# Brain Tumor MRI Classification

A simple, reproducible implementation of **Disci, Gurcan & Soylu (2025)** for classifying MRI slices as glioma, meningioma, no tumor, or pituitary tumor. The notebook is the main deliverable for our capstone M2/M3 evaluation.

**Team:** Vishesh Rao, Pranay Vishwakarma, Manu Vahan
**Institution:** Sajjan Agarwal School of Technology, Rishihood University

- [Executed research notebook](notebooks/disci2025_reproduction.ipynb)
- [Original executed Colab notebook](notebooks/disci2025_colab_executed.ipynb)
- [M2/M3 presentation](deliverables/evaluation/Brain_MRI_M2_M3_Presentation_Final.pptx)
- [Evaluation report PDF](deliverables/evaluation/Brain_MRI_M2_M3_Report.pdf)
- [Editable evaluation report](deliverables/evaluation/Brain_MRI_M2_M3_Report.docx)
- [Reproduction report and hypothesis](docs/report/Disci2025_Reproduction_Addendum.md)
- [Original literature-survey report](docs/report/Brain_Tumor_MRI_Classification_Phase1_Report.pdf)
- [Approved implementation plan](docs/superpowers/plans/2026-10-05-disci-paper-reproduction.md)
- [Measured results](results/disci2025/summary.csv)

## Paper and correct dataset

R. Disci, F. Gurcan, and A. Soylu, “Advanced Brain Tumor Classification in MR Images Using Transfer Learning and Pre-Trained Deep CNN Models,” *Cancers*, **17(1), 121**, 2025. [DOI / paper](https://doi.org/10.3390/cancers17010121).

Use **version 1** of [Masoud Nickparvar's Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1). The current version 2 was updated in February 2026 and contains 7,200 images; it does **not** match the paper. The download command pins version 1 and preparation checks all eight class counts.

| Partition | Glioma | Meningioma | No tumor | Pituitary | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 1,321 | 1,339 | 1,595 | 1,457 | 5,712 |
| Testing | 300 | 306 | 405 | 300 | 1,311 |

The original data has **263 training images identical to test images**, 207 redundant training images, and 30 redundant test images, using decoded-pixel hashes. Perceptual near-matches remain explicitly flagged. No reliable patient IDs are supplied.

## What is implemented

**Selected baseline: Xception. Proposed smaller replacement: MobileNetV2.** The other four architectures remain supported in code, but are outside the agreed experiment scope. Each uses ImageNet weights and the paper's head: Flatten, Dropout(0.3), Dense(128, ReLU), Dropout(0.2), Dense(4, Softmax).

Paper settings: **128 × 128 × 3**, Adam **0.0001**, batch **20**, **5 epochs**, brightness/contrast factors **0.8–1.2**, and pixel scaling to **[0,1]**. Grayscale slices are blurred, cropped using the largest foreground contour, and replicated into three channels. Threshold, blur kernel, seed, and backbone trainability are disclosed assumptions because the paper leaves them unspecified. A foreground crop is not a validated tumor segmentation.

Two separate experiment protocols:

- `paper_based`: retain official folders; evaluate both clean images and a fixed seeded transformed test set, reflecting the paper's stated test augmentation. All backbone layers are trainable; final-epoch weights are used.
- `leakage_audited`: remove exact train/test duplicates and redundant training copies, reserve 15% of the remaining training pool for validation, augment only training images, and use clean evaluation. Near-matches remain unresolved unless a reviewed decisions CSV is supplied. This reduces known leakage but cannot establish patient independence.

**Hypothesis:** Under the same audited protocol, MobileNetV2's test macro-F1 will be at most 0.02 below Xception's while using fewer total model parameters. A single seed provides preliminary evidence, not a statistical non-inferiority claim.

**Metric correction:** The paper's headline Xception “weighted accuracy” **98.73%** is consistent with a combination of training and testing scores (inferred from its tables). Its Table 8 held-out test accuracy is **95.27%**. The implementation reports ordinary test accuracy, macro/weighted F1, balanced accuracy, recall, specificity, ROC-AUC, confusion matrices, and learning curves separately. Published and measured numbers are never conflated.

## Current measured status

Four Colab runs are verified from finite five-epoch histories, data manifests and recomputed test predictions. The selected **Xception baseline** achieves **93.67% clean test accuracy**, macro-F1 **0.9354** (paper accuracy: 95.27%); its fixed transformed view achieves **93.29%**. This is an approximate selected-model replication, not the paper's entire six-model benchmark.

For the proposed replacement, the matched audited comparison gives **Xception macro-F1 0.9176** versus **MobileNetV2 0.7372**. MobileNetV2 uses **80.52% fewer parameters**, but the **0.1804** macro-F1 gap exceeds the 0.02 margin: **the hypothesis is not supported in this single-seed experiment**. No further training is required for the agreed deliverable. The additional paper-based MobileNetV2 run is supplementary; the earlier Mac evidence remains separately preserved in `results/disci2025/local_mac`.

## M2 and M3 presentation

Use the report and PPT under `deliverables/evaluation`. Present the selected Xception baseline first, then the proposed MobileNetV2 replacement and its unsupported hypothesis outcome. This is an approximate selected-model replication, with the paper's unspecified choices disclosed.

For the demonstration, open the executed research notebook with `RUN_TRAINING=False`. Show the baseline table, Xception confusion matrix, saved-prediction verification and audited hypothesis comparison. Outputs are already stored; no GPU or new training is required to display them. The compact evidence includes predictions rather than the exported cloud checkpoints, so a live cloud-model inference demo is outside this package.

After the evaluation, investigate normalization and fine-tuning with separately labeled configurations, repeat across seeds and seek patient metadata or external test data. Preserve the current negative result as the original hypothesis test.

## Run locally

Use Python **3.11 or 3.12**, not Python 3.14.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mac.txt  # Apple Silicon; includes Metal
# On Linux/Colab use requirements.txt instead.

python scripts/reproduce_disci2025.py download
python scripts/reproduce_disci2025.py prepare
python scripts/reproduce_disci2025.py train --models Xception --resume
python scripts/reproduce_disci2025.py train --protocol leakage_audited --models MobileNetV2 Xception --resume
python scripts/reproduce_disci2025.py analyze
```

If images are already downloaded, **skip download** and pass `--data-dir /path/to/dataset`; retain its `Training/` and `Testing/` directories. Existing image folders are never overwritten. The download endpoint is pinned to version 1; if it requires Kaggle authentication in your environment, download that version manually and pass its path.

One model runs at a time. The first GPU graph compilation takes longer than later batches. Metal requires native GPU access; restricted execution environments may expose only the CPU. Colab GPU is the fallback for larger batches or repeated experiments. Bitwise-identical GPU results across hardware are not guaranteed.

For a fast integration check:

```bash
python scripts/reproduce_disci2025.py train --smoke --models MobileNetV2 --resume
python -m pytest -q
```

Smoke runs use 128 training/32 test images and one epoch, are stored separately, and are excluded from reproduced-results summaries. Interrupted model runs restart from scratch; `--resume` skips complete runs only when settings/data identities and required artifacts agree.

## Notebook / free Colab

Open `notebooks/disci2025_reproduction.ipynb`. It defaults to **analyzing saved results**, so tomorrow's presentation does not retrain the models. Set `RUN_TRAINING=True` only when starting new experiments. The notebook calls the same functions as the CLI.

`notebooks/disci2025_colab_executed.ipynb` preserves the user-downloaded Colab notebook exactly, including training logs and saved outputs from the completed cloud experiments. Its saved settings enable training; use the research notebook above for the presentation without retraining. `deliverables/disci2025_colab_run.ipynb` is the clean starter notebook for future runs.

For Colab: open `deliverables/disci2025_colab_run.ipynb` (training enabled), select **T4 GPU and runtime version 2026.07 (Python 3.12)** under Runtime > Change runtime type, and upload `deliverables/brain_mri_colab.zip` using the Files sidebar, then upload/open the notebook and run its setup cell. The bundle contains code and compact evidence, not MRI images or heavy model checkpoints; its downloader fetches the exact version 1 dataset. Changes in this local branch have not been pushed to GitHub, so cloning the old remote is insufficient.

## Bring Colab results back to the Mac

The GPU notebook downloads `disci2025_results.zip` after training. Extract it into a separate folder so the cloud evidence stays separate from local checkpoints:

```bash
unzip /path/to/disci2025_results.zip -d outputs/colab
python scripts/reproduce_disci2025.py analyze --output-dir outputs/colab/outputs/disci2025
```

Analysis accepts only complete runs with finite five-epoch histories, matching data manifests, and metrics recomputed from saved predictions. It copies small evidence files to `results/disci2025/runs`. Re-run the main notebook with `RUN_TRAINING=False` to refresh its tables, figures and paired hypothesis outcome. Update the report's measured-results section from that verified summary when importing any future experiment.

## Repository structure

```text
configs/disci2025.yaml           paper settings and disclosed assumptions
src/reproduction/               data, preprocessing, models, training, metrics
scripts/reproduce_disci2025.py   download / prepare / train / analyze
notebooks/                      executed notebook
results/disci2025/              source checksum, audit, compact measured-run evidence
outputs/disci2025/              local checkpoints, predictions, histories, figures (ignored)
docs/report/                    original PDF and reproduction addendum
tests/                         focused data/model/training/metric checks
```

Unused PyTorch/radiomics/Grad-CAM placeholder modules were removed. The original Phase 1 PDF records future work; EfficientNetB0, radiomics, masks, external validation, and patient-level CV are outside the selected baseline and proposed replacement experiment.

For audited near-duplicate review, supply `--duplicate-decisions reviewed.csv` with `train_path,test_path,decision` (`confirmed_duplicate`, `distinct`, or `unresolved`). Only confirmed training copies are excluded. Optional patient metadata must contain complete `path,patient_id` rows; grouped partitions are used only when those IDs are actually supplied.
