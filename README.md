# Brain Tumor MRI Classification

A simple, reproducible implementation of **Disci, Gurcan & Soylu (2025)** for classifying MRI slices as glioma, meningioma, no tumor, or pituitary tumor. The notebooks cover baseline replication and the final capstone experiments.

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
- [Final evaluation report PDF](deliverables/final_evaluation/Brain_MRI_Final_Report.pdf)
- [Final evaluation presentation](deliverables/final_evaluation/Brain_MRI_Final_Presentation.pptx)

## Research report in LaTeX

- [Compiled research report](deliverables/final_evaluation/Brain_MRI_Research_Report.pdf)
- [LaTeX source and build instructions](docs/latex/README.md)
- [Overleaf-ready source ZIP](deliverables/brain_mri_latex_report.zip)

The report includes verified results, numbered tables and figures, references,
and a correct prediction plus a retained error using traced dataset examples.
The four Colab/reproduction notebooks are tracked under `notebooks/`.
Representative report images are included as figure assets; the complete raw
dataset and model checkpoints remain outside Git.

The final presentation has 16 slides and follows the previous milestone's
format: selected baseline and original outcome, work continued, final verified
results, prediction examples and remaining limitations. Evaluation dates and
command-line details are omitted from the slides. The previous milestone deck
is preserved separately under `deliverables/evaluation`.

## Final evaluation continuation

**Completed final extension (2 November):** both new Colab runs are imported and
verified. The compact two-stage MobileNetV2 reaches **92.68% accuracy**, **0.9256
macro-F1** and **2,422,468 parameters**. The saved audited Xception reference is
91.76%, 0.9176 and 25,056,428 parameters. The proposed model uses **90.33% fewer
parameters** and meets the exploratory improvement/efficiency target in this run.
The original five-epoch hypothesis remains unsupported.

| Audited experiment | Test accuracy | Macro-F1 | Parameters |
| --- | ---: | ---: | ---: |
| Historical Xception | 91.76% | 0.9176 | 25,056,428 |
| Normalized MobileNetV2 five epochs | 81.92% | 0.8062 | 4,880,068 |
| A longer-training control | 88.71% | 0.8841 | 4,880,068 |
| B compact two-stage proposal | **92.68%** | **0.9256** | **2,422,468** |

Use the [executed final notebook](notebooks/training_extension_colab.ipynb),
[upload bundle](deliverables/brain_mri_training_source.zip),
[measured comparison](results/training_extension/comparison.csv) and
[experiment plan/code map](docs/training_extension_plan.md).

A keeps Flatten, completes 19 epochs and selects epoch 15. B uses Global Average
Pooling, five frozen-backbone warm-up epochs and fifteen low-rate fine-tuning
epochs with frozen BatchNorm; it selects epoch 17. Both select the lowest
validation-loss checkpoint. B also has lower validation loss and is used for the
demo. Training takes about 10.72 minutes for A and 6.37 minutes for B.

This is an exploratory combined-method comparison on a reused test set, not a
new algorithm, statistical superiority or a result exceeding the published paper
under its protocol. Xception uses a different training policy. New evidence is
separate in `outputs/training_extension` and `results/training_extension`.
All 1,311 prediction metrics are recomputed; saved checkpoints load on the Mac.
B gets three of four deterministic local examples correct; the no-tumor error
matches Colab. Softmax scores differ across platforms. See the verification JSON
files for the full checks.

For the presentation keep `RUN_TRAINING=False`; no new training is needed.
Restore the complete model ZIP to run prediction. Raw images and checkpoints
are excluded from Git.

```bash
python scripts/training_extension.py analyze
python scripts/training_extension.py predict --experiment proposed --image /path/to/mri.jpg
# Optional new runs, not needed for the demo:
python scripts/training_extension.py train --resume
```

### Earlier normalization study

Before the final training extension, we completed one controlled normalization experiment and a fresh-image prediction demo. The completed October results below stay preserved, including the original unsupported hypothesis. [Continuation plan and code explanation](docs/final_evaluation_plan.md).

- [New Colab notebook](notebooks/final_project_colab.ipynb)
- [Upload bundle for Colab](deliverables/brain_mri_final_source.zip)
- [Experiment configuration](configs/mobilenet_imagenet.yaml)

The new run changes only MobileNetV2 input normalization from `[0,1]` to the documented ImageNet `[-1,1]` range. It reuses the same audited split, architecture, full fine-tuning and five epochs. The full Colab run and exported checkpoint are now verified. Reusing the previously evaluated test set limits this to an exploratory follow-up.

| Audited model | Test accuracy | Macro-F1 |
| --- | ---: | ---: |
| Xception control | 91.76% | 0.9176 |
| Original MobileNetV2 | 74.29% | 0.7372 |
| Normalized MobileNetV2 | **81.92%** | **0.8062** |

The observed gain is **7.63 accuracy percentage points** and **0.0690 macro-F1** over original MobileNetV2. It still uses 80.52% fewer total parameters than Xception. The remaining **0.1114** macro-F1 gap exceeds the **0.02** margin, so the joint exploratory target is **not met**. The original unsupported hypothesis remains unchanged. The recorded runtime platform/kernel differs, so the improvement is not attributed solely to normalization.

The exported weights load on the Mac. Four deterministic first-per-class test examples have matching true/predicted labels and Colab classes. CPU probabilities differ by up to about 0.0302; no CPU/GPU probability equivalence or separately reproduced full CPU test accuracy is claimed. See [final report source](docs/report/Final_Evaluation_Report.md) and [inference verification](results/final_project/inference_verification.json).

Open the new notebook in Colab, choose **T4 GPU and runtime 2026.07 / Python 3.12**, upload the source ZIP in the Files sidebar, then run all cells. The last cell downloads `brain_mri_final_demo.zip` containing code, measured evidence **and model weights**, without raw MRI images. The notebook also demonstrates prediction on an existing test image or an uploaded image.

The normalization notebook contains executed results and a fresh-image example; it is a historical fallback; use the completed training-extension notebook for the final demo. Locally, `RUN_TRAINING` defaults to false. Keep the returned model ZIP: weights are stored under ignored `outputs/final_project` and are not in Git. Only four test images were restored locally for inference checks, not a full training dataset; the new extension uses the fresh `data/training_extension` directory.

Equivalent local commands after installing the project requirements:

```bash
python scripts/final_project.py train --resume
python scripts/final_project.py analyze
python scripts/final_project.py predict --image /path/to/mri.jpg
```

The new run is saved under `outputs/final_project`; its compact evidence and comparison go to `results/final_project`. Keep the downloaded model bundle before Colab disconnects. Extract that returned bundle into a separate folder on the Mac to use the same prediction command without retraining. Prediction needs `model.keras` and matching run metadata; the old compact result ZIP alone has no weights. Training preprocessing is reused, and model-specific normalization lives inside the saved model.

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

## Original October results

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

For the original baseline notebook: open `deliverables/disci2025_colab_run.ipynb` (training enabled), select **T4 GPU and runtime version 2026.07 (Python 3.12)** under Runtime > Change runtime type, and upload `deliverables/brain_mri_colab.zip` using the Files sidebar, then upload/open the notebook and run its setup cell. The bundle contains code and compact evidence, not MRI images or heavy model checkpoints; its downloader fetches the exact version 1 dataset. Baseline work is merged into GitHub. Use the final-evaluation notebook above for the new single-model extension.

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
