# Brain Tumor MRI Classification: Paper Reproduction

**Team:** Vishesh Rao, Pranay Vishwakarma, Manu Vahan
**Purpose:** Capstone M2/M3 evaluation, 6 October 2026
**Source report:** [Phase 1 literature survey](Brain_Tumor_MRI_Classification_Phase1_Report.pdf)

## Introduction

This project studies four-class classification of brain MRI slices: glioma, meningioma, pituitary tumor, and no tumor. Transfer learning adapts features learned from ImageNet to a smaller MRI dataset and offers a practical starting point when training resources are limited. Our literature survey motivates evaluating predictive performance together with computational cost and the reliability of the evaluation protocol.

We replicate Xception, the best-performing model reported by Disci, Gurcan, and Soylu (2025), using the original version of its public dataset and the reported five-epoch training settings. We then test a proposed replacement with the smaller MobileNetV2 under matched leakage-audited conditions. This is a selected-model replication, not a reproduction of the paper's entire six-model benchmark. This provides a reproducible baseline before attempting the broader preprocessing, radiomics, or explainability extensions proposed in Phase 1. Ordinary held-out accuracy, macro-F1, per-class performance, and saved predictions form the principal evidence.

Evaluation reliability remains central. Identical images can occur in both training and testing, and slices from the same patient may be correlated. We therefore audit decoded-pixel duplicates and compare the original paper partition with a separate experiment that removes known exact overlap. The available collection does not expose reliable patient IDs, so these experiments measure image-level performance and cannot establish patient-independent generalization. Foreground cropping is also not assumed to identify a tumor without annotations.

## Research hypothesis

The report's resource-conscious motivation is operationalized as:

> Under the same leakage-audited protocol, MobileNetV2 achieves test macro-F1 no more than 0.02 below Xception while using fewer total model parameters.

The margin is our proposed experimental criterion, not a value stated in Disci et al. It is evaluated with one seed initially; the result is preliminary evidence rather than a statistical non-inferiority conclusion. Parameter counts include the classifier head, and runtime comparisons use the same hardware.

## Paper selection and reproducibility

**Selected reference:** R. Disci, F. Gurcan, and A. Soylu, “Advanced Brain Tumor Classification in MR Images Using Transfer Learning and Pre-Trained Deep CNN Models,” *Cancers*, 17(1), 121, 2025. [Paper](https://doi.org/10.3390/cancers17010121).

The paper directly matches our four-class task, names six standard pretrained CNNs, uses publicly available images, specifies a common classifier head, and reports only five epochs. It is a suitable implementation baseline under our deadline; this is a selection based on fit and accessible methods, not a claim that an exhaustive review has established it as the best paper in the field.

**Dataset:** [Masoud Nickparvar Brain Tumor MRI Dataset, version 1](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1). Version 1 contains 7,023 images, matching the paper. The current version 2 contains 7,200 and must not be silently substituted. Source URL, version, image counts, and archive SHA256 are saved in `results/disci2025/dataset_source.json`.

| Partition | Glioma | Meningioma | No tumor | Pituitary | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 1,321 | 1,339 | 1,595 | 1,457 | 5,712 |
| Testing | 300 | 306 | 405 | 300 | 1,311 |

## Implementation and assumptions

The selected baseline is Xception; the proposed smaller replacement is MobileNetV2. The implementation supports the other four paper architectures, but they are outside the agreed experiment scope and are not required deliverables. An additional paper-based MobileNetV2 run completed before the scope was narrowed and is retained as supplementary evidence. ImageNet backbones are followed by Flatten, Dropout(0.3), Dense(128, ReLU), Dropout(0.2), and Dense(4, Softmax). Training uses Adam at 0.0001, sparse categorical cross-entropy, batch size 20, and five epochs. Final-epoch weights are used without selecting checkpoints on test performance.

Input images are converted to grayscale, blurred, thresholded to derive a largest-contour foreground crop, resized to 128 x 128, copied to three channels, and divided by 255. Training brightness and contrast factors range from 0.8 to 1.2. Gaussian kernel 5 x 5, threshold 45, bounding-rectangle crop, retention of blurred grayscale intensities, seed 42, and full backbone fine-tuning are disclosed choices because the paper does not completely specify them. A missing contour falls back to the full slice and is counted.

The `paper_based` protocol retains official partitions and reports clean test images alongside a fixed seeded transformed test view, since the paper describes augmentation on test images. The `leakage_audited` protocol removes exact overlap and redundant training images, then reserves 15% of the remaining training pool for diagnostic validation (4,518 training and 798 validation images after exact-only exclusion; the same 1,311 test images). Augmentation applies only to training; validation and testing are clean. Changes between protocols affect multiple factors and cannot isolate leakage removal as the sole cause of a performance difference.

Optional ImageNet-specific normalization or frozen-backbone training changes the experiment and must be labeled separately. They are not quietly substituted for the paper's preprocessing or training assumptions.

## Dataset audit

Decoded-pixel hashing found **263 training images identical to test images**, **207 redundant training images**, and **30 redundant test images**, with no conflicting labels among identical images. The original test partition is retained for comparison, including its internal duplicates, so test images are not independent observations.

A 64-bit perceptual hash at Hamming distance <= 4 identified **1,254 train/test candidate pairs** beyond exact pixel matches. These remain unresolved candidates unless individually reviewed; similarity alone is not proof of duplicated patients. Exact-only exclusion reduces known overlap and is explicitly limited. Patient-level and external-data validation remain future work.

## Results and hypothesis outcome

All four Colab runs contain five finite training epochs and accepted saved predictions. Each view has 1,311 ordered test predictions; the metrics below were independently recomputed from those predictions. The main results use the Colab GPU environment (Python 3.12.13, TensorFlow 2.18.1, Keras 3.8.0, seed 42).

### Stage 1: selected baseline replication

Xception achieves **93.67% clean test accuracy** and **0.9354 macro-F1**, compared with the paper's **95.27% accuracy** and **0.9491 macro-F1**. Clean accuracy is **1.60 percentage points below** the published value. The fixed transformed test view, closer to the paper's stated test augmentation, achieves **93.29% accuracy**, **1.98 percentage points below** the published value. This establishes a measured baseline under the disclosed assumptions; it does not exactly reproduce the published numbers.

| Model | Protocol / view | Test accuracy | Macro-F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: |
| MobileNetV2 | leakage_audited / clean | 74.29% | 0.7372 | 0.7391 |
| Xception | leakage_audited / clean | 91.76% | 0.9176 | 0.9176 |
| MobileNetV2 | paper_based / clean | 72.08% | 0.6798 | 0.6941 |
| MobileNetV2 | paper_based / transformed | 71.17% | 0.6691 | 0.6841 |
| Xception | paper_based / clean | 93.67% | 0.9354 | 0.9369 |
| Xception | paper_based / transformed | 93.29% | 0.9315 | 0.9331 |

Paper-based runs use 5,712 training images; audited runs use 4,518 training and 798 diagnostic validation images. Both retain the same 1,311 test images. The paper-based MobileNetV2 row is supplementary rather than the selected baseline.

### Stage 2: proposed replacement and hypothesis outcome

Under identical audited data/configuration, MobileNetV2 achieves macro-F1 **0.7372** and Xception **0.9176**. Their gap is **0.1804**, exceeding the proposed **0.02** margin. **The hypothesis is not supported in this single-seed experiment:** MobileNetV2 is smaller and faster, but fails the accuracy-preservation criterion.

| Audited model | Total parameters | Training time | Median batch-1 inference |
| --- | ---: | ---: | ---: |
| Xception | 25,056,428 | 239.2 s | 10.02 ms |
| MobileNetV2 | 4,880,068 | 179.4 s | 5.32 ms |

MobileNetV2 uses **80.52% fewer total parameters**. Timings come from the same Colab runtime and exclude downloads and final evaluation. Inference timing uses 20 warmed-up, synchronized batch-1 measurements and is indicative of that runtime, not a universal speed claim. The audited pair has matching configuration, ordered data fingerprints and environment metadata. These checks make the comparison interpretable; one seed still cannot establish statistical non-inferiority.

The earlier Mac M2/Metal MobileNetV2 run achieved 68.27% clean accuracy and macro-F1 0.5789 after 45.2 minutes of training. Its compact evidence remains separately archived under `results/disci2025/local_mac`; it is not mixed into the Colab hypothesis comparison. The hardware-dependent difference has no established causal explanation. No model checkpoints were included in the user-exported ZIP, so the imported evidence supports prediction-based verification and presentation, not rerunning inference from cloud weights.

The selected baseline and proposed comparison are complete. The four other architectures are outside the revised scope, not pending experiments. Published targets, measured test scores, and combined transformed train/test statistics remain separate. Smoke and failed compatibility attempts are excluded.

An initial local compatibility test using TensorFlow 2.18.1 with Keras 3.15.1 on Metal produced an empty epoch history and was rejected. Keras is pinned to 3.8.0, XLA is disabled for Metal, and each accepted training run must contain finite loss and accuracy for every requested epoch. This failed attempt is not evidence about classification performance.

## Interpreting the paper's numbers

Xception's Table 8 **testing accuracy is 95.27%**, not the headline 98.73%. The latter is consistent with `(5712 * training_accuracy + 1311 * testing_accuracy) / 7023`, using training accuracy 99.52%. This formula is our reconstruction from reported numbers and sample counts. It is not a held-out metric. The paper's “macro accuracy average” similarly averages training and testing accuracy rather than class recalls.

There are also internal inconsistencies between some per-class tables and the summary: VGG16's weighted recall is 0.9476, while Table 8 lists testing accuracy as 0.9504. Source values are preserved and flagged, rather than adjusted to force agreement. Reproduction differences are reported transparently.

## Limitations and next experiments

Exact numerical matching cannot be guaranteed because crop details, trainability, seeds, and augmentation sampling are underspecified, and hardware/library versions differ. One seed and one dataset do not establish statistical robustness or clinical validity. Near-duplicates and shared-patient slices may remain; retaining duplicate test images also affects the effective sample size.

Next steps after the evaluation are repeated-seed experiments, reliable patient metadata or external testing, controlled normalization/class-weighting ablations, and annotation-based crop/explanation evaluation. EfficientNetB0 and radiomics belong to the broader Phase 1 roadmap, outside the selected baseline and proposed replacement experiment.
