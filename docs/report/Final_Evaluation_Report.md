# Brain MRI classification final evaluation

**Evaluation:** 3 November 2026  
**Team:** Vishesh Rao, Pranay Vishwakarma and Manu Vahan  
**Institution:** Sajjan Agarwal School of Technology, Rishihood University

## Project outcome

We implemented a selected Xception baseline from Disci, Gurcan and Soylu (2025), evaluated a smaller MobileNetV2 replacement, and completed a controlled normalization follow-up. Xception reached 93.67% clean test accuracy against the paper's 95.27%. The normalization follow-up improved audited MobileNetV2 accuracy from 74.29% to 81.92% and macro-F1 from 0.7372 to 0.8062. It retained 80.52% fewer total parameters than Xception but did not meet the proposed 0.02 macro-F1 gap criterion.

The deliverable includes executed notebooks, auditable results, exported model weights and fresh-image inference. The original and follow-up performance-preservation hypotheses remain unsupported. This is a completed experiment with a measured improvement and a negative joint hypothesis outcome, not an exact numerical replication or a clinical diagnostic system.

## Introduction and selected paper

The project classifies MRI slices into glioma, meningioma, no tumor and pituitary tumor. Transfer learning adapts features learned from ImageNet to a smaller MRI collection. The Phase 1 literature survey motivates measuring predictive performance together with computational cost and evaluation reliability. This implementation establishes the baseline and tests a limited extension before broader radiomics or explainability work.

The selected paper is R. Disci, F. Gurcan and A. Soylu, “Advanced Brain Tumor Classification in MR Images Using Transfer Learning and Pre-Trained Deep CNN Models,” *Cancers*, 17(1), 121, 2025, DOI 10.3390/cancers17010121. We select Xception, its best reported model, rather than reproduce its entire six-model benchmark. MobileNetV2 supplies the proposed smaller replacement. Four other paper architectures remain supported but outside this experiment's scope.

## Dataset and audit

We use version 1 of Masoud Nickparvar's Brain Tumor MRI Dataset. It contains the 7,023 images used by the paper. The version 2 collection has 7,200 images and is excluded. The pinned archive SHA256 and class counts are recorded.

| Partition | Glioma | Meningioma | No tumor | Pituitary | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 1,321 | 1,339 | 1,595 | 1,457 | 5,712 |
| Testing | 300 | 306 | 405 | 300 | 1,311 |

Decoded-pixel hashing found 263 training images identical to test images, 207 redundant training images and 30 redundant test images. Identical images have no conflicting labels. Another 1,254 perceptual near-match candidate pairs remain unresolved. Reliable patient identifiers are unavailable. Foreground cropping succeeded on all 7,023 images with zero contour fallbacks.

The paper-based protocol retains the official folders and evaluates clean and fixed seeded transformed test views. The audited protocol excludes exact training/test overlap and redundant training copies, then uses 4,518 training and 798 diagnostic validation images. It retains all 1,311 test images, including test duplicates. This reduces known exact leakage but does not establish patient independence.

## Model and training method

Images are converted to grayscale, blurred, cropped using the largest foreground contour, resized to 128 by 128, copied into three channels and scaled to [0,1]. Training brightness and contrast factors are 0.8 to 1.2. The crop extracts foreground, not an annotation-validated tumor region.

ImageNet backbones use Flatten, Dropout(0.3), Dense(128, ReLU), Dropout(0.2) and Dense(4, Softmax). All backbone layers remain trainable. Adam uses learning rate 0.0001, sparse categorical cross-entropy, batch size 20 and five epochs. The final epoch supplies the evaluated checkpoint; test accuracy does not select it.

The paper does not fully specify preprocessing, trainability or seeds. Gaussian kernel 5, threshold 45, bounding-rectangle cropping retaining blurred grayscale, seed 42 and full fine-tuning are disclosed assumptions. These limit claims of exact numerical replication.

## Stage 1 baseline replication

| Xception evaluation | Test accuracy | Macro-F1 |
| --- | ---: | ---: |
| Published Table 8 | 95.27% | 0.9491 |
| Our paper-based clean view | 93.67% | 0.9354 |
| Our paper-based transformed view | 93.29% | 0.9315 |

The clean result is 1.60 percentage points below the published accuracy; the transformed view is 1.98 points below it. This establishes an approximate selected baseline. The headline 98.73% is consistent with a sample-weighted combination of training and testing accuracy, inferred from the paper's tables; it is not ordinary held-out accuracy.

## Stage 2 original hypothesis

**H1:** Under identical audited settings, MobileNetV2 will achieve test macro-F1 no more than 0.02 below Xception while using fewer total model parameters.

The audited Xception reached macro-F1 0.9176 and original MobileNetV2 0.7372. Their gap, 0.1804, exceeded the 0.02 margin. MobileNetV2 had fewer parameters, but the joint hypothesis was not supported. This historical result remains unchanged.

## Stage 3 normalization follow-up

We changed only `input_scaling` from the paper's [0,1] interpretation to MobileNetV2's documented ImageNet [-1,1] preprocessing. The transformation lives inside the saved model. We retained the same audited data fingerprint, crop, augmentation, seed, full fine-tuning, classifier, optimizer settings, batch size and five epochs.

**Exploratory target:** improve over the original MobileNetV2 and come within 0.02 macro-F1 of the preserved audited Xception while retaining fewer parameters. This follows an already evaluated test set; it is not independent confirmation or a statistical non-inferiority test.

| Audited model | Test accuracy | Macro-F1 | Total parameters |
| --- | ---: | ---: | ---: |
| Xception control | 91.76% | 0.9176 | 25,056,428 |
| Original MobileNetV2 | 74.29% | 0.7372 | 4,880,068 |
| MobileNetV2 with ImageNet normalization | 81.92% | 0.8062 | 4,880,068 |

Accuracy improved by 7.63 percentage points and macro-F1 by 0.0690 relative to the original MobileNetV2. The remaining Xception gap is 0.1114, so the joint exploratory target was not met. The parameter criterion remains satisfied at 80.52% fewer total parameters.

The new run took 168.6 seconds for training and measured median batch-1 inference at 5.41 milliseconds in Colab. Timing excludes downloads and final evaluation; inference uses 20 warmed-up synchronized calls. The new platform/kernel differs from the earlier Colab environment although the recorded core package versions match. Therefore we do not attribute the observed improvement solely to normalization or claim a controlled cross-run speed comparison.

The final validation accuracy is 84.84%, compared with 88.72% at epoch 4. We retain epoch 5 as specified rather than changing the selection rule after observing the results. Repeated runs and validation-driven training policies are future work.

## Model export and fresh-image inference

The returned ZIP includes the trained `.keras` model, unlike the earlier compact evidence export. We loaded the checkpoint on the Mac CPU, verified the parameter count and checked four deterministic first-per-class test images against their decoded-pixel hashes. All four predicted labels match their true classes and the Colab predictions. This small implementation check is not a separate estimate of test accuracy.

CPU and Colab GPU probabilities differ, with a maximum absolute difference of approximately 0.0302 across these examples. A strict 0.001 probability comparison was not satisfied. The cause is unestablished. Full 1,311-image metrics refer to the saved Colab predictions, not a separately reproduced CPU evaluation.

`scripts/final_project.py` exposes `train`, `analyze` and `predict`. Prediction applies the same crop used in training, then calls the saved model in inference mode. It does not normalize inputs twice. Output consists of the predicted class and four softmax scores; these scores are not calibrated diagnostic probabilities.

The complete model export can be kept locally for a demonstration without retraining. Raw MRI images and weights remain outside Git; compact evidence and the executed notebook are reviewable in the repository.

## Verification and limitations

The full five-epoch run, ordered manifests and 1,311 saved test predictions pass evidence checks. Metrics were independently recomputed from probabilities. The new data fingerprint and all shared settings match the original audited MobileNetV2 after allowing only normalization to differ. The project has 36 passing tests, including normalization serialization, prediction preprocessing and comparison guards.

Limitations remain one seed, previously evaluated test data, differing runtime environments, missing patient identifiers, unresolved near-matches and internal test duplicates. No clinical, patient-independent, tumor-localization or statistical non-inferiority conclusion follows. Differences between paper-based and audited protocols involve multiple factors and cannot isolate leakage removal.

Next work should use validation-driven checkpoint rules defined before testing, controlled fine-tuning or BatchNorm experiments, repeated seeds and reliable patient-level or external validation. Further changes should preserve the current baseline and both measured hypothesis outcomes.

## Sources and evidence

- [Disci et al. paper](https://doi.org/10.3390/cancers17010121)
- [Dataset version 1](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1)
- [TensorFlow MobileNetV2 preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input)
- Baseline evidence: `results/disci2025`; new evidence: `results/final_project`.
- Import/checkpoint checksums and local inference checks: `import_verification.json` and `inference_verification.json` in the new evidence directory.
- The original Phase 1 PDF and October evaluation report and slides are preserved.
