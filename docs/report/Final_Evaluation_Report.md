# Brain MRI classification final evaluation

**Evaluation:** 3 November 2026
**Team:** Vishesh Rao, Pranay Vishwakarma and Manu Vahan
**Institution:** Sajjan Agarwal School of Technology, Rishihood University

## Project outcome

We replicated the selected Xception baseline from Disci et al. (2025), then evaluated a smaller MobileNetV2 replacement. The final proposed model combines Global Average Pooling with frozen-backbone warm-up and low-rate fine-tuning while keeping BatchNorm statistics frozen. Its verified Colab test accuracy is **92.68%**, macro-F1 **0.9256**, with **2,422,468 parameters**. This is 90.33% fewer parameters than the saved audited Xception reference, which reached 91.76% accuracy and 0.9176 macro-F1.

The final exploratory improvement and efficiency targets are met in this single run. The original five-epoch replacement hypothesis remains unsupported; a later successful extension does not change that historical outcome. The project includes executed analysis notebooks, saved model checkpoints, ordered predictions and local inference. The comparison does not establish a new algorithm, statistically significant superiority or performance exceeding the published paper under its original protocol.

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

## Baseline model and training method

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

The final validation accuracy is 84.84%, compared with 88.72% at epoch 4. We retain epoch 5 as specified rather than changing the selection rule after observing the results. This motivated the validation-guided training policy evaluated in Stage 4; repeated seeds remain future work.

## Stage 4 compact model and training control

The normalization experiment left a substantial generalization gap and unstable validation scores. We therefore defined two further runs before training, preserving the same audited split, crop, augmentation, seed 42, 128px input, normalization, batch 20, optimizer family and loss. Both start afresh from ImageNet weights. Test images never enter the training or checkpoint-selection callbacks.

**Exploratory hypothesis H2:** Global Average Pooling with staged fine-tuning will improve macro-F1 over the normalized five-epoch MobileNetV2 while reducing total parameters. The secondary target is to come within 0.02 macro-F1 of the saved audited Xception while retaining fewer parameters.

| Setting | A training control | B proposed compact model |
| --- | --- | --- |
| Feature aggregation | Existing Flatten | Global Average Pooling |
| Epoch budget | Up to 20 | Five warm-up plus up to 15 fine-tuning |
| Backbone adaptation | Fully trainable at 1e-4 | Initially frozen; then non-BatchNorm layers at 1e-5 |
| BatchNorm statistics | Updated | Frozen throughout |
| Checkpoint | Lowest validation loss | Lowest validation loss across both stages |

Both runs stop fine-tuning after four consecutive epochs without lower validation loss. B completes its fixed five-epoch warm-up before fine-tuning; its stopping counter resets at the stage boundary. Dense(128), dropout and the four-class output remain unchanged. The global pooling head reduces parameters from 4,880,068 to 2,422,468, or 50.36% relative to the original MobileNetV2.

A completed 19 epochs and selected epoch 15, with validation loss 0.2617. B completed the full 20-epoch budget and selected epoch 17, with validation loss 0.0937. The proposed model is also selected for the demo by lower validation loss, independently of test ranking. Recorded core packages, platform and device descriptions match between A, B and the normalization run; fixed seeds still do not guarantee bitwise GPU reproducibility.

A tests the combined longer-budget and checkpoint-selection policy. A versus B tests a package of pooling, staged learning and BatchNorm handling. Individual component effects cannot be separated without additional ablations. These techniques are established transfer-learning practices; our contribution is their evaluated application and reproducible efficiency comparison.

## Final measured comparison

| Audited model | Test accuracy | Macro-F1 | Total parameters |
| --- | ---: | ---: | ---: |
| Historical Xception reference | 91.76% | 0.9176 | 25,056,428 |
| Normalized MobileNetV2 five epochs | 81.92% | 0.8062 | 4,880,068 |
| A longer-training control | 88.71% | 0.8841 | 4,880,068 |
| B proposed compact model | **92.68%** | **0.9256** | **2,422,468** |

B improves accuracy by 10.76 percentage points and macro-F1 by 0.1194 over the normalized five-epoch model, with 50.36% fewer parameters. Relative to A, B improves accuracy by 3.97 points and macro-F1 by 0.0415. Relative to saved audited Xception, B is 0.92 accuracy points and 0.0080 macro-F1 higher while using 90.33% fewer parameters. H2's observed improvement/size criterion and the secondary 0.02-margin target are therefore met. This is an exploratory observed outcome, not a statistical non-inferiority or superiority finding.

| Class | A recall | B recall |
| --- | ---: | ---: |
| Glioma | 91.00% | 93.67% |
| Meningioma | 84.97% | 82.35% |
| No tumor | 96.30% | 96.05% |
| Pituitary | 80.00% | 97.67% |

Pituitary recall rises from 61.67% in the normalized five-epoch model to 97.67% in B. Improvement is not uniform: meningioma and no-tumor recall are slightly lower than A. B misclassifies 38 of 306 meningioma slices as no tumor; this remains a priority for error analysis.

A took 643.1 seconds, about 10.72 minutes, and B took 382.3 seconds, about 6.37 minutes, for training including validation and checkpoint writes. Their warmed batch-1 median inference times were 5.29 and 5.19 milliseconds. These are recorded runs, not repeated controlled timing benchmarks. Historical Xception took 239.2 seconds for five epochs; B therefore does not train faster than that saved reference. Different training policies and runtime history prevent a controlled architecture-speed claim.

The paper-based Xception replication remains 93.67% versus the paper's 95.27%. B uses a different audited protocol and training policy: its result must not be presented as exceeding the paper's reported accuracy.

## Model export and fresh-image inference

Both returned checkpoints load on the Mac CPU and have the expected parameter counts. For B, all 52 BatchNorm layers are frozen in the selected saved model. We recompute metrics and per-class results from all 1,311 Colab predictions and check four deterministic first-per-class images against their decoded-pixel hashes. Local inference results are recorded separately from full Colab metrics. B correctly predicts three of these four images; the no-tumor example is classified as meningioma, matching its Colab error. A correctly predicts two of four. All eight local class predictions agree with their corresponding Colab classes. Maximum absolute CPU/GPU score differences are approximately 0.0145 for B and 0.0538 for A. Their cause is unestablished; probability equivalence is not claimed.

The demo uses the validation-selected proposed checkpoint and the same preprocessing as training. Normalization remains inside the saved model, preventing duplicate scaling. Softmax scores are not calibrated medical probabilities. Four demonstration images are implementation checks, not an independent accuracy estimate; full CPU test accuracy is not claimed.

`scripts/training_extension.py` exposes train, analyze and predict. `notebooks/training_extension_colab.ipynb` displays verified comparisons, both learning curves, confusion matrices and a saved-model prediction. Its presentation mode disables training. The complete results ZIP includes checkpoints; raw images and weights remain outside Git. The old normalization-only demo and its CPU score discrepancy remain documented in the historical evidence.

## Verification and limitations

Both new full-data runs pass ordered manifest, settings and prediction checks. The checkpoint-selected epochs equal the minimum validation loss in the stored histories. The archived training code and configurations match the local implementation. All summary metrics, confusion matrices and per-class values are independently recomputed from saved probabilities. The project has 44 passing tests covering baseline behavior, staged training, checkpoint selection, BatchNorm handling, evidence checks and resume behavior.

Limitations remain one seed, a previously examined test set, missing patient identifiers, unresolved near-matches and internal test duplicates. The global pooling and training changes are bundled rather than individually ablated. Historical Xception uses a different training budget and checkpoint policy. No patient-independent, clinical or statistical non-inferiority conclusion follows. Differences between paper-based and audited protocols also cannot isolate leakage removal.

The evaluation deliverable is complete with a positive exploratory efficiency outcome, an explained control experiment and preserved earlier negative results. Next work should repeat seeds, isolate component effects and validate on reliable patient-level or external data. The present result supports further investigation rather than a claim of general superiority.

## Sources and evidence

- [Disci et al. paper](https://doi.org/10.3390/cancers17010121)
- [Dataset version 1](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1)
- [TensorFlow MobileNetV2 preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input)
- Baseline evidence: `results/disci2025`; normalization: `results/final_project`; final experiments: `results/training_extension`.
- Import/checkpoint checksums and local inference checks: `import_verification.json` and `inference_verification.json` in each extension evidence directory.
- [Keras transfer learning and BatchNorm guidance](https://keras.io/guides/transfer_learning/)
- The original Phase 1 PDF and October evaluation report and slides are preserved.
