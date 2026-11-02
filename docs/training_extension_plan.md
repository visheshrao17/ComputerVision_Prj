# Compact, validation-guided MobileNetV2 extension

Approved on 2 November 2026. Both full-data Colab runs are now **verified**. B reaches 92.68% accuracy,
macro-F1 0.9256 with 2,422,468 parameters; A reaches 88.71% and 0.8841.
The exploratory improvement/size target and secondary Xception-margin target
are met in this run. The original baseline and normalization evidence stay
unchanged. The final report/PPT include this extension; earlier October artifacts remain unchanged.

Implementation verification: 44 tests pass, including a synthetic two-stage
end-to-end run, saved-checkpoint selection, frozen BatchNorm statistics,
evidence validation and resume checks. All seven notebook code cells execute
locally with training disabled. These checks are not reproduced MRI results.

## Research question

Can a compact classification head with staged fine-tuning improve macro-F1 over
the normalized five-epoch MobileNetV2 while using fewer parameters?
The secondary target is a gap of at most 0.02 from the saved audited Xception.
Both parts must be measured; neither is assumed to succeed.

Xception from Disci et al. remains the selected paper baseline. MobileNetV2 is
our proposed efficient replacement, not a second paper baseline. Global pooling
and staged fine-tuning are established techniques; the contribution is this
evaluated extension and reproducible comparison, not a claim of algorithmic novelty.

## Fixed data and settings

Pinned Kaggle version 1, 7,023 images; same exact-deduplicated audited split:
4,518 training, 798 validation and 1,311 test. The split fingerprint must equal
the preserved normalization experiment before training starts. Keep seed 42,
128px foreground crop, augmentation, ImageNet normalization, batch 20,
Adam, loss and dense/dropout layers unchanged. Use fresh ImageNet weights for
each experiment. Do not continue from the old MRI checkpoint.

## Two experiments

| Setting | A: control | B: proposed |
| --- | --- | --- |
| Feature aggregation | Existing Flatten | Global Average Pooling |
| Maximum total epochs | 20 | 20 |
| Warm-up | None | 5 epochs; backbone frozen, head LR 1e-4 |
| Fine-tuning | All layers, LR 1e-4 | All non-BatchNorm backbone layers, LR 1e-5 |
| BatchNorm moving statistics | Updated as before | Frozen throughout |
| Fine-tuning early stopping | 4 consecutive epochs without lower validation loss | Same |
| Selected checkpoint | Lowest validation loss | Lowest validation loss across both stages |

Warm-up always completes five epochs; fine-tuning starts from its final weights.
The early-stopping counter resets at the start of fine-tuning. The minimum
validation loss across both stages still determines the exported checkpoint;
it can therefore come from warm-up. Recompile with a fresh Adam optimizer when
changing trainability. Dropout remains active during head training. Only the
backbone is called in inference mode for B, keeping BatchNorm statistics fixed.

A tests the combined longer-budget/validation-selection policy. A versus B tests
the combined pooling/staged-learning/BatchNorm proposal. These comparisons cannot
isolate each individual component. Their actual selected epochs may differ.

## Run in Colab

1. Open `notebooks/training_extension_colab.ipynb`.
2. Select a T4 GPU and Python 3.11/3.12 runtime (the earlier 2026.07 runtime works
   with the pinned TensorFlow 2.18.1 environment).
3. Upload `deliverables/brain_mri_training_source.zip` in the Files sidebar.
4. For presentation keep `RUN_TRAINING=False` and run all cells. For a deliberate
   new experiment set it to True; A and B run separately and complete runs are skipped.
5. Download `brain_mri_training_results.zip` before disconnecting. It includes
   both accepted checkpoints and compact evidence, without raw images.

Allow roughly 20–40 minutes for setup, data audit, two runs and evaluation; this
is an estimate, not a measured result. Early stopping can shorten training.
If interrupted, an unfinished experiment restarts from ImageNet. Resume skips
only accepted complete runs; it is not optimizer/epoch recovery.

Equivalent CLI:

```bash
python scripts/training_extension.py train --resume
python scripts/training_extension.py analyze
python scripts/training_extension.py predict --experiment proposed --image /path/to/mri.jpg
python scripts/package_training_extension.py --include-models --output deliverables/brain_mri_training_results.zip
```

The fresh data directory `data/training_extension` avoids the partial local demo
dataset. To reuse a complete Colab dataset, pass `--data-dir /path/to/data/raw`;
inventory and fingerprint checks still apply. Outputs live under
`outputs/training_extension/{control,proposed}` and compact evidence under
`results/training_extension/{control,proposed}`.

## What to show after the run

The notebook reports measured accuracy, macro-F1, pituitary recall, parameter
count, actual epochs, selected epoch, validation loss, training time and inference
time. It shows both learning curves and confusion matrices. Select the demo model
using validation loss between A and B, not test accuracy. Test scores are reported
after selection and do not change the predetermined configuration or checkpoint.

The first target requires macro-F1 above 0.8062 and fewer than 4,880,068 total
parameters. The secondary target requires macro-F1 at least approximately 0.8976
and fewer than Xception's 25,056,428 parameters. Both outcomes are now measured and verified from the returned checkpoints and
1,311 predictions per run. A selects epoch 15 of 19; B selects epoch 17 of 20.
B has lower validation loss, 0.0937 versus 0.2617, and is the demo model.

This is exploratory: the test set has already been examined, only one seed is
scheduled, near-matches remain unresolved, and patient IDs are missing. Comparison
to historical Xception uses a different training policy and potentially different
runtime; it does not prove architecture superiority or controlled speed gains.
Do not claim performance exceeding the paper using a different protocol.

The returned ZIP, predictions, metrics and checkpoints are verified. The final
report and presentation record the actual outcomes. Preserve the old hypothesis
outcomes. Further ablations, repeated seeds and patient-level/external validation
follow after the evaluation.

## Simple code map

- `configs/training_control.yaml`, `training_proposed.yaml`: experiment settings.
- `models.py`: optional average pooling; old Flatten default remains unchanged.
- `finetuning.py`: warm-up, fine-tuning, validation checkpoint and early stopping.
- `training.py`: reuses the existing data/evaluation pipeline; dispatches the new policy.
- `training_comparison.py`: checks expected settings and builds the results table.
- `scripts/training_extension.py`: train, analyze or predict.

Reference: [Keras transfer learning and BatchNorm guidance](https://keras.io/guides/transfer_learning/).
