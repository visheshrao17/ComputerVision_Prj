# Verified training extension results

Both full-data Colab runs are accepted. A reaches accuracy 0.887109 and macro-F1
0.884134, selecting epoch 15 of 19. B reaches accuracy 0.926773 and macro-F1
0.925609, selecting epoch 17 of 20, with 2,422,468 parameters. B meets the
exploratory improvement/size target and saved-Xception margin in this one run.
The original five-epoch hypothesis outcome remains unchanged.

`comparison.csv` contains recomputed measured outcomes. `control/` and `proposed/`
contain ordered manifests, predictions, metrics and histories. Import checksums
and Mac CPU checkpoint/example checks are recorded in verification JSON files.
Checkpoints remain under ignored `outputs/training_extension` and the local full
results ZIP; no raw MRI images or model weights are committed to Git.

This is a combined-method experiment on an already examined test set, one seed,
with unresolved near-matches and no reliable patient identifiers. Historical
Xception uses a different training budget. No statistical superiority, clinical
validity or new-algorithm claim follows. See the final evaluation report.
