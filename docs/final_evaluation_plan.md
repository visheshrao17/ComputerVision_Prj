# Final evaluation continuation

Historical normalization stage. The later compact-model experiment is completed
and verified; see [training extension](training_extension_plan.md) and the updated
final report/PPT. The original results and hypothesis outcomes below stay unchanged.

Evaluation date: 3 November 2026. This continues the merged October baseline work.

## Verified completion

The user returned the full Colab export on 2 November. The five-epoch normalization follow-up, 1,311 predictions and checkpoint passed verification. Accuracy is **81.92%**, macro-F1 **0.8062**, an observed gain of 7.63 accuracy percentage points and 0.0690 macro-F1 over original MobileNetV2. The gap to audited Xception is **0.1114**, exceeding 0.02: the joint exploratory target is **not met**. Runtime platforms differ, so normalization is not established as the sole cause of improvement.

The model loads on the Mac and predicts four deterministic first-per-class test images with matching true labels and Colab classes. Scores differ by up to 0.0302; strict probability parity is not established. Fresh-image prediction, compact comparison evidence and an executed notebook are complete. The final report and PPT record both negative hypothesis outcomes and the observed improvement. No further training is required for this evaluation.

## Completed evidence

The selected Disci et al. baseline is Xception, with 93.67% clean test accuracy versus the paper's 95.27%. This is an approximate selected-model replication. The original audited MobileNetV2 comparison gave macro-F1 0.7372 against Xception's 0.9176. It used 80.52% fewer parameters but failed the allowed 0.02 performance margin. That original hypothesis remains unsupported.

The report and PPT already propose controlled normalization/fine-tuning experiments as next work. The exported October ZIP lacks trained cloud checkpoints, so it supports saved-result analysis but cannot make fresh image predictions from those cloud weights.

## Extension for this evaluation

1. Run a single controlled MobileNetV2 experiment. Change only `input_scaling` from `paper` to `imagenet`. The existing model layer maps cropped `[0,1]` pixels to the documented MobileNetV2 `[-1,1]` input range.
2. Keep the exact audited split, crop, augmentation, seed, architecture, full backbone fine-tuning, Adam rate, batch size and five epochs. Save to separate output and result directories.
3. Compare actual metrics with both preserved audited models. Require the same data fingerprint and all other shared settings. Report hardware differences and the observed outcome rather than assuming improvement.
4. Export the trained `.keras` model and run a fresh-image prediction with the same preprocessing. Download the model bundle before Colab disconnects.
5. Incorporate the new evidence into the final evaluation report and slides after the run is verified. Present an unfavorable result if obtained.

Exploratory target: improve over the original MobileNetV2 macro-F1 and reach within 0.02 of the saved audited Xception with fewer total parameters. The target score from that control is approximately 0.8976. The original test set has already been evaluated. This is a hypothesis-generating follow-up, not a preregistered independent confirmation or a statistical non-inferiority claim. A successful follow-up cannot retrospectively change the original result.

## Code to explain

| File | Explanation |
| --- | --- |
| `configs/mobilenet_imagenet.yaml` | The same experiment settings with one normalization change. |
| `src/reproduction/preprocessing.py` | Reads, crops and resizes an MRI slice. Existing code is reused. |
| `src/reproduction/models.py` | Loads ImageNet features, applies normalization and adds the four-class head. Existing code is reused. |
| `src/reproduction/training.py` | Fits five epochs, saves weights, then evaluates. Existing code is reused. |
| `src/reproduction/final_project.py` | Checks the controlled comparison and predicts a new image. |
| `scripts/final_project.py` | Three commands: `train`, `analyze`, `predict`. |
| `notebooks/final_project_colab.ipynb` | Runs that pipeline in Colab and exports the complete demo. |

## Colab and local workflow

Select T4 GPU and runtime 2026.07 / Python 3.12. Upload `deliverables/brain_mri_final_source.zip` and open the new notebook. Run all cells. It trains one model, verifies its comparison, predicts a labeled example, and downloads `brain_mri_final_demo.zip` with code, evidence and weights. Raw MRI images are excluded.

On the Mac, extract the returned bundle into a separate directory. Install the requirements in that directory's virtual environment, then run:

```bash
python scripts/final_project.py analyze
python scripts/final_project.py predict --image /path/to/mri.jpg
```

Without training, the notebook displays the original baseline and explicitly labels the follow-up pending. Tests and tiny synthetic models verify implementation behavior; they are never used as reproduced research results.

## Completion boundary

Implementation checks, a verified full five-epoch run, an exported checkpoint, fresh-image inference, and a report/PPT reflecting the actual outcome constitute the final extension. These requirements are now fulfilled. The historical report and slides retain their October date and findings.

After this evaluation, prioritize repeated seeds, reviewed near-matches and reliable patient-level or external validation. A new architecture, radiomics, lesion localization, clinical validation and a broad model search are outside this immediate extension.

Sources: [MobileNetV2 preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input), [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html), the existing reproduction addendum and saved run evidence under `results/disci2025`.
