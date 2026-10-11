# Trained models

Put the trained linear probe here as `linear_probe_ViT-B-32.joblib` (about 20 KB). The notebook saves it to Google Drive in section 4.

Without this file the app still works: it falls back to zero-shot CLIP, which scored about the same on the test set (macro-F1 0.950 vs 0.946).

The fine-tuned ResNet-18 weights (45 MB) are not used by the app and are not stored in the repository.
