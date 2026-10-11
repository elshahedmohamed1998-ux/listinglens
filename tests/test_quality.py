"""Checks for the photo quality score: damaging a photo must lower its score and give the right reason.

Run from the project folder:  python -m pytest tests
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
from quality import quality_score  # noqa: E402

Q = json.loads((APP / "config.json").read_text(encoding="utf-8"))["quality"]


def detailed_photo(size=448, seed=0):
    """A synthetic 'busy room': many coloured blocks plus fine texture, so lots of edges and mid brightness."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 130.0)
    for _ in range(600):
        x, y = rng.integers(0, size - 24, 2)
        w, h = rng.integers(4, 24, 2)
        img[y:y + h, x:x + w] = rng.integers(20, 236) + rng.integers(-15, 16, 3)   # light and dark objects
    img += rng.normal(0, 12, img.shape)   # fine texture (fabric, wood grain)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def test_good_photo_scores_well():
    r = quality_score(detailed_photo(), Q)
    assert r["grade"] in {"good", "acceptable"}
    assert r["issues"] == ["none"]


def test_blur_lowers_score_and_is_flagged():
    photo = detailed_photo()
    blurred = photo.filter(ImageFilter.GaussianBlur(radius=6))
    before, after = quality_score(photo, Q), quality_score(blurred, Q)
    assert after["score"] < before["score"]
    assert "too blurry" in after["issues"]
    assert after["grade"] == "poor"


def test_darkening_is_flagged_as_too_dark():
    dark = ImageEnhance.Brightness(detailed_photo()).enhance(0.25)
    r = quality_score(dark, Q)
    assert "too dark" in r["issues"]
    assert r["score"] <= 49


def test_small_photo_is_flagged_low_resolution():
    tiny = detailed_photo().resize((120, 120))
    assert "low resolution" in quality_score(tiny, Q)["issues"]


def test_score_is_capped_when_any_rule_fails():
    r = quality_score(ImageEnhance.Brightness(detailed_photo()).enhance(0.25), Q)
    assert 0 <= r["score"] <= 49
