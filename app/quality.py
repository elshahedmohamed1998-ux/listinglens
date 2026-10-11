"""Photo quality score: sharpness, brightness and contrast combined into one 0–100 score.

Pure image processing (no deep learning), so it is fast and every number can be explained.
"""
import cv2
import numpy as np
from PIL import Image


def photo_measures(img: Image.Image, analysis_size: int = 224) -> dict:
    """Sharpness, brightness, contrast and original short side of one photo.

    The photo is resized to a fixed short side first, so one sharpness threshold works
    for small dataset images and large phone photos alike.
    """
    img = img.convert("RGB")
    short_side = min(img.size)
    scale = analysis_size / short_side
    small = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),  # variance of edges: low = blurry
        "brightness": float(gray.mean()),                           # 0 = black, 255 = white
        "contrast": float(gray.std()),                              # low = flat and grey
        "short_side": int(short_side),
    }


def quality_score(img: Image.Image, q: dict) -> dict:
    """0–100 score, grade and reasons. `q` is the "quality" section of config.json.

    Each measure becomes a 0–100 sub-score that is exactly 50 at its threshold.
    Score = weighted average; any failed threshold caps the score at 49 ("poor").
    """
    m = photo_measures(img, q["analysis_size"])

    sharp = np.interp(np.log10(max(m["sharpness"], 1)),
                      [np.log10(q["sharpness_min"]) - 0.3, np.log10(q["sharpness_min"]), np.log10(q["sharpness_good"])],
                      [0, 50, 100])
    bright = np.interp(m["brightness"],
                       [q["brightness_min"] - 30, q["brightness_min"], q["brightness_min"] + 30,
                        q["brightness_max"] - 30, q["brightness_max"], q["brightness_max"] + 30],
                       [0, 50, 100, 100, 50, 0])
    contrast = np.interp(m["contrast"], [q["contrast_min"] - 15, q["contrast_min"], q["contrast_good"]], [0, 50, 100])

    w = q["weights"]
    score = w["sharpness"] * sharp + w["brightness"] * bright + w["contrast"] * contrast

    issues = [msg for failed, msg in [
        (m["sharpness"] < q["sharpness_min"], "too blurry"),
        (m["brightness"] < q["brightness_min"], "too dark"),
        (m["brightness"] > q["brightness_max"], "too bright"),
        (m["contrast"] < q["contrast_min"], "low contrast"),
        (m["short_side"] < q["min_short_side"], "low resolution"),
    ] if failed]
    if issues:
        score = min(score, 49)

    grade = "good" if score >= 75 else "acceptable" if score >= 50 else "poor"
    return {
        "score": int(round(score)),
        "grade": grade,
        "issues": issues or ["none"],
        "sharpness": round(m["sharpness"]),
        "brightness": round(m["brightness"]),
        "contrast": round(m["contrast"]),
        "size": f"{img.width}×{img.height}",
    }
