"""ListingLens pipeline: room type, photo quality, amenity tags and cover choice for listing photos.

Everything is rebuilt from config.json. The room classifier is the trained linear probe
(models/linear_probe_ViT-B-32.joblib) when it is present; otherwise it falls back to
zero-shot CLIP, which scored about the same on the test set (macro-F1 0.950 vs 0.946).
"""
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageOps

from quality import quality_score

HERE = Path(__file__).parent


class ListingLens:
    def __init__(self, config_path: Path = HERE / "config.json"):
        self.cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.cfg["clip_model"], pretrained=self.cfg["clip_pretrained"])
        self.tokenizer = open_clip.get_tokenizer(self.cfg["clip_model"])
        self.model = self.model.to(self.device).eval()
        self.logit_scale = self.model.logit_scale.exp().item()

        self.classes = self.cfg["classes"]
        self.names = self.cfg["class_names"]

        # Room classifier: trained linear probe if available, zero-shot CLIP otherwise
        probe_path = HERE / self.cfg["probe_file"]
        self.probe = None
        if probe_path.exists():
            import joblib
            self.probe = joblib.load(probe_path)
            self.classes = list(self.probe.classes_)
        self.room_text = self._encode_text([self.cfg["room_template"].format(self.names[c]) for c in self.classes])
        self.room_method = "linear probe" if self.probe is not None else "zero-shot CLIP"

        # Amenity tags: one averaged "has it" and one "doesn't have it" vector per tag
        self.tag_names = list(self.cfg["tags"])
        self.tag_pos = torch.stack([self._encode_side(self.cfg["tags"][t]["pos"]) for t in self.tag_names])
        self.tag_neg = torch.stack([self._encode_side(self.cfg["tags"][t]["neg"]) for t in self.tag_names])

    # ---------- CLIP helpers ----------
    def _encode_text(self, prompts):
        with torch.no_grad():
            e = self.model.encode_text(self.tokenizer(prompts).to(self.device))
        return (e / e.norm(dim=-1, keepdim=True)).cpu()

    def _encode_side(self, prompts):
        e = self._encode_text(prompts).mean(dim=0)
        return e / e.norm()

    def embed(self, images):
        """Normalised CLIP embeddings for a list of PIL images."""
        batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        with torch.no_grad():
            e = self.model.encode_image(batch)
        return (e / e.norm(dim=-1, keepdim=True)).cpu()

    # ---------- Individual predictions ----------
    def room_probs(self, emb):
        """(N × classes) probabilities from the probe, or from zero-shot CLIP."""
        if self.probe is not None:
            return self.probe.predict_proba(emb.numpy())
        return (self.logit_scale * emb @ self.room_text.T).softmax(dim=-1).numpy()

    def tag_probs(self, emb):
        """(N × tags) probability that each tag is present (yes-vs-no softmax per tag)."""
        pair = torch.stack([emb @ self.tag_pos.T, emb @ self.tag_neg.T], dim=-1) * self.logit_scale
        return pair.softmax(dim=-1)[..., 0].numpy()

    # ---------- Public API ----------
    @staticmethod
    def load(path_or_image):
        """Open a photo and apply its EXIF rotation (phone photos are often stored sideways)."""
        img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
        return ImageOps.exif_transpose(img).convert("RGB")

    def analyze(self, image):
        """Room probabilities, quality report, tags and a one-line summary for one photo."""
        image = self.load(image)
        emb = self.embed([image])

        p = self.room_probs(emb)[0]
        rooms = {self.names[c]: float(prob) for c, prob in zip(self.classes, p)}
        ranked = sorted(rooms.items(), key=lambda kv: -kv[1])
        (top, top_p), (second, second_p) = ranked[0], ranked[1]
        room_text = f"{top} / {second}" if top_p - second_p < self.cfg["uncertain_gap"] else top

        yes = self.tag_probs(emb)[0]
        tags = {t: round(float(y), 2) for t, y in zip(self.tag_names, yes) if y >= self.cfg["tag_thresholds"][t]}
        tags = dict(sorted(tags.items(), key=lambda kv: -kv[1]))

        quality = quality_score(image, self.cfg["quality"])

        summary = (f"**{room_text}** ({top_p:.0%} sure) · quality **{quality['score']}/100 ({quality['grade']})**"
                   + ("" if quality["issues"] == ["none"] else f" — {', '.join(quality['issues'])}")
                   + (f" · tags: {', '.join(tags)}" if tags else " · no amenity tags"))
        return {"rooms": rooms, "top_room": top, "top_prob": top_p, "quality": quality, "tags": tags, "summary": summary}

    def choose_cover(self, photos):
        """Rank one listing's photos. Cover score = 70% quality + 30% room priority; never a poor photo
        unless every photo is poor. Returns (ranked list of dicts, reason)."""
        images = [self.load(p) for p in photos]
        folder_of = {nice: folder for folder, nice in self.names.items()}
        c = self.cfg["cover"]

        rows = []
        for img in images:
            a = self.analyze(img)
            priority = c["room_priority"][folder_of[a["top_room"]]]
            rows.append({**a, "image": img,
                         "cover_score": c["quality_weight"] * a["quality"]["score"] + c["room_weight"] * 100 * priority})

        usable = [r for r in rows if r["quality"]["grade"] != "poor"] or rows
        best = max(usable, key=lambda r: r["cover_score"])
        ranked = [best] + sorted((r for r in rows if r is not best), key=lambda r: -r["cover_score"])

        reason = (f"**Cover:** {best['top_room']} ({best['top_prob']:.0%} sure), "
                  f"quality {best['quality']['score']}/100 ({best['quality']['grade']}).")
        if all(r["quality"]["grade"] == "poor" for r in rows):
            reason += "\n\n⚠ All photos are poor quality. Ask the broker for better photos."
        return ranked, reason
