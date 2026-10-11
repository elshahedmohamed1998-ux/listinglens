"""ListingLens demo: run with `python app.py`, then open the local URL Gradio prints."""
from pathlib import Path

import gradio as gr

from pipeline import ListingLens

lens = ListingLens()
EXAMPLES_DIR = Path(__file__).parent / "examples"   # optional: put a few .jpg photos here
examples = [[str(p)] for p in sorted(EXAMPLES_DIR.glob("*.jpg"))] if EXAMPLES_DIR.exists() else []


def analyze_ui(image):
    if image is None:
        return "Upload a photo to analyse it.", {}, {}, {}
    a = lens.analyze(image)
    return a["summary"], a["rooms"], a["tags"], a["quality"]


def cover_ui(files):
    if not files:
        return [], "Upload 2–10 photos of the same listing."
    paths = [f if isinstance(f, str) else f.name for f in files]
    ranked, reason = lens.choose_cover(paths)
    gallery = [(r["image"], ("★ COVER · " if i == 0 else "") +
                f"{r['top_room']} · quality {r['quality']['score']} ({r['quality']['grade']})")
               for i, r in enumerate(ranked)]
    return gallery, reason


with gr.Blocks(title="ListingLens") as demo:
    gr.Markdown(
        "# ListingLens\n"
        "Room type, photo quality and amenity tags for real-estate listing photos. "
        f"Room classifier: CLIP ViT-B-32 + {lens.room_method}."
    )

    with gr.Tab("Analyse a photo"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Listing photo", height=360)
                btn = gr.Button("Analyse", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=inp)
            with gr.Column():
                summary_out = gr.Markdown()
                rooms_out = gr.Label(num_top_classes=5, label="Room type")
                tags_out = gr.Label(label="Amenity tags (above their threshold)")
                quality_out = gr.JSON(label="Photo quality")
        outputs = [summary_out, rooms_out, tags_out, quality_out]
        btn.click(analyze_ui, inputs=inp, outputs=outputs)
        inp.upload(analyze_ui, inputs=inp, outputs=outputs)

    with gr.Tab("Choose a cover"):
        files_in = gr.File(file_count="multiple", file_types=["image"], label="Photos of one listing (2–10)")
        cover_btn = gr.Button("Choose cover", variant="primary")
        reason_out = gr.Markdown()
        gallery_out = gr.Gallery(label="Ranked photos (cover first)", columns=4)
        cover_btn.click(cover_ui, inputs=files_in, outputs=[gallery_out, reason_out])

    gr.Markdown(
        "Limitations: trained on 5,250 small (224×224) photos of mostly Western homes; "
        "amenity tags are zero-shot and were checked by eye only; quality thresholds were tuned on that dataset."
    )

if __name__ == "__main__":
    demo.launch()
