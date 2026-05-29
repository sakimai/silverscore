"""
SiLVERScore — Gradio demo for HuggingFace Spaces.

Users upload a sign language video, enter a text description, choose a variant,
and optionally upload pre-trained CiCo/I3D checkpoints to compute the SiLVERScore.
"""

import logging
import os

import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

VARIANT_OPTIONS = {
    "Phoenix-2014T — German Sign Language (DGS)": "phoenix",
    "How2Sign — American Sign Language (ASL)": "how2sign",
    "CSL-Daily — Chinese Sign Language (CSL)": "csl",
}

CHECKPOINT_FILENAMES = {
    "phoenix": "ph_sota.pth",
    "how2sign": "h2s_sota.pth",
    "csl": "csl_sota.pth",
}

SCORE_SCALE = 3.5

SCORE_TABLE = """\
| Score range | Interpretation |
|---|---|
| ≥ 75 | 🟢 Strong alignment |
| 55 – 75 | 🟡 Good alignment |
| 35 – 55 | 🟠 Moderate alignment |
| 15 – 35 | 🔴 Weak alignment |
| < 15 | ⚫ Poor alignment |
"""

ABOUT_TEXT = """\
**SiLVERScore** (Sign Language Video Embedding Representation Score) is a
reference-free evaluation metric for sign language generation. It measures
the semantic alignment between a sign language video and a text description
using CLIP-based cross-modal embeddings.

The model architecture adapts [CiCo / CLCL](https://github.com/FangyunWei/SLRT/tree/main/CiCo)
(Cheng et al., CVPR 2023). I3D features are extracted with a sliding window
over 16-frame clips and projected into the CLIP embedding space.

For readability, displayed scores are scaled by **×3.5** (matching the paper analysis convention).

> **Paper:** [Imai et al., RANLP 2025](https://aclanthology.org/2025.ranlp-1.54/)  
> **Code:** [github.com/sakimai/silverscore](https://github.com/sakimai/silverscore)
"""


# ------------------------------------------------------------------
# Scoring logic
# ------------------------------------------------------------------

def _qualitative(score: float) -> str:
    # Heuristic labels on the scaled (x3.5) score range.
    if score >= 75:
        return "🟢 Strong alignment"
    if score >= 55:
        return "🟡 Good alignment"
    if score >= 35:
        return "🟠 Moderate alignment"
    if score >= 15:
        return "🔴 Weak alignment"
    return "⚫ Poor alignment"


def run_scorer(video_path, text, variant_label, model_ckpt, i3d_ckpt):
    """Compute SiLVERScore and return a Markdown-formatted result string."""
    if video_path is None:
        return "⚠️ Please upload a sign language video."
    if not text or not text.strip():
        return "⚠️ Please enter a text description."

    variant = VARIANT_OPTIONS[variant_label]

    # model_path: use uploaded checkpoint or fall back to base CLIP
    model_path = model_ckpt if model_ckpt is not None else "none"

    # i3d_checkpoint: optional — None means random I3D weights
    i3d_path = i3d_ckpt if i3d_ckpt is not None else None

    try:
        from silverscore import SiLVERScore  # imported lazily to keep startup fast

        scorer = SiLVERScore(
            variant=variant,
            model_path=model_path,
            i3d_checkpoint=i3d_path,
            score_scale=SCORE_SCALE,
        )
        score = float(scorer.score(video_path, text=text.strip()))
    except Exception as exc:
        hint = ""
        msg = str(exc)
        if "model_path" in msg or "checkpoint" in msg.lower():
            hint = (
                "\n\n💡 **Tip:** Download the CiCo checkpoints from "
                "[FangyunWei/SLRT](https://github.com/FangyunWei/SLRT/tree/main/CiCo) "
                f"and upload `{CHECKPOINT_FILENAMES[variant]}` above."
            )
        return f"**Error:** {msg}{hint}"

    qual = _qualitative(score)
    raw_score = score / SCORE_SCALE
    base_clip_note = (
        "\n\n> ⚠️ **Running with base CLIP weights only** — "
        "upload a CiCo checkpoint for best results."
        if model_path == "none"
        else ""
    )
    i3d_note = (
        "\n> ⚠️ **I3D using random weights** — "
        "upload a BSL-1K / CiCo I3D checkpoint for best feature extraction."
        if i3d_path is None
        else ""
    )

    return (
        f"## SiLVERScore: `{score:+.4f}`\n\n"
        f"(scaled by ×{SCORE_SCALE:.1f}; raw logit `{raw_score:+.4f}`)\n\n"
        f"**{qual}**\n\n"
        f"---\n"
        f"**Variant:** {variant_label}  \n"
        f"**Text:** *{text.strip()}*"
        f"{base_clip_note}"
        f"{i3d_note}"
    )


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------

CSS = """
.result-box { font-size: 1.05em; line-height: 1.6; }
.footer { text-align: center; color: #888; margin-top: 1.5em; font-size: 0.9em; }
"""

with gr.Blocks(
    title="SiLVERScore Demo",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    css=CSS,
) as demo:
    # --- Header ---
    gr.Markdown(
        """
        <div style="text-align:center; padding: 1em 0 0.5em;">
          <h1>🤟 SiLVERScore</h1>
          <p style="font-size:1.1em; color:#555;">
            Sign Language Video ↔ Text Semantic Similarity
          </p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # ---- Left column: inputs ----
        with gr.Column(scale=1, min_width=340):
            gr.Markdown("### Upload & Configure")

            video_input = gr.Video(label="Sign Language Video")

            text_input = gr.Textbox(
                label="Text Description",
                placeholder="e.g. 'a person signing about the weather tomorrow'",
                lines=2,
            )

            variant_input = gr.Radio(
                choices=list(VARIANT_OPTIONS.keys()),
                value=list(VARIANT_OPTIONS.keys())[0],
                label="Sign Language Variant",
            )

            with gr.Accordion("⚙️ Model Checkpoints (optional)", open=False):
                gr.Markdown(
                    "Upload pre-trained checkpoints for best results. "
                    "Without them, base CLIP / random I3D weights are used.\n\n"
                    "Download CiCo checkpoints from "
                    "[FangyunWei/SLRT](https://github.com/FangyunWei/SLRT/tree/main/CiCo)."
                )
                model_ckpt_input = gr.File(
                    label="CLCL Checkpoint (ph_sota.pth / h2s_sota.pth / csl_sota.pth)",
                )
                i3d_ckpt_input = gr.File(
                    label="I3D Checkpoint (.pth / .pth.tar)",
                )

            run_btn = gr.Button(
                "▶  Compute SiLVERScore",
                variant="primary",
                size="lg",
            )

        # ---- Right column: results + info ----
        with gr.Column(scale=1, min_width=340):
            gr.Markdown("### Result")
            score_output = gr.Markdown(
                value=(
                    "*Upload a video, enter text, then click "
                    "**Compute SiLVERScore**.*"
                ),
                elem_classes="result-box",
            )

            with gr.Accordion("📊 Score interpretation", open=True):
                gr.Markdown(SCORE_TABLE)

            with gr.Accordion("ℹ️ About SiLVERScore", open=False):
                gr.Markdown(ABOUT_TEXT)

    # --- Wire up the button ---
    run_btn.click(
        fn=run_scorer,
        inputs=[
            video_input,
            text_input,
            variant_input,
            model_ckpt_input,
            i3d_ckpt_input,
        ],
        outputs=score_output,
    )

    # --- Footer ---
    gr.Markdown(
        "---\n"
        "<div class='footer'>"
        "Built by <a href='https://huggingface.co/saki-imai'>saki-imai</a> · "
        "<a href='https://aclanthology.org/2025.ranlp-1.54/'>Paper (RANLP 2025)</a> · "
        "<a href='https://github.com/sakimai/silverscore'>GitHub</a>"
        "</div>",
        elem_classes="footer",
    )


demo.launch(server_name="0.0.0.0", show_api=False)
