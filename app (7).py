# ============================================================
#  app.py  —  Gradio UI with Grad-CAM · Hugging Face Spaces
#  Features: file upload + mobile camera · disease prediction
#            + heatmap overlay · top-3 confidence bar
# ============================================================

import os, json
import numpy as np
import gradio as gr
import tensorflow as tf
from grad_cam import explain_prediction

# ── Load model & class map ────────────────────────────────────
MODEL_PATH  = "plant_disease_model.h5"
CLASS_PATH  = "class_indices.json"

print("Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False
print("✓ Model loaded")

with open(CLASS_PATH) as f:
    idx_to_class: dict = json.load(f)
    # Keys may be strings from JSON; convert to int
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

# Make class labels nicer for display
def prettify(raw: str) -> str:
    """'Tomato___Early_blight' → 'Tomato — Early Blight'"""
    parts = raw.replace("___", "||").replace("_", " ").split("||")
    if len(parts) == 2:
        return f"{parts[0].strip()} — {parts[1].strip().title()}"
    return raw.replace("_", " ").title()

display_classes = {k: prettify(v) for k, v in idx_to_class.items()}

# ── Inference function ────────────────────────────────────────
def predict(image):
    """
    Gradio inference function with multi-level confidence filtering.
    """

    if image is None:
        return None, "⚠️ Please upload or capture a leaf image."

    label_raw, confidence, top3, heatmap_rgb = explain_prediction(
        image, model, display_classes
    )

    conf_percent = confidence * 100

    # ── ❌ VERY LOW CONFIDENCE: INVALID IMAGE ────────────────
    if confidence < 0.40:
        result_md = f"""
## 🌿 Diagnosis Result

❌ **Invalid Image / Not a Plant Leaf**

**Confidence:** {conf_percent:.1f}%

👉 The model is not able to recognize this as a valid plant leaf image.

### Try:
- Upload a plant leaf image
- Avoid objects, people, or background scenes
- Ensure proper focus and lighting
"""
        return heatmap_rgb, result_md

    # ── ⚠️ MEDIUM CONFIDENCE: UNCLEAR IMAGE ────────────────
    if confidence < 0.55:
        result_md = f"""
## 🌿 Diagnosis Result

⚠️ **Unclear Image**

**Confidence:** {conf_percent:.1f}%

👉 The model is unsure about this prediction.

### Improve result by:
- Taking a closer leaf image
- Increasing brightness / clarity
- Avoiding motion blur

👉 Try again for a more accurate diagnosis.
"""
        return heatmap_rgb, result_md

    # ── ✅ HIGH CONFIDENCE: FINAL PREDICTION ────────────────
    labels = [t[0] for t in top3]
    values = [round(t[1] * 100, 1) for t in top3]

    severity = "🔴 High" if confidence > 0.85 else "🟡 Medium" if confidence > 0.6 else "🟢 Low"

    result_md = f"""
## 🌿 Diagnosis Result
**Detected Disease:** `{label_raw}`
**Confidence:** `{conf_percent:.1f}%`  &nbsp;|&nbsp; **Certainty:** {severity}

---

### Top-3 Predictions
| Rank | Disease | Confidence |
|------|---------|-----------|
| 🥇 1st | {labels[0]} | {values[0]:.1f}% |
| 🥈 2nd | {labels[1]} | {values[1]:.1f}% |
| 🥉 3rd | {labels[2]} | {values[2]:.1f}% |

---

*Grad-CAM highlights the region of the leaf most responsible for prediction.*
"""

    return heatmap_rgb, result_md

# ── Gradio Interface ──────────────────────────────────────────
with gr.Blocks(
    title="🌿 Plant Disease Detector",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="lime"),
    css="""
    .header-banner { text-align: center; padding: 1rem 0 0.5rem; }
    .header-banner h1 { font-size: 2rem; font-weight: 700; color: #1a6b2a; }
    .header-banner p  { color: #555; font-size: 1rem; }
    .result-col { background: #f8fff8; border-radius: 12px; padding: 1rem; }
    """
) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="header-banner">
      <h1>🌿 Plant Disease Detector</h1>
      <p>Upload a leaf photo or use your camera &mdash; get instant AI diagnosis with explainable heatmap</p>
      <p><small>Powered by MobileNetV2 fine-tuned on PlantVillage · 38 disease classes · 54K images</small></p>
    </div>
    """)

    with gr.Row():
        # ── Left column: Input ──
        with gr.Column(scale=1):
            gr.Markdown("### 📷 Input Leaf Image")
            image_input = gr.Image(
                label="Upload or capture a leaf",
                sources=["upload", "webcam"],
                type="pil",
                height=300
    )
            
            submit_btn = gr.Button("🔍 Analyze Leaf", variant="primary", size="lg")

            gr.Markdown("""
**Tips for best results:**
- Use clear, well-lit images
- Fill the frame with the leaf
- Avoid blurry or dark photos
""")

        # ── Right column: Output ──
        with gr.Column(scale=1, elem_classes="result-col"):
            gr.Markdown("### 🔬 Grad-CAM Heatmap")
            heatmap_output = gr.Image(
                label="Heatmap — affected region highlighted",
                type="numpy",
                height=300
            )

    # ── Result markdown ──
    result_output = gr.Markdown("*Results will appear here after analysis.*")

    # ── Examples ──
    gr.Markdown("---\n### 📂 Try These Examples")
    gr.Examples(
        examples=[
            ["examples/tomato_blight.jpg"],
            ["examples/corn_rust.jpg"],
            ["examples/healthy_apple.jpg"],
        ],
        inputs=image_input,
        label="Example leaf images"
    )

    # ── Footer ──
    gr.HTML("""
    <div style="text-align:center;margin-top:2rem;color:#888;font-size:0.85rem">
      Built with TensorFlow · MobileNetV2 · Grad-CAM · Gradio<br>
      Dataset: <a href="https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset" target="_blank">PlantVillage on Kaggle</a>
    </div>
    """)

    # ── Event binding ──
    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[heatmap_output, result_output]
    )

    # Also run on image upload (optional UX improvement)
    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[heatmap_output, result_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False         # set True for a public Gradio link
    )
