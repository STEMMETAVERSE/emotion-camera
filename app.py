import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import tempfile
import os
# =========================
# CONFIG
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)
st.set_page_config(
    page_title="AI Emotion Detector",
    layout="centered"
)
st.title("😊 AI Emotion Detector")
st.info(
    "Upload a face image and AI will predict the emotion."
)
# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload face image",
    type=["jpg", "jpeg", "png"]
)
# =========================
# MAIN PIPELINE
# =========================
if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(
        image,
        caption="Uploaded Face",
        use_container_width=True
    )
    if st.button("Detect Emotion"):
        with st.spinner("Analyzing facial expression..."):
            try:
                # Save uploaded image temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                # Emotion Classification
                response = client.image_classification(
                    model="dima806/facial_emotions_image_detection",
                    image=tmp_path
                )
                # =========================
                # TOP EMOTION
                # =========================
                top_emotion = response[0]
                st.success(
                    f"Detected Emotion: {top_emotion.label.title()}"
                )
                st.metric(
                    "Confidence",
                    f"{top_emotion.score * 100:.2f}%"
                )
                # =========================
                # ALL PREDICTIONS
                # =========================
                st.subheader("📊 Emotion Scores")
                for item in response[:5]:
                    st.write(
                        f"**{item.label.title()}** — {item.score * 100:.2f}%"
                    )
            except Exception as e:
                st.error(f"Error: {e}")
