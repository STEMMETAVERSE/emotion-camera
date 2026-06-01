import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import os

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

st.title("😊 Emotion Detector")

uploaded_file = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Detect Emotion"):

        with st.spinner("Detecting emotion..."):

            response = client.image_classification(
                model="trpakov/vit-face-expression",
                image=uploaded_file.getvalue()
            )

        st.subheader("Emotion Result")
        st.write(response)
