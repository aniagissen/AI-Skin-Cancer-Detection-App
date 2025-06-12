import streamlit as st
from PIL import Image
import numpy as np
import torch
import base64
from io import BytesIO
import ollama

# Custom modules
from Classifier import load_model, predict
from Transforms import get_transform
from GradCam import get_heatmap
from Prompts import build_prompt

# Load model and preprocessing
model_path = 'skin_cancer_model.pth'
#"skin_cancer_model.pth"
model = load_model(model_path)
transform = get_transform()

# Label mappings
label_map = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma",
    4: "Melanocytic nevi",
    5: "Melanoma",
    6: "Vascular lesions"
}
malignant_classes = {"Actinic keratoses", "Basal cell carcinoma", "Melanoma"}

# Streamlit UI
st.title("AI Skin Cancer Detection")
st.write("This is a prototype. For medical concerns, please consult a GP or Doctor.")

name = st.text_input("Name:")
age = st.number_input("Age:", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender:", ["Prefer not to say", "Male", "Female", "Non-binary", "Other"])
country = st.text_input("Country of Residence:")
skin_type = st.selectbox("Skin Type:", ["Fair", "Medium", "Olive", "Dark"])
uploaded_file = st.file_uploader("Upload an image of a mole (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)
    image_np = np.array(image.resize((224, 224))) / 255.0

    # Inference
    predicted_idx, confidence_score, _ = predict(model, input_tensor)
    predicted_label = label_map[predicted_idx]
    risk_level = "Malignant (Cancerous)" if predicted_label in malignant_classes else "Benign (Non-Cancerous)"
    confidence_percentage = round(confidence_score * 100, 2)

    st.header("AI Diagnosis")
    if confidence_score < 0.65:
        st.warning("Low confidence prediction. Please try uploading a clearer image.")
        st.caption(f"Model confidence: {confidence_percentage}%")
        st.stop()
    else:
        st.subheader(f"Prediction: {predicted_label}")
        st.subheader(f"Final Diagnosis: {risk_level}")
        st.caption(f"Model confidence: {confidence_percentage}%")

    # Grad-CAM
    heatmap = get_heatmap(model, input_tensor, image_np)
    st.header("Heatmapped Risk Areas")
    st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)

    # Convert heatmap to base64
    buffered = BytesIO()
    Image.fromarray(heatmap).save(buffered, format="JPEG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Prompt LLM
    prompt = build_prompt(name, age, gender, country, skin_type, predicted_label, risk_level)

    response = ollama.chat(
        model="llava:7b",
        messages=[{"role": "user", "content": prompt, "image": heatmap_base64}]
    )

    # Display response
    st.subheader("AI Skin Care Advice")
    st.caption("Note: This advice is AI-generated and not a replacement for medical opinion.")
    st.write(response["message"]["content"])
