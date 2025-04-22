# Interface additions - https://www.youtube.com/watch?v=D0D4Pa22iG0&t=41s -- https://docs.streamlit.io/develop/api-reference

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import ollama

model_path = '/Users/aniagissen/Documents/GitHub/AI-4-Media-Project-Ania-Gissen/skin_cancer_model.pth'

# Pre-trained ResNet model
model = models.resnet50(pretrained=False)  # Load ResNet model
model.fc = nn.Linear(2048, 7)  # 7 classes in the model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load trained weights
model.eval()  # Set to evaluation mode

#Image preprocessing for resnet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dictionary of label names matching the dataset
label_map = {
    0: "Actinic keratoses",  # Malignant
    1: "Basal cell carcinoma",  # Malignant
    2: "Benign keratosis-like lesions",  # Benign
    3: "Dermatofibroma",  # Benign
    4: "Melanocytic nevi",  # Benign
    5: "Melanoma",  # Malignant
    6: "Vascular lesions"  # Benign
    }

# Map the 7-classes to Benign or Malignant
malignant_classes = {"Actinic keratoses", "Basal cell carcinoma", "Melanoma"}
benign_classes = {"Benign keratosis-like lesions", "Dermatofibroma", "Melanocytic nevi", "Vascular lesions"}

# Title and description
st.title("AI Skin Cancer Detection")
st.write("Please note this is not approved by a doctor and is a prototype - please consult your GP if you have any concerns")
st.write("In order to provide some relevant feedback for you, please provide some personal details in the text boxes below.")
name = st.text_input("Name:")
age = st.number_input("Age:", min_value=0, max_value=120, step=1) 
gender = st.selectbox("Gender:", ["Prefer not to say", "Male", "Female", "Non-binary", "Other"])
country = st.text_input("Country of Residence:")
skin_type = st.selectbox("Skin Type:", ["Fair", "Medium", "Olive", "Dark"])
st.write("Upload an image of a mole and we will try and detect any signs of skin cancer")

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL image to NumPy array for visualisation
    image_np = np.array(image.resize((224, 224))) / 255.0  # Normalize for visualisation

    # Preprocess the image for the model
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        torch_im = transform(image).unsqueeze(0)  # Preprocess image
        outputs = model(torch_im)  # Run model on input
        _, predicted = torch.max(outputs, 1)  # Get predicted class index

    # Get the class name from index
    predicted_class_name = label_map[predicted.item()]

    # Map the class to Benign or Malignant
    if predicted_class_name in malignant_classes:
        binary_result = "Malignant (Cancerous) Please visit your doctor immediately."
    else:
        binary_result = "Benign (Non-Cancerous)"

    # Display results
    st.header("AI Diagnosis")
    st.subheader(f"Prediction: {predicted_class_name}")
    st.subheader(f"Final Diagnosis: {binary_result}")

    # Apply Grad-CAM for visualisation - https://jacobgil.github.io/pytorch-gradcam-book/introduction.html
    target_layer = model.layer4[-1]  # Use the last convolutional layer in ResNet
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Generate heatmap
    grayscale_cam = cam(input_tensor)[0, :]
    heatmap = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    # Display heatmap
    st.header("Heatmapped Risk Areas")
    st.write("The highlighted regions indicate where the model focused on when making its prediction.")
    st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
 

    # Ollama LLM Feedback
    st.subheader("AI Skin Care Advice")
    st.caption("Please take note this is written by an AI and you should always visit your GP or doctor if you have serious concern about skin cancer, even if this model did not detect anything.")

    # Convert heatmap to base64 format for Ollama - ChatGPT
    import base64
    from io import BytesIO

    buffered = BytesIO()
    heatmap_pil = Image.fromarray(heatmap)
    heatmap_pil.save(buffered, format="JPEG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()

    # LLM Prompt
    prompt = f"""
    You are SkinCancerDetector, a dermatologist AI assistant. The user has provided the following details:
    - Name: {name}
    - Age: {age}
    - Gender: {gender}
    - Country of Residence: {country}
    - Skin Type: {skin_type}
    - Diagnosis: {predicted_class_name} ({binary_result})
    Here is an image of a mole. The AI model has diagnosed it as {predicted_class_name}.

    Attached is a Grad-CAM heatmap highlighting the high-risk areas of the mole. 
    - Please explain what the highlighted areas might indicate.

    Based on these details, provide personalized advice to the patient, including:
    - Risk factors for their skin type and age group.
    - UV protection recommendations based on their country.
    - General skincare tips to prevent skin cancer.
    Keep the response concise and user-friendly.
    """

    # Send the request to LLaVA-7B
    ollama_response = ollama.chat(
        model= "llava:7b",
        messages=[
            {"role": "user", "content": prompt, "image": heatmap_base64}
        ]
    )

    # Display Feedback
    st.write(ollama_response["message"]["content"])
