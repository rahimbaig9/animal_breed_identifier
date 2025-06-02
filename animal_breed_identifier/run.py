import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import urllib.request

# Load pretrained ResNet50 model
@st.cache_resource(show_spinner=False)
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()

# Load ImageNet labels
@st.cache_data(show_spinner=False)
def load_imagenet_labels():
    with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as f:
        classes = [line.decode("utf-8").strip() for line in f.readlines()]
    return classes

imagenet_classes = load_imagenet_labels()

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225],   # ImageNet std
    ),
])

def predict(image: Image.Image):
    input_tensor = preprocess(image).unsqueeze(0)  # batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probabilities, 0)
        class_name = imagenet_classes[top_idx] if top_idx < len(imagenet_classes) else "Unknown"
        return class_name, top_prob.item()

# Streamlit UI
st.set_page_config(page_title="🐶 Animal Breed Identifier", page_icon="🐕", layout="centered")

st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>🐶 Animal Breed Identifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:#306998;'>Welcome! Upload an animal image and get the predicted breed from ImageNet classes.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Predicting..."):
            class_name, confidence = predict(image)
            st.success(f"Prediction: **{class_name}** with confidence {confidence*100:.2f}%")

# Footer
footer_html = """
<style>
footer {visibility: hidden;}
#custom-footer {
    position: fixed;
    right: 15px;
    bottom: 15px;
    font-size: 13px;
    color: #888888;
    font-weight: 600;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(90deg, #4B8BBE, #306998);
    padding: 5px 10px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(48, 105, 152, 0.3);
    user-select: none;
}
</style>
<div id="custom-footer">Done by Rahim</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
