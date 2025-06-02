import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

# Load pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet labels from PyTorch GitHub repo
with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as f:
    imagenet_classes = [line.decode("utf-8").strip() for line in f.readlines()]

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # standard ImageNet mean
        std=[0.229, 0.224, 0.225],   # standard ImageNet std
    ),
])

def predict(image: Image.Image):
    input_tensor = preprocess(image).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probabilities, 0)
        class_name = imagenet_classes[top_idx] if top_idx < len(imagenet_classes) else "Unknown"
        return class_name, top_prob.item()
