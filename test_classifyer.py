import torch
from torchvision import transforms
from Classifier import load_model, predict
from PIL import Image
import os

def test_model_loads():
    model = load_model('skin_cancer_model.pth')
    assert model is not None
    assert hasattr(model, 'fc')

def test_prediction_output():
    model = load_model('skin_cancer_model.pth')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.new('RGB', (224, 224))  # create a blank test image
    input_tensor = transform(image).unsqueeze(0)

    pred, conf, probs = predict(model, input_tensor)
    assert isinstance(pred, int)
    assert 0 <= conf <= 1
    assert probs.shape[1] == 7  # 7 classes
