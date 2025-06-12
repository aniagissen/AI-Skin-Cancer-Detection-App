import torch
from torchvision import models
import numpy as np
from GradCam import generate_gradcam
from Transforms import transform
from PIL import Image

def test_gradcam_output_shape():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 7)
    model.eval()

    dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    input_tensor = transform(dummy_image).unsqueeze(0)
    
    heatmap = generate_gradcam(model, input_tensor)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape[-1] == 3  # RGB
