import torch
import torch.nn as nn
import torchvision.models as models

def load_model(model_path):
    model = models.resnet50(pretrained=False)  # Load ResNet model
    model.fc = nn.Linear(2048, 7)  # 7 classes in the model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load trained weights
    model.eval()  # Set to evaluation mode
    return model 

def predict(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        return prediction.item(), confidence.item(), probabilities
    


