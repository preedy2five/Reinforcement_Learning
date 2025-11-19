import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from .model import CubePositionNet

def load_model(model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CubePositionNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def play(imgA, imgB, imgC, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # This converts from HWC to CHW and scales to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert tensors to PIL Images for preprocessing
    # Assuming input tensors are in format [H, W, C] with values in [0, 255] or [0, 1]
    def tensor_to_pil(tensor):
        # If tensor is [B, H, W, C], take first batch
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        # Convert to numpy and ensure values are in [0, 255]
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        
        tensor = tensor.cpu().numpy().astype(np.uint8)
        return Image.fromarray(tensor)
    
    # Convert to PIL Images
    imgA_pil = tensor_to_pil(imgA)
    imgB_pil = tensor_to_pil(imgB)
    imgC_pil = tensor_to_pil(imgC)
    
    # Apply transforms
    imgA_processed = transform(imgA_pil).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    imgB_processed = transform(imgB_pil).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    imgC_processed = transform(imgC_pil).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    
    # Make prediction
    with torch.no_grad():
        pred = model(imgA_processed, imgB_processed, imgC_processed)

    position = pred[:, :3]  # Keep batch dimension [1, 3]
    
    print(f"Predicted Position: x={position[0,0]:.2f}, y={position[0,1]:.2f}, z={position[0,2]:.2f}")
    return position