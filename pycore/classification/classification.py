import json
import torch
import numpy as np
from tensorflow.keras.models import load_model
from torchvision import transforms
from PIL import Image

from pycore.training.pytorchmodel3 import CNNClassification

config_path = "pycore/setings.json"

cf = json.load(open(config_path, 'r'))


def get_Model(path):
    return load_model(path,compile=True)


def classify_image(image,model):

    prediction = model.predict(image)[0][0]  # Annahme: Binäre Klassifikation (eine Zahl als Output)


    
    # Falls das Modell eine Wahrscheinlichkeitsausgabe macht, kannst du einen Schwellenwert setzen:
    result = "Das PCB ist Defekt" if prediction > 0.5 else "Das PCB Funktioniert"

    
    #print(f"Klassifizierung: {result} (Wert: {prediction:.4f})")
    return result


def get_Custom_Model(path):

    model = CNNClassification()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

def classify_Custom_CNN(image_path, model, device):
    """
    Klassifiziert ein einzelnes Bild mit dem gegebenen Modell.
    
    Args:
        image_path (str): Pfad zum Bild.
        model (torch.nn.Module): Das geladene PyTorch-Modell.
        device (torch.device): CPU oder GPU.
    
    Returns:
        str: Klassifikationsergebnis
    """
    # Transformation wie bei den Testdaten
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert("RGB")



    # Transformation anwenden
    image = transform(image)
    image = image.to(device)

    print(image.shape)  # Ausgabe: torch.Size([4, 3, 224, 224])
    # Modell auf Evaluierungsmodus setzen
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()  # Falls das Modell Logits ausgibt
        
    # Binäre Klassifikation mit Schwellenwert
    result = "Das PCB ist Defekt" if prediction > 0.5 else "Das PCB funktioniert"
    return result
 