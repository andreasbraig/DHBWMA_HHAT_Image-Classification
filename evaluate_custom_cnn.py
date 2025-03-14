import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from pycore.training.pytorchmodel3 import CNNClassification

# Modellpfad und Testdaten
MODEL_PATH = "models/model30_NEWNET.state"
TEST_FOLDER = "Bilder/Pytorch/Test/"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 15

# Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Daten-Transformationen
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Testdaten laden
test_dataset = ImageFolder(TEST_FOLDER, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modell laden
model = CNNClassification()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Vorhersagen sammeln
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion Matrix erstellen
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix")
plt.show()

# Classification Report ausgeben
print("**Classification Report:**")
print(classification_report(all_labels, all_preds, target_names=["Bad", "Good"]))
