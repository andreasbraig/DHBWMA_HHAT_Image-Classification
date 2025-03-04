import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# **Schritt 1: Setup der Variablen**
MODEL_PATH = "mobilenetv3_PT_Newset.pth"  # Dein gespeichertes PyTorch-Modell
TEST_FOLDER = "Bilder/test"  # Pfad zum Test-Datensatz
IMAGE_SIZE = (224, 224)  # Bildgröße
BATCH_SIZE = 15  # Batch-Größe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Schritt 2: Laden des Modells**
model = torch.load(MODEL_PATH, map_location=device)
model.eval()
print("✅ Modell erfolgreich geladen!")

# **Schritt 3: Erstellung des DataFrames mit echten Labels**
def create_test_dataframe(test_folder):
    test_filenames = os.listdir(test_folder)
    filepaths = [os.path.join(test_folder, fname) for fname in test_filenames]
    labels = [0 if fname.lower().startswith("bad") else 1 for fname in test_filenames]  # 0 = Bad, 1 = Good
    return pd.DataFrame({"filename": test_filenames, "filepath": filepaths, "label": labels})

test_df = create_test_dataframe(TEST_FOLDER)

# **Schritt 4: Dataset und DataLoader für PyTorch**
class TestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.dataframe.iloc[idx]['label']

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = TestDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# **Schritt 5: Vorhersagen generieren**
predicted_labels = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        predicted_labels.extend(preds)
        true_labels.extend(labels.numpy())

predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)

print(predicted_labels)

# **Schritt 6: Erstellen der Confusion-Matrix**
cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix")
plt.show()

# **Schritt 8: Ausgabe des Classification Reports**
print("**Classification Report:**")
print(classification_report(true_labels, predicted_labels, target_names=["Bad", "Good"]))
