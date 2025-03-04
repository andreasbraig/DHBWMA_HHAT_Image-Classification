import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

# Pfade zu den Datensätzen
train_dir = "Bilder/train"
val_dir = "Bilder/validate"
test_dir = "Bilder/test"

# Transformationen für Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Eigene Dataset-Klasse, um Labels aus Dateinamen zu extrahieren
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = 1 if "Good" in img_name else 0

        if self.transform:
            image = self.transform(image)

        return image, label

# Datasets und Dataloader
def get_data_loaders():
    train_dataset = CustomDataset(train_dir, transform=transform)
    val_dataset = CustomDataset(val_dir, transform=transform)
    test_dataset = CustomDataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Modell laden
def get_model():
    model = models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)  # Binäre Klassifikation
    return model

def train_model(model, train_loader, num_epochs=30):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), "models/mobilenetv3_PT_Newset.pth")
    print("Modell gespeichert!")
    return model

def test_model(model, test_loader):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).int().squeeze()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    accuracy = np.trace(cm) / np.sum(cm) * 100
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print("Confusion Matrix:")
    print(cm)
    print(f'F1 Score: {f1:.4f}')

if __name__ == "__main__":
    choice = input("Möchtest du trainieren (T) oder nur testen (t)? ").strip().lower()
    
    train_loader, val_loader, test_loader = get_data_loaders()
    model = get_model()
    
    if choice == "t":
        model.load_state_dict(torch.load("models/mobilenetv3_PT_Newset.pth"))
        test_model(model, test_loader)
    else:
        model = train_model(model, train_loader)
        test_model(model, test_loader)
