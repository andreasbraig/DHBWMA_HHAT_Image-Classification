import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt   

from sklearn.model_selection import train_test_split

import cv2
import os

import os
import time
import csv
import shutil

# Definiert eine CNN-Klassifikation für Bilddatensätze
class CNNClassification(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(200704,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )


    def forward(self, xb):
        return self.network(xb)

    @torch.no_grad()  # Deaktiviert das Gradienten-Tracking (nützlich für Inferenz)
    def inferenzSet(self, dataset, device,logfile):

        self.eval()
        images = [sublist[0].to(device) for sublist in dataset]
        images = torch.stack(images).to(device)
        labels = [sublist[1] for sublist in dataset]
        labels = torch.tensor(labels).to(device)

        res = self(images)
    
        _, preds = torch.max(res, dim=1)

        accuracy = torch.sum(preds == labels).item() / len(preds)

        log_test_results(dataset,preds.cpu().tolist(),logfile)

        print("Erg: " + str(accuracy))

        return preds,accuracy


    def trainStart(self, epochs, lr, train_loader, device,modelname, opt_func=torch.optim.Adam, patience=5, lr_patience=3, lr_decay_factor=0.5):
        optimizer = opt_func(self.parameters(), lr)
        self.to(device)  # Verschiebt das Modell auf das angegebene Gerät
        self.train()

        best_acc = 0.0
        epochs_no_improve = 0
        lr_stagnation = 0
    
        
        log_file= modelname[:-6]+".csv"
        print("log saved to:",log_file)

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "epoch_loss", "epoch_acc", "timestamp"])

            for epoch in range(epochs):
                train_losses = []
                for batch in train_loader:
                    images, labels = batch
                    images = torch.stack([apply_augmentation(img) for img in images])
                    images, labels = images.to(device), labels.to(device)
                    loss = self.training_step((images, labels))
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    self.eval()
                    outputs = [self.validation_step(batch, device) for batch in train_loader]
                    batch_losses, batch_accs = zip(*outputs)
                    epoch_loss = torch.stack(batch_losses).mean().item()
                    epoch_acc = torch.stack(batch_accs).mean().item()
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f"Epoch {epoch}, loss: {epoch_loss}, acc: {epoch_acc}, Timestamp: {timestamp}")
                    writer.writerow([epoch, epoch_loss, epoch_acc, timestamp])
            

                        #Early Stopping & Reduce LR
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        epochs_no_improve = 0  # Reset counter
                        lr_stagnation = 0
                    elif epoch_acc>0.9:
                        epochs_no_improve += 1
                        lr_stagnation += 1

                        if lr_stagnation >= lr_patience:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= lr_decay_factor
                            print(f"Reducing LR to {optimizer.param_groups[0]['lr']}")
                            lr_stagnation = 0  # Reset LR stagnation counter

                        if epochs_no_improve >= patience:
                            print("Early stopping triggered!")
                            break

                    


    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch, device):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return (loss.detach(), acc)

def apply_augmentation(image):
    augmentation = transforms.Compose([
        #transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),  # Rotation, Verschiebung, Skalierung, Scherung
        #transforms.RandomHorizontalFlip(p=0.5),  # Horizontales Spiegeln
        #transforms.RandomVerticalFlip(p=0.2),  # Vertikales Spiegeln mit geringerer Wahrscheinlichkeit
        #transforms.RandomResizedCrop(size=(40, 40), scale=(0.7, 1.0), ratio=(0.75, 1.33)),  # Zufälliger Ausschnitt und Skalierung
        #transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Helligkeit und Kontrast variieren
        #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Leichte Unschärfe hinzufügen
        #transforms.RandomInvert(p=0.2),  # Zufälliges Invertieren von Pixelwerten
    ])
    return augmentation(image)

def log_test_results(test_dataset, predictions, filename="test_results.csv"):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "Label", "Prediction"])

            for (img_path, label), pred in zip(test_dataset.samples, predictions):
                filename = os.path.basename(img_path)
                writer.writerow([filename, label, pred])
        
        print(f"Test results saved to {filename}")


def train_model(data_dir, device,epochs=5,modelname = "model.state"):

    trans = [
        transforms.Resize((224, 224)),  # Bildgröße ändern (optional)
        transforms.Grayscale(num_output_channels=1),  # In Graustufen umwandeln (optional)
        transforms.ToTensor()  # Wandelt das Bild in einen Tensor um
    ]
    train_dataset = ImageFolder(data_dir, transform=transforms.Compose(trans))
    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    model = CNNClassification()


    try:
        model.load_state_dict(torch.load(modelname))
    except:
        print("No model found")

    trainEpochs = epochs
    if trainEpochs > 0:
        model.trainStart(trainEpochs, 0.001, train_dl, device,modelname)
        torch.save(model.state_dict(), modelname)


def test_model(test_data_dir, device,modelname,logfile):
    trans = [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]

    model = CNNClassification()
    test_dataset = ImageFolder(test_data_dir, transforms.Compose(trans))

    print(modelname)

    try:
        model.load_state_dict(torch.load(modelname,map_location="cpu"))
        model.to(device)
    except:
        print("No model found")
        return

    preds,_ = model.inferenzSet(test_dataset, device,logfile)



def get_device(preferred_device=None):
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def copy_misclassified_images(csv_file, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            filename, label, prediction = row
            if label != prediction:  # Copy only misclassified images
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, filename)
                if os.path.exists(source_path):
                    shutil.copy(source_path, target_path)

def rgba_loader(path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return transforms.ToTensor()(img)



def split_data(source,destination1,destination2,test_size=0.2,random_state=36):

  if not os.path.exists(destination2):
    os.makedirs(destination2)

  if not os.path.exists(destination1):
    os.makedirs(destination1)

  #Iterate through the folder and add anything to the list, that is a File and not a folder
  files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

  #If folder is Empty
  if len(files) == 0:
    print("Der Ordner ist Leer")
    return

  #Actual Split
  train_files, test_files = train_test_split(files,test_size = test_size,random_state = random_state)

  #Copying of the files "os.XXX" for Joining the Path and filename to the element in the train files list
  #Coping from first path to last path with soutil function
  for file in train_files:
    shutil.move(os.path.join(source,file),os.path.join(destination2,file))

  for file in test_files:
    shutil.move(os.path.join(source,file),os.path.join(destination1,file))

  print(f"Kopieren abgeschlossen. {len(train_files)} Dateien im Ersten Ordner, {len(test_files)} Dateien im Zweiten Ordner.")



def resize_images(input_folder, output_folder, scale_factor=0.3):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Überprüfen, ob es sich um eine Bilddatei handelt
        if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(file_path)
            

            if img is None:
                continue  # Überspringe ungültige Bilder
            
            # Neue Bildgröße berechnen
            new_width = int(img.shape[1] * scale_factor)
            new_height = int(img.shape[0] * scale_factor)
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            #img_gray=cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            
            # Bild speichern
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)
            
        
    print(f"Bilder wurden erfolgreich verkleinert und gespeichert.")

# Beispielaufruf:
# resize_images("input_bilder", "output_bilder", scale_factor=0.5)



def pytorchmodel():



    # resize_images(input_folder="Bilder/Pytorch/Source/Bad_Pictures",
    #               output_folder="Bilder/Pytorch/Learn/bad")
    
    # resize_images(input_folder="Bilder/Pytorch/Source/Good_Pictures",
    #               output_folder="Bilder/Pytorch/Learn/good")

    # split_data(source="Bilder/Pytorch/Learn/bad",
    #            destination1="Bilder/Pytorch/Test/bad",
    #            destination2="Bilder/Pytorch/Learn/bad")
    
    # split_data(source="Bilder/Pytorch/Learn/good",
    #            destination1="Bilder/Pytorch/Test/good",
    #            destination2="Bilder/Pytorch/Learn/good")
    

    data_dir = "Bilder/Pytorch/Learn"
    test_data_dir = "Bilder/Pytorch/Test/"
     

    model = "model30_NEWNET.state"

    logfile = model[:-6]+"_testlog.csv"

    fehl_data_dir = "Datensatz/"+model[:-6]+"fehl"

    # Geräteauswahl: "cuda",perplexityperp "mps" oder "cpu"
    preferred_device = "cuda"  # Beispiel: Manuelle Auswahl von MPS
    device = get_device(preferred_device)
    if not os.path.exists(data_dir):
        print(f"Fehler: Der Ordner {data_dir} existiert nicht!")

    print(f"Using device: {device}")

    train_model(data_dir, device, epochs=30,modelname=model)

    test_model(test_data_dir, device,model,logfile)

    #Sorge dafür, dass alle Bilder, bei denen es nicht geklappt hat, wegsortiert werden. 

    copy_misclassified_images(logfile,test_data_dir+"/bad",fehl_data_dir)
    copy_misclassified_images(logfile,test_data_dir+"/good",fehl_data_dir)


if __name__ == "__main__":
    print(cv2.imread("Bilder/Pytorch/Learn/good/Good (1).jpg"))
