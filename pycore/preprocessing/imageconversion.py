import cv2
import json
import os

 #Bild laden (im unterordner)
config_path = "../../pycore/setings.json"

def load_config(file_path):
    """Lädt die Konfigurationsparameter aus der JSON-Datei."""
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config






#Utilities 
def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Ausführbarer Bereich


cf = load_config(config_path)

image=cv2.imread(cf['filepaths']['good']+"/Good (1).jpg")


show_image("test",image)