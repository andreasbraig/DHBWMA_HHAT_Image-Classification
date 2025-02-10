import cv2
import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing import image as im

 #Bild laden (im unterordner)


def folder_to_grayscale(Origin_Path, Destination_Path):

#Check for existing Folder Othervise Create them
  if not os.path.exists(Destination_Path):
    os.makedirs(Destination_Path)

  for filename in os.listdir(Origin_Path):
    if filename.endswith(('.jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif')):
      image_path = os.path.join(Origin_Path, filename)

      single_to_grayscale(image_path,Destination_Path,filename)
      

  print("Conversion Complete")

#for use in the Later Version 
def single_to_grayscale(image_path,Destination,filename):
    
    img = Image.open(image_path).convert('L')

    output_path = os.path.join(Destination, filename)
    img.save(output_path)

    return output_path


#Image in Numpy Array Wandeln, sodass Klassifizierbar
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Lädt ein Bild, skaliert es und bereitet es für das Modell vor.
    """
    img = im.load_img(image_path, target_size=target_size, color_mode="rgb")  # WICHTIG: Graustufen-Modus
    img_array = im.img_to_array(img)  # In NumPy-Array umwandeln
    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen (TensorFlow erwartet 4D-Shape)
    img_array = img_array / 255.0  # Normalisierung (je nach Modell evtl. notwendig)
    return img_array


#Utilities 
def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Ausführbarer Bereich
