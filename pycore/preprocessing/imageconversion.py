import cv2
import json
import os
from PIL import Image

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






#Utilities 
def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Ausführbarer Bereich
