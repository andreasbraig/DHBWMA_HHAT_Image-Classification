import cv2
import json
import os

 #Bild laden (im unterordner)
 
config_path = "../../pycore/setings.json"

cf = json.load(open(config_path, 'r'))






#Utilities 
def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Ausführbarer Bereich



image=cv2.imread(cf['filepaths']['good']+"/Good (1).jpg")


show_image("test",image)