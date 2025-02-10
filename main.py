import json
import numpy as np
from pycore.preprocessing import imageconversion as uic
from pycore.classification import classification as cls

config_path = "pycore/setings.json"

cf = json.load(open(config_path, 'r'))

image_path="Bilder/Bad_Pictures/Bad (13).jpg"
#image_path="Bilder/Good_Pictures/Good (6).jpg"

model = cls.get_Model("mobilenet_v1.h5")

binary_latest = uic.single_to_grayscale(image_path,"Bilder/Cache_Grayscale","test.jpg")


img = preprocess_image(image_path)

result = cls.classify_image(img,model)

print(result)