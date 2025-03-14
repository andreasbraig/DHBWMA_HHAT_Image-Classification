import json
import pycore.preprocessing.imageconversion as uic
import pycore.preprocessing.filemanagement as ufm

#Nomenklatur: 

# u = utilities 
# ic = image
# fm = filemanagement

# ________________ DIESES FILE DIENT DER EINMALIGEN VORBEREITUNG ZUM TRAINING DES MODELLS_______________

#JSON Settings Laden

config_path = "pycore/setings.json"

cf = json.load(open(config_path, 'r'))


#___________Ausführender Bereich


#Bilder zu Grayscale
uic.folder_to_grayscale(cf["filepaths"]["good"],cf["filepaths"]["good_gray"])
uic.folder_to_grayscale(cf["filepaths"]["bad"],cf["filepaths"]["bad_gray"])

#Aufteilung der Daten
#Gut
ufm.split_data(source=cf["filepaths"]["good_gray"],
               destination1=cf["filepaths"]["test"],
               destination2=cf["filepaths"]["train"],
               test_size=cf["ttsplit"]["test_size"],
               random_state=cf["ttsplit"]["random_state"])

#Schlecht
ufm.split_data(source=cf["filepaths"]["bad_gray"],
               destination1=cf["filepaths"]["test"],
               destination2=cf["filepaths"]["train"],
               test_size=cf["ttsplit"]["test_size"],
               random_state=cf["ttsplit"]["random_state"])

#Aufteilung der Train Daten in Train und Validate
ufm.split_data(source=cf["filepaths"]["train"],
               destination1=cf["filepaths"]["validate"],
               destination2=cf["filepaths"]["train"],
               test_size=cf["ttsplit"]["test_size"],
               random_state=cf["ttsplit"]["random_state"])

#Funktioniert bis hierhin! 

