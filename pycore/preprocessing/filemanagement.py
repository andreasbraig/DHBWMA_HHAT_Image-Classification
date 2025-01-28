import shutil
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


config_path = "../pycore/setings.json"

cf = json.load(open(config_path, 'r'))


def split_data(source,destination1,destination2,test_size,random_state):

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


def make_df(images):
    labels = []

    for filename in images:
        label = filename.split(' ')[0]
        if label == 'Good':
            labels.append(0)
        else:
            labels.append(1)

    df = pd.DataFrame({'filename': images, 'label': labels})

    return df

#Ab hier imagedatagen ##TODO

def data_augmentation(df,filepath):
    datagen = ImageDataGenerator(
    rotation_range=20, #15
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2, #0.2
    horizontal_flip=True,
    width_shift_range=0.1, #0.1
    height_shift_range=0.1 #0.1
    )

    generator = datagen.flow_from_dataframe(
    df,
    filepath,
    x_col='filename',
    y_col='label',
    target_size=[cf["imagecfg"]["IMG_W"],cf["imagecfg"]["IMG_H"]],
    class_mode='categorical',
    batch_size=cf["train_augmentation"]["batch_size"]
    )
    return generator