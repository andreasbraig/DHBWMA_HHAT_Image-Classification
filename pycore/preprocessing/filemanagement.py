import shutil
import os
import pandas as pd
from sklearn.model_selection import train_test_split



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