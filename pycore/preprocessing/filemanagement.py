import shutil
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
            labels.append("good")
        else:
            labels.append("bad")

    df = pd.DataFrame({'filename': images, 'label': labels})

    return df

#Ab hier imagedatagen

def data_gen(mode, df, filepath, x_col, y_col, target_size, batch_size, config,shuffle=True,):
    """
    Erstellt einen ImageDataGenerator basierend auf den Parametern aus einer JSON-Datei.
    
    :param mode: 'train_augmentation', 'validation_augmentation' oder 'test_augmentation'
    :param df: DataFrame mit Bildinformationen
    :param filepath: Pfad zu den Bildern
    :param x_col: Spaltenname mit den Dateinamen
    :param y_col: Spaltenname mit den Labels (oder None für Testdaten)
    :param target_size: Zielgröße der Bilder (z. B. (150, 150))
    :param batch_size: Größe der Batches
    :param shuffle: Ob die Daten durchmischt werden sollen (für Test False setzen)
    :param config_path: Pfad zur JSON-Datei mit den Augmentationsparametern
    :return: ImageDataGenerator-Flow
    """
    if mode not in config:
        raise ValueError(f"Ungültiger Modus: {mode}. Erwarte 'train_augmentation', 'validation_augmentation' oder 'test_augmentation'.")

    # Erstelle den Generator mit den aus der JSON geladenen Parametern
    datagen = ImageDataGenerator(**config[mode])

    # Bestimme den class_mode
    class_mode = 'categorical' if y_col else None

    print(df)

    return datagen.flow_from_dataframe(
        df,
        filepath,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle
    )


# Nutzung:
# train_generator = create_data_generator("train_augmentation", df, filepath, "filename", "label", IMAGE_SIZE, batch_size)
# validation_generator = create_data_generator("validation_augmentation", df_validate, validation_filepath, "filename", "label", IMAGE_SIZE, batch_size)
# test_generator = create_data_generator("test_augmentation", test_df, test_filepath, "filename", None, IMAGE_SIZE, batch_size, shuffle=False)
