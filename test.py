import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# **Schritt 1: Setup der Variablen**
MODEL_PATH = "models/mobilenet_v2.h5"  # Dein gespeichertes Modell
TEST_FOLDER = "Bilder/test"  # Pfad zum Test-Datensatz
IMAGE_SIZE = (224, 224)  # Bildgröße
BATCH_SIZE = 15  # Batch-Größe

model = load_model(MODEL_PATH)

#--------NEXT CELL---------

test_filepath = TEST_FOLDER



def Test_Model(test_folder, test_filepath):
  test_filenames = os.listdir(test_filepath)
  test_df = pd.DataFrame({'filename': test_filenames})
  nb_samples = test_df.shape[0]

  test_gen = ImageDataGenerator(rescale=1./255)
  test_generator = test_gen.flow_from_dataframe(
    test_df,
    test_filepath,
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
  )
  steps=np.ceil(nb_samples/BATCH_SIZE)

  predict = model.predict(test_generator, steps= int(np.ceil(nb_samples/BATCH_SIZE)))

  test_df['label'] = np.argmax(predict, axis=-1)

  label_map = dict((v,k) for k,v in train_generator.class_indices.items())
  test_df['label'] = test_df['label'].replace(label_map)

  return test_df

test_df = Test_Model(test_folder, test_filepath)
test_df['label'].value_counts().plot.bar()

#--------NEXT CELL---------

sample_test = test_df.head(60)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['label']
    img = load_img(test_folder+"/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(12, 5, index+1)
    plt.imshow(img)
    plt.xlabel(filename + (' ') + 'Result-' + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()