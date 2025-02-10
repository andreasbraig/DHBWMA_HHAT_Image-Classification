#Step Zero: General Setup

from google.colab import drive
drive.mount('/content/drive')

import os

Origin_Good = '/content/drive/MyDrive/ML_Images/Good_Pictures'
Origin_Bad = '/content/drive/MyDrive/ML_Images/Bad_Pictures'

#--------NEXT CELL---------

#Funktion zum Plotten eines Bildes

import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img

filepath = r"/content/drive/MyDrive/ML_Images/Train/"

def Print_Image(sample):
  image = load_img(filepath+sample)
  plt.imshow(image)


#--------NEXT CELL---------

#Step One: Image Preprocessing

from PIL import Image


def Convert_Grayscale(Origin_Path, Destination_Path):

#Check for existing Folder Othervise Create them
  if not os.path.exists(Destination_Path):
    os.makedirs(Destination_Path)

    print("Warning, Destination Folder Does not Exist, Creating it for you...")

  for filename in os.listdir(Origin_Path):
    if filename.endswith(('.jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif')):
      image_path = os.path.join(Origin_Path, filename)
      img = Image.open(image_path).convert('L')

      output_path = os.path.join(Destination_Path, filename)
      img.save(output_path)

  print("Conversion Complete")

Good_Grayscale = '/content/drive/MyDrive/ML_Images/Good_Pictures_Grayscale'
Bad_Grayscale = '/content/drive/MyDrive/ML_Images/Bad_Pictures_Grayscale'

Convert_Grayscale(Origin_Good, Good_Grayscale)
Convert_Grayscale(Origin_Bad, Bad_Grayscale)

#--------NEXT CELL---------

#Step Two Train Test Split

import shutil
from sklearn.model_selection import train_test_split


def split_data(source,test,train,test_size=0.2):

  if not os.path.exists(train):
    os.makedirs(train)

    print("Warning, train Destination Folder Does not Exist, Creating it for you...")

  if not os.path.exists(test):
    os.makedirs(test)

    print("Warning, test Destination Folder Does not Exist, Creating it for you...")

  #Iterate through the folder and add anything to the list, that is a File and not a folder
  files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

  #If folder is Empty
  if len(files) == 0:
    print("Der Ordner ist Leer")
    return

  #Actual Split
  train_files, test_files = train_test_split(files,test_size = test_size,random_state=36)

  #Copying of the files "os.XXX" for Joining the Path and filename to the element in the train files list
  #Coping from first path to last path with soutil function
  for file in train_files:
    shutil.move(os.path.join(source,file),os.path.join(train,file))

  for file in test_files:
    shutil.move(os.path.join(source,file),os.path.join(test,file))

  print(f"Kopieren abgeschlossen. {len(train_files)} Dateien im Trainingsordner, {len(test_files)} Dateien im Testordner.")

#execute Train Test split

test_folder = '/content/drive/MyDrive/ML_Images/Test'
train_folder = '/content/drive/MyDrive/ML_Images/Train'

split_data(Good_Grayscale,test_folder,train_folder)
split_data(Bad_Grayscale,test_folder,train_folder)

#--------NEXT CELL---------

#execute the Train Test Split again to split the Train Data in Train and Validate

validate_folder = '/content/drive/MyDrive/ML_Images/Validate'

split_data(train_folder,validate_folder,train_folder)

#--------NEXT CELL---------

# Step Three Image Setup Variables

FAST_RUN = False
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=1

#--------NEXT CELL---------

#Step 4 Creating the Dataframe to

import pandas as pd

#Der Pfad muss nochmal als Raw String definiert werden, damit Pandas damit umghehen kann.
filenames = os.listdir(r'/content/drive/MyDrive/ML_Images/Train')

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

df = make_df(filenames)
df_validate = make_df(os.listdir(r'/content/drive/MyDrive/ML_Images/Validate'))

df.head(1000)

df['label'].value_counts().plot.bar()

#--------NEXT CELL---------

Print_Image(random.choice(filenames))

#--------NEXT CELL---------

#Step 5 Setup of the first ML model

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def Setup_MobilNet():
  base_model = MobileNet(weights='imagenet', include_top=False) # Importing the Base

  #Setup of the Layers for better Results
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  x = Dense(1024, activation='relu')(x)
  x = Dense(512, activation='relu')(x)
  preds = Dense(2, activation='softmax')(x)

  return base_model, x, preds

base_model, x, preds = Setup_MobilNet()

#--------NEXT CELL---------

#Step 5.1 First optimizing of the Base Model

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

for layer in model.layers[:20]:
  layer.trainable = False
for layer in model.layers[20:]:
  layer.trainable = True

#--------NEXT CELL---------

#Step 5.2 Setup EarlyStop and Learning Rate (Callbacks)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

#Renaming the dfs to have Good and Bad as Labels #why not in the first place?
df["label"] = df["label"].replace({0: 'Good', 1: 'Bad'})
df_validate["label"] = df_validate["label"].replace({0: 'Good', 1: 'Bad'})

df = df.reset_index(drop=True)
df_validate = df_validate.reset_index(drop=True)

total_train = df.shape[0]
total_validate = df_validate.shape[0]
batch_size = 15

#--------NEXT CELL---------

#Data Aufmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Creating the Train data
#Here Try different Options
train_datagen = ImageDataGenerator(
    rotation_range=20, #15
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2, #0.2
    horizontal_flip=True,
    width_shift_range=0.1, #0.1
    height_shift_range=0.1 #0.1
)

train_generator = train_datagen.flow_from_dataframe(
    df,
    filepath,
    x_col='filename',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

valitation_filepath = r"/content/drive/MyDrive/ML_Images/Validate/"

#Creating the Validation data

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    df_validate,
    valitation_filepath,
    x_col='filename',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

#--------NEXT CELL---------

#TRAINING
epochs=3 if FAST_RUN else 50
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

#--------NEXT CELL---------

import numpy as np

def Plot_Loss_Accuracy(history):
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
  ax1.plot(history.history['loss'], color='b', label="Training loss")
  ax1.plot(history.history['val_loss'], color='r', label="validation loss")
  ax1.set_xticks(np.arange(1, epochs, 1))
  ax1.set_yticks(np.arange(0, 1, 0.1))

  ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
  ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
  ax2.set_xticks(np.arange(1, epochs, 1))

  legend = plt.legend(loc='best', shadow=True)
  plt.tight_layout()
  plt.show()


Plot_Loss_Accuracy(history)

#--------NEXT CELL---------

#Saving the Model

model.save_weights(".weights.h5")
model.save("/content/drive/MyDrive/ML_Images/.weights.h5") 

#--------NEXT CELL---------

from tensorflow.keras.models import load_model

model = load_model(r"/content/drive/MyDrive/ML_Images/.weights.h5")

#--------NEXT CELL---------

test_filepath = r"/content/drive/MyDrive/ML_Images/Test/"



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
    batch_size=batch_size,
    shuffle=False
  )
  steps=np.ceil(nb_samples/batch_size)

  predict = model.predict(test_generator, steps= int(np.ceil(nb_samples/batch_size)))

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

