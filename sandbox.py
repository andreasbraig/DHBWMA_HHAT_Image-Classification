import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.preprocessing.image as image
import os
import numpy as np
import pandas as pd
from PIL import Image  # Für Bildverarbeitung
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Datenverzeichnisse
train_dir = "Bilder/train"
val_dir = "Bilder/validate"
test_dir = "Bilder/test"

# Bildparameter
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
MODEL_PATH = "models/best_modelv2.keras"

# Funktion zur Konvertierung der Bilder in Graustufen
def convert_to_grayscale(df):
    new_filenames = []
    for path in df["filename"]:
        img = Image.open(path).convert("L")  # "L" steht für Graustufen
        img = img.convert("RGB")  # Wieder in 3 Kanäle umwandeln für MobileNetV3
        new_path = path.replace(".jpg", "_gray.jpg").replace(".png", "_gray.png")
        img.save(new_path)
        new_filenames.append(new_path)
    df["filename"] = new_filenames
    return df

# Erstelle DataFrames für train, val, test
df_train = convert_to_grayscale(pd.DataFrame([(os.path.join(train_dir, f), "good" if "Good" in f else "bad") for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))], columns=["filename", "label"]))
df_val = convert_to_grayscale(pd.DataFrame([(os.path.join(val_dir, f), "good" if "Good" in f else "bad") for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))], columns=["filename", "label"]))
df_test = convert_to_grayscale(pd.DataFrame([(os.path.join(test_dir, f), "good" if "Good" in f else "bad") for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))], columns=["filename", "label"]))

print(df_train.head(10))  
print(df_train['label'].value_counts())  

# Verbesserte Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    preprocessing_function=lambda x: tf.image.random_contrast(x, 0.8, 1.2),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Daten-Generatoren
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(train_generator.class_indices)
print(val_generator.class_indices)
print(test_generator.class_indices)

# Modellaufbau
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-Tuning aktivieren: nur die letzten 50 Layer trainierbar machen
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Optimierung
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[lr_scheduler, early_stopping]
)

# Modell speichern
model.save(MODEL_PATH)

# Modell laden
model = tf.keras.models.load_model(MODEL_PATH)

# Test
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
