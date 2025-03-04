import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.preprocessing.image as image
import os
import numpy as np
import pandas as pd
from PIL import Image  # Für Bildverarbeitung
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
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
MODEL_PATH = "models/mobilnetv3L.keras"


# Erstelle DataFrames für train, val, test
df_train = pd.DataFrame([(os.path.join(train_dir, f), "good" if "Good" in f else "bad") for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))], columns=["filename", "label"])
df_val = pd.DataFrame([(os.path.join(val_dir, f), "good" if "Good" in f else "bad") for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))], columns=["filename", "label"])
df_test = pd.DataFrame([(os.path.join(test_dir, f), "good" if "Good" in f else "bad") for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))], columns=["filename", "label"])

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

# Modellaufbau
base_model = MobileNetV3Small(
    input_shape=(224,224,3),
    alpha=1.0,
    minimalistic=False,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation='softmax',
    include_preprocessing=True
)

print(base_model.output.shape)  


x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, "relu")(x)
x = layers.Dense(1024, "relu")(x)
x = layers.Dense(512, "relu")(x)
preds = layers.Dense(2,"softmax")(x)

# Fine-Tuning aktivieren: nur die letzten 50 Layer trainierbar machen
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model = Model(inputs=base_model.input, outputs=preds)

# Optimierung
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Training
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=60,
    callbacks=[lr_scheduler, early_stopping]
)

# Modell speichern
model.save(MODEL_PATH)

# Modell laden
model = tf.keras.models.load_model(MODEL_PATH)

# Test
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
