import os
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

config_path = "pycore/setings.json"
    
cf = json.load(open(config_path, 'r'))

# **Schritt 1: Setup der Variablen**
MODEL_PATH = cf["model"]["path"] #"models/mobilenetv2.keras"  # Dein gespeichertes Modell
TEST_FOLDER = "Bilder/test"  # Pfad zum Test-Datensatz
IMAGE_SIZE = (224, 224)  # Bildgröße
BATCH_SIZE = 15  # Batch-Größe

# **Schritt 2: Laden des Modells**
model = load_model(MODEL_PATH)
print("✅ Modell erfolgreich geladen!",MODEL_PATH)

# **Schritt 3: Erstellung des DataFrames mit echten Labels**
def create_test_dataframe(test_folder):
    test_filenames = os.listdir(test_folder)
    filepaths = [os.path.join(test_folder, fname) for fname in test_filenames]
    labels = [0 if "bad" in fname.lower() else 1 for fname in test_filenames]# 0 = Good, 1 = Bad
    return pd.DataFrame({"filename": test_filenames, "filepath": filepaths, "label": labels})

test_df = create_test_dataframe(TEST_FOLDER)

print(test_df.to_string())

# **Schritt 4: Erstellen des Testgenerators**
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    TEST_FOLDER,
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
  )

# **Schritt 5: Vorhersagen generieren**
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=-1)    # Falls das Modell eine einzelne Sigmoid-Ausgabe hat

#print(predicted_labels)

#print(test_df,len(test_df))

# **Schritt 6: Erstellen der Confusion-Matrix**
# **Confusion-Matrix berechnen**
cm = confusion_matrix(test_df["label"][1:], predicted_labels)

# **Confusion-Matrix visualisieren**
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel("Vorhergesagt")  # Predicted
plt.ylabel("Tatsächlich")   # Actual
plt.title("Confusion Matrix")
plt.show()


# **Schritt 8: Ausgabe des Classification Reports**
print(" **Classification Report:**")
print(classification_report(test_df["label"][1:], predicted_labels, target_names=["Bad", "Good"]))
