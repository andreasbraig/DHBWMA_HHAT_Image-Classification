import json
from tensorflow.keras.models import load_model



config_path = "pycore/setings.json"

cf = json.load(open(config_path, 'r'))


def get_Model(path):
    return load_model(path,compile=True)


def classify_image(image,model):

    prediction = model.predict(image)[0][0]  # Annahme: Binäre Klassifikation (eine Zahl als Output)

    
    # Falls das Modell eine Wahrscheinlichkeitsausgabe macht, kannst du einen Schwellenwert setzen:
    result = "Das PCB ist Defekt" if prediction > 0.5 else "Das PCB Funktioniert"


    
    print(f"Klassifizierung: {result} (Wert: {prediction:.4f})")
    return result