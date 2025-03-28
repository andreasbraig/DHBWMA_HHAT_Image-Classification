from flask import Flask, send_file, render_template
from pycore.preprocessing import imageconversion as uic
from pycore.classification import classification as cls
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask_socketio import SocketIO 
import threading
import json
import os

config_path = "pycore/setings.json"

cf = json.load(open(config_path, 'r'))

model_loaded_event = threading.Event()

model = None

global latest_image
latest_image = None  # Variable zum Speichern des neuesten Bildpfads

def load_model():
    global model
    print("Lade Modell...")
    model = cls.get_Model(cf["model"]["path"])
    print("Modell geladen")
    model_loaded_event.set()

def update_latest_image(image_path,):
    print("new image Detected")
    global latest_image
    latest_image = image_path

    print(image_path[-4:])

    if image_path[-4:]==".jpg":

        time.sleep(1)
        socketio.emit('update_image', {'image_url': '/latest_image'})

        img = uic.preprocess_image(image_path)

        result = cls.classify_image(img,model)

        socketio.emit('classification_result', {'result':result})
    else: 
        print(image_path, "iat kein Bild!")

    #print(result)

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:  # Sicherstellen, dass es keine Ordner sind
            update_latest_image(event.src_path)

def folder_monitor():
    path_to_watch = cf['filepaths']['new']  # Pfad zum zu überwachenden Ordner kommt aus JSON

    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)

    print(f"Überwache Ordner: {path_to_watch}")

    try:
        observer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Beendet.")
    observer.join()

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def main_page():
    return render_template('main.html')

@app.route('/latest_image')
def get_latest_image():
    if latest_image and os.path.exists(latest_image):
        return send_file(latest_image, mimetype='image/jpeg')
    return "No image available", 404

def start_watchdog():
    folder_monitor()

if __name__ == '__main__':

    model_thread = threading.Thread(target=load_model, daemon=True)
    model_thread.start()

    watchdog_thread = threading.Thread(target=start_watchdog, daemon=True)
    watchdog_thread.start()
    
    socketio.run(app, host="127.0.0.1", port=5000,debug=False,use_reloader=False)
