import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json

config_path = "pycore/setings.json"

cf = json.load(open(config_path, 'r'))

# Funktion, die bei einer neuen Datei ausgeführt wird
def on_new_file(file_path):
    print(f"Neue Datei entdeckt: {file_path}")
    print("Hello World")

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:  # Sicherstellen, dass es keine Ordner sind
            on_new_file(event.src_path)

def folder_monitor():
    path_to_watch = cf['filepaths']['bad']  # Pfad zum zu überwachenden Ordner kommt aus JSON

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
