import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Funktion, die bei einer neuen Datei ausgeführt wird
def on_new_file(file_path):
    print(f"Neue Datei entdeckt: {file_path}")
    print("Hello World")

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:  # Sicherstellen, dass es keine Ordner sind
            on_new_file(event.src_path)

if __name__ == "__main__":
    path_to_watch = "Bilder/Bad_Pictures"  # Pfad zum zu überwachenden Ordner

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
