import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Watcher:
    # DIRECTORY_TO_WATCH = os.path.dirname(os.path.abspath(__file__))
    def __init__(self):
        self.DIRECTORY_TO_WATCH = os.path.dirname(os.path.abspath(__file__))
        # List of directories- or files- names to exclude
        self.EXCLUDE_NAMES = [
            # pytest cache
            ".pytest_cache",
            # venv
            ".venv",
            # Python cache
            "__pycache__",
            # poetry lock file
            "poetry.lock",
            "project_to_txt.py"
        ]
        self.EXCLUDE_PATHS = [
            # pytest cache
            os.path.join(self.DIRECTORY_TO_WATCH, ".pytest_cache"),
            # venv
            os.path.join(self.DIRECTORY_TO_WATCH, ".venv"),
            # Python cache
            os.path.join(self.DIRECTORY_TO_WATCH, "src\\__pycache__"),
            os.path.join(self.DIRECTORY_TO_WATCH, "tests\\__pycache__"),
            # poetry lock file
            os.path.join(self.DIRECTORY_TO_WATCH, "poetry.lock"),
        ]
        # os.walk without the exclude list
        for subdir, dirs, files in os.walk(self.DIRECTORY_TO_WATCH):
            if any(exclude_name in subdir for exclude_name in self.EXCLUDE_NAMES):
                continue
            for file in files:
                if file in self.EXCLUDE_NAMES:
                    continue
                print(file)
        self.observer = Observer()

    def run(self):
        # event_handler = Handler(self.EXCLUDE_PATHS)
        event_handler = Handler(self)
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")
            # raise Exception("Watcher Stopped")
        self.observer.join()


class Handler(FileSystemEventHandler):
    def __init__(self, watcher):
        self.watcher = watcher
        self.exclude_paths = watcher.EXCLUDE_PATHS

    def on_any_event(self, event):
        if event.is_directory:
            for exclude_path in self.exclude_paths:
                if event.src_path.startswith(exclude_path):
                    return None
        elif any(exclude_path in event.src_path for exclude_path in self.exclude_paths):
            return None

        elif event.event_type == "modified":
            # Action when a file is modified
            print(f"File {event.src_path} has been modified")
            combine_files(self.watcher)


def is_readable_text(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            file.read(1024)  # Read first 1024 bytes
        return True
    except UnicodeDecodeError:
        return False
    except Exception as e:
        print(f"Error checking if file is text: {e}")
        return False


def combine_files(watcher):
    root_directory = watcher.DIRECTORY_TO_WATCH  # "/path/to/your/files"
    exclude_names = watcher.EXCLUDE_NAMES
    output_file = "combined.txt"

    with open(output_file, "w", encoding="utf-8") as outfile:
        for subdir, dirs, files in os.walk(root_directory):
            if any(exclude_name in subdir for exclude_name in exclude_names):
                continue
            for filename in files:
                if filename in exclude_names:
                    continue
                filepath = os.path.join(subdir, filename)
                if is_readable_text(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as infile:
                            outfile.write(f"----- Start of {filepath} -----\n")
                            outfile.write(infile.read() + "\n\n")
                            outfile.write(f"----- End of {filepath} -----\n\n")
                    except Exception as e:
                        print(f"Could not read file {filepath}: {e}")
    print(f"All files have been combined into {output_file}")


if __name__ == "__main__":
    w = Watcher()
    w.run()
