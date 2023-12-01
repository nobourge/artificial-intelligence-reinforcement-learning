from contextlib import contextmanager
from io import StringIO
import sys
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Context manager to capture output
@contextmanager
def capture_output():
    # StringIO objects are used to capture terminal output and errors
    new_out, new_err = StringIO(), StringIO()  # Create StringIO objects
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = (
            new_out,
            new_err,
        )  # Replace stdout and stderr with the StringIO objects
        # now sys.stdout and sys.stderr will be captured
        yield sys.stdout, sys.stderr  # Let the caller run the code
        print("Terminal Output:", new_out.getvalue())
    finally:  # Restore stdout and stderr
        sys.stdout, sys.stderr = old_out, old_err


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
            "project_to_txt.py",
            "combined.txt",
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
        event_src_path = event.src_path
        print("Event:", event.event_type, event_src_path)
        event_src_path_file_name = os.path.basename(event.src_path)
     
        # if event_src_path contains any of the exclude names, return early
        for exclude_name in self.watcher.EXCLUDE_NAMES:
            if exclude_name in event_src_path:
                print("Event for excluded name:", 
                      exclude_name,
                        "in path:",
                    event_src_path)
                return None
        # Check if the event is for any of the exclude paths or exclude names
        if any(
            exclude_path in event.src_path 
            for exclude_path in self.exclude_paths
        ) or any(
            exclude_name in event_src_path_file_name
            for exclude_name in self.watcher.EXCLUDE_NAMES
        ):
            print("Event for excluded path or name:", event.src_path)
            #  and return early if it is
            return None
       
        elif event.event_type == "modified":
            # Action when a file is modified
            print(f"File {event.src_path} has been modified")
            combine_files(self.watcher)
        elif event.event_type == "created":
            # Action when a file is created
            print(f"File {event.src_path} has been created")
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
        with capture_output() as (out, err):
            # Append captured terminal output and errors
            outfile.write("\n----- Terminal Output -----\n")
            outfile.write(out.getvalue())
            outfile.write("\n----- Terminal Errors -----\n")
            outfile.write(err.getvalue())

    print(f"All files have been combined into {output_file}")


if __name__ == "__main__":
    w = Watcher()
    w.run()
