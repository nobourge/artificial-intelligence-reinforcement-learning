import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Watcher:
    # watch all directories
    # DIRECTORY_TO_WATCH = "/path/to/your/files"
    # DIRECTORY_TO_WATCH = "./*"
    # watch current directory's all subdirectories files
    # DIRECTORY_TO_WATCH = "./**/*" # OSError: [WinError 123] The filename, directory name, or volume label syntax is incorrect: './**/*'

    DIRECTORY_TO_WATCH = "src"
    # DIRECTORY_TO_WATCH = "./src"
    # DIRECTORY_TO_WATCH = ".\\src"
    # print directories and files names
    for filename in os.listdir(DIRECTORY_TO_WATCH):
        print(filename)

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()


class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == "modified":
            # Take any action here when a file is modified.
            print(f"File {event.src_path} has been modified")
            combine_files()


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


def combine_files():
    root_directory = Watcher.DIRECTORY_TO_WATCH   # "/path/to/your/files"
    output_file = "combined.txt"

    with open(output_file, "w", encoding="utf-8") as outfile:
        for subdir, dirs, files in os.walk(root_directory):
            for filename in files:
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
