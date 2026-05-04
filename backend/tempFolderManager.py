import os
import shutil
import uuid
from datetime import datetime, timedelta


class TempFolderManager:
    def __init__(self, base_dir, history_duration_minutes=1):
        self.base_dir = base_dir
        self.history_duration = timedelta(minutes=history_duration_minutes)
        self.history_file = "folder_history.txt"

        os.makedirs(base_dir, exist_ok=True)
        self.load_history()

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as file:
                self.history = {
                    line.strip().split("|")[0]: line.strip().split("|")[1]
                    for line in file.readlines()
                }
        else:
            self.history = {}

    def save_history(self):
        with open(self.history_file, "w") as file:
            for folder, timestamp in self.history.items():
                file.write(f"{folder}|{timestamp}\n")

    def generate_unique_folder_name(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4()).replace("-", "")
        return f"project_{timestamp}_{unique_id}"

    def create_temp_folder(self):
        folder_name = self.generate_unique_folder_name()
        folder_path = os.path.join(self.base_dir, folder_name)

        os.makedirs(folder_path)
        self.history[folder_name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_history()

        return folder_path

    def clean_old_folders(self):
        current_time = datetime.now()
        for folder, timestamp in list(self.history.items()):
            folder_creation_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            if current_time - folder_creation_time > self.history_duration:
                folder_path = os.path.join(self.base_dir, folder)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                del self.history[folder]
        self.save_history()
