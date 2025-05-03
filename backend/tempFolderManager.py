"""
Temporary Folder Manager

This script defines a class `TempFolderManager` that is responsible for managing
temporary folders used to store project files. The key features include:

1. **Unique Folder Creation**: Each folder is created with a unique name that
   combines the current timestamp (formatted as YYYYMMDD_HHMMSS) and a UUID.
   This ensures that no folder names overlap.

2. **Folder History**: The history of created folders is stored in a file 
   (`folder_history.txt`). This file records the name and creation timestamp 
   of each folder. The history is loaded when the script starts and updated 
   whenever a new folder is created.

3. **Automatic Folder Cleanup**: The manager allows for automatic deletion of 
   folders that have been in existence for longer than a specified duration 
   (in minutes). This ensures that old folders are cleaned up and don't take up 
   unnecessary space.

4. **Configurable Retention Time**: The retention time for the folder history 
   is configurable. You can set the duration for how long folders are kept 
   before they are deleted.

Usage Example:
    - Create a new temporary folder: `create_temp_folder()`
    - Clean old folders that exceed the retention time: `clean_old_folders()`

This solution provides an efficient way to manage temporary folders, ensuring 
they are uniquely named, their history is tracked, and outdated folders are 
automatically deleted after a configurable duration.
"""

import os
import uuid
from datetime import datetime, timedelta
import shutil

class TempFolderManager:
    def __init__(self, base_dir, history_duration_minutes=60):
        self.base_dir = base_dir
        self.history_duration = timedelta(minutes=history_duration_minutes)
        self.history_file = "folder_history.txt"  # Store folder history
        
        # Ensure the base directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize history
        self.load_history()

    def load_history(self):
        """Load the history of created folders from the history file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as file:
                self.history = {line.strip().split('|')[0]: line.strip().split('|')[1] for line in file.readlines()}
        else:
            self.history = {}

    def save_history(self):
        """Save the current history to the history file"""
        with open(self.history_file, 'w') as file:
            for folder, timestamp in self.history.items():
                file.write(f"{folder}|{timestamp}\n")

    def generate_unique_folder_name(self):
        """Generate a unique folder name using UUID and timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4()).replace('-', '')
        folder_name = f"project_{timestamp}_{unique_id}"
        return folder_name

    def create_temp_folder(self):
        """Create a new temporary folder and store it in the history"""
        folder_name = self.generate_unique_folder_name()
        folder_path = os.path.join(self.base_dir, folder_name)

        # Create the folder
        os.makedirs(folder_path)

        # Record the folder in the history
        creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history[folder_name] = creation_time
        self.save_history()

        return folder_path

    def clean_old_folders(self):
        """Clean folders that are older than the specified duration"""
        current_time = datetime.now()
        for folder, timestamp in list(self.history.items()):
            folder_creation_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            if current_time - folder_creation_time > self.history_duration:
                folder_path = os.path.join(self.base_dir, folder)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}")
                del self.history[folder]  # Remove from history
        self.save_history()