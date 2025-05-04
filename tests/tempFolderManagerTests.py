import unittest
from unittest.mock import patch, MagicMock
import os
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
import uuid
import shutil

from backend.tempFolderManager import TempFolderManager  # Assuming the code is saved in temp_folder_manager.py


class TestTempFolderManager(unittest.TestCase):
    @patch('os.makedirs')  # Mock os.makedirs to prevent actual folder creation
    @patch('shutil.rmtree')  # Mock shutil.rmtree to prevent actual folder deletion
    @patch('builtins.open', new_callable=MagicMock)  # Mock open to simulate file reading and writing
    def test_create_temp_folder(self, mock_open, mock_rmtree, mock_makedirs):
        # Arrange
        base_dir = "test_base"
        manager = TempFolderManager(base_dir)
        
        # Simulate empty history file
        mock_open.return_value.readlines.return_value = []
        
        # Act
        folder_path = manager.create_temp_folder()

        # Assert folder path exists and is returned as expected
        self.assertTrue(folder_path.startswith(base_dir))
        self.assertTrue(os.path.exists(folder_path))

        # Ensure folder history file is written correctly
        mock_open.assert_called_once_with('folder_history.txt', 'w')
        mock_open.return_value.write.assert_called()

    @patch('os.makedirs')
    @patch('shutil.rmtree')
    @patch('builtins.open', new_callable=MagicMock)
    def test_clean_old_folders(self, mock_open, mock_rmtree, mock_makedirs):
        # Arrange
        base_dir = "test_base"
        history_data = {
            "project_20250314_100101_12345": "2023-03-14 10:00:00",
            "project_20250314_110101_12346": "2023-03-15 10:00:00",
            "project_20250314_120101_12347": "2023-03-16 10:00:00",
        }

        manager = TempFolderManager(base_dir, history_duration_minutes=30)
        manager.load_history()

        # Mock open to return our predefined history
        mock_open.return_value.readlines.return_value = [
            f"{folder}|{timestamp}\n" for folder, timestamp in history_data.items()
        ]

        # Simulate the folders
        for folder in history_data:
            mock_makedirs.return_value = None
        
        # Act: Simulate cleanup after 30 minutes retention
        manager.clean_old_folders()

        # Assert folders older than 30 minutes are deleted
        for folder in history_data:
            if folder != "project_20250314_120101_12347":
                mock_rmtree.assert_any_call(os.path.join(base_dir, folder))
            else:
                mock_rmtree.assert_not_called()

        # Assert the history is updated
        mock_open.return_value.write.assert_called_once()

    @patch('os.makedirs')
    @patch('shutil.rmtree')
    @patch('builtins.open', new_callable=MagicMock)
    def test_load_and_save_history(self, mock_open, mock_rmtree, mock_makedirs):
        # Arrange
        base_dir = "test_base"
        history_data = {
            "project_20250314_100101_12345": "2023-03-14 10:00:00",
            "project_20250314_110101_12346": "2023-03-15 10:00:00",
        }

        # Simulate the history file content
        mock_open.return_value.readlines.return_value = [
            f"{folder}|{timestamp}\n" for folder, timestamp in history_data.items()
        ]

        manager = TempFolderManager(base_dir, history_duration_minutes=60)

        # Act: Check if the history is loaded correctly
        self.assertEqual(manager.history, history_data)

        # Act: Add a new folder and ensure history is updated
        manager.create_temp_folder()

        # Assert the history file is saved
        mock_open.return_value.write.assert_called()

    @patch('os.makedirs')
    @patch('shutil.rmtree')
    def test_generate_unique_folder_name(self, mock_rmtree, mock_makedirs):
        # Arrange
        base_dir = "test_base"
        manager = TempFolderManager(base_dir)

        # Act: Generate a unique folder name
        folder_name = manager.generate_unique_folder_name()

        # Assert that the folder name contains a timestamp and UUID
        self.assertTrue(folder_name.startswith("project_"))
        self.assertTrue(len(folder_name.split("_")) >= 3)  # Timestamp, UUID, etc.

    @patch('os.makedirs')
    @patch('shutil.rmtree')
    def test_cleanup_no_old_folders(self, mock_rmtree, mock_makedirs):
        # Arrange
        base_dir = "test_base"
        manager = TempFolderManager(base_dir, history_duration_minutes=1)
        
        # Set current time to be very close to the folder creation time
        manager.history = {
            "project_20250314_100101_12345": (datetime.now() - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Act: Clean up folders
        manager.clean_old_folders()

        # Assert no folder is deleted
        mock_rmtree.assert_not_called()

    @patch('os.makedirs')
    @patch('shutil.rmtree')
    @patch('builtins.open', new_callable=MagicMock)
    def test_history_not_created_when_file_does_not_exist(self, mock_open, mock_rmtree, mock_makedirs):
        # Arrange
        base_dir = "test_base"
        
        # Simulate no history file exists
        mock_open.side_effect = FileNotFoundError
        
        # Act
        manager = TempFolderManager(base_dir)
        
        # Assert the history should be an empty dictionary when the file doesn't exist
        self.assertEqual(manager.history, {})


if __name__ == '__main__':
    unittest.main()
