import requests
import time
from websocket import create_connection

# Set the API key and the server URL
API_KEY = 'your_api_key'
SERVER_URL = 'http://localhost:5000'

# Start inference task
def start_inference():
    headers = {'Authorization': f'Bearer {API_KEY}'}
    response = requests.post(f'{SERVER_URL}/start_inference', headers=headers)
    
    if response.status_code == 200:
        task_id = response.json().get('task_id')
        print(f"Task started! Task ID: {task_id}")
        return task_id
    else:
        print("Failed to start inference")
        return None

# Monitor progress via WebSocket
def monitor_progress(task_id):
    ws = create_connection(f"ws://localhost:5000/socket.io/?token={API_KEY}")
    
    while True:
        result = ws.recv()
        print(f"Progress Update: {result}")
        time.sleep(1)
        if "completed" in result.lower():
            print("Inference task completed!")
            break
    ws.close()

# Download results
def download_result(task_id):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    response = requests.get(f'{SERVER_URL}/download_result/{task_id}', headers=headers)
    
    if response.status_code == 200:
        print("Files ready for download:", response.json().get('files'))
    else:
        print("Task not ready yet")

# Example usage
task_id = start_inference()
if task_id:
    monitor_progress(task_id)
    download_result(task_id)
