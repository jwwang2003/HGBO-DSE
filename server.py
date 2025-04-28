import os
from functools import wraps
from flask import Flask, request, jsonify, abort
from flask_socketio import SocketIO, emit
from celery import Celery
import time
from dotenv import load_dotenv

# ─── Load environment variables from .env ───────────────────────────────────────
load_dotenv()  # This will automatically load variables from .env

API_KEY = os.getenv("API_KEY", None)
if not API_KEY:
    raise RuntimeError("You must set the API_KEY environment variable in the .env file")

# ─── Flask & SocketIO setup ───────────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

app.config['CELERY_BROKER_URL']    = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# ─── Auth decorator ───────────────────────────────────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # 1) Try Bearer token in Authorization header
        auth = request.headers.get('Authorization', '')
        token = None
        if auth.startswith('Bearer '):
            token = auth.split()[1]
        # 2) Fallback to query‐param
        if token is None:
            token = request.args.get('api_key')

        if not token or token != API_KEY:
            abort(401, description="Invalid or missing API key")
        return f(*args, **kwargs)
    return decorated

# ─── ML inference stub ────────────────────────────────────────────────────────
def run_inference_model(config_params, file_data):
    for i in range(5):
        time.sleep(1)
        yield f"Progress: {20*(i+1)}%"

@celery.task(bind=True)
def run_inference_task(self, config_params, file_data):
    progress = 0
    for status in run_inference_model(config_params, file_data):
        progress += 20
        # emit over WebSocket
        socketio.emit('progress_update', {'current': progress, 'status': status})
    return {'status': 'Task completed!', 'result': 'Inference Results'}

# ─── REST endpoints ───────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return "Hello, world from HGBO inference backend!"

@app.route('/', methods=['POST'])
@require_api_key
def home_auth():
    return jsonify({'message': "Hello, world from HGBO inference backend!", "authorized": "true"})

@app.route('/start_inference', methods=['POST'])
@require_api_key
def start_inference():
    data = request.get_json(force=True)
    config_params = data.get('config_params')
    file_data     = data.get('file_data')
    task = run_inference_task.apply_async(args=[config_params, file_data])
    return jsonify({'task_id': task.id, 'status': 'started'})

@app.route('/download_result/<task_id>', methods=['GET'])
@require_api_key
def download_result(task_id):
    task = run_inference_task.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        return jsonify({'status': 'success', 'files': ['out1.txt','out2.txt']})
    else:
        return jsonify({'status': 'not ready'}), 400

# ─── WebSocket connection auth ────────────────────────────────────────────────
@socketio.on('connect')
def ws_connect():
    # expect ?token=<API_KEY> on WS URL
    token = request.args.get('token')
    if token != API_KEY:
        return False  # reject connection

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    socketio.run(app, debug=True)
