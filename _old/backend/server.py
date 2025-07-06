

# OS related libraries
import os
import time
from urllib.parse import urlparse

from functools import wraps
from flask import Flask, request, jsonify, abort
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from celery import Celery
from celery.contrib import rdb

# Environment
from dotenv import load_dotenv

# DB & storage
import redis

# Model
import optuna
from bome import hls_dse

# '''

# '''

# # Some constants (note that all paths are relative to the root project directory)
# EXPERIMENT_NAME = lambda case, mode: f"{case}_{mode}_dse"

# BENCHMARK_PATH = lambda case, ver : f"benchmark/remote/{case}/{ver}"            # Path for source code & yaml files
# RESULT_PATH = lambda case, algo, process: f"{case}_{algo}_prj_p{process}"       # Path for resulting compiled results
# DB_PATH = lambda case, mode: f"{EXPERIMENT_NAME(case, mode)}.db"                # Optimzation plots

# '''
# API ticket parameters:
# - Files
#     - C files
#     - config.yaml
#     - param.yaml
# - mode
# - bench (auto-defiend as "remote")
# - case  (project name)
# - ver   (project name, user-defined, or "don't care")

# - num   (the number of iterations)
# - alg   ()
#     - sa
#     - motpe_d
#     - motpe_fl
#     - nsga
#     - random
#     - else?
# - device (only one device is avaliable)
#     - xc7vx485tffg1761-2
# - clk   (在FPGA上运行的时钟周期约束)
# - encode (编译指令参数的编码方式)		float, discrete
#     - float
#     - discrete
# - space (参数空间构造方式)
#     - tree
#     - homo

# Optional parameters:
# - parallel  (是否并行化执行) 
#     - true
#     - false
# - process   (当前运行的进程) 1, 2, ...
# '''

# # ─── Load environment variables from .env ─────────────────────────────────────
# load_dotenv()  # This will automatically load variables from .env

# API_KEY = os.getenv("API_KEY", None)
# if not API_KEY:
#     raise RuntimeError("You must set the API_KEY environment variable in the .env file")

# # ─── Flask & SocketIO setup ───────────────────────────────────────────────────
# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*")

# # app.config['CELERY_BROKER_URL']    = 'redis://localhost:6379/0'
# # app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# BROKER_URL = "redis://localhost:6379/0"
# celery = Celery(app.name, broker=BROKER_URL, backend=BROKER_URL)
# celery.conf.update(app.config)

# parsed = urlparse(BROKER_URL)
# redis_client = redis.Redis(
#     host=parsed.hostname,
#     port=parsed.port,
#     db=int(parsed.path.lstrip('/')),
# )

# # ─── Auth decorator ───────────────────────────────────────────────────────────
# def require_api_key(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         # 1) Try Bearer token in Authorization header
#         auth = request.headers.get('Authorization', '')
#         token = None
#         if auth.startswith('Bearer '):
#             token = auth.split()[1]
#         # 2) Fallback to query‐param
#         if token is None:
#             token = request.args.get('api_key')

#         if not token or token != API_KEY:
#             abort(401, description="Invalid or missing API key")
#         return f(*args, **kwargs)
#     return decorated

# # ─── ML inference stub ────────────────────────────────────────────────────────
# def run_inference_model():
#     for i in range(5):
#         time.sleep(1)
#         yield f"Progress: {20*(i+1)}%"

# @celery.task(bind=True)
# def run_inference_task(self):
#     progress = 0

#     # class IterationCallback:
#     #     def __init__(self):
#     #         self.caseNumber = 0
            
#     #     def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
#     #         self.caseNumber = self.caseNumber + 1
#     #         socketio.emit('progress_update', {'current': self.caseNumber})
#     #         print("[DEBUG] Case:", self.caseNumber)
        
#     #     def getCaseNumber(self):
#     #         return self.caseNumber
    
#     # iterationCallback = IterationCallback()
#     # print("starting dse...")
#     # hls_dse.runDSE(
#     #     os.getcwd(),
#     #     "hgp",
#     #     "MachSuite",
#     #     "bfs",
#     #     100,
#     #     "motpe_fl",
#     #     "xc7vx485tffg1761-2",
#     #     "10",
#     #     "float",
#     #     "tree",
#     #     False,
#     #     1,
#     #     iterationCallback
#     # )

#     # for test in run_inference_model():
#     #     progress += test
#     #     rdb.set_trace()
#         # socketio.emit('progress_update', {'current': progress})
        
#     return {'status': 'Task completed!', 'result': 'Inference Results'}

# # ─── REST endpoints ───────────────────────────────────────────────────────────
# @app.route('/', methods=['GET'])
# def home():
#     return "Hello, world from HGBO inference backend!"

# @app.route('/', methods=['POST'])
# @require_api_key
# def home_auth():
#     return jsonify({'message': "Hello, world from HGBO inference backend!", "authorized": "true"})

# @app.route('/started', methods=['GET'])
# @require_api_key
# def started():
#     """
#     Ping Redis to confirm the broker is reachable.
#     Returns 200 {"status":"ok"} if Redis.ping() succeeds,
#     or 503 if it fails.
#     """
#     try:
#         if redis_client.ping():
#             return jsonify({'status': 'ok'}), 200
#         else:
#             # Should almost never hit this branch, ping() returns False only on weird setups
#             abort(503, description="Redis did not respond to PING")
#     except redis.RedisError as e:
#         abort(503, description=f"Redis connection error: {e}")

# @app.route('/start_inference', methods=['POST'])
# @require_api_key
# def start_inference():
#     data = request.get_json(force=True)
    
#     task = run_inference_task.apply_async()
#     return jsonify({'task_id': task.id, 'status': 'started'})

# @app.route('/download_result/<task_id>', methods=['GET'])
# @require_api_key
# def download_result(task_id):
#     task = run_inference_task.AsyncResult(task_id)
#     print(task)
#     if task.state == 'SUCCESS':
#         return jsonify({'status': 'success', 'files': ['out1.txt','out2.txt']})
#     else:
#         return jsonify({'status': 'not ready'}), 400

# # ─── WebSocket connection auth ────────────────────────────────────────────────
# @socketio.on('connect')
# def ws_connect():
#     # expect ?token=<API_KEY> on WS URL
#     token = request.args.get('token')
#     if token != API_KEY:
#         return False  # reject connection

# # ─── Run ──────────────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     celery.worker_main(["worker", "--loglevel=info"])
#     socketio.run(app, debug=True, host="0.0.0.0")
