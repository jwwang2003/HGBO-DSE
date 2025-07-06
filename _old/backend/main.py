"""
Entry point of backend inference server
- Flask Rest API
- Websocket for live updates
- Celery for a background task broker

How to run:
- First start up celery worker:
    > celery -A backend.main.celery worker --loglevel=DEBUG
- Start Python backend:
    > FLASK_ENV=production python -m backend.main
    > FLASK_ENV=development python -m backend.main
"""

import os
from flask import Flask, request, render_template, session, flash, redirect, \
    url_for, jsonify, abort, make_response
from celery import Celery
from celery.schedules import crontab

# Environment
from dotenv import load_dotenv

from backend.auth import require_auth_token

# Flash blueprints
from backend.api.auth import auth_bp
from backend.api.config import config_bp

# Some constants (note that all paths are relative to the root project directory)
EXPERIMENT_NAME = lambda case, mode: f"{case}_{mode}_dse"

BENCHMARK_PATH = lambda case, ver : f"benchmark/remote/{case}/{ver}"            # Path for source code & yaml files
RESULT_PATH = lambda case, algo, process: f"{case}_{algo}_prj_p{process}"       # Path for resulting compiled results
DB_PATH = lambda case, mode: f"{EXPERIMENT_NAME(case, mode)}.db"                # Optimzation plots

'''
API ticket parameters:
- Files
    - C files
    - config.yaml
    - param.yaml
- mode
- bench (auto-defiend as "remote")
- case  (project name)
- ver   (project name, user-defined, or "don't care")

- num   (the number of iterations)
- alg   ()
    - sa
    - motpe_d
    - motpe_fl
    - nsga
    - random
    - else?
- device (only one device is avaliable)
    - xc7vx485tffg1761-2
- clk   (在FPGA上运行的时钟周期约束)
- encode (编译指令参数的编码方式)		float, discrete
    - float
    - discrete
- space (参数空间构造方式)
    - tree
    - homo

Optional parameters:
- parallel  (是否并行化执行) 
    - true
    - false
- process   (当前运行的进程) 1, 2, ...
'''

# ─── Load environment variables from .env ─────────────────────────────────────
load_dotenv()  # This will automatically load variables from .env

# Check environment type
FLASK_ENV = os.getenv('FLASK_ENV', 'development')  # Default to development if not set
PORT = os.getenv('PORT', 5000)

SECRET_KEY = os.getenv('SECRET_KEY', '123456')

REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = os.getenv('REDIS_PORT')

CELERY_BROKER_REDIS_DB = os.getenv('CELERY_BROKER_REDIS_DB', '0')
AUTH_SESSION_MANAGER_REDIS_DB = os.getenv('AUTH_SESSION_MANAGER_REDIS_DB', '1')

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# Set configurations based on the environment
app.config['CELERY_BROKER_URL'] = \
    f'redis://{REDIS_HOST}:{REDIS_PORT}/{CELERY_BROKER_REDIS_DB}'
app.config['CELERY_RESULT_BACKEND'] = \
    f'redis://{REDIS_HOST}:{REDIS_PORT}/{CELERY_BROKER_REDIS_DB}'
if FLASK_ENV == 'production':
    app.config['DEBUG'] = False  # Disable debug in production
else:
    app.config['DEBUG'] = True  # Enable debug in development

# Initialize Celery with the configuration
celery = Celery('backend.main', broker=app.config['CELERY_BROKER_URL'])
celery.config_from_object('backend.celeryconfig')  # load default configs

celery.conf.update(
    beat_schedule={},
    include=['backend.tasks']
)

# Register the blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(config_bp, url_prefix='/config')

# ─── REST endpoints ───────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': "Hello, world from HGBO inference backend!"})

@app.route('/', methods=['POST'])
@require_auth_token
def home_auth():
    return jsonify({'message': "Hello, world from HGBO inference backend! (authorized)"})

@app.route('/upload', methods=['POST']) 
def upload(): 
    if request.method == 'POST': 
  
        # Get the list of files from webpage 
        files = request.files.getlist("file") 
  
        # Iterate for each file in the files List, and Save them 
        for file in files: 
            file.save(file.filename) 
        return "<h1>Files Uploaded Successfully.!</h1>"

if __name__ == "__main__":
    host = "0.0.0.0"
    if FLASK_ENV == "production":
        from waitress import serve
        print('Starting to serve in production mode...')
        serve(app, host=host, port=5000)
    else:
        app.run(host=host, port=PORT, debug=app.config['DEBUG'])

"""
Entry point of backend inference server
- Flask Rest API
- Websocket for live updates
- Celery for a background task broker
"""

from celery import Celery
from celery.schedules import crontab

app = Celery('backend.main', broker='redis://localhost:6379/0')
app.config_from_object('backend.celeryconfig')  # load default configs

app.conf.update(
    beat_schedule={},
    include=['backend.tasks']
)

# from backend.tasks import run_inference_task 

# if __name__ == "__main__":
#     result = run_inference_task.delay()
#     print('Task result: ', run_inference_task.get())