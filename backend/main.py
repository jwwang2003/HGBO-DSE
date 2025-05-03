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