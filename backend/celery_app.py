"""
Celery entry point of backend inference server
(note that this is just the celery instance, not the full backend)

> celery -A backend.main worker --loglevel=DEBUG
"""

from celery import Celery
from celery.schedules import crontab

app = Celery('backend.main', broker='redis://localhost:6379/0')
app.config_from_object('backend.celeryconfig')  # load default configs

app.conf.update(
    beat_schedule={},
    include=['backend.tasks']
)