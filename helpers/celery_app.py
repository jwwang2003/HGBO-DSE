import logging
from celery import Celery, Task
from .context import set_context_id, clear_context_id, get_context_id

# configure root logger to show INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextTask(Task):
    """Assign a fresh UUID to each task invocation."""
    def __call__(self, *args, **kwargs):
        set_context_id()
        try:
            return self.run(*args, **kwargs)
        finally:
            clear_context_id()

# adjust broker/backend URLs as needed:
app = Celery(
    'demo',
    broker   = 'redis://localhost:6379/0',
    backend  = 'redis://localhost:6379/0',
)
app.Task = ContextTask

@app.task
def report_context():
    """
    A trivial task that logs and returns its context UUID.
    """
    cid = get_context_id()
    logger.info(f"[report_context] context UUID = {cid}")
    return cid