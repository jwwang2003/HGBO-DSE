import os
import logging
from logging.handlers import RotatingFileHandler
from helpers.context import get_context_id

class ContextFilter(logging.Filter):
    """Inject the current or overridden context UUID into the LogRecord as `context_id`."""
    def __init__(self, override_context: str = None):
        super().__init__()
        self.override_context = override_context

    def filter(self, record):
        record.context_id = self.override_context or get_context_id()
        return True

class ContextualFileHandler(logging.Handler):
    """
    Routes each record into a RotatingFileHandler specific to its record.context_id.
    """
    def __init__(self, log_dir: str, maxBytes: int = 10*1024*1024, backupCount: int = 5):
        super().__init__()
        self.log_dir = log_dir
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.handlers: dict[str, RotatingFileHandler] = {}
        os.makedirs(self.log_dir, exist_ok=True)

    def emit(self, record: logging.LogRecord):
        cid = getattr(record, "context_id", get_context_id())
        if cid not in self.handlers:
            log_path = os.path.join(self.log_dir, f"{cid}.log")
            fh = RotatingFileHandler(
                log_path, maxBytes=self.maxBytes, backupCount=self.backupCount
            )
            fh.setFormatter(self.formatter)
            # this filter ensures fh.format(record) has record.context_id set
            fh.addFilter(ContextFilter(override_context=cid))
            self.handlers[cid] = fh
        self.handlers[cid].emit(record)

def setup_logger(
    log_dir: str = "./logs",
    level: int = logging.DEBUG,
    context: str = None
) -> logging.Logger:
    """
    Configure and return a logger named for this context (or 'root' if none).
    
    :param log_dir: Directory to hold log files.
    :param level:   Logging level.
    :param context: If provided, use this fixed context_id and logger name.
    """
    # Use the context as the logger name (so each context can coexist)
    logger_name = context or "root"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # ---- clear previous handlers & filters so override always applies ----
    for h in list(logger.handlers):
        logger.removeHandler(h)
    for f in list(logger.filters):
        logger.removeFilter(f)
    # ----------------------------------------------------------------------

    # 1) Context filter
    ctx_filter = ContextFilter(override_context=context)
    logger.addFilter(ctx_filter)

    # 2) Console handler
    console_fmt = logging.Formatter(
        '%(asctime)s - %(levelname)s - [ctx=%(context_id)s] - %(message)s'
    )
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(console_fmt)
    ch.addFilter(ctx_filter)
    logger.addHandler(ch)

    # 3) Contextual file handler
    file_fmt = logging.Formatter(
        '%(asctime)s - %(levelname)s - [ctx=%(context_id)s] - '
        '%(filename)s:%(lineno)d - %(message)s'
    )
    cfh = ContextualFileHandler(log_dir)
    cfh.setLevel(level)
    cfh.setFormatter(file_fmt)
    logger.addHandler(cfh)

    return logger
