import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# logging.basicConfig(
#     level=logging.DEBUG,  # Adjust this level to INFO, DEBUG, ERROR depending on verbosity
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s - Line: %(lineno)d',
#     handlers=[
#         # logging.FileHandler(self.log_file),
#         logging.StreamHandler()  # Also outputs to the console
#     ]
# )

def setup_logger(context: str, log_dir: str = './logs', level=logging.DEBUG):
    """
    Setup a logger for a specific context, which logs to a different file based on the context.
    
    :param context: The context for this logger (e.g., "task1", "task2", etc.)
    :param log_dir: Directory where log files will be saved. Defaults to './logs'.
    :param level: The logging level (e.g., DEBUG, INFO).
    :return: Configured logger instance.
    """
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Log file name based on the context
    log_file = os.path.join(log_dir, f"{context}.log")

    # Create a logger with the specified context
    logger = logging.getLogger(context)

    # Set the log level
    logger.setLevel(level)

    # Create a formatter for log entries
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s - Line: %(lineno)d')

    # Check if file handler already exists, if not, create and add it
    if not logger.hasHandlers():
        # Add file handler
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)  # 10 MB per log file, keep 5 backups
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        
        # Add stdout (console) handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
    
    return logger

# Example Usage:
if __name__ == "__main__":
    # Create different loggers for different tasks (contexts)
    task_logger = setup_logger('task_logger', './logs', logging.DEBUG)
    celery_logger = setup_logger('celery_logger', './logs', logging.INFO)

    # Log something to show the setup is working
    task_logger.debug("This is a debug message from task_logger.")
    celery_logger.info("This is an info message from celery_logger.")
