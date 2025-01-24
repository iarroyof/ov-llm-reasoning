# gpu_monitor.py
import torch
import time
import logging
from functools import wraps

def gpu_wait(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                if torch.cuda.is_available():
                    break
                logging.info("GPU unavailable, waiting 60s...")
                time.sleep(60)
            except Exception as e:
                logging.warning(f"GPU check error: {e}")
                time.sleep(60)
        return func(*args, **kwargs)
    return wrapper
