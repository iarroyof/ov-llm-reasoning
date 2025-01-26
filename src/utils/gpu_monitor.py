# gpu_monitor.py
import torch
import time
import logging
from functools import wraps
from system import os
PATIENCE = 3
SLEEP = 5
from pdb import set_trace as st

def gpu_wait(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        attempts = 0
        while True:
            try:
                if torch.cuda.is_available():
                    st()
                    break
                
                attempts += 1
                logging.info(f"GPU unavailable, waiting {SLEEP}s...")
                time.sleep(SLEEP)
                #if attempts > PATIENCE:
                #    os.run()
            except Exception as e:
                logging.warning(f"GPU check error: {e}")
                time.sleep(SLEEP)
        return func(*args, **kwargs)
    return wrapper
