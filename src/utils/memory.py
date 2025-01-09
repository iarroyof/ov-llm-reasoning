# src/utils/memory.py

import torch
import psutil

def log_gpu_memory_usage():
    """Log detailed GPU and RAM memory usage"""
    memory_stats = {}
    
    # GPU Memory
    for i in range(torch.cuda.device_count()):
        memory_stats.update({
            f'gpu_{i}_used_gb': torch.cuda.memory_allocated(i) / 1e9,
            f'gpu_{i}_cached_gb': torch.cuda.memory_reserved(i) / 1e9,
            f'gpu_{i}_free_gb': (torch.cuda.get_device_properties(i).total_memory - 
                                torch.cuda.memory_allocated(i)) / 1e9
        })
    
    # RAM Memory
    ram = psutil.virtual_memory()
    memory_stats.update({
        'ram_used_gb': ram.used / 1e9,
        'ram_free_gb': ram.free / 1e9,
        'ram_percent': ram.percent
    })
    
    return memory_stats
