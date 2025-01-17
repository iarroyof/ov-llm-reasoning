# src/utils/cache_utils.py

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional

def generate_cache_key(split_params: Dict) -> str:
    """Generate unique cache key based on parameters."""
    param_str = json.dumps(split_params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()[:16]

def save_split_cache(
        train_pairs: List[Tuple[str, int]],
        test_pairs: List[Tuple[str, int]],
        split_params: Dict,
        cache_dir: str = "cache") -> str:
    """Save train/test split of sentence-triplet pairs to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = generate_cache_key(split_params)
    cache_file = os.path.join(cache_dir, f"split_{cache_key}.json")
    
    cache_data = {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "split_params": split_params
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    
    return cache_key

def load_split_cache(
        split_params: Dict,
        cache_dir: str = "cache") -> Optional[Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]]:
    """Load train/test split from cache if available."""
    cache_key = generate_cache_key(split_params)
    cache_file = os.path.join(cache_dir, f"split_{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        if cache_data["split_params"] != split_params:
            return None
        
        return cache_data["train_pairs"], cache_data["test_pairs"]
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None
