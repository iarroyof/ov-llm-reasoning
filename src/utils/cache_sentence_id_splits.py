"""Utilities for caching dataset splits."""

import os
import json
import hashlib
from typing import Tuple, List, Dict, Optional

def generate_cache_key(split_params: Dict) -> str:
    """
    Generate a unique cache key based on split parameters.
    
    Args:
        split_params: Dictionary containing split parameters
        
    Returns:
        str: Cache key as hexadecimal string
    """
    # Create a sorted string representation of parameters for consistent hashing
    param_str = json.dumps(split_params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()[:16]

def save_split_cache(
    train_ids: List[str],
    test_ids: List[str],
    split_params: Dict,
    cache_dir: str = "cache"
) -> str:
    """
    Save train/test split to cache.
    
    Args:
        train_ids: List of training document IDs
        test_ids: List of test document IDs
        split_params: Parameters used to create the split
        cache_dir: Directory to store cache files
        
    Returns:
        str: Cache key used to save the split
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = generate_cache_key(split_params)
    cache_file = os.path.join(cache_dir, f"split_{cache_key}.json")
    
    cache_data = {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "split_params": split_params
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    
    return cache_key

def load_split_cache(
    split_params: Dict,
    cache_dir: str = "cache"
) -> Optional[Tuple[List[str], List[str]]]:
    """
    Load train/test split from cache if available.
    
    Args:
        split_params: Parameters to identify the split
        cache_dir: Directory containing cache files
        
    Returns:
        Tuple of (train_ids, test_ids) if cache exists, None otherwise
    """
    cache_key = generate_cache_key(split_params)
    cache_file = os.path.join(cache_dir, f"split_{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return None
        
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            
        # Verify parameters match
        if cache_data["split_params"] != split_params:
            return None
            
        return cache_data["train_ids"], cache_data["test_ids"]
        
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None
