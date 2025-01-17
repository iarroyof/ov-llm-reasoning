# src/utils/cache_utils.py

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from .triplet_filter import FilterMethod

def serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert parameters to JSON-serializable format.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Dictionary with all values converted to JSON-serializable types
    """
    serialized = {}
    for key, value in params.items():
        if isinstance(value, FilterMethod):
            serialized[key] = value.value  # Convert enum to string
        elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
            serialized[key] = value  # These types are already JSON-serializable
        else:
            serialized[key] = str(value)  # Convert other types to string
    return serialized

def deserialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert parameters back from JSON-serialized format.
    
    Args:
        params: Dictionary of serialized parameters
        
    Returns:
        Dictionary with values converted back to original types where possible
    """
    deserialized = params.copy()
    if 'filter_method' in params:
        # Convert string back to FilterMethod enum
        deserialized['filter_method'] = FilterMethod(params['filter_method'])
    return deserialized

def generate_cache_key(split_params: Dict) -> str:
    """Generate unique cache key based on parameters."""
    # Serialize parameters before generating key
    serialized_params = serialize_params(split_params)
    param_str = json.dumps(serialized_params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()[:16]

def save_split_cache(
        train_pairs: List[Tuple[str, int]],
        test_pairs: List[Tuple[str, int]],
        split_params: Dict,
        cache_dir: str = "cache") -> str:
    """Save train/test split to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = generate_cache_key(split_params)
    cache_file = os.path.join(cache_dir, f"split_{cache_key}.json")
    
    # Serialize parameters for storage
    cache_data = {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "split_params": serialize_params(split_params)
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
        
        # Deserialize and compare parameters
        stored_params = deserialize_params(cache_data["split_params"])
        current_params = serialize_params(split_params)
        
        if serialize_params(stored_params) != current_params:
            return None
        
        # Convert stored pairs back to tuples (JSON stores them as lists)
        train_pairs = [tuple(pair) for pair in cache_data["train_pairs"]]
        test_pairs = [tuple(pair) for pair in cache_data["test_pairs"]]
        
        return train_pairs, test_pairs
        
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None
