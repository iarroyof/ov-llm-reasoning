# src/utils/triplet_utils.py

from enum import Enum
from typing import Set, Optional, List, Dict, Tuple
import re

class FilterMethod(Enum):
    STOPWORDS = "stopwords"

# Basic stopwords excluding semantically meaningful ones
BASIC_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'the', 'this', 'those', 'these', 'there', 'here'
}

# Words to keep even if they might be considered stopwords
SEMANTIC_KEEPERS = {
    'who', 'what', 'when', 'where', 'why', 'how',
    'no', 'not', 'none', 'never',
    'all', 'every', 'some', 'any', 'many', 'few',
    'before', 'after', 'during', 'while',
    'because', 'therefore', 'however', 'although',
    'more', 'less', 'better', 'worse',
    'my', 'your', 'their', 'his', 'her', 'its'
}

def load_stopwords(filepath: Optional[str] = None) -> Set[str]:
    """Load stopwords from file or use defaults."""
    if filepath:
        try:
            with open(filepath, 'r') as f:
                custom_words = {word.strip().lower() for word in f if word.strip()}
            return custom_words - SEMANTIC_KEEPERS
        except Exception as e:
            print(f"Error loading custom stopwords: {e}. Using defaults.")
    return BASIC_STOPWORDS - SEMANTIC_KEEPERS

def is_stopwords_only(text: str, stopwords: Set[str]) -> bool:
    """Check if text contains only stopwords."""
    words = set(re.findall(r'\w+', text.lower()))
    return words and all(word in stopwords for word in words)

class TripletFilter:
    """Filter triplets based on different methods."""
    
    def __init__(
            self,
            method: FilterMethod = FilterMethod.STOPWORDS,
            stopwords_file: Optional[str] = None):
        self.method = method
        self.stopwords = load_stopwords(stopwords_file)
    
    def should_keep_triplet(
            self,
            subject: str,
            relation: str,
            object_: str) -> bool:
        """Determine if triplet should be kept."""
        if self.method == FilterMethod.STOPWORDS:
            subject_stops = is_stopwords_only(subject, self.stopwords)
            object_stops = is_stopwords_only(object_, self.stopwords)
            return not (subject_stops and object_stops)
        return True
