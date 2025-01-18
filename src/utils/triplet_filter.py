# src/utils/triplet_filter.py

from enum import Enum
from typing import Set, Dict, Any, List, Tuple, Optional
import re
from pathlib import Path

class FilterMethod(Enum):
    NONE = "none"
    STOPWORDS = "stopwords"
    # Future methods can be added here

class StopwordsConfig:
    # Basic stopwords that are typically not semantically meaningful
    BASIC_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'this', 'those', 'there',
        'here', 'they', 'them', 'these', 'their', 'which', 'when', 'what'
    }

    # Words to keep even if they might be considered stopwords
    SEMANTIC_KEEPERS = {
        # Question words (important for reasoning)
        'who', 'what', 'when', 'where', 'why', 'how',
        
        # Negations (critical for logic)
        'no', 'not', 'none', 'never', 'neither', 'nor',
        
        # Quantifiers
        'all', 'every', 'some', 'any', 'many', 'few', 'most', 'each',
        
        # Temporal indicators
        'before', 'after', 'during', 'while', 'until', 'since',
        
        # Logical connectors
        'because', 'therefore', 'however', 'although', 'despite', 'hence',
        
        # Comparatives and superlatives
        'more', 'less', 'better', 'worse', 'best', 'worst', 'most', 'least',
        
        # Entity references
        'he', 'she', 'it', 'they', 'we', 'you', 'who',
        
        # Important prepositions
        'through', 'within', 'without', 'between', 'among'
    }

class TripletFilter:
    """Filters triplets based on configurable criteria."""
    
    def __init__(
            self,
            method: FilterMethod = FilterMethod.NONE,
            stopwords_file: Optional[Path] = None,
            keep_semantic_stopwords: bool = True):
        """
        Initialize the triplet filter.
        
        Args:
            method: Filtering method to use
            stopwords_file: Optional path to custom stopwords file
            keep_semantic_stopwords: Whether to keep semantically meaningful stopwords
        """
        self.method = method
        self.stopwords = self._load_stopwords(stopwords_file, keep_semantic_stopwords)
    
    def _load_stopwords(self, stopwords_file: Optional[Path], keep_semantic: bool) -> Set[str]:
        """Load stopwords from file or use defaults."""
        stopwords = set(StopwordsConfig.BASIC_STOPWORDS)
        
        # Load custom stopwords if provided
        if stopwords_file and stopwords_file.exists():
            with open(stopwords_file, 'r') as f:
                custom_words = {word.strip().lower() for word in f if word.strip()}
            stopwords.update(custom_words)
        
        # Remove semantic keepers if specified
        if keep_semantic:
            stopwords = stopwords - StopwordsConfig.SEMANTIC_KEEPERS
            
        return stopwords
    
    def _has_non_stopwords(self, text: str) -> bool:
        """Check if text contains any non-stopword."""
        if not self.stopwords:
            return True
            
        # Tokenize and clean text
        words = set(re.findall(r'\w+', text.lower()))
        return any(word not in self.stopwords for word in words)
    
    def should_keep_triplet(
            self,
            subject: str,
            relation: str,
            object_: str) -> bool:
        """
        Determine if triplet should be kept based on filtering method.
        """
        if self.method == FilterMethod.NONE:
            return True
            
        if self.method == FilterMethod.STOPWORDS:
            # Keep triplet if either subject or object has non-stopwords
            return (self._has_non_stopwords(subject) or 
                   self._has_non_stopwords(object_))
        
        return True

def process_and_filter_triplets(
        doc_id: str,
        doc_source: Dict[str, Any],
        triplet_filter: TripletFilter) -> Tuple[List[Tuple[str, int]], int, int]:
    """
    Process document and return valid (sentence_id, triplet_idx) pairs plus counts.
    
    Returns:
        Tuple of (valid_pairs, total_triplets, valid_triplets)
    """
    valid_pairs = []
    total_triplets = 0
    
    if 'triplets' not in doc_source:
        return valid_pairs, 0, 0
        
    total_triplets = len(doc_source['triplets'])
    
    for idx, triplet in enumerate(doc_source['triplets']):
        if triplet_filter.should_keep_triplet(
            triplet['subject']['text'],
            triplet['relation']['text'],
            triplet['object']['text']
        ):
            valid_pairs.append((doc_id, idx))
            
    return valid_pairs, total_triplets, len(valid_pairs)
