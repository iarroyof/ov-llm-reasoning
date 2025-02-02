from torch.utils.data import IterableDataset
from typing import List, Any, Dict, Tuple, Callable, Optional
from elasticsearch import Elasticsearch
import random
import torch
from transformers import PreTrainedTokenizer
from collections import deque


class ElasticSearchDataset(IterableDataset):
    """
    Optimized ElasticSearch dataset with progressive loading and memory efficiency.
    """
    
    @staticmethod
    def create_train_test_split(
            url: str, 
            index: str, 
            max_docs2load: int,
            test_ratio: float = 0.3, 
            seed: Optional[int] = None,
            filter_article_ids: Optional[List[str]] = None,
            es_page_size: int = 500) -> Tuple[List[str], List[str]]:
        """
        Create train/test split of document IDs with optional article ID filtering.
        
        Args:
            url: Elasticsearch URL
            index: Index name
            max_docs2load: Maximum number of documents to load in total
            test_ratio: Ratio of documents to use for testing
            seed: Random seed for reproducibility
            filter_article_ids: Optional list of article IDs to filter by
            es_page_size: Number of documents to fetch in each Elasticsearch request
            
        Returns:
            Tuple of (train_ids, test_ids)
        """
        es_client = Elasticsearch(url)
        
        # Build query based on whether we have article IDs to filter
        if filter_article_ids:
            query = {
                "size": es_page_size,  # Using es_page_size instead of hardcoded value
                "query": {
                    "terms": {
                        "article_id.keyword": filter_article_ids
                    }
                },
                "_source": False
            }
        else:
            query = {
                "size": es_page_size,  # Using es_page_size instead of hardcoded value
                "query": {"match_all": {}},
                "_source": False
            }
        
        try:
            # Get initial scroll ID
            response = es_client.search(
                index=index,
                body=query,
                scroll='5m'
            )
            
            scroll_id = response['_scroll_id']
            all_ids = [hit['_id'] for hit in response['hits']['hits']]
            
            try:
                # Continue scrolling until we have all matching documents
                while True:
                    response = es_client.scroll(
                        scroll_id=scroll_id,
                        scroll='5m'
                    )
                    
                    hits = response['hits']['hits']
                    if not hits:
                        break
                        
                    all_ids.extend(hit['_id'] for hit in hits)
                    
            finally:
                # Clear scroll context
                try:
                    es_client.clear_scroll(scroll_id=scroll_id)
                except Exception as e:
                    print(f"Error clearing scroll: {e}")
            
            # If we have more documents than max_docs2load, randomly sample
            if len(all_ids) > max_docs2load:
                if seed is not None:
                    random.seed(seed)
                all_ids = random.sample(all_ids, max_docs2load)
            
            # Shuffle the IDs
            if seed is not None:
                random.seed(seed)
            random.shuffle(all_ids)
            
            # Split into train and test
            test_size = int(len(all_ids) * test_ratio)
            train_ids = all_ids[test_size:]
            test_ids = all_ids[:test_size]
            
            return train_ids, test_ids
            
        except Exception as e:
            print(f"Error fetching IDs: {e}")
            return [], []
        
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_ids)
        
        test_size = int(len(all_ids) * test_ratio)
        train_ids = all_ids[test_size:]
        test_ids = all_ids[:test_size]
        
        return train_ids, test_ids
                
    def __init__(
            self,
            url: str,
            index: str,
            tokenizer: PreTrainedTokenizer,
            es_page_size: int = 500,
            batch_size: int = 8,
            filter_article_ids: Optional[List[str]] = None,
            source_len: int = 20,
            target_len: int = 30,
            true_sample_f: Optional[Callable] = None,
            seed: Optional[int] = None,
            selected_doc_ids: Optional[List[str]] = None,
            cache_size_limit: int = 5000):
        
        self.es_client = Elasticsearch(url)
        self.index = index
        self.tokenizer = tokenizer
        self.es_page_size = es_page_size
        self.batch_size = batch_size
        self.filter_article_ids = filter_article_ids
        self.source_len = source_len
        self.target_len = target_len
        self.true_sample_f = true_sample_f or (lambda x: x)
        self.seed = seed
        self.cache_size_limit = cache_size_limit
        
        # Progressive loading state
        self.current_page_index = 0
        self.data_buffer = deque(maxlen=self.cache_size_limit)
        self.selected_doc_ids = selected_doc_ids
        
        # Initialize document IDs
        self._initialize_document_ids()
        
        if seed is not None:
            random.seed(seed)

    def _initialize_document_ids(self):
        """Initialize document IDs using pre-selected IDs if available"""
        self.document_ids = list(self.selected_doc_ids) if self.selected_doc_ids else []
        self.total_docs = len(self.document_ids)

    def _build_base_query(self) -> Dict:
        """Build base query with article ID filtering if needed"""
        if not self.filter_article_ids:
            return {"match_all": {}}
            
        return {
            "bool": {
                "must": [{
                    "terms": {
                        "article_id.keyword": self.filter_article_ids
                    }
                }]
            }
        }

    def __len__(self) -> int:
        """Return number of batches"""
        return self.total_docs // self.batch_size

    def __iter__(self):
        """Reset iterator state"""
        self.current_page_index = 0
        self.data_buffer.clear()
        return self

    def __next__(self):
        """Returns next batch of processed data"""
        # Load next page if buffer is getting low
        if len(self.data_buffer) < self.batch_size:
            self.load_next_page()
            
        if not self.data_buffer:
            raise StopIteration
        
        # Create batch
        batch = []
        for _ in range(min(self.batch_size, len(self.data_buffer))):
            if self.data_buffer:
                batch.append(self.data_buffer.popleft())
            
        if not batch:
            raise StopIteration
            
        return {
            'source_ids': torch.stack([x['source_ids'] for x in batch]),
            'source_masks': torch.stack([x['source_mask'] for x in batch]),  # Changed to plural
            'target_ids': torch.stack([x['target_ids'] for x in batch]),
            'target_masks': torch.stack([x['target_mask'] for x in batch])  # Changed to plural
        }

    def load_next_page(self) -> None:
        """Load next page of documents and process them"""
        start_index = self.current_page_index * self.es_page_size
        end_index = start_index + self.es_page_size
        
        if end_index > len(self.document_ids):
            end_index = len(self.document_ids)
            
        if start_index >= len(self.document_ids):
            return
            
        # Get document IDs for this page
        page_ids = self.document_ids[start_index:end_index]
        
        try:
            # Fetch documents in smaller batches
            BATCH_SIZE = 100
            all_docs = []
            
            for i in range(0, len(page_ids), BATCH_SIZE):
                batch_ids = page_ids[i:i + BATCH_SIZE]
                response = self.es_client.mget(
                    index=self.index,
                    body={"ids": batch_ids}
                )
                all_docs.extend(doc['_source'] for doc in response['docs'] 
                              if doc.get('found'))
            
            # Process documents and add to buffer
            processed_examples = self._process_documents(all_docs)
            for example in processed_examples:
                if len(self.data_buffer) < self.cache_size_limit:
                    self.data_buffer.append(example)
                    
        except Exception as e:
            print(f"Error loading page: {e}")
            
        self.current_page_index += 1

    def get_triplets_from_sentence(
            self, es_sentence: Dict) -> List[Tuple[str, str, str, str]]:
        """Extract triplets from an Elasticsearch document"""
        triplets_and_sentence: List[Tuple[str, str, str, str]] = []
        for triplet in es_sentence['triplets']:
            sample = list(self.flatten_triplet_keys(triplet))
            sample.append(es_sentence['sentence_text'])
            triplets_and_sentence.append(tuple(sample))
            
        return triplets_and_sentence

    @staticmethod
    def flatten_triplet_keys(doc: Dict[str, Any]) -> Tuple[str, str, str]:
        """Flatten the nested structure of a document triplet"""
        subject_text: str = doc['subject']['text']
        relation_text: str = doc['relation']['text']
        object_text: str = doc['object']['text']
        return subject_text, relation_text, object_text

    def _process_documents(self, documents: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        """Process documents into model-ready format"""
        batch_samples = []
        
        for doc in documents:
            triplets = self.get_triplets_from_sentence(doc)
            for triplet in triplets:
                processed_sample = self.true_sample_f(triplet)
                if isinstance(processed_sample, tuple) and len(processed_sample) >= 2:
                    source_text, target_text = processed_sample
                    batch_samples.append((source_text, target_text))
        
        if not batch_samples:
            return []
            
        source_texts, target_texts = zip(*batch_samples)
        
        source_encodings = self.tokenizer.batch_encode_plus(
            source_texts,
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encodings = self.tokenizer.batch_encode_plus(
            target_texts,
            max_length=self.target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return [{
            'source_ids': source_encodings['input_ids'][i],
            'source_mask': source_encodings['attention_mask'][i],  # Keep singular here as it's internal
            'target_ids': target_encodings['input_ids'][i],
            'target_mask': target_encodings['attention_mask'][i]   # Keep singular here as it's internal
        } for i in range(len(batch_samples))]

    def get_stats(self) -> Dict[str, Any]:
        """Get current dataset statistics"""
        return {
            "total_documents": self.total_docs,
            "current_page": self.current_page_index,
            "documents_per_page": self.es_page_size,
            "buffer_size": len(self.data_buffer),
            "cache_limit": self.cache_size_limit,
            "has_article_filter": bool(self.filter_article_ids),
            "batch_size": self.batch_size
        }
