from torch.utils.data import IterableDataset
from typing import List, Any, Dict, Tuple, Callable, Optional
from elasticsearch import Elasticsearch
import random
import torch
from transformers import PreTrainedTokenizer
from collections import deque


class OptimizedElasticSearchDataset(IterableDataset):
    """
    Optimized ElasticSearch dataset for PyTorch training with efficient train/test splitting.
    """
    
    @staticmethod
    def create_train_test_split(url: str, index: str, max_docs2load: int, 
                               test_ratio: float = 0.3, seed: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Create train/test split of document IDs.
        
        Args:
            url: Elasticsearch URL
            index: Index name
            max_docs2load: Maximum number of documents to load in total
            test_ratio: Ratio of documents to use for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_ids, test_ids)
        """
        es_client = Elasticsearch(url)
        
        query = {
            "size": 1000,
            "query": {"match_all": {}},
            "_source": False,
            "sort": [{"_id": "asc"}]
        }
        
        all_ids = []
        last_sort = None
        
        while True:
            if last_sort:
                query["search_after"] = last_sort
                
            try:
                response = es_client.search(index=index, body=query)
                hits = response['hits']['hits']
                
                if not hits:
                    break
                    
                all_ids.extend(hit['_id'] for hit in hits)
                last_sort = hits[-1]['sort']
                
                if len(all_ids) >= max_docs2load:
                    all_ids = all_ids[:max_docs2load]
                    break
                    
            except Exception as e:
                print(f"Error fetching IDs: {e}")
                break
        
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
            batch_size: int = 8,
            prefetch_batches: int = 10,
            filter_article_ids: Optional[List[str]] = None,
            source_len: int = 20,
            target_len: int = 30,
            true_sample_f: Optional[Callable] = None,
            seed: Optional[int] = None,
            selected_doc_ids: Optional[List[str]] = None):
        
        self.es_client = Elasticsearch(url)
        self.index = index
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prefetch_size = batch_size * prefetch_batches
        self.filter_article_ids = filter_article_ids
        self.source_len = source_len
        self.target_len = target_len
        self.true_sample_f = true_sample_f or (lambda x: x)
        self.seed = seed
        
        # Track documents
        self.seen_docs = set()
        self.data_buffer = deque(maxlen=self.prefetch_size * 2)
        self.selected_doc_ids = selected_doc_ids
        
        # Initialize document IDs
        self._initialize_document_ids()
        
        if seed is not None:
            random.seed(seed)

    def _initialize_document_ids(self):
        """Initialize document IDs using pre-selected IDs if available"""
        if self.selected_doc_ids:
            self.document_ids = list(self.selected_doc_ids)
            self.total_docs = len(self.document_ids)
            return
            
        query = {
            "size": 10000,  # Default max size
            "query": self._build_base_query(),
            "_source": False,
            "sort": [{"_id": "asc"}]
        }
        
        try:
            response = self.es_client.search(
                index=self.index,
                body=query
            )
            
            self.document_ids = [hit['_id'] for hit in response['hits']['hits']]
            self.total_docs = len(self.document_ids)
            
        except Exception as e:
            print(f"Error initializing document IDs: {e}")
            self.document_ids = []
            self.total_docs = 0

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
        """Iterator implementation with buffer management"""
        self.seen_docs.clear()
        self.data_buffer.clear()
        return self

    def __next__(self):
        """Returns next batch of processed data"""
        if len(self.data_buffer) < self.batch_size:
            raw_docs = self._fetch_batch()
            if not raw_docs and not self.data_buffer:
                raise StopIteration
                
            if raw_docs:
                processed_examples = self._process_documents(raw_docs)
                self.data_buffer.extend(processed_examples)
        
        batch = []
        for _ in range(min(self.batch_size, len(self.data_buffer))):
            if self.data_buffer:
                batch.append(self.data_buffer.popleft())
            
        if not batch:
            raise StopIteration
            
        return {
            'source_ids': torch.stack([x['source_ids'] for x in batch]),
            'source_mask': torch.stack([x['source_mask'] for x in batch]),
            'target_ids': torch.stack([x['target_ids'] for x in batch]),
            'target_mask': torch.stack([x['target_mask'] for x in batch])
        }

    def _fetch_batch(self) -> List[Dict]:
        """Fetch a batch of documents using stored document IDs"""
        if not self.document_ids:
            return []
            
        batch_ids = []
        for doc_id in self.document_ids:
            if doc_id not in self.seen_docs:
                batch_ids.append(doc_id)
                self.seen_docs.add(doc_id)
            if len(batch_ids) >= self.prefetch_size:
                break
                
        if not batch_ids:
            return []
            
        try:
            response = self.es_client.mget(
                index=self.index,
                body={"ids": batch_ids}
            )
            return [doc['_source'] for doc in response['docs'] if doc.get('found')]
        except Exception as e:
            print(f"Error fetching batch: {e}")
            return []

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
            'source_mask': source_encodings['attention_mask'][i],
            'target_ids': target_encodings['input_ids'][i],
            'target_mask': target_encodings['attention_mask'][i]
        } for i in range(len(batch_samples))]

    def get_stats(self) -> Dict[str, Any]:
        """Get current dataset statistics"""
        return {
            "total_documents": self.total_docs,
            "seen_documents": len(self.seen_docs),
            "remaining_documents": self.total_docs - len(self.seen_docs),
            "buffer_size": len(self.data_buffer),
            "has_article_filter": bool(self.filter_article_ids),
            "batch_size": self.batch_size,
            "prefetch_size": self.prefetch_size
        }
