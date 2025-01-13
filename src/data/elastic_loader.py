from queue import Queue
from threading import Thread
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Any, Dict, Tuple, Callable
from elasticsearch import Elasticsearch
import random
import time
from transformers import PreTrainedTokenizer
import torch


class ElasticSearchDataset(IterableDataset):
    """
    ElasticSearchDataset class with enhanced filtering capabilities.
    Provides an IterableDataset implementation for fetching and processing documents 
    from Elasticsearch to train a Huggingface model.
    """
    def __init__(self, url: str, index: str, tokenizer: PreTrainedTokenizer,
                 es_page_size: int = 500, batch_size: int = 8, yield_raw_triplets: bool = False,
                 async_loading: bool = False, shuffle: bool = True, seed: int = None,
                 true_sample_f: Callable[[tuple[str, ...]], tuple[str, ...]] = None,
                 max_documents: int = 10000, cache_size_limit: int = 5000,
                 source_len: int = 20, target_len: int = 30, exclude_docs: List[str] = None,
                 filter_article_ids: List[str] = None):
        """
        Initialize the dataset with enhanced filtering capabilities.
        
        Args:
            filter_article_ids: Optional list of article IDs to filter documents by
            (all other parameters remain as in original implementation)
        """
        self.index = index
        self.es_page_size = es_page_size
        self.batch_size = batch_size
        self.async_loading = async_loading
        self.shuffle = shuffle
        self.yield_raw_triplets = yield_raw_triplets
        self.seed = seed
        self.es_client: Elasticsearch = Elasticsearch(url)
        self.tokenizer = tokenizer
        self.exclude_docs = exclude_docs
        self.filter_article_ids = filter_article_ids
        self.true_sample_f = true_sample_f
        self.cache_size_limit = cache_size_limit
        self.source_len = source_len
        self.target_len = target_len
        self.loading_thread: Thread = None
        self.stop_loading: bool = False
        self.data_cache: Queue = Queue()
        self.current_page_index: int = 0
        self.document_ids: List[str] = self._get_all_document_ids(max_documents)

        if self.async_loading:
            self.start_async_loading()

    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.
        """
        return (len(self.document_ids) + self.batch_size - 1) // self.batch_size

    def _build_search_query(self, size: int) -> Dict[str, Any]:
        """
        Builds the Elasticsearch query based on configuration.
        
        Args:
            size: Number of documents to retrieve per page
            
        Returns:
            Dict containing the Elasticsearch query
        """
        query: Dict[str, Any] = {
            "bool": {
                "must": [{"match_all": {}}]
            }
        }

        if self.filter_article_ids:
            query["bool"]["must"].append({
                "terms": {
                    "article_id.keyword": self.filter_article_ids
                }
            })

        search_body = {
            "size": size,
            "query": query,
            "_source": ["article_id"] if self.filter_article_ids else False
        }

        return search_body

    def _get_all_document_ids(self, max_documents: int) -> List[str]:
        """
        Retrieve document IDs using scroll API with support for article ID filtering.
        
        Args:
            max_documents: Maximum number of documents to retrieve
            
        Returns:
            List of document IDs
        """
        search_body = self._build_search_query(1000)
        
        try:
            response = self.es_client.search(
                index=self.index,
                body=search_body,
                scroll='5m'
            )
            
            scroll_id = response['_scroll_id']
            document_ids = [hit['_id'] for hit in response['hits']['hits']]
            
            try:
                while len(document_ids) < max_documents:
                    response = self.es_client.scroll(
                        scroll_id=scroll_id,
                        scroll='5m'
                    )
                    
                    if not response['hits']['hits']:
                        break
                    
                    document_ids.extend(hit['_id'] for hit in response['hits']['hits'])
                    
            finally:
                try:
                    self.es_client.clear_scroll(scroll_id=scroll_id)
                except Exception as e:
                    print(f"Error clearing scroll: {e}")
            
            if len(document_ids) > max_documents:
                if self.seed is not None:
                    random.seed(self.seed)
                document_ids = random.sample(document_ids, max_documents)
            
            if self.exclude_docs:
                document_ids = list(set(document_ids) - set(self.exclude_docs))
            
            return document_ids
            
        except Exception as e:
            print(f"Error in document ID retrieval: {e}")
            return []

    def start_async_loading(self) -> None:
        """
        Starts asynchronous loading of documents in a separate thread.
        """
        def load_data():
            while not self.stop_loading:
                if self.data_cache.qsize() * self.batch_size < self.cache_size_limit:
                    self.load_next_page()

        self.loading_thread = Thread(target=load_data)
        self.loading_thread.start()

    def __iter__(self) -> 'ElasticSearchDataset':
        """
        Returns the iterator object.
        """
        self.current_page_index = 0
        self.data_cache.queue.clear()
        self.stop_loading = False
        if self.async_loading:
            self.start_async_loading()
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        """
        Returns the next batch of processed data.
        """
        if (not self.async_loading and not self.stop_loading
                and self.data_cache.qsize() < self.cache_size_limit):
            self.load_next_page()

        if self.data_cache.empty():
            raise StopIteration

        if self.data_cache.qsize() < self.batch_size:
            batchz = self.data_cache.qsize()
        else:
            batchz = self.batch_size
        
        batch_data: List[Any] = [
            self.true_sample_f(self.data_cache.get()) for _ in range(batchz)]
            
        if self.shuffle:
            random.shuffle(batch_data)

        if self.yield_raw_triplets:
            return batch_data
            
        assert self.tokenizer is not None
        
        source_text, target_text = zip(*batch_data)
        source = self.tokenizer.batch_encode_plus(
                    source_text, max_length=self.source_len,
                    return_tensors='pt', pad_to_max_length=True, truncation=True)
        target = self.tokenizer.batch_encode_plus(
                    target_text, max_length=self.target_len,
                    return_tensors='pt', pad_to_max_length=True, truncation=True)
            
        source_ids = source['input_ids'].squeeze()
        source_masks = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_masks = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(torch.long),
            'source_masks': source_masks.to(torch.long),
            'target_ids': target_ids.to(torch.long),
            'target_masks': target_masks.to(torch.long)
        }

    def load_next_page(self) -> None:
        """
        Loads the next batch of documents with improved error handling and retries.
        """
        MAX_RETRIES = 3
        RETRY_DELAY = 1
        BATCH_SIZE = 100
        
        start_index: int = self.current_page_index * self.es_page_size 
        end_index: int = start_index + self.es_page_size
        if end_index > len(self.document_ids):
            end_index = len(self.document_ids)
            
        ids_to_fetch: List[str] = self.document_ids[start_index:end_index]
        
        for attempt in range(MAX_RETRIES):
            try:
                all_hits = []
                
                for i in range(0, len(ids_to_fetch), BATCH_SIZE):
                    batch_ids = ids_to_fetch[i:i + BATCH_SIZE]
                    response: Dict[str, Any] = self.es_client.mget(
                        index=self.index,
                        body={"ids": batch_ids}
                    )
                    all_hits.extend(response['docs'])
                
                for hit in all_hits:
                    if hit['found']:
                        triplets = self.get_triplets_from_sentence(hit['_source'])
                        for s in triplets:
                            self.data_cache.put(s)
                            
                break
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to fetch documents after {MAX_RETRIES} attempts: {e}")
                    self.stop_loading = True
                    return
                time.sleep(RETRY_DELAY * (attempt + 1))
        
        self.current_page_index += 1
        
        if start_index >= len(self.document_ids) or not self.test_next_sliding():
            self.stop_loading = True
        if self.async_loading and self.loading_thread and self.loading_thread.is_alive():
            self.loading_thread.join()

    def __del__(self) -> None:
        """
        Cleanup method to ensure the loading thread is properly terminated.
        """
        self.stop_loading = True
        if self.async_loading and hasattr(self, 'loading_thread') and self.loading_thread and self.loading_thread.is_alive():
            self.loading_thread.join()

    def test_next_sliding(self) -> bool:
        """
        Tests if there are more documents to fetch in the current window.
        
        Returns:
            bool indicating if there are more documents to fetch
        """
        start_index: int = self.current_page_index * self.es_page_size 
        end_index: int = start_index + self.es_page_size
        if end_index > len(self.document_ids):
            end_index = len(self.document_ids)
        
        ids_to_fetch: List[str] = self.document_ids[start_index:end_index]
        
        return bool(len(ids_to_fetch))

    def get_triplets_from_sentence(
            self, es_sentence: Dict) -> List[Tuple[str, str, str, str]]:
        """
        Extracts triplets from an Elasticsearch document.
        
        Args:
            es_sentence: Document from Elasticsearch containing triplets
            
        Returns:
            List of tuples containing subject, relation, object, and sentence text
        """
        triplets_and_sentence: List[Tuple[str, str, str, str]] = []
        for triplet in es_sentence['triplets']:
            sample = list(self.flatten_triplet_keys(triplet))
            sample.append(es_sentence['sentence_text'])
            triplets_and_sentence.append(tuple(sample))
            
        return triplets_and_sentence

    @staticmethod
    def flatten_triplet_keys(doc: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Flattens the nested structure of a document triplet.
        
        Args:
            doc: Document containing subject, relation, and object
            
        Returns:
            Tuple of (subject_text, relation_text, object_text)
        """
        subject_text: str = doc['subject']['text']
        relation_text: str = doc['relation']['text']
        object_text: str = doc['object']['text']

        return subject_text, relation_text, object_text

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the dataset state.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "total_documents": len(self.document_ids),
            "cache_size": self.data_cache.qsize(),
            "current_page": self.current_page_index,
            "is_loading": not self.stop_loading,
            "has_article_filter": bool(self.filter_article_ids),
            "batch_size": self.batch_size,
            "page_size": self.es_page_size
        }
        return stats
