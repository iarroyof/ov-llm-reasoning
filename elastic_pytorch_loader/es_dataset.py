from queue import Queue
from threading import Thread
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Any, Dict, Tuple, Callable
from elasticsearch import Elasticsearch
import random
from transformers import PreTrainedTokenizer
import torch

from pdb import set_trace as st

class ElasticSearchDataset(IterableDataset):
    def __init__(self, url: str, index: str, tokenizer: PreTrainedTokenizer,
            es_page_size: int = 500, batch_size: int = 8,
            async_loading: bool = False, shuffle: bool = True, seed: int = None,
            true_sample_f: Callable[[tuple[str, ...]], tuple[str, ...]] = None,
            max_documents: int = 10000, cache_size_limit: int = 5000, source_len:int=20, target_len:int=30):
        self.index: str = index
        self.es_page_size: int = es_page_size
        self.batch_size: int = batch_size
        self.async_loading: bool = async_loading
        self.shuffle: bool = shuffle
        self.seed: int = seed
        self.es_client: Elasticsearch = Elasticsearch(url)
        self.tokenizer = tokenizer
        self.document_ids: List[str] = self._get_all_document_ids(max_documents)
        self.current_page_index: int = 0
        self.data_cache: Queue = Queue()
        self.loading_thread: Thread = None
        self.stop_loading: bool = False
        self.true_sample_f: Callable = true_sample_f
        self.cache_size_limit = cache_size_limit
        self.source_len = source_len
        self.target_len = target_len        

        if self.async_loading:
            self.start_async_loading()
    
    def _get_all_document_ids(self, max_documents: int) -> List[str]:
        search_query = {
            "size": max_documents,
            "query": {"match_all": {}},  # You can adjust the query as needed
            "sort": [
            {
                "_script": {
                    "type": "number",
                    "script": {
                        "source": "Math.random()",
                        "lang": "painless"
                    },
                    "order": "asc"
                }
            }
            ],
            "_source": False  # Exclude document content, only return document IDs
        }
        # Retrieve random document IDs from the index
        response: Dict[str, Any] = self.es_client.search(
            index=self.index,
            body=search_query
        )
        document_ids: List[str] = [hit['_id'] for hit in response['hits']['hits']]
        
        return document_ids

    def start_async_loading(self) -> None:
        def load_data():
            while not self.stop_loading:
                if self.data_cache.qsize() * self.batch_size < self.cache_size_limit:
                    self.load_next_page()

        self.loading_thread = Thread(target=load_data)
        self.loading_thread.start()

    def __iter__(self) -> 'ElasticSearchDataset':
        self.current_batch_index: int = 0
        self.data_cache: Queue = Queue()
        self.stop_loading: bool = False
        if self.async_loading:
           self.start_async_loading()
        return self

    def __next__(self) -> List[Any]:
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

        source_text, target_text = zip(*batch_data)
        source = self.tokenizer.batch_encode_plus(
                    source_text, max_length=self.source_len,
                    return_tensors='pt', pad_to_max_length=True)
        target = self.tokenizer.batch_encode_plus(
                    target_text, max_length=self.target_len,
                    return_tensors='pt', pad_to_max_length=True)
            
        source_ids = source['input_ids'].squeeze()
        source_masks = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_masks = target['attention_mask'].squeeze()

        return {
            'source_ids':source_ids.to(torch.long),
            'source_masks':source_masks.to(torch.long),
            'target_ids':target_ids.to(torch.long),
            'target_masks':target_masks.to(torch.long)
        }

    def load_next_page(self) -> None:
        start_index: int = self.current_page_index * self.es_page_size 
        end_index: int = start_index + self.es_page_size
        if end_index > len(self.document_ids):
            end_index = len(self.document_ids)
            
        ids_to_fetch: List[str] = self.document_ids[start_index:end_index]
        try:
            response: Dict[str, Any] = self.es_client.mget(
                index=self.index,
                body={"ids": ids_to_fetch}
            )
        except:
            print(f"No documents to fetch from index {self.index}.")
        hits: List[Dict[str, Any]] = response['docs']
        for hit in hits:
            if hit['found']:
                triplets = self.get_triplets_from_sentence(hit['_source'])
                for s in triplets:
                    self.data_cache.put(s)

        self.current_page_index += 1
        
        if start_index >= len(self.document_ids) or not self.test_next_sliding():
            self.stop_loading = True
        if self.async_loading:
            self.loading_thread.join()

    def __del__(self) -> None:
        self.stop_loading = True
        if self.async_loading and self.loading_thread.is_alive():
            self.loading_thread.join()

    def test_next_sliding(self):
        start_index: int = self.current_page_index * self.es_page_size 
        end_index: int = start_index + self.es_page_size
        if end_index > len(self.document_ids):
            end_index = len(self.document_ids)
        
        ids_to_fetch: List[str] = self.document_ids[start_index:end_index]
        
        return bool(len(ids_to_fetch))

    def get_triplets_from_sentence(
            self, es_sentence: Dict) -> List[Tuple[str, str, str, str]]:
        triplets_and_sentence: List[Tuple[str, str, str, str]] = []
        for triplet in es_sentence['triplets']:
            sample = list(self.flatten_triplet_keys(triplet))
            sample.append(es_sentence['sentence_text'])
            triplets_and_sentence.append(tuple(sample))
            
        return triplets_and_sentence

    @staticmethod
    def flatten_triplet_keys(
            doc: Tuple[Dict[str, Any], str]) -> Tuple[str, str, str, str]:
        subject_text: str = doc['subject']['text']
        relation_text: str = doc['relation']['text']
        object_text: str = doc['object']['text']

        return subject_text, relation_text, object_text
        