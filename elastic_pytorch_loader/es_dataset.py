from queue import Queue
from threading import Thread
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Any, Dict, Tuple, Callable
from elasticsearch import Elasticsearch
import random
from transformers import PreTrainedTokenizer
import torch


class ElasticSearchDataset(IterableDataset):
    """
    **ElasticSearchDataset class**

    This class provides an `IterableDataset` implementation for fetching and 
    processing documents from Elasticsearch to train a Huggingface model. It 
    retrieves documents in batches and supports asynchronous loading for efficiency.

    **Attributes:**

    * `url` (str): URL of the Elasticsearch instance.
    * `index` (str): Name of the Elasticsearch index containing the documents.
    * `tokenizer` (PreTrainedTokenizer): Huggingface tokenizer for processing text.
    * `es_page_size` (int, optional): Number of documents to retrieve per Elasticsearch request (default: 500).
    * `batch_size` (int, optional): Batch size for training the model (default: 8).
    * `async_loading` (bool, optional): Enable asynchronous loading of documents using a separate thread (default: False).
    * `shuffle` (bool, optional): Shuffle documents within each batch (default: True).
    * `seed` (int, optional): Seed for random number generation (for reproducibility, default: None).
    * `true_sample_f` (Callable[[tuple[str, ...]], tuple[str, ...]], optional): Custom function to transform document triplets (default: None).
    * `max_documents` (int, optional): Maximum number of documents to retrieve from Elasticsearch (default: 10000).
    * `cache_size_limit` (int, optional): Maximum size of the data cache to prevent memory overload (default: 5000).
    * `source_len` (int, optional): Maximum source sequence length for the model (default: 20).
    * `target_len` (int, optional): Maximum target sequence length for the model (default: 30).
    * `exclude_docs` (List[str], optional): List of document IDs to exclude from the dataset (default: None).

    **Methods:**

    * `__init__` (constructor): Initializes the class with the provided parameters.
    * `_get_all_document_ids` (private): Retrieves a random subset of document IDs from the Elasticsearch index.
    * `start_async_loading` (private): Starts a separate thread to asynchronously load documents.
    * `__iter__` (iterator): Returns the iterator object for the dataset.
    * `__next__` (iterator): Returns the next batch of processed data for training.
    * `load_next_page` (private): Fetches the next batch of documents from Elasticsearch based on document IDs.
    * `__del__` (destructor): Stops the asynchronous loading thread if running. (Optional cleanup)
    * `test_next_sliding` (private): Checks if there are more documents to fetch in the current page window. (Used for stopping loading)
    * `get_triplets_from_sentence` (private): Extracts triplets (subject, relation, object) from a document returned by Elasticsearch. (Assuming your documents have a 'triplets' key)
    * `flatten_triplet_keys` (static): Flattens the nested structure of a document triplet for easier processing. (Assuming your documents have a specific structure)

    **Notes:**

    * This class utilizes Elasticsearch's scroll API for efficient retrieval of large datasets.
    * The data cache helps to store preprocessed data for faster access during training.
    * The `true_sample_f` function allows for custom data transformations before feeding data to the model.
    """
    def __init__(self, url: str, index: str, tokenizer: PreTrainedTokenizer,
            es_page_size: int = 500, batch_size: int = 8, yield_raw_triplets: bool=False, 
            async_loading: bool = False, shuffle: bool = True, seed: int = None,
            true_sample_f: Callable[[tuple[str, ...]], tuple[str, ...]] = None,
            max_documents: int = 10000, cache_size_limit: int = 5000, 
            source_len:int=20, target_len:int=30, exclude_docs: List[str]=None):
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
    
    def _get_all_document_ids(self, max_documents: int) -> List[str]:
        search_query = {
            "size": max_documents,
            "query": {"match_all": {}},  # You can adjust the query as needed
            "sort": [
            {
                "_script": {
                    "type": "number",
                    "script": {
                        "source": "Math.random()", # Get random documents from
                        "lang": "painless"         # from the whole database.
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
        # Check if document to be excluded from this dataset are given in the constructor.
        # This is for ensuring that test/validation data is not in the training data
        if self.exclude_docs: 
            document_ids = list(set(document_ids) - set(self.exclude_docs))

        return document_ids

    def start_async_loading(self) -> None:
        def load_data():
            while not self.stop_loading:
                if self.data_cache.qsize() * self.batch_size < self.cache_size_limit:
                    self.load_next_page()

        self.loading_thread = Thread(target=load_data)
        self.loading_thread.start()

    def __iter__(self) -> 'ElasticSearchDataset':
        self.current_page_index = 0
        self.data_cache.queue.clear()
        self.stop_loading = False
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

		# Check if we should yield raw triplets or continue to tokenize sequences
        if self.yield_raw_triplets:
            return batch_data
            
        assert self.tokenizer is not None # Valid tokenizer is required to continue
        # Otherwise, return tokenized sequences
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
        