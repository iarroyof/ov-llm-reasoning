from torch.utils.data import IterableDataset
from typing import List, Any, Dict, Tuple, Callable, Optional
from elasticsearch import Elasticsearch
import random
import torch
from transformers import PreTrainedTokenizer
from collections import deque
from pathlib import Path
from ..utils.triplet_filter import FilterMethod, TripletFilter, process_and_filter_triplets

class ElasticSearchDataset(IterableDataset):
    """
    Optimized ElasticSearch dataset with progressive loading and memory efficiency.
    """
    
    @staticmethod
    def create_train_test_split(
            url: str, 
            index: str, 
            n_sentences: int,
            test_ratio: float = 0.3, 
            seed: Optional[int] = None,
            filter_article_ids: Optional[List[str]] = None,
            es_page_size: int = 500,
            filter_method: FilterMethod = FilterMethod.NONE,
            stopwords_file: Optional[Path] = None,
            keep_semantic_stopwords: bool = True) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Create train/test split with optional triplet filtering.
        Returns (sentence_id, triplet_idx) pairs for valid triplets.
        """
        es_client = Elasticsearch(url)
        
        # Initialize triplet filter if needed
        triplet_filter = TripletFilter(
            method=filter_method,
            stopwords_file=stopwords_file,
            keep_semantic_stopwords=keep_semantic_stopwords
        )
        
        # Build base query
        if filter_article_ids:
            query = {
                "size": es_page_size,
                "query": {
                    "terms": {
                        "article_id.keyword": filter_article_ids
                    }
                }
            }
        else:
            query = {
                "size": es_page_size,
                "query": {"match_all": {}}
            }
        
        try:
            # Get initial response
            response = es_client.search(
                index=index,
                body=query,
                scroll='5m'
            )
            
            valid_pairs = []  # List of (sentence_id, triplet_idx) pairs
            scroll_id = response['_scroll_id']
            
            try:
                while len(valid_pairs) < n_sentences:
                    # Process current batch
                    for hit in response['hits']['hits']:
                        doc_id = hit['_id']
                        
                        # Filter and process triplets
                        batch_pairs = process_and_filter_triplets(
                            doc_id,
                            hit['_source'],
                            triplet_filter
                        )
                        valid_pairs.extend(batch_pairs)
                        
                        if len(valid_pairs) >= n_sentences:
                            break
                    
                    # Get next batch
                    response = es_client.scroll(
                        scroll_id=scroll_id,
                        scroll='5m'
                    )
                    
                    if not response['hits']['hits']:
                        break
                    
            finally:
                es_client.clear_scroll(scroll_id=scroll_id)
            
            # Limit to max_docs2load if needed
            if len(valid_pairs) > n_sentences:
                if seed is not None:
                    random.seed(seed)
                valid_pairs = random.sample(valid_pairs, n_sentences)
            
            # Shuffle and split
            if seed is not None:
                random.seed(seed)
            random.shuffle(valid_pairs)
            
            test_size = int(len(valid_pairs) * test_ratio)
            train_pairs = valid_pairs[test_size:]
            test_pairs = valid_pairs[:test_size]
            
            return train_pairs, test_pairs
            
        except Exception as e:
            print(f"Error in create_train_test_split: {e}")
            return [], []

    def _fetch_batch(self) -> List[Dict]:
        """Fetch batch using sentence-triplet pairs."""
        logger = logging.getLogger(__name__)
        
        if not self.document_ids:
            logger.info("No document_ids available")
            return []
        
        try:
            # Log document_ids information
            logger.info(f"document_ids type: {type(self.document_ids)}")
            logger.info(f"First few document_ids: {self.document_ids[:2]}")
            
            batch_pairs = []
            for pair in self.document_ids:
                logger.info(f"Processing pair: {pair}, type: {type(pair)}")
                
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    sent_id, triplet_idx = pair
                    if (sent_id, triplet_idx) not in self.seen_docs:
                        batch_pairs.append((str(sent_id), int(triplet_idx)))
                        self.seen_docs.add((sent_id, triplet_idx))
                        if len(batch_pairs) >= self.prefetch_size:
                            break
            
            if not batch_pairs:
                logger.warning("No batch_pairs collected")
                return []
            
            logger.info(f"Collected batch_pairs: {batch_pairs[:2]}")
            
            # Group by sentence ID
            sent_id_to_triplets = {}
            unique_ids = []
            for sent_id, triplet_idx in batch_pairs:
                if sent_id not in sent_id_to_triplets:
                    sent_id_to_triplets[sent_id] = []
                    unique_ids.append(sent_id)
                sent_id_to_triplets[sent_id].append(triplet_idx)
            
            # Log request structure and add assertion
            mget_body = {"ids": unique_ids}
            logger.info(f"mget request body: {mget_body}")
            logger.info(f"First few unique_ids: {unique_ids[:2]}")
            
            # Add assertions to check data
            assert len(unique_ids) > 0, "No unique IDs collected"
            assert all(isinstance(id_, str) for id_ in unique_ids), "Non-string IDs found"
            assert isinstance(mget_body["ids"], list), "ids must be a list"
            
            # Try the mget request
            try:
                response = self.es_client.mget(
                    index=self.index,
                    body=mget_body
                )
                
                # Process results
                processed_docs = []
                for hit in response.get('docs', []):
                    if not hit.get('found'):
                        logger.warning(f"Document not found for ID: {hit.get('_id')}")
                        continue
                    
                    sent_id = hit['_id']
                    source = hit.get('_source', {})
                    
                    if 'triplets' not in source:
                        logger.warning(f"No triplets found in document: {sent_id}")
                        continue
                    
                    triplet_indices = sent_id_to_triplets[sent_id]
                    for idx in triplet_indices:
                        if idx < len(source['triplets']):
                            processed_docs.append({
                                '_id': sent_id,
                                'sentence_text': source.get('sentence_text', ''),
                                'triplets': [source['triplets'][idx]]
                            })
                        else:
                            logger.warning(f"Triplet index {idx} out of range for document {sent_id}")
                
                return processed_docs
                
            except Exception as e:
                logger.error(f"mget request failed: {e}")
                logger.error(f"Request body was: {mget_body}")
                return []
                
        except Exception as e:
            logger.error(f"Error in _fetch_batch: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        
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
        print("=============== ENTERING load_next_page ===============")
        start_index = self.current_page_index * self.es_page_size
        end_index = start_index + self.es_page_size
        
        if end_index > len(self.document_ids):
            end_index = len(self.document_ids)
            
        if start_index >= len(self.document_ids):
            print("No more documents to load - reached end")
            return
                
        # Get document IDs for this page
        page_ids = self.document_ids[start_index:end_index]
        print(f"Debug: page_ids type: {type(page_ids)}")
        print(f"Debug: First few page_ids: {page_ids[:2]}")
        
        try:
            # Fetch documents in smaller batches
            BATCH_SIZE = 100
            all_docs = []
            
            for i in range(0, len(page_ids), BATCH_SIZE):
                batch_ids = page_ids[i:i + BATCH_SIZE]
                print(f"Debug: Processing batch {i//BATCH_SIZE + 1}")
                print(f"Debug: batch_ids type: {type(batch_ids)}")
                print(f"Debug: First few batch_ids: {batch_ids[:2]}")
                
                # Format check before request
                if not all(isinstance(id_, str) for id_ in batch_ids):
                    print(f"Debug: Warning - non-string IDs found in batch")
                    # Convert to strings if needed
                    batch_ids = [str(id_) for id_ in batch_ids]
                
                mget_body = {"ids": batch_ids}
                print(f"Debug: mget request body: {mget_body}")
                
                try:
                    response = self.es_client.mget(
                        index=self.index,
                        body=mget_body
                    )
                    print(f"Debug: mget request successful, got {len(response.get('docs', []))} documents")
                    all_docs.extend(doc['_source'] for doc in response['docs'] 
                                  if doc.get('found'))
                except Exception as e:
                    print(f"Debug: mget request failed: {e}")
                    print(f"Debug: Failed request body was: {mget_body}")
                    continue  # Try next batch instead of failing completely
            
            # Process documents and add to buffer
            print(f"Debug: Processing {len(all_docs)} documents")
            processed_examples = self._process_documents(all_docs)
            added_to_buffer = 0
            for example in processed_examples:
                if len(self.data_buffer) < self.cache_size_limit:
                    self.data_buffer.append(example)
                    added_to_buffer += 1
            print(f"Debug: Added {added_to_buffer} examples to buffer")
                        
        except Exception as e:
            print(f"Error loading page: {e}")
            import traceback
            traceback.print_exc()
                
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
