"""
Main training script for neural text-to-text models on reasoning tasks.
Supports various architectures (T5, BART, PEGASUS) with configurable training parameters.
"""

import logging
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Type

import torch
import wandb
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.nn.modules import Module
from transformers import PreTrainedTokenizer

import pandas as pd
from torch.utils.data import Dataset

# Local imports
from src.trainers import (
    T5ReasoningTrainer, 
    T5LargeReasoningTrainer,
    BartReasoningTrainer,
    PegasusReasoningTrainer,
    BaseNeuralReasoningTrainer  # Fixed class name
)
from src.data import ElasticSearchDataset
from src.utils.memory import log_gpu_memory_usage
from src.utils import ClearCache
from src.utils import es_settings
from src.utils.cache_utils import save_split_cache, load_split_cache
from src.utils.triplet_filter import FilterMethod
from src.utils.gpu_monitor import gpu_wait
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DETAILED_LOG_WB = False

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    source_len: int
    target_len: int
    model_name: str
    epochs: int
    learning_rate: float
    batch_size: int
    optimizer: str
    quantization: Optional[str] = None
    max_memory: Optional[dict] = None

@dataclass
class ElasticSearchConfig:
    """Configuration for ElasticSearch data source."""
    url: str
    index: str
    page_size: int
    n_sentences: int
    n_articles: int
    article_ids_file: str

@dataclass
class LocalDataConfig:
    """Configuration for LocalData source"""
    file_path: str
    chunk_size: int
    test_ratio: float = 0.3
    seed: int = 42
    # n_samples: int = 10000

class JSONLDataset(Dataset):

    # Funcion para inicializar parametros
    def __init__(self, Df, tokenizer, source_len, target_len):
        """
        Args:
            file_path (str): Path to the JSONL file.
            chunk_size (int): Number of lines to read at a time.
        """
        self.Df = Df
        self.tokenizer = tokenizer
        self.source_len = source_len,
        self.target_len = target_len,

    def __len__(self):
        return len(self.Df)

    def __getitem__(self, index):
        """Obtiene un numero n de elementos del chunk."""
        sample_data = self.Df.loc[index, 'Article']
        sample_target = self.Df.loc[index, 'Abstract']
        source_encodings = self.tokenizer.batch_encode_plus(
            sample_data,
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encodings = self.tokenizer.batch_encode_plus(
            sample_target,
            max_length=self.target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return source_encodings, target_encodings

class LargeJSONLDataset(Dataset):

    # Funcion para inicializar parametros
    def __init__(self, file_path, chunk_size, tokenizer, source_len, target_len, test_ratio):
        """
        Args:
            file_path (str): Path to the JSONL file.
            chunk_size (int): Number of lines to read at a time.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.current_chunk = None
        self.current_chunk_index = 0
        self.total_samples = self._count_total_samples()
        self.tokenizer = tokenizer
        self.source_len = source_len,
        self.target_len = target_len,
        self.test_ratio = test_ratio,
        self.train = None,
        self.test = None

    def _count_total_samples(self):
        """Cuenta el numero de lineas en el archivo."""
        # Mejorar la manera en la que se le por chunks
        n = 0
        if ".jsonl" in self.file_path:
            chunk_iterator = pd.read_json(self.file_path, lines=True)
            #chunk_iterator = pd.read_json(self.file_path, lines=True, chunksize=self.chunk_size)
            self.train, self.test = self.create_train_test_split(self.test_ratio, chunk_iterator)
            return len(chunk_iterator)
        elif ".csv" in self.file_path:
            chunk_iterator = pd.read_csv(self.file_path, chunksize=self.chunk_size)

        for chunk in chunk_iterator:
            n = len(chunk) + n
            print(n)
        return n
        #with open(self.file_path, 'r') as f:
        #    return sum(1 for _ in f)

    def create_train_test_split(test_ratio, chunk_iterator):
        test_size = int(len(chunk_iterator) * test_ratio)
        train = chunk_iterator[test_size:]
        test = chunk_iterator[:test_size]

        return train, test
    
    def _load_chunk(self):
        """carga el siguiente chunk del archivo."""
        if ".jsonl" in self.file_path:
            chunk_iterator = pd.read_json(self.file_path, lines=True)
            # chunk_iterator = pd.read_json(self.file_path, lines=True, chunksize=self.chunk_size)
            return chunk_iterator
        elif ".csv" in self.file_path:
            chunk_iterator = pd.read_csv(self.file_path, chunksize=self.chunk_size)
        
        for chunk in chunk_iterator:
            yield chunk

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        """Obtiene un numero n de elementos del chunk."""

        if ".jsonl" in self.file_path:

            if self.train == None:
                chunk_iterator = self._load_chunk()
                self.train,_ = self.create_train_test_split(self.test_ratio, chunk_iterator)
            
            sample_data = self.train.loc[index, 'Article']
            sample_target = self.train.loc[index, 'Abstract']
            source_encodings = self.tokenizer.batch_encode_plus(
                sample_data,
                max_length=self.source_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            target_encodings = self.tokenizer.batch_encode_plus(
                sample_target,
                max_length=self.target_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else: 
            if self.current_chunk is None or self.current_chunk_index >= len(self.current_chunk):
                # Load the next chunk
                self.current_chunk = next(self._load_chunk())
                self.current_chunk_index = 0

            # Get the sample from the current chunk
            sample = self.current_chunk.iloc[index]
            self.current_chunk_index = self.current_chunk_index + 1

            source_encodings = self.tokenizer.batch_encode_plus(
                sample['Article'],
                max_length=self.source_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            target_encodings = self.tokenizer.batch_encode_plus(
                sample['Abstract'],
                max_length=self.target_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        return source_encodings, target_encodings


def get_trainer_class(model_name: str) -> Type[BaseNeuralReasoningTrainer]:  # Fixed return type
    """
    Determine appropriate trainer class based on model architecture.
    """
    model_name_lower = model_name.lower()
    if '11b' in model_name_lower:
        return T5LargeReasoningTrainer
    elif any(name in model_name_lower for name in ['t5', 'flan', 'mt5', 'umt5']):
        return T5ReasoningTrainer
    elif 'bart' in model_name_lower:
        return BartReasoningTrainer
    elif 'pegasus' in model_name_lower:
        return PegasusReasoningTrainer
    
    raise ValueError(f"Unsupported model architecture: {model_name}")

def setup_datasets(
    config: ElasticSearchConfig,
    trainer: BaseNeuralReasoningTrainer,
    batch_size: int,
    source_len: int,
    target_len: int,
    force_recollect: bool = False,  # New parameter
    cache_dir: str = "cache"  # New parameter
) -> Tuple[DataLoader, DataLoader]:
    """
    Set up training and validation datasets with caching support.
    
    Args:
        config: ElasticSearch configuration
        trainer: Model trainer instance
        batch_size: Batch size for training
        source_len: Maximum source sequence length
        target_len: Maximum target sequence length
        force_recollect: If True, ignore cache and recollect IDs
        cache_dir: Directory for caching splits
    """
    # Prepare split parameters
    split_params = {
        'url': config.url,
        'index': config.index,
        'n_sentences': config.n_sentences,
        'test_ratio': 0.3,
        'seed': 42,
        'filter_method': FilterMethod.STOPWORDS
    }
    
    # Handle article ID filtering
    if config.article_ids_file not in [None, '', 'Not Found']:
        try:
            with open(config.article_ids_file, 'r') as f:
                article_ids = [line.strip() for line in f]
                
            if len(article_ids) <= 10 or len(article_ids) < config.n_articles:
                raise ValueError(
                    f"Insufficient articles ({len(article_ids)}) for analysis. "
                    f"Minimum required: max(10, {config.n_articles})"
                )
                
            random.seed(42)
            random.shuffle(article_ids)
            split_params['filter_article_ids'] = article_ids[:config.n_articles]
            
        except FileNotFoundError:
            logger.warning(f"Article IDs file not found: {config.article_ids_file}")
        except Exception as e:
            logger.error(f"Error processing article IDs: {str(e)}")
            raise
    
    # Try to load from cache if not force_recollect
    train_ids = test_ids = None
    if not force_recollect:
        cache_result = load_split_cache(split_params, cache_dir)
        if cache_result is not None:
            train_ids, test_ids = cache_result
            logger.info("Successfully loaded split from cache")
    
    # Create new split if necessary
    if train_ids is None or test_ids is None:
        logger.info("Collecting new train/test split...")
        train_ids, test_ids = ElasticSearchDataset.create_train_test_split(**split_params)
        # Cache the new split
        save_split_cache(train_ids, test_ids, split_params, cache_dir)
        logger.info("New split saved to cache")
    
    logger.info(f"Dataset split - Train: {len(train_ids)}, Test: {len(test_ids)}")
    
    # Rest of the function remains the same...
    true_sample = lambda x: (' '.join((x[0], x[1])), x[2]) if len(x) >= 3 else x
    
    dataset_params = {
        'url': config.url,
        'index': config.index,
        'tokenizer': trainer.tokenizer,
        'true_sample_f': true_sample,
        'es_page_size': config.page_size,
        'batch_size': batch_size,
        'source_len': source_len,
        'target_len': target_len,
        'cache_size_limit': config.page_size,
        'seed': 42
    }
    
    train_dataset = ElasticSearchDataset(
        **dataset_params,
        selected_doc_ids=train_ids
    )
    val_dataset = ElasticSearchDataset(
        **dataset_params,
        selected_doc_ids=test_ids
    )
    
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)
    
    return train_loader, val_loader

def setup_local_datasets(
    config: LocalDataConfig,
    trainer: BaseNeuralReasoningTrainer,
    batch_size: int,
    source_len: int,
    target_len: int
) -> Tuple[DataLoader, DataLoader]:
    
    # Parameters for local splits
    split_params = {
        'file_path':config.file_path,
        'test_ratio': config.test_ratio,
        'seed':config.seed,
        'chunk_size': config.chunk_size
    }

    if ".jsonl" in config.file_path:
        chunk_iterator = pd.read_json(config.file_path, lines=True)
        test_size = int(len(chunk_iterator) * config.test_ratio)
        train = chunk_iterator[test_size:]
        test = chunk_iterator[:test_size]
        train_dataset = JSONLDataset(
            train, 
            trainer.tokenizer, 
            source_len, 
            target_len)
        
        testn_dataset = JSONLDataset(
            test,
            trainer.tokenizer, 
            source_len, 
            target_len)
        
    # For data in csv format and diferent files to load
    #train_dataset = LargeJSONLDataset(name_arch_train ,batch_size, trainer.tokenizer, source_len, target_len)
    #val_dataset = LargeJSONLDataset(name_arch_val ,batch_size, trainer.tokenizer, source_len, target_len)

    # Configurar DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0
    )
    
    val_loader = DataLoader(
        testn_dataset,
        batch_size=batch_size,
        num_workers=0
    )
    
    return train_loader, val_loader

def train_model(
    trainer: BaseNeuralReasoningTrainer,  # Fixed type hint
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig
) -> Tuple[float, dict]:
    """
    Train the model and evaluate performance.
    """
    # Setup optimizer
    optimizer_class = AdamW if config.optimizer == "adamw" else Adam
    optimizer = optimizer_class(
        trainer.model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01 if config.optimizer == "adamw" else 0
    )
    
    # Configure wandb monitoring
    if DETAILED_LOG_WB:
        wandb.watch(
            trainer.model,
            log="all",
            log_freq=100,
            log_graph=True
        )
    
    # Set training parameters
    trainer.score_type = 'all'
    trainer.gen_method = 'beam'
    
    logger.info("Starting training...")
    for epoch in range(config.epochs):
        with ClearCache():# Get configurations from wandb
            trainer.train(optimizer, train_loader, epoch)
    
    logger.info("Training completed. Running final evaluation...")
    with ClearCache():# Get configurations from wandb
        final_loss, final_scores = trainer.test(val_loader)
    
    return final_loss, final_scores

#@gpu_wait
def main():
    """Main training pipeline."""
    with wandb.init() as run:
        with ClearCache():
            # Get configurations from wandb
            training_config = TrainingConfig(
                source_len=wandb.config["source_seq_len"],
                target_len=wandb.config["target_seq_len"],
                model_name=wandb.config["hf_model_name"],
                epochs=wandb.config["epochs"],
                learning_rate=wandb.config["learning_rate"],
                batch_size=wandb.config["batch_size"],
                optimizer=wandb.config["optimizer"],
                quantization=wandb.config.get("quantization"),
                max_memory=wandb.config.get("max_memory")
            )
            #es_config = ElasticSearchConfig(
            #    url=es_settings.get("url", "http://192.168.241.210:9200"),
            #    index=es_settings.get("index", "triplets"),
            #    page_size=es_settings.get("es_page_size", 500),
            #    n_sentences=es_settings.get("n_sentences", 10000),
            #    n_articles=es_settings.get("n_articles", 10000),
            #    article_ids_file=es_settings.get("article_ids_file", "Not Found")
            #)
            ld_config = LocalDataConfig(
                file_path = '/mnt/sda2/Datos_cancer_pulmon',
                chunk_size = 900,
                test_ratio = 0.3,
                seed = 42
            )            
            # Get caching options from settings or wandb config
            force_recollect = wandb.config.get("force_recollect", False)
            cache_dir = es_settings.get("cache_dir", "data/cache_art_ids")
            
            #logger.info(f"ElasticSearch configuration: {es_config}")
            logger.info(f"Cache settings - Dir: {cache_dir}, Force recollect: {force_recollect}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            logger.info(f"Initial system state: {log_gpu_memory_usage()}")
        
            trainer_class = get_trainer_class(training_config.model_name)
            trainer = (
                trainer_class.from_pretrained(
                    model_name=training_config.model_name,
                    device=device,
                    quantization=training_config.quantization,
                    max_memory=training_config.max_memory
                )
                if issubclass(trainer_class, T5LargeReasoningTrainer)
                else trainer_class.from_pretrained(training_config.model_name, device)
            )
            
            # Setup datasets with caching options for elasticsearch
            #train_loader, val_loader = setup_datasets(
            #    es_config,
            #    trainer,
            #    training_config.batch_size,
            #    training_config.source_len,
            #    training_config.target_len,
            #    force_recollect=force_recollect,
            #    cache_dir=cache_dir
            #)
            # Setup datasets for local datasets
            train_loader, val_loader = setup_local_datasets(
                ld_config,
                trainer,
                training_config.batch_size,
                training_config.source_len,
                training_config.target_len
            )
            
            final_loss, final_scores = train_model(
                trainer,
                train_loader,
                val_loader,
                training_config
            )
            
            wandb.run.summary.update({
                "final_test_loss": final_loss,
                "final_test_scores": final_scores
            })
            
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training failed with error:")
        raise
    finally:
        logger.info(f"Final system state: {log_gpu_memory_usage()}")
