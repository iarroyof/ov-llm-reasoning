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
            es_config = ElasticSearchConfig(
                url=es_settings.get("url", "http://192.168.241.210:9200"),
                index=es_settings.get("index", "triplets"),
                page_size=es_settings.get("es_page_size", 500),
                n_sentences=es_settings.get("n_sentences", 10000),
                n_articles=es_settings.get("n_articles", 10000),
                article_ids_file=es_settings.get("article_ids_file", "Not Found")
            )            
            # Get caching options from settings or wandb config
            force_recollect = wandb.config.get("force_recollect", False)
            cache_dir = es_settings.get("cache_dir", "data/cache_art_ids")
            
            logger.info(f"ElasticSearch configuration: {es_config}")
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
            
            # Setup datasets with caching options
            train_loader, val_loader = setup_datasets(
                es_config,
                trainer,
                training_config.batch_size,
                training_config.source_len,
                training_config.target_len,
                force_recollect=force_recollect,
                cache_dir=cache_dir
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
