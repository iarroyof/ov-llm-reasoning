# main.py

import wandb
import torch
import logging
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

# Local imports
from src.trainers import T5ReasoningTrainer, T5LargeReasoningTrainer
from src.trainers import BartReasoningTrainer
from src.trainers import PegasusReasoningTrainer
from src.data import ElasticSearchDataset
from src.utils.memory import log_gpu_memory_usage
from src.utils import ClearCache
from src.utils import es_settings
from pdb import set_trace as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_trainer_class(model_name):
    if '11b' in model_name.lower():
        return T5LargeReasoningTrainer
    elif any(name in model_name.lower() for name in ['t5', 'flan', 'mt5', 'umt5']):
        return T5ReasoningTrainer
    elif 'bart' in model_name.lower():
        return BartReasoningTrainer
    elif 'pegasus' in model_name.lower():
        return PegasusReasoningTrainer
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def main():
    with wandb.init() as run:
        # Get hyperparameters
        source_len = wandb.config["source_seq_len"]
        target_len = wandb.config["target_seq_len"]
        model_name = wandb.config["hf_model_name"]
        epochs = wandb.config["epochs"]
        lr = wandb.config["learning_rate"]
        batch_size = wandb.config["batch_size"]
        optimizer_name = wandb.config["optimizer"]

        # ElasticSearch settings loaded from config/es_config.yaml
        url = es_settings.get("url") #"http://192.168.241.210:9200"
        index = es_settings.get("index") # 'triplets'
        es_page_size = es_settings.get("es_page_size") #100
        n_sentences = es_settings.get("max_docs2load") #1500
        logger.info(f"ES URL: {url}; ES index: {index}; ES page size: {es_page_size}; Total sentences: {n_sentences}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device found: {device}")
        # Log initial system state
        logger.info({"initial_system_state": log_gpu_memory_usage()})

        with ClearCache():
            # Initialize appropriate trainer
            trainer_class = get_trainer_class(model_name)
            # Special handling for large models
            if type(trainer_class).__name__ ==  "T5LargeReasoningTrainer":
                quantization = wandb.config["quantization"]
                reasoning_trainer = trainer_class.from_pretrained(
                    model_name=model_name,
                    device=device,
                    quantization=quantization,
                    max_memory=max_memory,
                )
            else:
                reasoning_trainer = trainer_class.from_pretrained(model_name, device)
            # Prepare dataset
            # Create train/test split
            train_ids, test_ids = ElasticSearchDataset.create_train_test_split(
                url=url,
                index=index,
                max_docs2load=n_sentences,
                test_ratio=0.3,
                seed=42
            )
            logger.info(f"Number of training sentences: {len(train_ids)}")
            logger.info(f"Number of test sentences: {len(test_ids)}")
            # Define transformation function
            true_sample = lambda x: (' '.join((x[0], x[1])), x[2]) if len(x) >= 3 else x
            
            # Create training dataset
            train_dataset = ElasticSearchDataset(
                url=url,
                index=index,
                tokenizer=reasoning_trainer.tokenizer,
                true_sample_f=true_sample,
                es_page_size=es_page_size,  # Control memory usage with page size
                batch_size=batch_size,
                source_len=source_len,
                target_len=target_len,
                selected_doc_ids=train_ids,
                cache_size_limit=es_page_size,  # Control memory usage with cache limit
                seed=42
            )
            
            # Create validation dataset similarly
            val_dataset = ElasticSearchDataset(
                url=url,
                index=index,
                tokenizer=reasoning_trainer.tokenizer,
                true_sample_f=true_sample,
                es_page_size=es_page_size,
                batch_size=batch_size,
                source_len=source_len,
                target_len=target_len,
                selected_doc_ids=test_ids,
                cache_size_limit=es_page_size,
                seed=42
            )           
            # Create data loaders
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=None,  # Batching is handled by the dataset
                num_workers=0
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=None,
                num_workers=0
            )
            # Initialize optimizer
            optimizer_class = AdamW if optimizer_name == "adamw" else Adam
            optimizer = optimizer_class(
                reasoning_trainer.model.parameters(),
                lr=lr,
                weight_decay=0.01 if optimizer_name == "adamw" else 0
            )
            # Configure model watching with gradients and parameters
            wandb.watch(
                reasoning_trainer.model,
                log="all",  # Log gradients and parameters
                log_freq=100,  # Log every 100 batches
                log_graph=True  # Log model graph
            )
            
            logger.info("Training started.")
            logger.info({"initial_system_state": log_gpu_memory_usage()})
            logger.info(f"Training epoch with hyperparameters {wandb.config}")
            for epoch in range(epochs):
                reasoning_trainer.train(optimizer, train_loader, epoch)

            logger.info("Training completed.")
            logger.info("Starting final evaluation...")
            
            # Run evaluation and get metrics
            final_loss, final_scores = reasoning_trainer.test(val_loader)
            logger.info("Evaluation completed.")
            
            # Log final summary metrics
            wandb.run.summary.update({
                "final_test_loss": final_loss,
                "final_test_scores": final_scores
            })

if __name__ == "__main__":
    main()
    logger.info({"Final system state": log_gpu_memory_usage()})
