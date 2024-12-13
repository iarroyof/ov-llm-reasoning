# main.py

# main.py

import wandb
import torch
import logging
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

# Local imports
from src.trainers.trainer_t5 import T5ReasoningTrainer
from src.trainers.trainer_bart import BartReasoningTrainer
from src.trainers.trainer_pegasus import PegasusReasoningTrainer
from src.data.elastic_loader import ElasticSearchDataset
from src.utils.cache import ClearCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_trainer_class(model_name):
    if any(name in model_name.lower() for name in ['t5', 'flan', 'mt5', 'umt5']):
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

        # ElasticSearch settings
        url = "http://192.168.241.210:9200"
        index = 'triplets'
        es_page_size = 100
        max_docs2load = 1000

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device found: {device}")

        with ClearCache():
            # Initialize appropriate trainer
            trainer_class = get_trainer_class(model_name)
            reasoning_trainer = trainer_class.from_pretrained(model_name, device)

            # Prepare dataset
            true_sample = lambda x: (' '.join((x[0], x[1])), x[2]) if len(x) >= 3 else x
            train_dataset = ElasticSearchDataset(
                url=url, index=index, es_page_size=es_page_size,
                tokenizer=reasoning_trainer.tokenizer,
                true_sample_f=true_sample, max_documents=max_docs2load,
                source_len=source_len, target_len=target_len, batch_size=1)
            
            val_dataset = ElasticSearchDataset(
                url=url, index=index, es_page_size=es_page_size,
                tokenizer=reasoning_trainer.tokenizer,
                true_sample_f=true_sample, max_documents=int(max_docs2load * 0.3),
                shuffle=False, source_len=source_len, target_len=target_len,
                batch_size=1, exclude_docs=train_dataset.document_ids)

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=0)

            # Initialize optimizer
            if optimizer_name == "adam":
                optimizer = Adam(reasoning_trainer.model.parameters(), lr=lr, amsgrad=True)
            elif optimizer_name == "adamw":
                optimizer = AdamW(reasoning_trainer.model.parameters(), lr=lr, amsgrad=True)

            wandb.watch(reasoning_trainer.model, log='all')
            
            # Training loop
            logger.info("Training started.")
            for epoch in range(epochs):
                reasoning_trainer.train(optimizer, train_loader, epoch)
                reasoning_trainer.test(val_loader, epoch)
                logger.info(f"Training epoch with parameters {wandb.config}")
            
            logger.info("Training finished.")

if __name__ == "__main__":
    main()
