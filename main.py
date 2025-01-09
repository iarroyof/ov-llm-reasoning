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
        max_docs2load = es_settings.get("max_docs2load") #1500

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device found: {device}")
        # Log initial system state
        wandb.log({"initial_system_state": log_gpu_memory_usage()})

        with ClearCache():
            # Initialize appropriate trainer
            trainer_class = get_trainer_class(model_name)
            # Special handling for large models
            if type(trainer_class).__name__ ==  "T5LargeReasoningTrainer":
                quantization = wandb.config["quantization"]
                max_memory = wandb.config["max_memory"]
                
                reasoning_trainer = trainer_class.from_pretrained(
                    model_name=model_name,
                    device=device,
                    quantization=quantization,
                    #use_gradient_checkpointing=gradient_checkpointing,
                    max_memory=max_memory,
                    #model_max_length=model_max_length
                )
                # Force smaller batch size for large models
                batch_size = min(batch_size, 2 if quantization == "4bit" else 1)
            else:
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
            optimizer_class = AdamW if optimizer_name == "adamw" else Adam
            optimizer = optimizer_class(
                reasoning_trainer.model.parameters(),
                lr=lr,
                weight_decay=0.01 if optimizer_name == "adamw" else 0
            )

            wandb.watch(reasoning_trainer.model, log='all')
            
            # Training loop
            logger.info("Training started.")
            wandb.watch(reasoning_trainer.model, log='all')
            for epoch in range(epochs):
                reasoning_trainer.train(optimizer, train_loader, epoch)
                reasoning_trainer.test(val_loader, epoch)
                logger.info(f"Training epoch with hyperparameters {wandb.config}")
            
            logger.info("Training completed.")

if __name__ == "__main__":
    main()
