import wandb
import yaml
from pathlib import Path
from datasets import load_dataset
from torch.optim import Adam, AdamW, SGD, ASGD
import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU #from sacrebleu import BLEU  # Install sacrebleu library: pip install sacrebleu
from rouge import Rouge  # Install py-rouge library: pip install rouge
from torch.utils.data import DataLoader,Dataset
from transformers import T5ForConditionalGeneration # SentencePiece library is required to download pretrained t5tokenizer
# Let's try T5TokenizerFast
from transformers.models.t5 import T5TokenizerFast
from elastic_pytorch_loader.es_dataset import ElasticSearchDataset

from pdb import set_trace as st


class OVNeuralReasoningPipeline:
    def __init__(self, model, tokenizer, device, gen_test_score='bleu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.score_type = gen_test_score

    def get_data(self, data):
        source_ids, source_mask, target_ids = (data["source_ids"].to(self.device),
                                               data["source_masks"].to(self.device),
                                               data["target_ids"].to(self.device))
        y_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids[:, 1:].clone().detach()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

        return source_ids, source_mask, y_ids, lm_labels
        
    def train(self, optimizer, loader, epoch):
        self.model.train()
        for step, data in enumerate(loader):
            source_ids, source_mask, y_ids, lm_labels = self.get_data(data)

            outputs = self.model(
                input_ids=source_ids,
                attention_mask=source_mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            loss = outputs[0]

            if step % 10 == 0:
                wandb.log({"training_loss": loss})
                print(f"Epoch: {epoch} | Train Loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(self, loader, epoch):
        self.model.eval()
        for step, data in enumerate(loader):
            source_ids, source_mask, y_ids, lm_labels = self.get_data(data)

            outputs = self.model(
                input_ids=source_ids,
                attention_mask=source_mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            loss = outputs[0]
            
            logits = outputs.logits
            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            try:
                test_score = self.calculate_validation_score(data, preds)
            except:
                if self.score_type == 'all':
                    test_score = [-1] * 3 # Model didn't learn so outputs
                else:                     # invalid predictions metrics 
                    test_score = -1       # are incable of operate on.

            if step % 10 == 0:
                wandb.log({"test_loss": loss})
                if self.score_type == 'all':
                    wandb.log({f"test_score (bleu)": test_score[0]})
                    wandb.log({f"test_score (rouge)": test_score[1]})
                    wandb.log({f"test_score (combined)": test_score[2]})
                else:    
                    wandb.log({f"test_score ({self.score_type})": test_score})
                print(f"Epoch: {epoch} | Test Loss: {loss} | Test score ({'bleu, rouge, combined' if self.score_type=='all' else self.score_type}): {test_score}")
                

    def generate(self, loader, epoch, return_predictions=False, evaluate=False):
        self.model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for step, data in enumerate(loader):
                source_ids, source_mask, target_ids = (data["source_ids"].to(self.device),
                                                       data["source_masks"].to(self.device),
                                                       data["target_ids"].to(self.device))
                generated_ids = self.model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    num_beams=2,
                    max_length=170,
                    repetition_penalty=2.5,
                    early_stopping=True,
                    length_penalty=1.0,
                )
                
                if evaluate:
                    gen_score = self.calculate_validation_score(data, generated_ids)
                    if step % 10 == 0:
                        #wandb.log({"generation_score": gen_score})
                        print(f"Epoch: {epoch} | Gen score ({self.score_type}): {gen_score}")

                if return_predictions:
                    preds = [self.tokenizer.decode(
                        p, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            for p in generated_ids]
                    target = [self.tokenizer.decode(
                        t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            for t in target_ids]

                    predictions.extend(preds)
                    targets.extend(target)

                    return predictions, targets

    def calculate_validation_score(self, data, generated_ids):
        """
        Calculates validation loss using BLEU and ROUGE scores.

        Args:
            data: A dictionary containing source_ids, source_masks, and target_ids.
            generated_ids: A list of generated summaries (tensor after decoding).

        Returns:
            score: A float/List[float] representing the validation score; either
            'bleu', 'rouge', 'combined', 'all' (this latter returns a list with all scores).
        """
        target_ids = data["target_ids"].to(self.device)
        # Decode target summaries
        target_text = [self.tokenizer.decode(t, skip_special_tokens=True) for t in target_ids]
        # Decode generated summaries
        generated_text = [self.tokenizer.decode(p, skip_special_tokens=True) for p in generated_ids]
        # Calculate BLEU score
        if self.score_type in ['all', 'bleu', 'combined']:
            bleu = BLEU(smooth_method='floor')
            bleu_score = bleu.corpus_score([ref for ref in target_text], generated_text).score
        # Calculate ROUGE score (using py-rouge)
        if self.score_type in ['all', 'rouge', 'combined']:
            rouge = Rouge()
            rouge_score = rouge.get_scores(target_text, generated_text, avg=True)["rouge-l"]["f"]

        if self.score_type in ['combined', 'all']:
        # Combine BLEU and ROUGE scores (weighted average is common)
            score = (bleu_score + rouge_score) / 2.0  # Adjust weights as needed
        if self.score_type == 'all':
            score = [bleu_score, rouge_score, score]
        elif self.score_type == 'bleu':
            score = bleu_score
        elif self.score_type == 'rouge':
            score = rouge_score
        return score


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


def get_sweep_config(path2sweep_config: str) -> dict:
    """ Get sweep config from path """
    config_path = Path(path2sweep_config).expanduser()
    with open(config_path, 'r') as file:
        sweep_config = yaml.safe_load(file)
    return sweep_config


def main():
  wandb.init()
  
  source_len=wandb.config["source_seq_len"]
  target_len=wandb.config["target_seq_len"] 
  model_name = wandb.config["hf_model_name"]
  epochs = wandb.config["epochs"]
  lr = wandb.config["learning_rate"]
  batch_size = wandb.config["batch_size"]
  optimizer_name = wandb.config["optimizer"]
  url = "http://192.168.241.210:9200"
  index = 'triplets'
  # Amount of sentences to load from ElasticSearch in memory 
  es_page_size = 100
  # Total amount of sentences to get their triplets
  max_docs2load = 10000
  generate = False
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  with ClearCache():
    ## Prepare Dataset ##
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16  # flotante de precisión media para cargar modelos grandes
        ).to(device)
        
    true_sample = lambda x: (' '.join((x[0], x[1])), x[2]) if len(x) >= 3 else x
    train_dataset = ElasticSearchDataset(
        url=url, index=index, es_page_size=es_page_size, tokenizer=tokenizer,
        true_sample_f=true_sample, max_documents=max_docs2load,
        source_len=source_len, target_len=target_len, batch_size=1)
    val_dataset = ElasticSearchDataset(
        url=url, index=index, es_page_size=es_page_size, tokenizer=tokenizer,
        true_sample_f=true_sample, max_documents=int(max_docs2load * 0.3),
        shuffle=False, source_len=source_len, target_len=target_len, batch_size=1)
  
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=0)
  
    reasoning_pipeline = OVNeuralReasoningPipeline(model, tokenizer, device, 'all')
 
    if optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr, amsgrad=True)
    elif optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)
    elif optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_name == "asgd":
        optimizer = ASGD(model.parameters(), lr=lr)

    wandb.watch(model, log='all')
  # Call train function
    for epoch in range(epochs):
        reasoning_pipeline.train(optimizer, train_loader, epoch)
        reasoning_pipeline.test(val_loader, epoch)
        if generate:
            reasoning_pipeline.generate(val_loader, epoch, return_predictions=True)


#project_name = 'nli_T5'
path2sweep_config = "config_seq2seq_T5.yaml"
sweep_configuration = get_sweep_config(path2sweep_config)
# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration)
# Start sweep job.
wandb.agent(sweep_id,
            function=main,
            count=2
            )

# En caso de necesitar entrenar Bloom para Text2Text Generation
#https://huggingface.co/bigscience/bloom/discussions/234
