import wandb
import yaml
from pathlib import Path
from datasets import load_dataset
from torch.optim import Adam, AdamW
import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU #from sacrebleu import BLEU  # Install sacrebleu library: pip install sacrebleu
from rouge import Rouge  # Install py-rouge library: pip install rouge
from torch.utils.data import DataLoader,Dataset
from transformers import T5ForConditionalGeneration # SentencePiece library is required to download pretrained t5tokenizer
# Let's try T5TokenizerFast
from transformers.models.t5 import T5TokenizerFast
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
            try:
                logits = outputs.logits
                preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            #generated_output = self.tokenizer.batch_decode(sequences=preds, skip_special_tokens=True)
                test_score = self.calculate_validation_score(data, preds)
            except:
                st()        
            if step % 10 == 0:
                wandb.log({"test_loss": loss})    
                wandb.log({f"test_score ({self.score_type})": test_score})
                print(f"Epoch: {epoch} | Test Loss: {loss} | Test score ({self.score_type}): {test_score}")
                
    #def generate_batch(self, loader, epoch, return_predictions=False)
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
            score_type: select either 'bleu', 'rouge', 'combined', 'all' (this 
                latter returns a list with all scores)

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

        if self.score_type == 'combined':
        # Combine BLEU and ROUGE scores (weighted average is common)
            score = (bleu_score + rouge_score) / 2.0  # Adjust weights as needed
        if self.score_type == 'all':
            score = [bleu_score, rouge_score, score]
        elif self.score_type == 'bleu':
            score = bleu_score
        elif self.score_type == 'rouge':
            score = rouge_score
        return score

class CustomDataset(Dataset):
  def __init__(self, dataset, tokenizer, source_len, target_len, source_key, target_key):
    self.dataset = dataset 
    self.tokenizer = tokenizer
    self.text_len = source_len
    self.summ_len = target_len
    self.text = self.dataset[source_key]
    self.summary = self.dataset[target_key]

  def __len__(self):
    return len(self.text)

  def __getitem__(self,i):
    summary = str(self.summary[i])
    summary = ' '.join(summary.split())
    text = str(self.text[i])
    text = ' '.join(text.split())
    source = self.tokenizer.batch_encode_plus(
        [text], max_length=self.text_len, return_tensors='pt', pad_to_max_length=True) # Each source sequence is encoded and padded to max length in batches
    target = self.tokenizer.batch_encode_plus(
        [summary], max_length=self.summ_len, return_tensors='pt', pad_to_max_length=True) # Each target sequence is encoded and padded to max lenght in batches

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
  wandb.init(project=project_name)
  
  source_len=270
  target_len=160
  model_name = wandb.config["hf_model_name"]#'t5-base'
  epochs = wandb.config["epochs"]
  lr = wandb.config["learning_rate"] #3e-4
  batch_size = wandb.config["batch_size"]
  optimizer_name = wandb.config["optimizer"]

  source_key = 'article'
  target_key = 'highlights'
  generate = False
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
  ## Prepare Dataset ##
  ##  We will use cnn_dailymail summarization dataset for abstractive summarization #
  dataset = load_dataset('cnn_dailymail','3.0.0')
  # As we can observe, dataset is too large so for now we will consider just 8k rows for training
  #  and 4k rows for validation
  train_dataset = dataset['train'][:800]
  val_dataset = dataset['validation'][:400]

  with ClearCache():
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    train_dataset = CustomDataset(
      train_dataset, tokenizer, source_len, target_len, source_key, target_key)
    val_dataset = CustomDataset(
      val_dataset,tokenizer, source_len, target_len, source_key, target_key)
  
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=0)
  
    reasoning_pipeline = OVNeuralReasoningPipeline(model, tokenizer, device)
 
    if optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr, amsgrad=True)
    elif optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)

    wandb.watch(model, log='all')
  # Call train function
    for epoch in range(epochs): #optimizer, train_loader, epoch
        reasoning_pipeline.train(optimizer, train_loader, epoch)
        reasoning_pipeline.test(val_loader, epoch)
        if generate:
            reasoning_pipeline.generate(val_loader, epoch, return_predictions=True)


project_name = 'huggingface'
path2sweep_config = "config_seq2seq_T5.yaml"
sweep_configuration = get_sweep_config(path2sweep_config)
# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
# Start sweep job.
wandb.agent(sweep_id,
            function=main,
            count=2,
            project=project_name)