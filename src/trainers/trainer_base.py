# src/trainers/trainer_base.py

import torch
from torch.utils.data import DataLoader
import wandb
import logging
import torch.nn.functional as F
from rouge import Rouge
from sacrebleu.metrics import BLEU

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseNeuralReasoningTrainer:
    def __init__(self, model, tokenizer, device, gen_test_score='bleu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.score_type = gen_test_score

    def train(self, optimizer, train_loader, epoch, val_loader=None):
        self.model.train()
        if val_loader is not None:
            iterator = zip(train_loader, val_loader)
        else:
            iterator = train_loader

        for step, data in enumerate(iterator):
            if val_loader is not None:
                train_batch, val_batch = data
                # Use both train_batch and val_batch
            else:
                train_batch = data

            self.train_step(step, train_batch, epoch, optimizer)

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    self.test_step(step, val_batch, epoch)
                    self.model.train()

    def train_step(self, step, train_batch, epoch, optimizer):
        source_ids, source_mask, y_ids, lm_labels = self.get_data(train_batch)
        optimizer.zero_grad()
        outputs = self.forward_pass(source_ids, source_mask, y_ids, lm_labels)
        loss = outputs[0]
        wandb.log({"train_batch_loss": loss})
        # Aggregate training loss per epoch (this creates the segmented plot)
        wandb.log({
            "training_loss": loss,
            "epoch": epoch
            })

        if step % 10 == 0:
            print(f"Epoch: {epoch} | Train Loss: {loss}")
        
        loss.backward()
        optimizer.step()
    
    def test(self, loader):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(loader):
                self.test_step(step, data)
                
    def test_step(self, step, data):
            source_ids, source_mask, y_ids, lm_labels = self.get_data(data)

            outputs = self.forward_pass(source_ids, source_mask, y_ids, lm_labels)
            loss = outputs[0]
            log_dict = {
                "test_loss": loss
            }
            logits = outputs.logits
            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            try:
                test_score = self.calculate_validation_score(data, preds)
            except:
                if self.score_type == 'all':
                    test_score = [-1] * 3
                else:
                    test_score = -1

            if self.score_type == 'all':
                log_dict.update({
                    "test_score (bleu)": test_score[0],
                    "test_score (rouge)": test_score[1],
                    "test_score (combined)": test_score[2]
                })
            else:
                log_dict.update({f"test_score ({self.score_type})": test_score})

            wandb.log(log_dict)

            if step % 10 == 0:
                print(f"Batch test Loss: {loss} | Batch test score: {test_score}")

    def get_data(self, data):
        """To be implemented by specific model trainers"""
        raise NotImplementedError

    def forward_pass(self, source_ids, source_mask, y_ids, lm_labels):
        """To be implemented by specific model trainers"""
        raise NotImplementedError

    def calculate_validation_score(self, data, generated_ids):
        target_ids = data["target_ids"].to(self.device)
        target_text = [
            self.tokenizer.decode(t, skip_special_tokens=True) for t in target_ids]
        generated_text = [
            self.tokenizer.decode(p, skip_special_tokens=True) for p in generated_ids]

        if self.score_type in ['all', 'bleu', 'combined']:
            bleu = BLEU(smooth_method='floor')
            bleu_score = bleu.corpus_score(
                [ref for ref in target_text], generated_text).score

        if self.score_type in ['all', 'rouge', 'combined']:
            rouge = Rouge()
            rouge_score = rouge.get_scores(
                target_text, generated_text, avg=True)["rouge-l"]["f"]

        if self.score_type in ['combined', 'all']:
            score = (bleu_score + rouge_score) / 2.0
        if self.score_type == 'all':
            score = [bleu_score, rouge_score, score]
        elif self.score_type == 'bleu':
            score = bleu_score
        elif self.score_type == 'rouge':
            score = rouge_score
        return score
