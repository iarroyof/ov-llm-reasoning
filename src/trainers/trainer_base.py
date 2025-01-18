# src/trainers/trainer_base.py

import torch
import numpy as np
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
    def __init__(self, model, tokenizer, device, gen_test_score='bleu', gen_method='beam', temp=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.score_type = gen_test_score
        self.gen_method = gen_method
        self.temp = temp

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
            logger.info(f"Epoch: {epoch} | Step {step} | Train Loss: {loss}")
        
        loss.backward()
        optimizer.step()
        
    def test(self, loader):
        """Run evaluation"""
        self.model.eval()
        total_loss = 0
        all_scores = []
        num_batches = 0
        
        with torch.no_grad():
            for step, data in enumerate(loader):
                batch_metrics = self.test_step(step, data)
                total_loss += batch_metrics['loss']
                all_scores.append(batch_metrics['score'])
                num_batches += 1
                
                if step % 10 == 0:
                    logger.info(f"Testing batch {step} | Batch test Loss: {batch_metrics['loss']} | Batch test score: {batch_metrics['score']}")
        
        # Calculate averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_scores = self._aggregate_scores(all_scores)
        
        # Log final metrics
        self._log_final_metrics(avg_loss, avg_scores, all_scores)
        
        return avg_loss, avg_scores
    
    def test_step(self, step, data):
        """Process single test batch with configurable generation method.
        
        Args:
            step: Current step number
            data: Batch data
            generation_method: Either 'beam' for beam search or 'sample' for sampling-based generation
        """
        source_ids, source_mask, y_ids, lm_labels = self.get_data(data)
        outputs = self.forward_pass(source_ids, source_mask, y_ids, lm_labels)
        loss = outputs[0]
        # Common generation parameters
        generation_config = {
            'input_ids': source_ids,
            'attention_mask': source_mask,
            'max_length': y_ids.shape[1],
            'min_length': 5,
            'no_repeat_ngram_size': 2,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        # Method-specific parameters
        if self.gen_method == 'beam':
            generation_config.update({
                'num_beams': 4,
                'early_stopping': True,
            })
        elif self.gen_method == 'sample':
            generation_config.update({
                'do_sample': True,
                'temperature': self.temp,
                'top_p': 0.9,
            })
        else:
            raise ValueError(f"Unknown generation method: {gen_method}. Available options: ['beam', 'sample']")
        # Generate text
        generated_ids = self.model.generate(**generation_config)        
        try:
            test_score = self.calculate_validation_score(data, generated_ids)
        except Exception as e:
            logger.error(f"Error in test scoring: {str(e)}")
            if self.score_type == 'all':
                test_score = [-1] * 3
            else:
                test_score = -1
        test_loss = loss.item()
        wandb.log({
            "test_loss": test_loss,
            "step": step
            })
        return {'loss': test_loss, 'score': test_score}

    def _aggregate_scores(self, scores):
        """Average scores across batches, ignoring error cases"""
        if not scores:
            logger.warning("No scores to aggregate")
            return [-1] * 3 if self.score_type == 'all' else -1
        
        if self.score_type == 'all':
            scores_array = np.array(scores)
            # Filter out error cases (all -1)
            valid_mask = ~np.all(scores_array == -1, axis=1)
            valid_scores = scores_array[valid_mask]
            
            if len(valid_scores) == 0:
                logger.warning("No valid scores found after filtering")
                return [-1] * 3
            
            # Apply weights to give more importance to non-zero scores
            weights = (valid_scores != 0).astype(float)
            weights = weights / weights.sum(axis=0, keepdims=True)
            avg_scores = np.sum(valid_scores * weights, axis=0)
            
            # Log statistics
            logger.info(f"Valid scores ({len(valid_scores)}/{len(scores)} batches):")
            logger.info(f"BLEU mean: {avg_scores[0]:.4f}")
            logger.info(f"ROUGE mean: {avg_scores[1]:.4f}")
            logger.info(f"Combined mean: {avg_scores[2]:.4f}")
            
            return avg_scores
        else:
            valid_scores = [s for s in scores if s != -1 and s != 0]
            if not valid_scores:
                return -1
            return sum(valid_scores) / len(valid_scores)
            
    def _get_summary_statistics(self, data):
        if not isinstance(data, list):
            raise ValueError("Input data should be a list or a list of lists.")
        # Check if it's a list of lists
        if all(isinstance(i, list) for i in data):
            data_np = np.array(data)
        else:
            data_np = np.array([data])
        
        if data_np.ndim != 2:
            raise ValueError("Input data should be a 2-dimensional list of lists.")
    
        # Calculate summary statistics
        #mean = np.mean(data_np, axis=0)
        std = np.std(data_np, axis=0)
        min_val = np.min(data_np, axis=0)
        max_val = np.max(data_np, axis=0)
        percentiles = np.percentile(data_np, [25, 50, 75], axis=0)
        mode = stats.mode(data_np, axis=0)
    
        # Display the results
        return {#"Mean": mean,
            "Test Score Standard Deviation": std,
            "Test Score Min": min_val,
            "Test Score Max": max_val,
            " Test Score Percentiles (25th, 50th, 75th)": percentiles,
            "Test Score Mode": mode}
    
    def _log_final_metrics(self, avg_loss, avg_scores, all_scores):
        """Log final metrics to wandb"""
        log_dict = {
            "final_test_loss": avg_loss,
        }
        
        if self.score_type == 'all':
            log_dict.update({
                "final_test_score (bleu)": avg_scores[0],
                "final_test_score (rouge)": avg_scores[1],
                "final_test_score (combined)": avg_scores[2]
            })
            logger.info(f"Final Test Loss: {avg_loss} | Final Test Scores (Bleu, Rouge, Combined): {avg_scores}")
            logger.info(f"Test Scores Sumary Statistics:")
            logger.info()
        else:
            log_dict[f"final_test_score ({self.score_type})"] = avg_scores
            logger.info(f"Final Test Loss: {avg_loss} | Final Test Score: {avg_scores}")
            
        log_dict.update(self._get_summary_statistics(all_scores))
        wandb.log(log_dict)

    def calculate_validation_score(self, data, generated_ids):
        try:
            target_ids = data["target_ids"].to(self.device)
            
            # Decode and clean sequences
            target_text = [self.tokenizer.decode(t, skip_special_tokens=True).strip() for t in target_ids]
            generated_text = [self.tokenizer.decode(p, skip_special_tokens=True).strip() for p in generated_ids]
            
            # Filter out empty sequences from both lists simultaneously
            paired_texts = [(g, t) for g, t in zip(generated_text, target_text) 
                           if g.strip() and t.strip()]
            
            if not paired_texts:
                logger.warning("No valid text pairs found")
                return [-1] * 3 if self.score_type == 'all' else -1
                
            generated_text, target_text = zip(*paired_texts)
            
            if self.score_type in ['all', 'bleu', 'combined']:
                bleu = BLEU(smooth_method='exp')  # Changed to exp smoothing
                references = [[t] for t in target_text]  # Proper reference format
                bleu_score = bleu.corpus_score(list(generated_text), references).score
                
            if self.score_type in ['all', 'rouge', 'combined']:
                rouge = Rouge()
                # Convert tuples to lists and ensure non-empty
                try:
                    rouge_scores = rouge.get_scores(
                        list(generated_text), 
                        list(target_text), 
                        avg=True
                    )
                    rouge_score = rouge_scores["rouge-l"]["f"]
                except Exception as e:
                    logger.error(f"ROUGE calculation error: {str(e)}")
                    rouge_score = 0.0
    
            # Return scores
            if self.score_type == 'combined':
                score = (bleu_score + rouge_score) / 2.0
            elif self.score_type == 'all':
                score = [bleu_score, rouge_score, (bleu_score + rouge_score) / 2.0]
            elif self.score_type == 'bleu':
                score = bleu_score
            elif self.score_type == 'rouge':
                score = rouge_score
                
            return score
    
        except Exception as e:
            logger.error(f"Error in validation scoring: {str(e)}", exc_info=True)
            return [-1] * 3 if self.score_type == 'all' else -1
        
    def get_data(self, data):
        """To be implemented by specific model trainers"""
        raise NotImplementedError

    def forward_pass(self, source_ids, source_mask, y_ids, lm_labels):
        """To be implemented by specific model trainers"""
        raise NotImplementedError
