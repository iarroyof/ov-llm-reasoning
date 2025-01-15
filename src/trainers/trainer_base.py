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
        self._log_final_metrics(avg_loss, avg_scores)
        
        return avg_loss, avg_scores
    
    def test_step(self, step, data):
        """Process single test batch"""
        source_ids, source_mask, y_ids, lm_labels = self.get_data(data)
        outputs = self.forward_pass(source_ids, source_mask, y_ids, lm_labels)
        loss = outputs[0]
        
        logits = outputs.logits
        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        
        try:
            test_score = self.calculate_validation_score(data, preds)
        except:
            if self.score_type == 'all':
                test_score = [-1] * 3
            else:
                test_score = -1

        return {'loss': loss.item(), 'score': test_score}
    
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
            
            # Log statistics about valid/invalid scores
            logger.info(f"Valid scores before averaging ({len(valid_scores)}/{len(scores)} batches):")
            logger.info(f"Valid scores: {valid_scores}")
            
            avg_scores = np.mean(valid_scores, axis=0)
            logger.info(f"Averaged scores (excluding errors): {avg_scores}")
            
            # Add success rate to wandb
            wandb.log({
                "validation_success_rate": len(valid_scores) / len(scores),
                "valid_batch_count": len(valid_scores),
                "total_batch_count": len(scores)
            })
            
            return avg_scores
        else:
            valid_scores = [s for s in scores if s != -1]
            if not valid_scores:
                logger.warning("No valid scores found after filtering")
                return -1
                
            logger.info(f"Valid scores before averaging ({len(valid_scores)}/{len(scores)} batches): {valid_scores}")
            avg_score = sum(valid_scores) / len(valid_scores)
            logger.info(f"Averaged score (excluding errors): {avg_score}")
            
            # Add success rate to wandb
            wandb.log({
                "validation_success_rate": len(valid_scores) / len(scores),
                "valid_batch_count": len(valid_scores),
                "total_batch_count": len(scores)
            })
            
            return avg_score
            
    def _log_final_metrics(self, avg_loss, avg_scores):
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
        else:
            log_dict[f"final_test_score ({self.score_type})"] = avg_scores
            logger.info(f"Final Test Loss: {avg_loss} | Final Test Score: {avg_scores}")
            
        wandb.log(log_dict)

    def calculate_validation_score(self, data, generated_ids):
        """Calculate validation scores (BLEU, ROUGE) for generated text"""
        try:
            # Log input shapes and content
            target_ids = data["target_ids"].to(self.device)
            logger.info(f"Target ids shape: {target_ids.shape}")
            logger.info(f"Generated ids shape: {generated_ids.shape}")
            
            # Decode and log sample outputs
            target_text = [
                self.tokenizer.decode(t, skip_special_tokens=True) for t in target_ids]
            generated_text = [
                self.tokenizer.decode(p, skip_special_tokens=True) for p in generated_ids]
            
            logger.info(f"Sample target text (first item): {target_text[0]}")
            logger.info(f"Sample generated text (first item): {generated_text[0]}")
            
            # Validate inputs
            if not any(target_text) or not any(generated_text):
                logger.warning("Empty sequences detected in validation")
                return [-1] * 3 if self.score_type == 'all' else -1
                
            # Strip whitespace and validate again
            target_text = [t.strip() for t in target_text if t.strip()]
            generated_text = [g.strip() for g in generated_text if g.strip()]
            
            logger.info(f"Number of non-empty target sequences: {len(target_text)}")
            logger.info(f"Number of non-empty generated sequences: {len(generated_text)}")
            
            if not target_text or not generated_text:
                logger.warning("All sequences empty after cleaning")
                return [-1] * 3 if self.score_type == 'all' else -1
    
            # Calculate BLEU if needed
            if self.score_type in ['all', 'bleu', 'combined']:
                bleu = BLEU(smooth_method='floor')
                try:
                    bleu_score = bleu.corpus_score(
                        generated_text, 
                        [[ref] for ref in target_text]
                    ).score
                    logger.info(f"BLEU calculation successful: {bleu_score}")
                except Exception as e:
                    logger.error(f"BLEU calculation failed: {str(e)}")
                    bleu_score = 0.0
    
            # Calculate ROUGE if needed
            if self.score_type in ['all', 'rouge', 'combined']:
                rouge = Rouge()
                try:
                    rouge_scores = rouge.get_scores(
                        generated_text,
                        target_text, 
                        avg=True
                    )
                    rouge_score = rouge_scores["rouge-l"]["f"]
                    logger.info(f"ROUGE calculation successful: {rouge_score}")
                except Exception as e:
                    logger.error(f"ROUGE calculation failed: {str(e)}")
                    rouge_score = 0.0
    
            # Return appropriate score format
            if self.score_type == 'combined':
                score = (bleu_score + rouge_score) / 2.0
            elif self.score_type == 'all':
                score = [bleu_score, rouge_score, (bleu_score + rouge_score) / 2.0]
                logger.info(f"Batch scores - BLEU: {bleu_score:.4f}, ROUGE: {rouge_score:.4f}, Combined: {score[2]:.4f}")
            elif self.score_type == 'bleu':
                score = bleu_score
            elif self.score_type == 'rouge':
                score = rouge_score
            
            return score
    
        except Exception as e:
            import traceback
            logger.error(f"Error in validation scoring: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [-1] * 3 if self.score_type == 'all' else -1
    def get_data(self, data):
        """To be implemented by specific model trainers"""
        raise NotImplementedError

    def forward_pass(self, source_ids, source_mask, y_ids, lm_labels):
        """To be implemented by specific model trainers"""
        raise NotImplementedError
