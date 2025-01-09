# src/trainers/trainer_t5_xxl.py

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BitsAndBytesConfig
)
from src.trainers.trainer_base import BaseNeuralReasoningTrainer
from src.utils.memory import log_gpu_memory_usage
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

class T5LargeReasoningTrainer(BaseNeuralReasoningTrainer):
    def __init__(self, model, tokenizer, device, gen_test_score='bleu', quantization='8bit'):
        super().__init__(model, tokenizer, device, gen_test_score)
        self.quantization = quantization
        # Log initial memory state
        wandb.log({"initial_memory_state": log_gpu_memory_usage()})

    @classmethod
    def from_pretrained(cls, model_name, device, quantization='8bit', max_memory=None):
        """
        Args:
            model_name: Name or path of the model
            device: Device to load the model on
            quantization: '8bit' or '4bit' for different quantization schemes
        """
        use_8bit = quantization == '8bit'
        
        # Adjust memory handling based on quantization
        if max_memory is None:
            max_memory = {
            0: "40GB",
            1: "40GB",
            "cpu": "30GB"
            }
        
        # Optimized quantization config based on bit size
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            load_in_4bit=not use_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=not use_8bit,
            bnb_4bit_compute_dtype=torch.float16 if not use_8bit else None,
            llm_int8_threshold=6.0 if use_8bit else None,
            llm_int8_has_fp16_weight=True if use_8bit else None
        )

        # Load model with adjusted settings
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
        )

        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Prepare model with optimized settings
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
        model.config.use_cache = False  # Disable KV cache for training

        # Adjust LoRA config based on quantization
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
            bits=4 if not use_8bit else 8,
            double_quant=not use_8bit
        )
        model = get_peft_model(model, peft_config)

        return cls(model, tokenizer, device, quantization=quantization)

    def train(self, optimizer, loader, epoch):
        self.model.train()
        total_steps = len(loader)
        
        for step, data in enumerate(loader):
            source_ids, source_mask, y_ids, lm_labels = self.get_data(data)
            
            outputs = self.forward_pass(source_ids, source_mask, y_ids, lm_labels)
            loss = outputs[0]

            # Enhanced logging with memory tracking
            wandb.log({
                "train_batch_loss": loss,
                "global_step": epoch * total_steps + step,
                "quantization": self.quantization,
                **log_gpu_memory_usage()
            })
            
            if step % 10 == 0:
                print(f"Epoch: {epoch} | Step: {step} | Loss: {loss} | Quantization: {self.quantization}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(self, loader, epoch):
        self.model.eval()
        total_steps = len(loader)
        
        for step, data in enumerate(loader):
            source_ids, source_mask, y_ids, lm_labels = self.get_data(data)
            
            outputs = self.forward_pass(source_ids, source_mask, y_ids, lm_labels)
            loss = outputs[0]
            
            # Add memory tracking during evaluation
            log_dict = {
                "test_loss": loss,
                "epoch": epoch,
                "quantization": self.quantization,
                **log_gpu_memory_usage()
            }
            
            wandb.log(log_dict)
            
            if step % 10 == 0:
                print(f"Epoch: {epoch} | Test Step: {step} | Loss: {loss}")

    def get_data(self, data):
        """Same as before but with memory tracking"""
        source_ids, source_mask, target_ids = (data["source_ids"].to(self.device),
                                             data["source_masks"].to(self.device),
                                             data["target_ids"].to(self.device))
        y_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids[:, 1:].clone().detach()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100
        
        # Log memory after data preparation
        wandb.log({"data_preparation_memory": log_gpu_memory_usage()})
        
        return source_ids, source_mask, y_ids, lm_labels
