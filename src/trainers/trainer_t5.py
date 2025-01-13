# src/trainers/trainer_t5.py

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    T5Tokenizer
)
from src.trainers.trainer_base import BaseNeuralReasoningTrainer

class T5ReasoningTrainer(BaseNeuralReasoningTrainer):
    def get_data(self, data):
        source_ids, source_mask, target_ids = (data["source_ids"].to(self.device),
                                             data["source_masks"].to(self.device),
                                             data["target_ids"].to(self.device))
        y_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids[:, 1:].clone().detach()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100
        return source_ids, source_mask, y_ids, lm_labels

    def forward_pass(self, source_ids, source_mask, y_ids, lm_labels):
        return self.model(
            input_ids=source_ids,
            attention_mask=source_mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )

    @classmethod
    def from_pretrained(cls, model_name, device):
        # Use regular T5Tokenizer for v1_1 models, FastTokenizer for others
        if 'v1_1' in model_name or 'b-ssm-nq' in model_name:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
        else:
            tokenizer = T5TokenizerFast.from_pretrained(model_name, legacy=False)

        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)

        return cls(model, tokenizer, device)
