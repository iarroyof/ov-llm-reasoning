from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config
from huggingface_hub import login
import torch
import sys

push_model = False
# Authenticate with Hugging Face
login(token=sys.argv[1])  # Replace with your actual token in commandline

model_dir = 'sharded_t5'  # Local directory containing the model files
repo_name = 'iarroyof/t5-11b-ssm-nq-sharded'  # Hugging Face repository name

# Load model and tokenizer
model_id = 'google/t5-11b-ssm-nq'

# Push model and tokenizer to Hugging Face Hub
if push_model:
    config = T5Config.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        config=config,
        device_map='auto',
        max_memory={0: '40GB', 1: '40GB', 'cpu': '30GB'},
        low_cpu_mem_usage=True,
        quantization_config=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model.push_to_hub(repo_name, use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)  # Use slow tokenizer
tokenizer.push_to_hub(repo_name, use_auth_token=True)
