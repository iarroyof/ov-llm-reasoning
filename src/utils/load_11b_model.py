from transformers import T5ForConditionalGeneration, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import torch
from pdb import set_trace as st
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    load_in_8bit_fp32_cpu_offload=True
)
#model = T5ForConditionalGeneration.from_pretrained(
#    'google-t5/t5-11b',
#    quantization_config=bnb_config,
#    torch_dtype=torch.float16,
#    device_map='auto')
model = AutoModelForSeq2SeqLM.from_pretrained(
    #"google/ul2",
    'ybelkada/t5-11b-sharded',
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map='auto')


st()
