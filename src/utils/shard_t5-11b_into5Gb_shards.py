import torch
from transformers import T5Config, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import os
import json
import gc
import shutil

DTYPE_MAPPING = {
    'F32': torch.float32,
    'F16': torch.float16,
    'BF16': torch.bfloat16,
    'I32': torch.int32,
    'I64': torch.int64,
}

def read_header_only(file_path):
    with open(file_path, 'rb') as f:
        header_length = int.from_bytes(f.read(8), byteorder='little')
        header_json = f.read(header_length).decode('utf-8')
        return json.loads(header_json), 8 + header_length

def read_tensor_data(file_obj, offset, length, dtype, shape):
    file_obj.seek(offset)
    data = file_obj.read(length)
    return torch.frombuffer(data, dtype=dtype).reshape(shape).clone()

def create_shards(source_dir, target_dir, max_shard_size=5*1024*1024*1024):  # 5GB max
    os.makedirs(target_dir, exist_ok=True)
    
    source_file = os.path.join(source_dir, "model.safetensors")
    print(f"Reading header from {source_file}")
    
    # Read header
    header, data_offset = read_header_only(source_file)
    
    # Initialize trackers
    weight_map = {}
    current_shard = {}
    current_shard_size = 0
    shard_index = 1
    
    def save_current_shard():
        nonlocal current_shard, current_shard_size, shard_index
        if current_shard:
            shard_file = f"pytorch_model-{shard_index:05d}-of-{num_shards:05d}.bin"
            print(f"Saving shard {shard_file}")
            torch.save(current_shard, os.path.join(target_dir, shard_file))
            current_shard = {}
            current_shard_size = 0
            shard_index += 1
    
    # Calculate total size and number of shards needed
    total_size = os.path.getsize(source_file)
    num_shards = max(8, (total_size + max_shard_size - 1) // max_shard_size)
    print(f"Will create {num_shards} shards")
    
    # First pass: Read tensor info to estimate sizes
    tensor_sizes = {}
    for tensor_name, tensor_info in header.items():
        if tensor_name == "__metadata__":
            continue
        tensor_sizes[tensor_name] = (tensor_info['data_offsets'][1] - tensor_info['data_offsets'][0])
    
    # Sort tensors by size for better distribution
    sorted_tensors = sorted(tensor_sizes.items(), key=lambda x: x[1], reverse=True)
    
    print("Reading tensor data...")
    with open(source_file, 'rb') as f:
        for tensor_name, _ in sorted_tensors:
            tensor_info = header[tensor_name]
            
            # Read tensor data
            tensor_offset = tensor_info['data_offsets'][0] + data_offset
            tensor_length = tensor_info['data_offsets'][1] - tensor_info['data_offsets'][0]
            tensor_dtype = DTYPE_MAPPING[tensor_info['dtype']]
            tensor_shape = tensor_info['shape']
            
            print(f"Reading tensor {tensor_name}")
            tensor = read_tensor_data(f, tensor_offset, tensor_length, tensor_dtype, tensor_shape)
            
            # If adding this tensor would exceed shard size, save current shard
            if current_shard_size + tensor_length > max_shard_size and current_shard:
                save_current_shard()
            
            # Add tensor to current shard
            current_shard[tensor_name] = tensor
            current_shard_size += tensor_length
            weight_map[tensor_name] = f"pytorch_model-{shard_index:05d}-of-{num_shards:05d}.bin"
            
            # Clean up
            if len(current_shard) >= 20:  # Save more frequently
                save_current_shard()
                gc.collect()
                torch.cuda.empty_cache()
    
    # Save last shard
    save_current_shard()
    
    # Create index file
    index = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }
    
    with open(os.path.join(target_dir, "pytorch_model.bin.index.json"), "w") as f:
        json.dump(index, f)
    
    # Copy config files
    for file in os.listdir(source_dir):
        if file.endswith(('.json', '.model', '.tokenizer')):
            src_path = os.path.join(source_dir, file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, target_dir)

# Clean up previous attempts
sharded_dir = "sharded_t5"
if os.path.exists(sharded_dir):
    shutil.rmtree(sharded_dir)

# Get source directory
cache_dir = "/root/.cache/huggingface/hub/models--google--t5-11b-ssm-nq/snapshots"
snapshot_dirs = os.listdir(cache_dir)
source_dir = os.path.join(cache_dir, snapshot_dirs[0])

# Create shards
print("Creating sharded structure...")
create_shards(source_dir, sharded_dir)

# Load model
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# First load config to get correct shapes
config = T5Config.from_pretrained("google/t5-11b-ssm-nq")

model = AutoModelForSeq2SeqLM.from_pretrained(
    sharded_dir,
    config=config,  # Use original config to ensure correct shapes
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "40GB", 1: "40GB", "cpu": "30GB"},
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
