from transformers import utils
utils.clean_cache()  # Cleans the entire cache

# Or remove specific model:
from huggingface_hub import HfApi
api = HfApi()
api.delete_cache_file('models--google--t5-11b-ssm-nq')
