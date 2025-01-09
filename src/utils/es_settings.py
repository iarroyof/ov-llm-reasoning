import yaml
import os

def load_config(config_path="configs/es_config.yaml"):
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Load configuration at the module level
config = load_config()

# Access ElasticSearch settings
es_settings = config.get("ElasticSearch", {})
