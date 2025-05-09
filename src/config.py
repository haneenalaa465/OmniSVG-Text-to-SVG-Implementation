"""
Configuration settings for OmniSVG.

This module provides configuration settings for the OmniSVG model,
training pipeline, and inference.
"""
import os
from typing import Dict, Any

# Default model configuration
MODEL_CONFIG = {
    "base_model_name": "Qwen/Qwen2.5-VL-3B",
    "svg_vocab_size": 40000,
    "max_svg_len": 8192,
    "viewbox_size": 200
}

# Default training configuration
TRAINING_CONFIG = {
    "batch_size": 4,
    "num_epochs": 5,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    "save_steps": 1000,
    "eval_steps": 500,
    "log_steps": 100
}

# Default inference configuration
INFERENCE_CONFIG = {
    "max_length": 1024,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "num_samples": 1
}

# Data paths
DATA_PATHS = {
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
    "train_file": "train.json",
    "val_file": "val.json",
    "test_file": "test.json"
}

# Model paths
MODEL_PATHS = {
    "output_dir": "models/omnisvg",
    "final_model_dir": "models/omnisvg/final_model",
    "best_model_dir": "models/omnisvg/best_model"
}

def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    import json
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_config_to_file(config: Dict[str, Any], config_file: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to save the configuration file
    """
    import json
    
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    
    return merged_config

def get_config(config_file: str = None) -> Dict[str, Any]:
    """
    Get configuration settings.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    config = {
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "inference": INFERENCE_CONFIG,
        "data_paths": DATA_PATHS,
        "model_paths": MODEL_PATHS
    }
    
    if config_file is not None:
        override_config = load_config_from_file(config_file)
        config = merge_configs(config, override_config)
    
    return config