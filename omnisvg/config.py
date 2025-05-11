"""
Configuration settings for OmniSVG
"""

# SVG Canvas Size
CANVAS_SIZE = 200

# Special tokens for SVG tokenization
SPECIAL_TOKENS = ["<PAD>", "<SOP>", "<EOS>", "<COLOR>"] + list("MLCAZF")

# Base model configurations
BASE_MODEL = "Qwen/Qwen2.5-3B"
LARGE_MODEL = "Qwen/Qwen2.5-7B"

# Training parameters
TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "num_train_epochs": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "save_total_limit": 2,
    "fp16": True,
    "save_safetensors": True,
    "learning_rate": 3e-4,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.1,
    "optim": "adamw_torch",
}

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Generation parameters
GENERATION_CONFIG = {
    "max_new_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.95,
    "do_sample": True,
    "repetition_penalty": 1.1
}

# Dataset paths
MMSVG_ICON_DATASET = "OmniSVG/MMSVG-Icon"
MMSVG_ILLUSTRATION_DATASET = "OmniSVG/MMSVG-Illustration"
MMSVG_CHARACTER_DATASET = "OmniSVG/MMSVG-Character"