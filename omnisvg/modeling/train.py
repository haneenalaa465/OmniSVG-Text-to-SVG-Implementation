"""
Training utilities for OmniSVG models
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Trainer, 
    TrainingArguments
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model, 
    PeftModel
)

from omnisvg.config import BASE_MODEL, TRAINING_ARGS, LORA_CONFIG
from omnisvg.dataset import load_svg_datasets, PaddingCollator
from omnisvg.tokenizer import SVGTokenizer


def setup_training(model, text_tokenizer, svg_tokenizer):
    """
    Set up the model for QLoRA training
    
    Args:
        model: Base LLM model
        text_tokenizer: Text tokenizer
        svg_tokenizer: SVG tokenizer
        
    Returns:
        Model prepared for training
    """
    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # Freeze text embeddings (original vocab)
    embedding_layer = model.get_input_embeddings()
    orig_vocab_size = len(text_tokenizer) - svg_tokenizer.vocab_size
    embedding_layer.weight[:orig_vocab_size].requires_grad = False
    
    # Ensure SVG embeddings are trainable
    embedding_layer.weight[-svg_tokenizer.vocab_size:].requires_grad = True

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=LORA_CONFIG["task_type"]
    )

    # Apply LoRA configuration to model
    model = get_peft_model(model, lora_config)
    
    return model


def train_model(model, text_tokenizer, svg_tokenizer, output_dir="./models/omnisvg", num_samples=16000):
    """
    Train the model with the enhanced token handling
    
    Args:
        model: Model prepared for training
        text_tokenizer: Text tokenizer
        svg_tokenizer: SVG tokenizer
        output_dir: Directory to save model checkpoints
        num_samples: Number of samples to train on
    
    Returns:
        Trained model
    """
    # Load datasets
    print("Loading and processing training data...")
    train_dataset = load_svg_datasets(
        text_tokenizer, 
        svg_tokenizer, 
        num_samples=num_samples, 
        split="train"
    )
    
    # Sample evaluation set
    eval_dataset = load_svg_datasets(
        text_tokenizer, 
        svg_tokenizer, 
        num_samples=min(2000, num_samples//4), 
        split="validation"
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        **TRAINING_ARGS
    )
    
    # Initialize trainer with custom collator
    data_collator = PaddingCollator(text_tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save the trained model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    
    return model


def init_model_for_training(use_8bit=True):
    """
    Initialize a new model for training from scratch
    
    Args:
        use_8bit: Whether to use 8-bit quantization
        
    Returns:
        model: Initialized model
        text_tokenizer: Text tokenizer
        svg_tokenizer: SVG tokenizer
    """
    # Initialize SVG tokenizer
    svg_tokenizer = SVGTokenizer()
    
    # Set up quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model and tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=quant_config if use_8bit else None,
        trust_remote_code=True
    )
    
    # Add SVG tokens to text tokenizer
    text_tokenizer.add_tokens([f"<svg_{i}>" for i in range(svg_tokenizer.vocab_size)], special_tokens=False)
    
    # Resize model embeddings to accommodate SVG tokens
    model.resize_token_embeddings(len(text_tokenizer))
    
    # Set up model for training
    model = setup_training(model, text_tokenizer, svg_tokenizer)
    
    return model, text_tokenizer, svg_tokenizer