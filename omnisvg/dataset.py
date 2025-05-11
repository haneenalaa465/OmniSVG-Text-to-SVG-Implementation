"""
Dataset handling for OmniSVG
"""

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from typing import Dict, List, Any

from omnisvg.config import MMSVG_ICON_DATASET, MMSVG_ILLUSTRATION_DATASET, MMSVG_CHARACTER_DATASET


def preprocess_function(examples, text_tokenizer, svg_tokenizer):
    """
    Process examples for training with instruction formatting
    
    Args:
        examples: Raw dataset examples
        text_tokenizer: Text tokenizer for processing prompts
        svg_tokenizer: SVG tokenizer for processing SVG content
        
    Returns:
        Processed examples with input_ids and labels
    """
    result = {"input_ids": [], "labels": []}

    caption_field = "description" if "description" in examples else "caption"

    for i in range(len(examples[caption_field])):
        caption = examples[caption_field][i]
        svg_text = examples["svg"][i]

        # Skip examples with empty captions
        if caption is None or caption.strip() == "":
            continue

        # Add a clear instruction format
        instruction = f"You are a helpful SVG Generation assistant, designed to generate SVG. Generate an SVG icon for: {caption}"

        # Tokenize the instruction
        prefix_ids = text_tokenizer(instruction, add_special_tokens=True).input_ids

        # Encode SVG
        svg_ids = svg_tokenizer.encode(svg_text)

        # Skip examples with minimal SVG content
        if len(svg_ids) <= 2:  # Only <SOP> and <EOS>
            continue

        # Concatenate instruction tokens and SVG tokens
        input_ids = prefix_ids + svg_ids

        # Set labels to -100 for instruction tokens (no loss) and keep SVG tokens for loss
        labels = [-100] * len(prefix_ids) + svg_ids

        result["input_ids"].append(input_ids)
        result["labels"].append(labels)

    return result


def load_svg_datasets(text_tokenizer, svg_tokenizer, num_samples=None, split="train"):
    """
    Load and prepare SVG datasets for training or evaluation
    
    Args:
        text_tokenizer: The text tokenizer
        svg_tokenizer: The SVG tokenizer
        num_samples: Number of samples to load (None = all)
        split: Dataset split to use ("train", "validation", "test")
        
    Returns:
        Processed dataset ready for training
    """
    subset = f"{split}[:{num_samples}]" if num_samples else split
    
    # Load datasets
    icon_dataset = load_dataset(MMSVG_ICON_DATASET, split=subset)
    illustration_dataset = load_dataset(MMSVG_ILLUSTRATION_DATASET, split=subset)
    
    # Process datasets
    icon_processed = icon_dataset.map(
        lambda examples: preprocess_function(examples, text_tokenizer, svg_tokenizer),
        batched=True,
        remove_columns=icon_dataset.column_names,
        num_proc=4
    )
    
    illustration_processed = illustration_dataset.map(
        lambda examples: preprocess_function(examples, text_tokenizer, svg_tokenizer),
        batched=True,
        remove_columns=illustration_dataset.column_names,
        num_proc=4
    )
    
    # Combine datasets
    combined_dataset = concatenate_datasets([icon_processed, illustration_processed])
    
    return combined_dataset


class PaddingCollator:
    """Collator that handles padding and attention masks for variable length sequences"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        
    def __call__(self, features):
        # Get max length in batch
        max_length = max(len(feature["input_ids"]) for feature in features)

        # Pad all sequences to max length
        batch = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }

        for feature in features:
            # Padding for input_ids
            padding_length = max_length - len(feature["input_ids"])
            padded_input_ids = feature["input_ids"] + [self.pad_token_id] * padding_length
            batch["input_ids"].append(padded_input_ids)

            # Padding for labels (-100 to ignore in loss)
            padded_labels = feature["labels"] + [-100] * padding_length
            batch["labels"].append(padded_labels)

            # Create attention mask
            attention_mask = [1] * len(feature["input_ids"]) + [0] * padding_length
            batch["attention_mask"].append(attention_mask)

        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch


def analyze_training_data(dataset):
    """
    Analyze SVG training data for balance of command types
    
    Args:
        dataset: Processed dataset with tokenized SVGs
        
    Returns:
        stats: Statistics about the dataset
        recommendations: Recommendations based on the analysis
    """
    from omnisvg.tokenizer import BASE_ID
    
    command_counts = {cmd: 0 for cmd in ["M", "L", "C", "A", "Z", "F"]}
    total_svgs = 0
    token_length_distribution = []
    
    for item in dataset:
        total_svgs += 1
        svg_tokens = item["input_ids"]
        
        # Count command types
        for cmd in ["M", "L", "C", "A", "Z", "F"]:
            cmd_token = BASE_ID[cmd]
            command_counts[cmd] += svg_tokens.count(cmd_token)
            
        # Track token length
        token_length_distribution.append(len(svg_tokens))
    
    # Calculate statistics
    stats = {
        "total_svgs": total_svgs,
        "command_distribution": {cmd: count / sum(command_counts.values()) for cmd, count in command_counts.items()},
        "token_length": {
            "min": min(token_length_distribution),
            "max": max(token_length_distribution),
            "avg": sum(token_length_distribution) / len(token_length_distribution),
            "p25": np.percentile(token_length_distribution, 25),
            "p50": np.percentile(token_length_distribution, 50),
            "p75": np.percentile(token_length_distribution, 75),
            "p90": np.percentile(token_length_distribution, 90)
        }
    }
    
    # Recommendations based on the analysis
    recommendations = []
    
    # Check for under-represented commands
    for cmd, ratio in stats["command_distribution"].items():
        if ratio < 0.05:  # Less than 5% of commands
            recommendations.append(f"Command '{cmd}' is underrepresented ({ratio:.1%}). Consider adding more examples.")
    
    # Check for token length distribution
    if stats["token_length"]["p90"] > 10000:
        recommendations.append("90% of examples have more than 10000 tokens. Consider adding more simple examples for better training.")
    
    return stats, recommendations