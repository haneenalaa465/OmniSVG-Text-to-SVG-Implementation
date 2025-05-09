"""
Training Pipeline for OmniSVG.

This module provides the training pipeline for the OmniSVG model,
including data loading, training loops, and evaluation.
"""
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from tqdm import tqdm

from .model import OmniSVGModel
from .data_processing import SVGProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SVGDataset(Dataset):
    """
    Dataset for text-to-SVG training.
    """
    
    def __init__(
        self,
        data_path: str,
        svg_processor: SVGProcessor,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the SVG dataset.
        
        Args:
            data_path: Path to the JSON file containing text-SVG pairs
            svg_processor: SVG processor for tokenization
            max_samples: Maximum number of samples to load
        """
        self.svg_processor = svg_processor
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Limit number of samples if requested
        if max_samples is not None:
            self.data = self.data[:max_samples]
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing input_ids and labels
        """
        item = self.data[idx]
        text = item['text']
        svg_content = item['svg']
        
        # Prepare training data
        input_ids, labels = self.svg_processor.prepare_training_data(text, svg_content)
        
        # Create SVG token indices
        svg_token_indices = torch.zeros_like(input_ids, dtype=torch.long)
        svg_token_indices[labels != -100] = 1  # SVG tokens are not masked in labels
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'svg_token_indices': svg_token_indices,
            'attention_mask': torch.ones_like(input_ids)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of batch items
        
    Returns:
        Batched inputs
    """
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = torch.full((len(batch), max_len), 
                           0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), 
                        -100, dtype=torch.long)
    svg_token_indices = torch.zeros((len(batch), max_len), dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        labels[i, :seq_len] = item['labels']
        svg_token_indices[i, :seq_len] = item['svg_token_indices']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'svg_token_indices': svg_token_indices
    }


def train_model(
    model: OmniSVGModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    output_dir: str = "models/omnisvg",
    num_epochs: int = 5,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
    save_steps: int = 1000,
    eval_steps: int = 500,
    log_steps: int = 100,
    device: Optional[str] = None
) -> OmniSVGModel:
    """
    Train the OmniSVG model.
    
    Args:
        model: OmniSVG model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        output_dir: Directory to save checkpoints to
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm
        save_steps: Steps between checkpoint saves
        eval_steps: Steps between evaluations
        log_steps: Steps between logging
        device: Device to train on
        
    Returns:
        Trained model
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    total_steps = len(train_dataloader) * num_epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate * 0.1
    )
    
    # Initialize tracking variables
    global_step = 0
    best_val_loss = float('inf')
    train_losses = []
    
    # Training loop
    model.train()
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                svg_token_indices=batch['svg_token_indices']
            )
            
            loss = outputs['loss']
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimization step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Logging
            if global_step % log_steps == 0:
                logger.info(f"Step {global_step} - Loss: {loss.item():.6f}")
                train_losses.append(loss.item())
            
            # Evaluation
            if val_dataloader is not None and global_step % eval_steps == 0:
                eval_loss = evaluate_model(model, val_dataloader, device)
                logger.info(f"Step {global_step} - Eval Loss: {eval_loss:.6f}")
                
                # Save best model
                if eval_loss < best_val_loss:
                    best_val_loss = eval_loss
                    model.save_model(os.path.join(output_dir, "best_model"))
                    logger.info(f"New best model saved with eval loss: {best_val_loss:.6f}")
                
                model.train()  # Switch back to train mode
            
            # Save checkpoint
            if global_step % save_steps == 0:
                model.save_model(os.path.join(output_dir, f"checkpoint-{global_step}"))
                logger.info(f"Checkpoint saved at step {global_step}")
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f} - Time: {epoch_time:.2f}s")
        
        # Save epoch checkpoint
        model.save_model(os.path.join(output_dir, f"epoch-{epoch+1}"))
    
    # Save final model
    model.save_model(os.path.join(output_dir, "final_model"))
    logger.info("Training completed!")
    
    # Save training losses
    with open(os.path.join(output_dir, "train_losses.json"), 'w') as f:
        json.dump(train_losses, f)
    
    return model


def evaluate_model(
    model: OmniSVGModel,
    eval_dataloader: DataLoader,
    device: str = 'cuda'
) -> float:
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        eval_dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Evaluation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                svg_token_indices=batch['svg_token_indices']
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
    
    avg_loss = total_loss / len(eval_dataloader)
    return avg_loss


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Main training function.
    """
    # Set random seed
    set_seed(42)
    
    # Load configuration
    config = {
        "data_path": "data/processed/",
        "train_file": "train.json",
        "val_file": "val.json",
        "output_dir": "models/omnisvg",
        "base_model_name": "Qwen/Qwen2.5-VL-3B",
        "svg_vocab_size": 40000,
        "max_svg_len": 8192,
        "batch_size": 4,
        "num_epochs": 5,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "max_samples": None  # Set to a number for debugging
    }
    
    # Initialize SVG processor
    svg_processor = SVGProcessor(
        base_tokenizer_name=config["base_model_name"],
        max_svg_len=config["max_svg_len"]
    )
    
    # Create datasets
    train_dataset = SVGDataset(
        os.path.join(config["data_path"], config["train_file"]),
        svg_processor,
        max_samples=config["max_samples"]
    )
    
    val_dataset = SVGDataset(
        os.path.join(config["data_path"], config["val_file"]),
        svg_processor,
        max_samples=config["max_samples"]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = OmniSVGModel(
        base_model_name=config["base_model_name"],
        svg_vocab_size=config["svg_vocab_size"]
    )
    
    # Train model
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_dir=config["output_dir"],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )


if __name__ == "__main__":
    main()