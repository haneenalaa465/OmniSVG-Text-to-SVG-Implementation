"""
OmniSVG Model Implementation.

This module implements the OmniSVG architecture for text-to-SVG generation
based on the Qwen2.5-VL 3B model.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel, 
    PreTrainedTokenizer
)

class OmniSVGModel(nn.Module):
    """
    OmniSVG Model for Text-to-SVG generation.
    
    This model leverages the Qwen2.5-VL 3B model as a base and extends
    it with SVG-specific token embeddings and decoding capabilities.
    """
    
    def __init__(
        self, 
        base_model_name: str = "Qwen/Qwen2.5-VL-3B",
        svg_vocab_size: int = 40000,
        device: Optional[str] = None
    ):
        """
        Initialize the OmniSVG model.
        
        Args:
            base_model_name: Name of the pretrained VL model to use
            svg_vocab_size: Size of the SVG token vocabulary
            device: Device to use for model computation
        """
        super().__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load base tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32
        )
        
        # Get dimensions from base model
        self.embed_dim = self.base_model.config.hidden_size
        
        # Create SVG token embeddings
        self.svg_token_embeddings = nn.Embedding(svg_vocab_size, self.embed_dim)
        self.svg_vocab_size = svg_vocab_size
        
        # Initialize the SVG token embeddings with random values
        nn.init.normal_(self.svg_token_embeddings.weight, std=0.02)
        
        # SVG token output projection
        self.svg_output_projection = nn.Linear(self.embed_dim, svg_vocab_size, bias=False)
        
        # Move model to device
        self.to(self.device)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        svg_token_indices: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input tokens
            labels: Labels for computing the language modeling loss
            svg_token_indices: Indices of SVG tokens in the input
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Dictionary containing loss and logits
        """
        batch_size, seq_length = input_ids.shape
        
        # Create embeddings for all tokens
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # Replace embeddings for SVG tokens
        if svg_token_indices is not None:
            for i in range(batch_size):
                for j in range(seq_length):
                    if svg_token_indices[i, j] == 1:
                        svg_token_id = input_ids[i, j] - self.tokenizer.vocab_size
                        if 0 <= svg_token_id < self.svg_vocab_size:
                            embeddings[i, j] = self.svg_token_embeddings(
                                torch.tensor([svg_token_id], device=self.device)
                            )
        
        # Run the base model
        outputs = self.base_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=None,  # We'll compute the loss ourselves
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
        
        # Project to SVG token space for SVG tokens, use base model outputs for text tokens
        logits = outputs.logits.clone()
        
        # Replace logits for SVG tokens
        if svg_token_indices is not None:
            for i in range(batch_size):
                for j in range(seq_length):
                    if svg_token_indices[i, j] == 1:
                        svg_logits = self.svg_output_projection(hidden_states[i, j])
                        logits[i, j, :self.tokenizer.vocab_size] = float('-inf')  # Mask text tokens
                        logits[i, j, self.tokenizer.vocab_size:self.tokenizer.vocab_size + self.svg_vocab_size] = svg_logits
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Only compute loss for SVG tokens (not masked with -100)
            loss_mask = (labels != -100)
            if loss_mask.sum() > 0:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1))[loss_mask.view(-1)],
                    labels.view(-1)[loss_mask.view(-1)]
                )
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits
            }
        else:
            return (loss, logits)
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for generation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input tokens
            
        Returns:
            Dictionary of model inputs
        """
        # Determine which tokens are SVG tokens
        svg_token_indices = torch.zeros_like(input_ids, dtype=torch.long)
        svg_token_indices[input_ids >= self.tokenizer.vocab_size] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'svg_token_indices': svg_token_indices
        }
    
    def generate_svg(
        self,
        text: str,
        svg_processor: Any,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate SVG from text prompt.
        
        Args:
            text: Text prompt to generate SVG from
            svg_processor: SVG processor for tokenization and detokenization
            max_length: Maximum length of generated sequence
            temperature: Generation temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_return_sequences: Number of SVGs to generate
            
        Returns:
            List of generated SVG strings
        """
        # Encode the text prompt
        encoded_text = svg_processor.encode_text_for_svg_generation(text).to(self.device)
        
        # Add start of SVG special token
        start_token = torch.tensor([svg_processor.special_to_token['<SOP>']], device=self.device)
        input_ids = torch.cat([encoded_text[0], start_token])
        input_ids = input_ids.unsqueeze(0)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Generate SVG tokens
        with torch.no_grad():
            svg_token_indices = torch.zeros_like(input_ids, dtype=torch.long)
            svg_token_indices[:, -1] = 1  # Mark the start token as an SVG token
            
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=svg_processor.special_to_token['<EOS>'] + self.tokenizer.vocab_size
            )
        
        # Extract SVG tokens from generated sequences
        generated_svgs = []
        
        for i in range(num_return_sequences):
            # Get the generated sequence
            generated_seq = outputs[i]
            
            # Find where the SVG tokens start (after the text tokens)
            svg_start = len(encoded_text[0])
            
            # Extract SVG tokens
            svg_tokens = generated_seq[svg_start:].tolist()
            
            # Adjust token IDs to match SVG processor's token space
            adjusted_tokens = [t - self.tokenizer.vocab_size for t in svg_tokens]
            
            # Convert tokens back to SVG
            svg_content = svg_processor.tokens_to_svg(adjusted_tokens)
            generated_svgs.append(svg_content)
        
        return generated_svgs
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the model and its configuration.
        
        Args:
            output_dir: Directory to save the model to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(os.path.join(output_dir, "base_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        
        # Save SVG-specific parameters
        torch.save(self.svg_token_embeddings.state_dict(), 
                  os.path.join(output_dir, "svg_token_embeddings.pt"))
        torch.save(self.svg_output_projection.state_dict(), 
                  os.path.join(output_dir, "svg_output_projection.pt"))
        
        # Save model configuration
        config = {
            "base_model_name": self.base_model.config._name_or_path,
            "svg_vocab_size": self.svg_vocab_size,
            "embed_dim": self.embed_dim
        }
        
        torch.save(config, os.path.join(output_dir, "config.pt"))
    
    @classmethod
    def load_model(cls, model_dir: str, device: Optional[str] = None) -> "OmniSVGModel":
        """
        Load a saved model.
        
        Args:
            model_dir: Directory containing the saved model
            device: Device to load the model to
            
        Returns:
            Loaded OmniSVGModel
        """
        config = torch.load(os.path.join(model_dir, "config.pt"))
        
        # Create model instance
        model = cls(
            base_model_name=os.path.join(model_dir, "base_model"),
            svg_vocab_size=config["svg_vocab_size"],
            device=device
        )
        
        # Load SVG-specific parameters
        model.svg_token_embeddings.load_state_dict(
            torch.load(os.path.join(model_dir, "svg_token_embeddings.pt"))
        )
        model.svg_output_projection.load_state_dict(
            torch.load(os.path.join(model_dir, "svg_output_projection.pt"))
        )
        
        return model