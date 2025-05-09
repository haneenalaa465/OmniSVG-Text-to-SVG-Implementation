"""
Inference Pipeline for OmniSVG.

This module provides utilities for generating SVGs from text prompts
using a trained OmniSVG model.
"""
import os
import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from .model import OmniSVGModel
from .data_processing import SVGProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OmniSVGGenerator:
    """
    Generator for Text-to-SVG using OmniSVG.
    """
    
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        svg_processor: Optional[SVGProcessor] = None
    ):
        """
        Initialize the SVG generator.
        
        Args:
            model_dir: Directory containing the trained model
            device: Device to run inference on
            svg_processor: SVG processor for tokenization and detokenization
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading model from {model_dir}")
        self.model = OmniSVGModel.load_model(model_dir, device=self.device)
        self.model.eval()
        
        # Initialize SVG processor if not provided
        if svg_processor is None:
            base_tokenizer_path = os.path.join(model_dir, "tokenizer")
            if os.path.exists(base_tokenizer_path):
                logger.info(f"Loading SVG processor with tokenizer from {base_tokenizer_path}")
                self.svg_processor = SVGProcessor(base_tokenizer_name=base_tokenizer_path)
            else:
                # Try to use the base model name from model config
                config_path = os.path.join(model_dir, "config.pt")
                if os.path.exists(config_path):
                    config = torch.load(config_path)
                    base_model_name = config.get("base_model_name", "Qwen/Qwen2.5-VL-3B")
                    logger.info(f"Loading SVG processor with tokenizer from {base_model_name}")
                    self.svg_processor = SVGProcessor(base_tokenizer_name=base_model_name)
                else:
                    logger.info("Using default tokenizer")
                    self.svg_processor = SVGProcessor()
        else:
            self.svg_processor = svg_processor
    
    def generate(
        self,
        text: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate SVGs from a text prompt.
        
        Args:
            text: Text prompt
            max_length: Maximum length of generated SVG sequence
            temperature: Generation temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_samples: Number of SVGs to generate
            
        Returns:
            List of generated SVG strings
        """
        logger.info(f"Generating SVG for prompt: '{text}'")
        start_time = time.time()
        
        # Generate SVGs
        with torch.no_grad():
            svgs = self.model.generate_svg(
                text=text,
                svg_processor=self.svg_processor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_samples
            )
        
        generation_time = time.time() - start_time
        logger.info(f"Generated {len(svgs)} SVG(s) in {generation_time:.2f}s")
        
        return svgs
    
    def generate_and_save(
        self,
        text: str,
        output_dir: str,
        prefix: str = "generated",
        **kwargs
    ) -> List[str]:
        """
        Generate SVGs and save them to files.
        
        Args:
            text: Text prompt
            output_dir: Directory to save SVGs to
            prefix: Prefix for SVG filenames
            **kwargs: Generation parameters
            
        Returns:
            List of file paths to saved SVGs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate SVGs
        svgs = self.generate(text, **kwargs)
        
        # Save SVGs to files
        file_paths = []
        for i, svg in enumerate(svgs):
            file_path = os.path.join(output_dir, f"{prefix}_{i+1}.svg")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(svg)
            file_paths.append(file_path)
            logger.info(f"Saved SVG to {file_path}")
        
        return file_paths
    
    @staticmethod
    def optimize_svg(svg_content: str) -> str:
        """
        Optimize SVG for better rendering and smaller file size.
        
        Args:
            svg_content: SVG content to optimize
            
        Returns:
            Optimized SVG content
        """
        # Basic optimization - remove unnecessary whitespace
        svg_content = svg_content.strip()
        
        # TODO: Add more optimizations as needed
        
        return svg_content
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[List[str]]:
        """
        Generate SVGs for multiple prompts.
        
        Args:
            prompts: List of text prompts
            **kwargs: Generation parameters
            
        Returns:
            List of lists of generated SVG strings
        """
        results = []
        
        for prompt in prompts:
            svgs = self.generate(prompt, **kwargs)
            results.append(svgs)
        
        return results


def main():
    """
    Example usage of OmniSVGGenerator.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SVGs from text")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = OmniSVGGenerator(args.model_dir)
    
    # Generate and save SVGs
    generator.generate_and_save(
        text=args.prompt,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()