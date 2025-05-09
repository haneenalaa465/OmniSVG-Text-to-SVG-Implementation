"""
Utility functions for OmniSVG.

This module provides utility functions for various tasks related to
SVG processing, data handling, and model evaluation.
"""
import os
import json
import re
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image

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

def save_json(data: Any, file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file to
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def visualize_svg(svg_content: str, output_path: Optional[str] = None) -> None:
    """
    Visualize an SVG.
    
    Args:
        svg_content: SVG content to visualize
        output_path: Path to save the visualization to (optional)
    """
    try:
        # Use cairosvg if available
        import cairosvg
        import io
        
        png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data))
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        
        if output_path is not None:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        
        plt.show()
    except ImportError:
        # Fallback to matplotlib
        from matplotlib.pyplot import figure
        import matplotlib.image as mpimg
        import tempfile
        
        temp_svg = tempfile.NamedTemporaryFile(suffix='.svg', delete=False)
        temp_svg.write(svg_content.encode('utf-8'))
        temp_svg.close()
        
        temp_png = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_png.close()
        
        # Convert SVG to PNG using Inkscape if available
        try:
            import subprocess
            subprocess.run(['inkscape', '--export-filename', temp_png.name, temp_svg.name], 
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            image = mpimg.imread(temp_png.name)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')
            
            if output_path is not None:
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            
            plt.show()
        except (ImportError, subprocess.SubprocessError):
            print("Could not visualize SVG. Please install cairosvg or Inkscape.")
        
        # Clean up temporary files
        os.unlink(temp_svg.name)
        os.unlink(temp_png.name)

def plot_training_loss(losses: List[float], output_path: Optional[str] = None) -> None:
    """
    Plot training loss.
    
    Args:
        losses: List of loss values
        output_path: Path to save the plot to (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if output_path is not None:
        plt.savefig(output_path)
    
    plt.show()

def validate_svg(svg_content: str) -> bool:
    """
    Validate SVG content.
    
    Args:
        svg_content: SVG content to validate
        
    Returns:
        True if the SVG is valid, False otherwise
    """
    # Basic validation
    if not svg_content.strip().startswith('<svg'):
        return False
    
    if not svg_content.strip().endswith('</svg>'):
        return False
    
    # Check for required SVG attributes
    if not re.search(r'<svg[^>]*xmlns=', svg_content):
        return False
    
    # Check for balanced tags
    stack = []
    tag_pattern = re.compile(r'<(/?)([a-zA-Z][a-zA-Z0-9:_.-]*)([^>]*)>')
    
    for match in tag_pattern.finditer(svg_content):
        closing, tag_name, _ = match.groups()
        
        if closing:
            if not stack or stack[-1] != tag_name:
                return False
            stack.pop()
        elif not tag_name.lower() in ['path', 'rect', 'circle', 'line', 'polyline', 'polygon', 'ellipse']:
            # Only track non-self-closing tags
            if not match.group(0).endswith('/>'):
                stack.append(tag_name)
    
    return len(stack) == 0

def estimate_model_size(model: torch.nn.Module) -> Dict[str, Union[int, float]]:
    """
    Estimate the size of a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'parameters': sum(p.numel() for p in model.parameters()),
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb
    }

def count_tokens(text: str, tokenizer: Any) -> int:
    """
    Count the number of tokens in a text.
    
    Args:
        text: Text to count tokens in
        tokenizer: Tokenizer to use
        
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text))

def optimize_svg_size(svg_content: str) -> str:
    """
    Optimize SVG size by removing unnecessary elements.
    
    Args:
        svg_content: SVG content to optimize
        
    Returns:
        Optimized SVG content
    """
    # Remove comments
    svg_content = re.sub(r'<!--[\s\S]*?-->', '', svg_content)
    
    # Remove unnecessary whitespace
    svg_content = re.sub(r'\s+', ' ', svg_content)
    svg_content = re.sub(r'>\s+<', '><', svg_content)
    
    # Simplify decimal places in coordinates
    def round_coords(match):
        value = float(match.group(0))
        return str(round(value, 2))
    
    svg_content = re.sub(r'-?\d+\.\d+', round_coords, svg_content)
    
    return svg_content