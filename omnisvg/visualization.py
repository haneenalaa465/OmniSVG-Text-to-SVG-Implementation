"""
Visualization utilities for SVG generation
"""

import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from IPython.display import display, HTML, SVG
from typing import List, Dict, Any


def display_svg(svg_text):
    """
    Display SVG in a notebook
    
    Args:
        svg_text: SVG markup text
    """
    display(SVG(svg_text))


def compare_svgs(svgs, titles=None, figsize=(15, 5)):
    """
    Display multiple SVGs side by side for comparison
    
    Args:
        svgs: List of SVG text strings
        titles: Optional list of titles for each SVG
        figsize: Figure size as (width, height)
    """
    if not titles:
        titles = [f"SVG {i+1}" for i in range(len(svgs))]
    
    svg_data = []
    for svg in svgs:
        # Convert SVG to data URL for embedding in HTML
        svg_bytes = svg.encode('utf-8')
        b64 = base64.b64encode(svg_bytes).decode('utf-8')
        svg_data.append(f'data:image/svg+xml;base64,{b64}')
    
    # Create HTML for displaying SVGs in a row
    html = '<div style="display: flex; flex-direction: row; flex-wrap: wrap;">'
    for i, (data, title) in enumerate(zip(svg_data, titles)):
        html += f'''
        <div style="margin: 10px; text-align: center;">
            <img src="{data}" style="height: {figsize[1]*80}px; border: 1px solid #ddd;">
            <p>{title}</p>
        </div>
        '''
    html += '</div>'
    
    display(HTML(html))


def visualize_token_distribution(tokens, figsize=(12, 6)):
    """
    Visualize the distribution of token types in an SVG
    
    Args:
        tokens: List of token IDs from the SVG tokenizer
        figsize: Figure size as (width, height)
    """
    from omnisvg.tokenizer import BASE_ID
    
    # Count command types
    command_counts = {cmd: tokens.count(BASE_ID[cmd]) for cmd in ["M", "L", "C", "A", "Z", "F"]}
    
    # Count coordinate and color tokens
    coord_tokens = sum(1 for t in tokens if len(BASE_ID) <= t < len(BASE_ID) + 40000)
    color_tokens = sum(1 for t in tokens if t >= len(BASE_ID) + 40000)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Command distribution
    cmd_names = list(command_counts.keys())
    cmd_values = list(command_counts.values())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(cmd_names)))
    
    ax1.bar(cmd_names, cmd_values, color=colors)
    ax1.set_title('Command Distribution')
    ax1.set_ylabel('Count')
    
    # Token type distribution
    token_types = ['Commands', 'Coordinates', 'Colors']
    token_counts = [sum(command_counts.values()), coord_tokens, color_tokens]
    
    ax2.pie(token_counts, labels=token_types, autopct='%1.1f%%', 
            colors=plt.cm.tab10(np.linspace(0, 0.3, len(token_types))))
    ax2.set_title('Token Type Distribution')
    
    plt.tight_layout()
    plt.show()


def plot_training_progress(log_history, figsize=(12, 6)):
    """
    Plot training progress from saved logs
    
    Args:
        log_history: Training log history from Trainer
        figsize: Figure size as (width, height)
    """
    # Extract metrics
    train_loss = [x.get('loss') for x in log_history if 'loss' in x]
    eval_loss = [x.get('eval_loss') for x in log_history if 'eval_loss' in x]
    
    steps = list(range(len(train_loss)))
    eval_steps = [step * 100 for step in range(len(eval_loss))]  # Adjust based on eval frequency
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(steps, train_loss, 'b-', label='Training Loss')
    if eval_loss:
        ax.plot(eval_steps, eval_loss, 'r-', label='Validation Loss')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()