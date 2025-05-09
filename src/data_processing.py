"""
SVG Tokenization and Processing Module.

This module provides utilities for tokenizing SVG commands to a format 
compatible with the OmniSVG model and converting them back to SVG format.
Following the SVG parameterization approach from the OmniSVG paper.
"""
import os
import re
import math
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Union, Any
import torch
from transformers import AutoTokenizer
import picosvg.svg as picosvg

# Define SVG command types
SVG_COMMANDS = {
    'M': 'MoveTo',
    'L': 'LineTo',
    'C': 'CubicBezier', 
    'A': 'EllipticalArc',
    'Z': 'ClosePath',
    'F': 'Fill'
}

# Special tokens for SVG sequence
SPECIAL_TOKENS = {
    'SOP': '<SOP>',  # Start of Path
    'EOS': '<EOS>'   # End of SVG
}

class SVGProcessor:
    """
    Process and tokenize SVG for the OmniSVG model.
    
    This class handles SVG simplification, tokenization, and detokenization
    following the approach described in the OmniSVG paper.
    """
    
    def __init__(self, 
                 base_tokenizer_name: str = "Qwen/Qwen2.5-VL-3B", 
                 max_svg_len: int = 8192,
                 viewbox_size: int = 200):
        """
        Initialize SVG processor.
        
        Args:
            base_tokenizer_name: Pretrained tokenizer to use as base
            max_svg_len: Maximum SVG token length
            viewbox_size: Size of SVG viewbox (default 200x200)
        """
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        self.max_svg_len = max_svg_len
        self.viewbox_size = viewbox_size
        self.svg_vocab_size = 40000  # As mentioned in the paper
        
        # Command token mappings
        self.cmd_to_token = {cmd: i for i, cmd in enumerate(SVG_COMMANDS.keys())}
        self.token_to_cmd = {i: cmd for i, cmd in enumerate(SVG_COMMANDS.keys())}
        
        # Special token mappings
        self.special_to_token = {token: i + len(SVG_COMMANDS) for i, token in enumerate(SPECIAL_TOKENS.values())}
        self.token_to_special = {i + len(SVG_COMMANDS): token for i, token in enumerate(SPECIAL_TOKENS.values())}
        
        # Fill color token mappings (will be populated as colors are encountered)
        self.color_to_token = {}
        self.token_to_color = {}
        self.next_color_token_id = len(SVG_COMMANDS) + len(SPECIAL_TOKENS)
        
        # Coordinate token mappings (will be created dynamically)
        self.coord_to_token = {}
        self.token_to_coord = {}
        self.next_coord_token_id = self.next_color_token_id + 1000  # Reserve space for colors
    
    def simplify_svg(self, svg_content: str) -> str:
        """
        Simplify SVG using picosvg to standardize the format.
        
        Args:
            svg_content: Raw SVG content
        
        Returns:
            Simplified SVG content with only atomic commands
        """
        try:
            # Use picosvg to simplify the SVG
            svg = picosvg.SVG.fromstring(svg_content)
            svg = svg.topicosvg()
            
            # Adjust viewBox to standard size
            svg.viewBox = (0, 0, self.viewbox_size, self.viewbox_size)
            
            return svg.tostring()
        except Exception as e:
            print(f"Error simplifying SVG: {e}")
            return svg_content
    
    def extract_paths(self, svg_content: str) -> List[Dict[str, Any]]:
        """
        Extract path elements from SVG content.
        
        Args:
            svg_content: SVG content to extract paths from
        
        Returns:
            List of dictionaries containing path data and attributes
        """
        try:
            root = ET.fromstring(svg_content)
            paths = []
            
            # Find all path elements
            for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
                path_data = {
                    'd': path.get('d', ''),
                    'fill': path.get('fill', '#000000'),
                    'stroke': path.get('stroke', 'none'),
                    'stroke-width': path.get('stroke-width', '1')
                }
                paths.append(path_data)
            
            return paths
        except Exception as e:
            print(f"Error extracting paths: {e}")
            return []
    
    def parse_path_commands(self, path_data: str) -> List[Dict[str, Any]]:
        """
        Parse SVG path data into commands and coordinates.
        
        Args:
            path_data: SVG path data string
        
        Returns:
            List of command dictionaries
        """
        # Regular expression to match SVG path commands
        command_pattern = re.compile(r'([MLCAZ])([^MLCAZ]*)', re.IGNORECASE)
        matches = command_pattern.findall(path_data)
        
        commands = []
        for cmd, params in matches:
            cmd = cmd.upper()  # Normalize command to uppercase
            if cmd not in SVG_COMMANDS:
                continue
                
            # Parse parameters based on command type
            if cmd == 'Z':
                # ClosePath has no parameters
                commands.append({'type': cmd, 'params': []})
            else:
                # Extract numeric parameters
                params = [float(p) for p in re.findall(r'-?\d+\.?\d*', params)]
                
                # Group parameters by command type
                if cmd == 'M' or cmd == 'L':
                    # MoveTo and LineTo: (x, y)
                    for i in range(0, len(params), 2):
                        if i+1 < len(params):
                            commands.append({
                                'type': cmd, 
                                'params': [params[i], params[i+1]]
                            })
                elif cmd == 'C':
                    # CubicBezier: (x1, y1, x2, y2, x, y)
                    for i in range(0, len(params), 6):
                        if i+5 < len(params):
                            commands.append({
                                'type': cmd,
                                'params': [
                                    params[i], params[i+1],    # First control point
                                    params[i+2], params[i+3],  # Second control point
                                    params[i+4], params[i+5]   # End point
                                ]
                            })
                elif cmd == 'A':
                    # EllipticalArc: (rx, ry, x-axis-rotation, large-arc-flag, sweep-flag, x, y)
                    for i in range(0, len(params), 7):
                        if i+6 < len(params):
                            commands.append({
                                'type': cmd,
                                'params': [
                                    params[i], params[i+1],    # Radii
                                    params[i+2],               # x-axis-rotation
                                    params[i+3], params[i+4],  # flags
                                    params[i+5], params[i+6]   # End point
                                ]
                            })
        
        return commands
    
    def svg_to_commands(self, svg_content: str) -> List[Dict[str, Any]]:
        """
        Convert SVG content to a list of commands.
        
        Args:
            svg_content: SVG content
        
        Returns:
            List of command dictionaries with path information
        """
        simplified_svg = self.simplify_svg(svg_content)
        paths = self.extract_paths(simplified_svg)
        
        all_commands = []
        for path in paths:
            # Add start of path token
            all_commands.append({'type': 'SOP', 'params': []})
            
            # Add path commands
            path_commands = self.parse_path_commands(path['d'])
            all_commands.extend(path_commands)
            
            # Add fill color as a command
            fill_color = path.get('fill', '#000000')
            all_commands.append({'type': 'F', 'params': [fill_color]})
        
        # Add end of SVG token
        all_commands.append({'type': 'EOS', 'params': []})
        
        return all_commands
    
    def _get_or_create_coord_token(self, x: float, y: float) -> int:
        """
        Map coordinate to token ID or create a new token.
        
        Args:
            x: X coordinate
            y: Y coordinate
        
        Returns:
            Token ID for the coordinate
        """
        # Using the mapping function from the paper: <x,y> → x × w + y
        coord_val = int(x * self.viewbox_size + y)
        
        if coord_val not in self.coord_to_token:
            token_id = self.next_coord_token_id
            self.coord_to_token[coord_val] = token_id
            self.token_to_coord[token_id] = (x, y)
            self.next_coord_token_id += 1
        
        return self.coord_to_token[coord_val]
    
    def _get_or_create_color_token(self, color: str) -> int:
        """
        Map color to token ID or create a new token.
        
        Args:
            color: Color in hex format
        
        Returns:
            Token ID for the color
        """
        if color not in self.color_to_token:
            token_id = self.next_color_token_id
            self.color_to_token[color] = token_id
            self.token_to_color[token_id] = color
            self.next_color_token_id += 1
        
        return self.color_to_token[color]
    
    def commands_to_tokens(self, commands: List[Dict[str, Any]]) -> List[int]:
        """
        Convert SVG commands to token IDs.
        
        Args:
            commands: List of command dictionaries
        
        Returns:
            List of token IDs
        """
        tokens = []
        
        for cmd in commands:
            cmd_type = cmd['type']
            params = cmd['params']
            
            # Add command token
            if cmd_type in SPECIAL_TOKENS.values():
                # Special tokens (SOP, EOS)
                tokens.append(self.special_to_token[cmd_type])
            elif cmd_type in SVG_COMMANDS:
                # Regular command tokens
                tokens.append(self.cmd_to_token[cmd_type])
                
                # Add parameter tokens based on command type
                if cmd_type == 'F':
                    # Fill color token
                    tokens.append(self._get_or_create_color_token(params[0]))
                elif cmd_type == 'M' or cmd_type == 'L':
                    # MoveTo or LineTo: (x, y)
                    tokens.append(self._get_or_create_coord_token(params[0], params[1]))
                elif cmd_type == 'C':
                    # CubicBezier: (x1, y1, x2, y2, x, y)
                    tokens.append(self._get_or_create_coord_token(params[0], params[1]))  # Control point 1
                    tokens.append(self._get_or_create_coord_token(params[2], params[3]))  # Control point 2
                    tokens.append(self._get_or_create_coord_token(params[4], params[5]))  # End point
                elif cmd_type == 'A':
                    # EllipticalArc: (rx, ry, x-axis-rotation, large-arc-flag, sweep-flag, x, y)
                    # We simplify by combining all parameters into a single token
                    arc_params = f"{params[0]},{params[1]},{params[2]},{int(params[3])},{int(params[4])}"
                    arc_token = hash(arc_params) % 1000 + self.next_coord_token_id  # Reserve 1000 tokens for arc params
                    tokens.append(arc_token)
                    tokens.append(self._get_or_create_coord_token(params[5], params[6]))  # End point
        
        return tokens
    
    def tokens_to_commands(self, tokens: List[int]) -> List[Dict[str, Any]]:
        """
        Convert token IDs back to SVG commands.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            List of command dictionaries
        """
        commands = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Handle special tokens
            if token in self.token_to_special:
                cmd_type = self.token_to_special[token]
                commands.append({'type': cmd_type, 'params': []})
                i += 1
                continue
            
            # Handle command tokens
            if token in self.token_to_cmd:
                cmd_type = self.token_to_cmd[token]
                i += 1
                
                # Parse parameters based on command type
                if cmd_type == 'F':
                    # Fill color
                    color_token = tokens[i]
                    color = self.token_to_color.get(color_token, '#000000')
                    commands.append({'type': cmd_type, 'params': [color]})
                    i += 1
                elif cmd_type == 'M' or cmd_type == 'L':
                    # MoveTo or LineTo: (x, y)
                    coord_token = tokens[i]
                    x, y = self.token_to_coord.get(coord_token, (0, 0))
                    commands.append({'type': cmd_type, 'params': [x, y]})
                    i += 1
                elif cmd_type == 'C':
                    # CubicBezier: (x1, y1, x2, y2, x, y)
                    if i + 2 < len(tokens):
                        cp1_token = tokens[i]
                        cp2_token = tokens[i+1]
                        end_token = tokens[i+2]
                        
                        x1, y1 = self.token_to_coord.get(cp1_token, (0, 0))
                        x2, y2 = self.token_to_coord.get(cp2_token, (0, 0))
                        x, y = self.token_to_coord.get(end_token, (0, 0))
                        
                        commands.append({
                            'type': cmd_type,
                            'params': [x1, y1, x2, y2, x, y]
                        })
                        i += 3
                    else:
                        i += 1  # Skip if not enough tokens remain
                elif cmd_type == 'A':
                    # EllipticalArc: (rx, ry, x-axis-rotation, large-arc-flag, sweep-flag, x, y)
                    if i + 1 < len(tokens):
                        # Skip arc parameters token for now (simplified)
                        i += 1
                        
                        end_token = tokens[i]
                        x, y = self.token_to_coord.get(end_token, (0, 0))
                        
                        commands.append({
                            'type': cmd_type,
                            'params': [1, 1, 0, 0, 1, x, y]  # Default values for simplicity
                        })
                        i += 1
                    else:
                        i += 1  # Skip if not enough tokens remain
                elif cmd_type == 'Z':
                    # ClosePath has no parameters
                    commands.append({'type': cmd_type, 'params': []})
            else:
                # Skip unknown tokens
                i += 1
        
        return commands
    
    def commands_to_svg(self, commands: List[Dict[str, Any]]) -> str:
        """
        Convert commands back to SVG content.
        
        Args:
            commands: List of command dictionaries
        
        Returns:
            SVG content
        """
        svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.viewbox_size} {self.viewbox_size}">'
        svg_footer = '</svg>'
        
        paths = []
        current_path = None
        current_path_data = ""
        current_fill = "#000000"
        
        for cmd in commands:
            cmd_type = cmd['type']
            params = cmd['params']
            
            if cmd_type == 'SOP':
                # Start a new path
                if current_path_data:
                    # Save the previous path if it exists
                    paths.append(f'<path d="{current_path_data}" fill="{current_fill}" />')
                
                current_path_data = ""
                current_fill = "#000000"
            elif cmd_type == 'EOS':
                # End of SVG, add the last path if it exists
                if current_path_data:
                    paths.append(f'<path d="{current_path_data}" fill="{current_fill}" />')
            elif cmd_type == 'F':
                # Set fill color for the current path
                current_fill = params[0]
            elif cmd_type == 'M':
                # MoveTo
                x, y = params
                current_path_data += f"M{x},{y} "
            elif cmd_type == 'L':
                # LineTo
                x, y = params
                current_path_data += f"L{x},{y} "
            elif cmd_type == 'C':
                # CubicBezier
                x1, y1, x2, y2, x, y = params
                current_path_data += f"C{x1},{y1} {x2},{y2} {x},{y} "
            elif cmd_type == 'A':
                # EllipticalArc
                rx, ry, x_axis_rot, large_arc, sweep, x, y = params
                current_path_data += f"A{rx},{ry} {x_axis_rot} {int(large_arc)},{int(sweep)} {x},{y} "
            elif cmd_type == 'Z':
                # ClosePath
                current_path_data += "Z "
        
        # Combine all paths and return the full SVG
        paths_str = "\n".join(paths)
        return f"{svg_header}\n{paths_str}\n{svg_footer}"
    
    def svg_to_tokens(self, svg_content: str) -> List[int]:
        """
        Convert SVG content to tokens.
        
        Args:
            svg_content: SVG content
        
        Returns:
            List of token IDs
        """
        commands = self.svg_to_commands(svg_content)
        tokens = self.commands_to_tokens(commands)
        return tokens
    
    def tokens_to_svg(self, tokens: List[int]) -> str:
        """
        Convert tokens back to SVG content.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            SVG content
        """
        commands = self.tokens_to_commands(tokens)
        svg_content = self.commands_to_svg(commands)
        return svg_content
    
    def encode_text_for_svg_generation(self, text: str) -> torch.Tensor:
        """
        Encode text for SVG generation.
        
        Args:
            text: Text prompt to encode
        
        Returns:
            Tensor of token IDs
        """
        # Encode the text using the base tokenizer
        encoded = self.base_tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
        return encoded
    
    def prepare_training_data(self, text: str, svg_content: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data for a text-SVG pair.
        
        Args:
            text: Text description
            svg_content: SVG content
        
        Returns:
            Tuple of (input_ids, labels)
        """
        # Encode text
        text_ids = self.encode_text_for_svg_generation(text)[0]
        
        # Encode SVG
        svg_tokens = self.svg_to_tokens(svg_content)
        svg_ids = torch.tensor(svg_tokens, dtype=torch.long)
        
        # Combine text and SVG tokens
        input_ids = torch.cat([text_ids, svg_ids])
        
        # Labels are -100 for text tokens (not included in loss) and SVG tokens
        labels = torch.full_like(input_ids, -100)
        labels[-len(svg_ids):] = svg_ids
        
        # Truncate if too long
        if len(input_ids) > self.max_svg_len:
            input_ids = input_ids[:self.max_svg_len]
            labels = labels[:self.max_svg_len]
        
        return input_ids, labels