"""
Utility functions for SVG processing and validation
"""

import re
import math
from typing import Tuple, Optional

def validate_token_sequence(tokens):
    """Validate that tokens follow valid SVG path grammar rules"""
    from omnisvg.tokenizer import BASE_ID
    
    valid_sequences = {
        BASE_ID["<SOP>"]: [BASE_ID["M"]],  # SOP must be followed by MoveTo
        BASE_ID["M"]: list(range(len(BASE_ID), len(BASE_ID) + 40000)),  # MoveTo followed by coordinate
        BASE_ID["L"]: list(range(len(BASE_ID), len(BASE_ID) + 40000)),  # LineTo followed by coordinate
        BASE_ID["C"]: list(range(len(BASE_ID), len(BASE_ID) + 40000)),  # CubicBezier followed by coordinate
        BASE_ID["A"]: list(range(len(BASE_ID), len(BASE_ID) + 40000)),  # Arc followed by coordinate
        # Z can be followed by either F (Fill) or another command
        BASE_ID["Z"]: [BASE_ID["F"], BASE_ID["M"], BASE_ID["L"], BASE_ID["C"], BASE_ID["A"]],
        BASE_ID["F"]: [BASE_ID["<COLOR>"]],  # Fill followed by color token
    }
    
    state_counts = {
        BASE_ID["M"]: 1,  # MoveTo requires 1 coordinate pair
        BASE_ID["L"]: 1,  # LineTo requires 1 coordinate pair
        BASE_ID["C"]: 3,  # CubicBezier requires 3 coordinate pairs
        BASE_ID["A"]: 4,  # Arc requires 4 parameters
        BASE_ID["<COLOR>"]: 3  # Color requires 3 values (R,G,B)
    }
    
    errors = []
    command_stack = []
    expected_params = 0
    param_count = 0
    
    for i, token in enumerate(tokens):
        if token in [BASE_ID["M"], BASE_ID["L"], BASE_ID["C"], BASE_ID["A"], BASE_ID["Z"], BASE_ID["F"]]:
            # Starting a new command
            if expected_params > 0 and param_count < expected_params:
                errors.append(f"Incomplete parameters for command at position {i-param_count-1}")
            
            command_stack.append(token)
            expected_params = state_counts.get(token, 0)
            param_count = 0
            
        elif token == BASE_ID["<COLOR>"]:
            if not command_stack or command_stack[-1] != BASE_ID["F"]:
                errors.append(f"COLOR token without preceding Fill command at position {i}")
            expected_params = 3  # R,G,B values
            param_count = 0
            
        elif token == BASE_ID["<EOS>"]:
            # End of sequence
            if expected_params > 0 and param_count < expected_params:
                errors.append(f"Incomplete parameters at end of sequence")
                
        else:
            # This should be a parameter
            param_count += 1
    
    return errors

def post_process_svg(svg_text):
    """Fix common issues in generated SVG markup"""
    # Ensure all paths have both 'd' and 'fill' attributes
    path_pattern = re.compile(r'<path([^>]*)>')
    
    def fix_path(match):
        attrs = match.group(1)
        
        # Ensure 'd' attribute exists
        if 'd=' not in attrs:
            attrs += ' d="M0,0"'  # Add a minimal path
            
        # Ensure 'fill' attribute exists
        if 'fill=' not in attrs:
            attrs += ' fill="#000000"'  # Add default black fill
            
        return f'<path{attrs}>'
    
    svg_text = path_pattern.sub(fix_path, svg_text)
    
    # Ensure SVG has proper namespace
    if '<svg' in svg_text and 'xmlns=' not in svg_text:
        svg_text = svg_text.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
    
    # Ensure viewBox is present
    if 'viewBox=' not in svg_text:
        svg_text = svg_text.replace('<svg', '<svg viewBox="0 0 200 200"')
    
    # Fix unclosed tags
    if '<svg' in svg_text and '</svg>' not in svg_text:
        svg_text += '</svg>'
        
    # Fix malformed or missing path data
    svg_text = re.sub(r'd="\s*"', 'd="M0,0"', svg_text)
    svg_text = re.sub(r'd=""', 'd="M0,0"', svg_text)
    
    return svg_text

def validate_svg(svg_text) -> Tuple[bool, Optional[str]]:
    """Check if SVG is valid by attempting to parse it with a standard parser"""
    try:
        from lxml import etree
        parser = etree.XMLParser()
        etree.fromstring(svg_text.encode('utf-8'), parser)
        return True, None
    except Exception as e:
        return False, str(e)
    
def is_valid_svg_content(svg_text) -> bool:
    """Check if SVG has meaningful content (paths, shapes, etc.)"""
    min_path_count = 1
    min_command_count = 3  # At least a move and a couple of other commands
    
    # Check for path elements
    path_count = svg_text.count('<path')
    
    # Check for path commands
    command_count = sum(1 for cmd in "MLCAZ" if cmd in svg_text)
    
    # Check for minimal size
    is_valid_size = len(svg_text) > 100
    
    return (path_count >= min_path_count and 
            command_count >= min_command_count and 
            is_valid_size)

def create_fallback_svg(prompt):
    """Create a simple SVG based on the text prompt when generation fails"""
    prompt = prompt.lower()

    # Default values
    shape = "circle"
    color = "#ff0000"  # Red
    cx, cy = 100, 100  # Center of viewBox
    size = 50          # Default size

    # Determine shape from prompt
    if "circle" in prompt:
        shape = "circle"
    elif "square" in prompt or "rectangle" in prompt or "rect" in prompt:
        shape = "rect"
    elif "triangle" in prompt:
        shape = "polygon"
    elif "star" in prompt:
        shape = "star"
    elif "heart" in prompt:
        shape = "heart"

    # Determine color from prompt
    if "red" in prompt:
        color = "#ff0000"
    elif "blue" in prompt:
        color = "#0000ff"
    elif "green" in prompt:
        color = "#00ff00"
    elif "yellow" in prompt:
        color = "#ffff00"
    elif "orange" in prompt:
        color = "#ffa500"
    elif "purple" in prompt:
        color = "#800080"
    elif "pink" in prompt:
        color = "#ffc0cb"
    elif "black" in prompt:
        color = "#000000"
    elif "white" in prompt:
        color = "#ffffff"

    # Generate SVG based on shape
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">'

    if shape == "circle":
        svg += f'<circle cx="{cx}" cy="{cy}" r="{size}" fill="{color}"/>'
    elif shape == "rect":
        svg += f'<rect x="{cx-size}" y="{cy-size}" width="{size*2}" height="{size*2}" fill="{color}"/>'
    elif shape == "polygon":
        svg += f'<polygon points="{cx},{cy-size} {cx+size},{cy+size} {cx-size},{cy+size}" fill="{color}"/>'
    elif shape == "star":
        points = []
        for i in range(10):
            angle = math.pi * i / 5
            r = size if i % 2 == 0 else size / 2
            x = cx + r * math.sin(angle)
            y = cy - r * math.cos(angle)
            points.append(f"{x},{y}")
        svg += f'<polygon points="{" ".join(points)}" fill="{color}"/>'
    elif shape == "heart":
        svg += f'''<path d="M {cx} {cy+size/2}
                   C {cx-size} {cy-size/2} {cx-size*1.5} {cy-size} {cx} {cy-size/3}
                   C {cx+size*1.5} {cy-size} {cx+size} {cy-size/2} {cx} {cy+size/2} Z"
                   fill="{color}"/>'''

    svg += '</svg>'
    return svg