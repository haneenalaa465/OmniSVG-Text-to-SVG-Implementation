"""
SVG Tokenizer for converting between SVG markup and token sequences.
"""

import re
import torch
import xml.etree.ElementTree as ET
from svgpathtools import parse_path

from omnisvg.config import CANVAS_SIZE, SPECIAL_TOKENS

# Create base token ID mapping
BASE_ID = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}

class SVGTokenizer:
    def __init__(self, canvas_size=CANVAS_SIZE):
        """
        Initialize the SVG tokenizer.
        
        Args:
            canvas_size: Size of the SVG canvas (both width and height)
        """
        self.canvas_size = canvas_size
        self.special_tokens = SPECIAL_TOKENS
        
        # Vocabulary size: special tokens + coordinate tokens (canvas^2) + color tokens (256^3 RGB)
        self.vocab_size = len(SPECIAL_TOKENS) + canvas_size**2 + 16777216  # 16777216 = 256^3 for RGB colors
        
        # For tracking command parameters during decoding
        self._bezier_points = []
        self._arc_params = []

    def coord_token(self, x, y):
        """Convert x,y coordinates to a token ID"""
        x = max(0, min(int(round(x)), self.canvas_size-1))
        y = max(0, min(int(round(y)), self.canvas_size-1))
        return len(BASE_ID) + (x * self.canvas_size) + y

    def color_token(self, r, g, b):
        """Convert RGB components to token IDs"""
        color_base = len(BASE_ID) + self.canvas_size**2
        r_token = color_base + r
        g_token = color_base + 256 + g
        b_token = color_base + 512 + b
        return [r_token, g_token, b_token]

    def hex_to_rgb(self, hex_color):
        """Convert hex color string to RGB tuple with fallbacks for special cases"""
        # Handle special color keywords
        if not hex_color or hex_color == "none":
            return (0, 0, 0)  # Default for no color

        if hex_color == "currentColor" or hex_color == "transparent":
            return (0, 0, 0)  # Use black for currentColor or transparent

        try:
            if hex_color.startswith("#"):
                hex_color = hex_color[1:]

            if len(hex_color) == 3:
                # Short form #RGB
                r = int(hex_color[0] + hex_color[0], 16)
                g = int(hex_color[1] + hex_color[1], 16)
                b = int(hex_color[2] + hex_color[2], 16)
            else:
                # Normal form #RRGGBB
                r = int(hex_color[0:2], 16) if len(hex_color) >= 2 else 0
                g = int(hex_color[2:4], 16) if len(hex_color) >= 4 else 0
                b = int(hex_color[4:6], 16) if len(hex_color) >= 6 else 0

            return (r, g, b)
        except ValueError:
            # Handle named colors
            color_map = {
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "yellow": (255, 255, 0),
                "cyan": (0, 255, 255),
                "magenta": (255, 0, 255),
                "gray": (128, 128, 128),
                "orange": (255, 165, 0),
                "purple": (128, 0, 128)
            }

            if hex_color.lower() in color_map:
                return color_map[hex_color.lower()]

            print(f"Invalid color format: {hex_color}, defaulting to black")
            return (0, 0, 0)

    def to_atomic(self, path_str):
        """
        Convert an SVG path string (from 'd' attribute) into atomic commands + coordinates as strings.
        """
        atomic_tokens = []
        token_re = re.compile(r"([MmLlHhVvCcSsQqTtAaZzZ])|([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
        tokens = token_re.findall(path_str)

        current_command = None
        current_args = []

        for cmd, num in tokens:
            if cmd:
                if current_command:
                    atomic_tokens.append((current_command, current_args))
                current_command = cmd.upper()  # Normalize to uppercase
                current_args = []
            elif num:
                current_args.append(float(num))

        if current_command:
            atomic_tokens.append((current_command, current_args))

        return atomic_tokens

    def encode(self, svg_text):
        """Convert SVG text to token IDs with error handling"""
        tokens = []
        try:
            # Parse the SVG XML directly
            root = ET.fromstring(svg_text)

            # Find all path elements
            paths = []
            # Try with namespace
            try:
                namespaces = {'svg': 'http://www.w3.org/2000/svg'}
                paths = root.findall('.//svg:path', namespaces)
            except:
                pass

            # If no paths found, try without namespace
            if not paths:
                paths = root.findall('.//path')

            for path in paths:
                tokens.append(BASE_ID["<SOP>"])

                # Get the path data
                path_d = path.get('d', '')

                if path_d:
                    try:
                        commands = parse_path(path_d)

                        for cmd in commands:
                            cmd_type = cmd.__class__.__name__

                            if cmd_type == "Move":
                                tokens.append(BASE_ID["M"])
                                tokens.append(self.coord_token(cmd.end.real, cmd.end.imag))

                            elif cmd_type == "Line":
                                tokens.append(BASE_ID["L"])
                                tokens.append(self.coord_token(cmd.end.real, cmd.end.imag))

                            elif cmd_type == "CubicBezier":
                                tokens.append(BASE_ID["C"])
                                tokens.append(self.coord_token(cmd.control1.real, cmd.control1.imag))
                                tokens.append(self.coord_token(cmd.control2.real, cmd.control2.imag))
                                tokens.append(self.coord_token(cmd.end.real, cmd.end.imag))

                            elif cmd_type == "Arc":
                                tokens.append(BASE_ID["A"])
                                tokens.append(self.coord_token(cmd.radius.real, cmd.radius.imag))
                                tokens.append(self.coord_token(cmd.rotation, 1 if cmd.large_arc else 0))
                                tokens.append(self.coord_token(1 if cmd.sweep else 0, 0))
                                tokens.append(self.coord_token(cmd.end.real, cmd.end.imag))

                            elif cmd_type == "Close":
                                tokens.append(BASE_ID["Z"])

                    except Exception as e:
                        print(f"Error parsing path commands: {e}")

                # Process fill color
                tokens.append(BASE_ID["F"])
                tokens.append(BASE_ID["<COLOR>"])

                # Get fill color
                fill = path.get('fill', '#000000')
                if fill.startswith("url("):
                    # For gradient or pattern references, default to black
                    r, g, b = 0, 0, 0
                else:
                    r, g, b = self.hex_to_rgb(fill)
                    
                color_tokens = self.color_token(r, g, b)
                tokens.extend(color_tokens)

            tokens.append(BASE_ID["<EOS>"])

            # If no paths were found, return a minimal token sequence
            if len(tokens) <= 1:
                tokens = [BASE_ID["<SOP>"], BASE_ID["<EOS>"]]

        except Exception as e:
            print(f"Error encoding SVG: {e}")
            # Fallback to empty SVG
            tokens = [BASE_ID["<SOP>"], BASE_ID["<EOS>"]]

        return tokens

    def _decode_coord(self, coord_token):
        """Convert a coordinate token back to x,y values with error checking"""
        coord_value = coord_token - len(BASE_ID)
        if coord_value < 0 or coord_value >= self.canvas_size**2:
            return 0, 0  # Default for invalid token
        x = coord_value // self.canvas_size
        y = coord_value % self.canvas_size
        return x, y

    def decode(self, token_ids):
        """Convert token IDs back to SVG text"""
        svg_paths = []
        current_commands = []
        
        # State tracking
        expecting_coord = False
        coord_count = 0
        expected_coords = 0
        last_command = None
        
        i = 0
        while i < len(token_ids):
            token = token_ids[i]
            
            if token == BASE_ID["<SOP>"]:
                # Start a new path
                if current_commands:
                    # If we have commands but no path created, add them with default fill
                    svg_paths.append(f'<path d="{" ".join(current_commands)}" fill="#000000"/>')
                    current_commands = []
                
            elif token == BASE_ID["<EOS>"]:
                # End the sequence, add any remaining path
                if current_commands:
                    svg_paths.append(f'<path d="{" ".join(current_commands)}" fill="#000000"/>')
                break
                
            elif token == BASE_ID["M"]:
                last_command = "M"
                expecting_coord = True
                expected_coords = 1
                coord_count = 0
                
            elif token == BASE_ID["L"]:
                last_command = "L"
                expecting_coord = True
                expected_coords = 1
                coord_count = 0
                
            elif token == BASE_ID["C"]:
                last_command = "C"
                expecting_coord = True
                expected_coords = 3  # Control point 1, control point 2, end point
                coord_count = 0
                self._bezier_points = []
                
            elif token == BASE_ID["A"]:
                last_command = "A"
                expecting_coord = True
                expected_coords = 4  # rx, ry, x-axis-rotation, large-arc-flag, sweep-flag, end-x, end-y
                coord_count = 0
                self._arc_params = []
                
            elif token == BASE_ID["Z"]:
                current_commands.append("Z")
                expecting_coord = False
                
            elif token == BASE_ID["F"]:
                # Fill command, expect color token next
                i += 1
                if i < len(token_ids) and token_ids[i] == BASE_ID["<COLOR>"]:
                    i += 1
                    if i + 2 < len(token_ids):
                        # Extract RGB values
                        r_token = token_ids[i]
                        i += 1
                        g_token = token_ids[i]
                        i += 1
                        b_token = token_ids[i]
                        
                        # Calculate RGB values
                        color_base = len(BASE_ID) + self.canvas_size**2
                        r = max(0, min(255, r_token - color_base if r_token >= color_base else 0))
                        g = max(0, min(255, g_token - (color_base + 256) if g_token >= color_base + 256 else 0))
                        b = max(0, min(255, b_token - (color_base + 512) if b_token >= color_base + 512 else 0))
                        
                        # Create hex color
                        color = f"#{r:02x}{g:02x}{b:02x}"
                        if current_commands:
                            svg_paths.append(f'<path d="{" ".join(current_commands)}" fill="{color}"/>')
                            current_commands = []
                    else:
                        # Missing color components, use default black
                        if current_commands:
                            svg_paths.append(f'<path d="{" ".join(current_commands)}" fill="#000000"/>')
                            current_commands = []
                else:
                    # Missing COLOR token, use default black
                    if current_commands:
                        svg_paths.append(f'<path d="{" ".join(current_commands)}" fill="#000000"/>')
                        current_commands = []
                    i -= 1  # Back up to process the token after F
                    
            elif expecting_coord and coord_count < expected_coords:
                # This should be a coordinate
                if len(BASE_ID) <= token < len(BASE_ID) + self.canvas_size**2:
                    x, y = self._decode_coord(token)
                    
                    if last_command == "M":
                        current_commands.append(f"M{x},{y}")
                    elif last_command == "L":
                        current_commands.append(f"L{x},{y}")
                    elif last_command == "C":
                        if coord_count == 0:
                            # First control point
                            self._bezier_points = [(x, y)]
                        elif coord_count == 1:
                            # Second control point
                            self._bezier_points.append((x, y))
                        elif coord_count == 2:
                            # End point, create the full command
                            x1, y1 = self._bezier_points[0]
                            x2, y2 = self._bezier_points[1]
                            current_commands.append(f"C{x1},{y1} {x2},{y2} {x},{y}")
                            self._bezier_points = []
                    elif last_command == "A":
                        if coord_count == 0:
                            # rx, ry
                            self._arc_params = [(x, y)]
                        elif coord_count == 1:
                            # rotation, large_arc
                            self._arc_params.append((x, y))
                        elif coord_count == 2:
                            # sweep, placeholder
                            self._arc_params.append((x, y))
                        elif coord_count == 3:
                            # End point, create the full command
                            rx, ry = self._arc_params[0]
                            rotation, large_arc = self._arc_params[1]
                            sweep, _ = self._arc_params[2]
                            current_commands.append(f"A{rx},{ry} {rotation} {large_arc},{sweep} {x},{y}")
                            self._arc_params = []
                    
                    coord_count += 1
                    
                    if coord_count >= expected_coords:
                        expecting_coord = False
                else:
                    # Invalid coordinate token, use default
                    if last_command in ["M", "L"]:
                        current_commands.append(f"{last_command}0,0")
                        expecting_coord = False
                    elif last_command == "C":
                        if coord_count == 0:
                            self._bezier_points = [(0, 0)]
                        elif coord_count == 1:
                            self._bezier_points.append((0, 0))
                        elif coord_count == 2:
                            x1, y1 = self._bezier_points[0]
                            x2, y2 = self._bezier_points[1]
                            current_commands.append(f"C{x1},{y1} {x2},{y2} 0,0")
                            self._bezier_points = []
                        coord_count += 1
                    elif last_command == "A":
                        if coord_count == 0:
                            self._arc_params = [(0, 0)]
                        elif coord_count == 1:
                            self._arc_params.append((0, 0))
                        elif coord_count == 2:
                            self._arc_params.append((0, 0))
                        elif coord_count == 3:
                            rx, ry = self._arc_params[0]
                            rotation, large_arc = self._arc_params[1]
                            sweep, _ = self._arc_params[2]
                            current_commands.append(f"A{rx},{ry} {rotation} {large_arc},{sweep} 0,0")
                            self._arc_params = []
                        coord_count += 1
                    
                    if coord_count >= expected_coords:
                        expecting_coord = False
            
            i += 1
        
        # Create the final SVG
        viewbox = f"0 0 {self.canvas_size} {self.canvas_size}"
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" width="{self.canvas_size}" height="{self.canvas_size}">' + "".join(svg_paths) + '</svg>'
        
        # Apply post-processing to ensure valid SVG
        from omnisvg.utils import post_process_svg
        return post_process_svg(svg)

    def constrained_next_token(self, prev_tokens, logits):
        """Apply constraints to next token generation based on SVG grammar"""
        # Get the last token
        last_token = prev_tokens[-1] if prev_tokens else None
        
        # Apply constraints based on the last token
        if last_token in [BASE_ID["M"], BASE_ID["L"]]:
            # Must be followed by a coordinate token
            valid_range = (len(BASE_ID), len(BASE_ID) + self.canvas_size**2)
            logits_mask = torch.zeros_like(logits, dtype=torch.bool)
            logits_mask[valid_range[0]:valid_range[1]] = True
            logits[~logits_mask] = float('-inf')
            
        elif last_token == BASE_ID["C"]:
            # Track how many coordinates we've seen
            c_indices = [i for i, t in enumerate(prev_tokens) if t == BASE_ID["C"]]
            last_c_index = c_indices[-1] if c_indices else -1
            coords_after_c = len(prev_tokens) - last_c_index - 1
            
            if coords_after_c < 3:  # Need 3 coordinate pairs for C
                valid_range = (len(BASE_ID), len(BASE_ID) + self.canvas_size**2)
                logits_mask = torch.zeros_like(logits, dtype=torch.bool)
                logits_mask[valid_range[0]:valid_range[1]] = True
                logits[~logits_mask] = float('-inf')
        
        elif last_token == BASE_ID["F"]:
            # Must be followed by COLOR token
            logits_mask = torch.zeros_like(logits, dtype=torch.bool)
            logits_mask[BASE_ID["<COLOR>"]] = True
            logits[~logits_mask] = float('-inf')
        
        elif last_token == BASE_ID["<COLOR>"]:
            # Must be followed by a color value token
            valid_range = (len(BASE_ID) + self.canvas_size**2, len(BASE_ID) + self.canvas_size**2 + 16777216)
            logits_mask = torch.zeros_like(logits, dtype=torch.bool)
            logits_mask[valid_range[0]:valid_range[1]] = True
            logits[~logits_mask] = float('-inf')
        
        return logits

    def debug_tokens(self, tokens):
        """Visualize token sequence for debugging"""
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Handle special tokens and commands
            if token in BASE_ID.values():
                cmd = [k for k, v in BASE_ID.items() if v == token][0]
                result.append(f"CMD: {cmd}")
            elif token == BASE_ID["<COLOR>"]:
                # Extract color components
                if i + 3 < len(tokens):
                    r = tokens[i+1] - (len(BASE_ID) + self.canvas_size**2)
                    g = tokens[i+2] - (len(BASE_ID) + self.canvas_size**2 + 256)
                    b = tokens[i+3] - (len(BASE_ID) + self.canvas_size**2 + 512)
                    result.append(f"COLOR: #{r:02x}{g:02x}{b:02x}")
                    i += 3  # Skip color components
                else:
                    result.append("COLOR: [INCOMPLETE]")
            elif len(BASE_ID) <= token < len(BASE_ID) + self.canvas_size**2:
                # This is a coordinate token
                x, y = self._decode_coord(token)
                result.append(f"COORD: ({x},{y})")
            else:
                result.append(f"UNKNOWN: {token}")
            
            i += 1
        
        return result

    def inspect_svg_token_pattern(self, tokens):
        """Analyze token patterns for common issues"""
        patterns = {
            "missing_sop": BASE_ID["<SOP>"] not in tokens,
            "missing_eos": BASE_ID["<EOS>"] not in tokens,
            "commands": {cmd: tokens.count(BASE_ID[cmd]) for cmd in ["M", "L", "C", "A", "Z", "F"]},
            "total_coords": sum(1 for t in tokens if len(BASE_ID) <= t < len(BASE_ID) + self.canvas_size**2),
            "color_tokens": sum(1 for t in tokens if t > len(BASE_ID) + self.canvas_size**2)
        }
        
        # Check for balance
        expected_coords = (
            patterns["commands"]["M"] + 
            patterns["commands"]["L"] + 
            patterns["commands"]["C"] * 3 + 
            patterns["commands"]["A"] * 4
        )
        
        patterns["coord_balance"] = {
            "expected": expected_coords,
            "actual": patterns["total_coords"],
            "missing": expected_coords - patterns["total_coords"] if expected_coords > patterns["total_coords"] else 0
        }
        
        patterns["fill_color_balance"] = {
            "fill_commands": patterns["commands"]["F"],
            "color_tokens": patterns["color_tokens"],
            "ratio": patterns["color_tokens"] / patterns["commands"]["F"] if patterns["commands"]["F"] > 0 else 0
        }
        
        return patterns