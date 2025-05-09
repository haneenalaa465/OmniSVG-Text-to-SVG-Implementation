"""
Tests for the SVG data processing module.
"""
import os
import sys
import unittest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import SVGProcessor

class TestSVGProcessor(unittest.TestCase):
    """Test the SVGProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.svg_processor = SVGProcessor(
            base_tokenizer_name="Qwen/Qwen2.5-VL-3B",
            max_svg_len=8192,
            viewbox_size=200
        )
        
        # Sample SVG content for testing
        self.sample_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
            <path d="M50,50 L150,50 L150,150 L50,150 Z" fill="#FF0000" />
        </svg>"""
    
    def test_svg_to_commands(self):
        """Test converting SVG to commands."""
        commands = self.svg_processor.svg_to_commands(self.sample_svg)
        
        # Check that we have commands
        self.assertTrue(len(commands) > 0)
        
        # Check that we have the expected special tokens
        self.assertEqual(commands[0]['type'], 'SOP')
        self.assertEqual(commands[-1]['type'], 'EOS')
        
        # Check that we have the expected commands (MoveTo, LineTo, ClosePath, Fill)
        command_types = [cmd['type'] for cmd in commands]
        self.assertIn('M', command_types)
        self.assertIn('L', command_types)
        self.assertIn('Z', command_types)
        self.assertIn('F', command_types)
    
    def test_commands_to_svg(self):
        """Test converting commands back to SVG."""
        commands = self.svg_processor.svg_to_commands(self.sample_svg)
        svg_output = self.svg_processor.commands_to_svg(commands)
        
        # Check that we have an SVG output
        self.assertTrue(svg_output.startswith('<svg'))
        self.assertTrue(svg_output.endswith('</svg>'))
        
        # Check that we have a path element
        self.assertIn('<path', svg_output)
        
        # Check that we have the fill attribute
        self.assertIn('fill="#FF0000"', svg_output)
    
    def test_svg_to_tokens(self):
        """Test converting SVG to tokens."""
        tokens = self.svg_processor.svg_to_tokens(self.sample_svg)
        
        # Check that we have tokens
        self.assertTrue(len(tokens) > 0)
        
        # Check that all tokens are integers
        for token in tokens:
            self.assertIsInstance(token, int)
    
    def test_tokens_to_svg(self):
        """Test converting tokens back to SVG."""
        tokens = self.svg_processor.svg_to_tokens(self.sample_svg)
        svg_output = self.svg_processor.tokens_to_svg(tokens)
        
        # Check that we have an SVG output
        self.assertTrue(svg_output.startswith('<svg'))
        self.assertTrue(svg_output.endswith('</svg>'))
        
        # Check that we have a path element
        self.assertIn('<path', svg_output)
    
    def test_round_trip(self):
        """Test round-trip conversion (SVG -> tokens -> SVG)."""
        tokens = self.svg_processor.svg_to_tokens(self.sample_svg)
        svg_output = self.svg_processor.tokens_to_svg(tokens)
        
        # Convert the output SVG back to tokens
        tokens2 = self.svg_processor.svg_to_tokens(svg_output)
        
        # Check that the tokens are the same
        self.assertEqual(len(tokens), len(tokens2))
        self.assertEqual(tokens, tokens2)

if __name__ == '__main__':
    unittest.main()