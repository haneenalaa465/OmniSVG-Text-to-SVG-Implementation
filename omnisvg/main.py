"""
Main CLI interface for OmniSVG
"""

import argparse
import os
import sys
from IPython.display import display

from omnisvg.modeling import load_model, generate_svg_from_text, test_generation, train_model, setup_training
from omnisvg.tokenizer import SVGTokenizer
from omnisvg.visualization import display_svg


def main():
    parser = argparse.ArgumentParser(description="OmniSVG: A Framework for AI-Generated Vector Graphics")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate SVG from text prompt")
    generate_parser.add_argument("prompt", help="Text prompt for SVG generation")
    generate_parser.add_argument("--model", default=None, help="Path to trained model (None for base model)")
    generate_parser.add_argument("--output", default="generated.svg", help="Output SVG file path")
    generate_parser.add_argument("--max-tokens", type=int, default=500, help="Maximum new tokens to generate")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run multiple test generations")
    test_parser.add_argument("--model", default=None, help="Path to trained model (None for base model)")
    test_parser.add_argument("--output-dir", default="./generated", help="Output directory for generated SVGs")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model on SVG dataset")
    train_parser.add_argument("--output-dir", default="./models/omnisvg", help="Output directory for model")
    train_parser.add_argument("--num-samples", type=int, default=8000, help="Number of training samples")
    train_parser.add_argument("--base-model", default=None, help="Path to base model to continue training")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        # Generate a single SVG from text prompt
        print(f"Generating SVG from prompt: '{args.prompt}'")
        
        model, text_tokenizer, svg_tokenizer = load_model(args.model)
        svg = generate_svg_from_text(args.prompt, model, text_tokenizer, svg_tokenizer, args.max_tokens)
        
        # Save to file
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(svg)
        
        print(f"SVG saved to {args.output}")
        
        # Try to display if in a notebook environment
        try:
            display_svg(svg)
        except:
            pass
            
    elif args.command == "test":
        # Run test generations
        print("Running test generations")
        
        model, text_tokenizer, svg_tokenizer = load_model(args.model)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Use default test prompts
        test_prompts = [
            "A simple red circle icon",
            "A blue star icon",
            "A black and white checkered flag", 
            "A green leaf icon",
            "A yellow lightning bolt"
        ]
        
        svgs = test_generation(model, text_tokenizer, svg_tokenizer, test_prompts)
        
        # Save to files
        for i, svg in enumerate(svgs):
            with open(os.path.join(args.output_dir, f"test_{i}.svg"), "w") as f:
                f.write(svg)
                
        print(f"Generated {len(svgs)} SVGs in {args.output_dir}")
        
    elif args.command == "train":
        # Train the model
        print("Setting up training...")
        
        # Load or initialize model
        if args.base_model:
            model, text_tokenizer, svg_tokenizer = load_model(args.base_model)
            model = setup_training(model, text_tokenizer, svg_tokenizer)
        else:
            from omnisvg.modeling.train import init_model_for_training
            model, text_tokenizer, svg_tokenizer = init_model_for_training()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Train the model
        train_model(model, text_tokenizer, svg_tokenizer, args.output_dir, args.num_samples)
        
        print(f"Training complete. Model saved to {args.output_dir}")
        
    else:
        parser.print_help()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())