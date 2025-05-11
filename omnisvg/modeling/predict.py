"""
Inference utilities for OmniSVG models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Optional

from omnisvg.config import BASE_MODEL, GENERATION_CONFIG
from omnisvg.tokenizer import SVGTokenizer
from omnisvg.utils import validate_svg, is_valid_svg_content, post_process_svg, create_fallback_svg


def load_model(model_path=None, device_map="auto", use_4bit=True):
    """
    Load the model for inference
    
    Args:
        model_path: Path to trained LoRA adapter (None to use base model)
        device_map: Device mapping strategy
        use_4bit: Whether to use 4-bit quantization
        
    Returns:
        model: Loaded model
        text_tokenizer: Text tokenizer
        svg_tokenizer: SVG tokenizer
    """
    # Initialize SVG tokenizer
    svg_tokenizer = SVGTokenizer()

    # Load the base model
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map=device_map,
        quantization_config=quant_config,
        trust_remote_code=True
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Add SVG tokens to text tokenizer
    tokenizer.add_tokens([f"<svg_{i}>" for i in range(svg_tokenizer.vocab_size)], special_tokens=False)

    # Ensure the model has the correct embedding size
    model.resize_token_embeddings(len(tokenizer))

    if model_path:
        # Load the trained LoRA weights
        model = PeftModel.from_pretrained(model, model_path)

    return model, tokenizer, svg_tokenizer


def generate_svg_from_text(text_prompt, model, text_tokenizer, svg_tokenizer, max_new_tokens=500):
    """
    Generate an SVG from a text prompt
    
    Args:
        text_prompt: The text description prompt
        model: The LLM model
        text_tokenizer: Text tokenizer
        svg_tokenizer: SVG tokenizer
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        svg: The generated SVG text
    """
    # Prepare the prompt
    prompt = f"""You are a helpful SVG Generation assistant, designed to generate SVG. 
    We provide the text description as input, generate SVG based on the text.
    {text_prompt}
    """
    
    # Tokenize the text prompt
    text_tokens = text_tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Set up generation parameters
    generation_params = {
        **GENERATION_CONFIG,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": text_tokenizer.pad_token_id,
        "eos_token_id": text_tokenizer.eos_token_id,
    }

    # Generate tokens
    with torch.no_grad():
        outputs = model.generate(**text_tokens, **generation_params)

    # Extract the generated text
    generated_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Try to extract SVG from the generated text
    svg_start = generated_text.find("<svg")
    svg_end = generated_text.find("</svg>")

    if svg_start >= 0 and svg_end >= 0:
        svg = generated_text[svg_start:svg_end+6]  # +6 to include "</svg>"
        
        # Validate the SVG
        is_valid, error = validate_svg(svg)
        if is_valid and is_valid_svg_content(svg):
            return svg
        else:
            print(f"Generated SVG was invalid: {error}")
            # Apply post-processing to fix common issues
            svg = post_process_svg(svg)
            is_valid, _ = validate_svg(svg)
            if is_valid:
                return svg

    # If we don't have a valid SVG at this point, create a fallback
    print("Using fallback SVG generation")
    return create_fallback_svg(text_prompt)


def test_generation(model, text_tokenizer, svg_tokenizer, test_prompts=None):
    """
    Test SVG generation with different prompts
    
    Args:
        model: The LLM model
        text_tokenizer: Text tokenizer
        svg_tokenizer: SVG tokenizer
        test_prompts: List of test prompts (uses defaults if None)
        
    Returns:
        generated_svgs: List of generated SVG text
    """
    # Default test prompts if none provided
    if test_prompts is None:
        test_prompts = [
            "A simple red circle icon",
            "A blue star icon",
            "A black and white checkered flag",
            "A green leaf icon",
            "A yellow lightning bolt"
        ]

    generated_svgs = []
    for i, prompt in enumerate(test_prompts):
        print(f"\nGenerating SVG for: {prompt}")
        svg = generate_svg_from_text(prompt, model, text_tokenizer, svg_tokenizer)
        
        # Validate the SVG
        is_valid, error = validate_svg(svg)
        print(f"Is valid SVG? {is_valid}")
        
        if is_valid:
            # Check if it has meaningful content
            has_content = is_valid_svg_content(svg)
            print(f"Has meaningful content? {has_content}")
            
            # Save the SVG
            filename = f"generated_svg_{i}.svg"
            with open(filename, "w") as f:
                f.write(svg)
            print(f"SVG saved to {filename}")
            
            generated_svgs.append(svg)
        else:
            print(f"Error in SVG: {error}")
    
    return generated_svgs