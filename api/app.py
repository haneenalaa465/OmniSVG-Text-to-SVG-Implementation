"""
Flask API for OmniSVG.

This module implements a Flask web application that serves predictions
from the OmniSVG model.
"""
import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to import OmniSVG modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, Response, render_template
from src.inference import OmniSVGGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize generator
MODEL_DIR = os.getenv("MODEL_DIR", "models/omnisvg/final_model")
generator = None


def load_model():
    """
    Load the OmniSVG model.
    """
    global generator
    
    if generator is None:
        logger.info(f"Loading model from {MODEL_DIR}")
        generator = OmniSVGGenerator(MODEL_DIR)
        logger.info("Model loaded successfully")


@app.route("/")
def index():
    """
    Render the index page.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Generate SVG from text prompt.
    
    Request JSON format:
    {
        "prompt": "Text prompt",
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "num_samples": 1
    }
    
    Response JSON format:
    {
        "svg": ["SVG content"],
        "generation_time": 1.23
    }
    """
    # Load model if not already loaded
    load_model()
    
    # Get request data
    if request.is_json:
        data = request.get_json()
    else:
        # Try to parse form data
        data = request.form.to_dict()
    
    # Extract parameters
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.7))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.95))
    num_samples = int(data.get("num_samples", 1))
    
    if not prompt:
        return jsonify({"error": "Missing prompt parameter"}), 400
    
    # Generate SVG
    start_time = time.time()
    try:
        svgs = generator.generate(
            text=prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_samples=num_samples
        )
        generation_time = time.time() - start_time
        
        return jsonify({
            "svg": svgs,
            "generation_time": generation_time
        })
    except Exception as e:
        logger.error(f"Error generating SVG: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/svg", methods=["POST"])
def get_svg():
    """
    Generate SVG and return it directly with proper content type.
    
    Request JSON format:
    {
        "prompt": "Text prompt",
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95
    }
    
    Response: SVG content with content type "image/svg+xml"
    """
    # Load model if not already loaded
    load_model()
    
    # Get request data
    if request.is_json:
        data = request.get_json()
    else:
        # Try to parse form data
        data = request.form.to_dict()
    
    # Extract parameters
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.7))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.95))
    
    if not prompt:
        return jsonify({"error": "Missing prompt parameter"}), 400
    
    # Generate SVG
    try:
        svgs = generator.generate(
            text=prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_samples=1
        )
        
        # Return the first generated SVG
        svg_content = svgs[0]
        return Response(svg_content, content_type="image/svg+xml")
    except Exception as e:
        logger.error(f"Error generating SVG: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """
    Health check endpoint.
    """
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="OmniSVG API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the API server on")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Path to model directory")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Set model directory
    MODEL_DIR = args.model_dir
    
    # Load model before starting server
    load_model()
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug)