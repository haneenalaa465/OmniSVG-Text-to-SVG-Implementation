#!/usr/bin/env python
"""
Quick start script for the OmniSVG API.

This script provides a convenient way to start the Flask API for the OmniSVG model.
It also creates the necessary directories and helps set up the environment.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for the API."""
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Check if templates directory exists
    api_templates_dir = Path("api/templates")
    if not api_templates_dir.exists():
        logger.warning(f"Templates directory '{api_templates_dir}' not found. Creating...")
        api_templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a basic index.html if missing
        index_html = api_templates_dir / "index.html"
        if not index_html.exists():
            logger.warning(f"Index template '{index_html}' not found. Creating a basic template...")
            with open(index_html, "w") as f:
                f.write("""<!DOCTYPE html>
<html>
<head>
    <title>OmniSVG API</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input, textarea { width: 100%; padding: 8px; }
        button { background-color: #3498db; color: white; border: none; padding: 10px 15px; cursor: pointer; }
        #result { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>OmniSVG: Text-to-SVG Generation</h1>
    <div class="form-group">
        <label for="prompt">Text Description:</label>
        <textarea id="prompt" rows="3" placeholder="Describe the SVG you want to generate..."></textarea>
    </div>
    <button onclick="generateSVG()">Generate SVG</button>
    <div id="result"></div>

    <script>
        async function generateSVG() {
            const prompt = document.getElementById('prompt').value;
            const resultDiv = document.getElementById('result');
            
            if (!prompt) {
                resultDiv.innerHTML = '<p style="color: red;">Please enter a text description</p>';
                return;
            }
            
            resultDiv.innerHTML = '<p>Generating SVG...</p>';
            
            try {
                const response = await fetch('/svg', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ prompt })
                });
                
                if (response.ok) {
                    const svgData = await response.text();
                    resultDiv.innerHTML = `
                        <h2>Generated SVG:</h2>
                        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0;">
                            ${svgData}
                        </div>
                        <a href="data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgData)}" 
                           download="generated.svg" style="display: inline-block; margin-top: 10px; padding: 5px 10px; background-color: #2ecc71; color: white; text-decoration: none;">
                            Download SVG
                        </a>
                    `;
                } else {
                    const errorData = await response.json();
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${errorData.error || 'Failed to generate SVG'}</p>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>""")
    
    return True

def main():
    """Main function to run the API."""
    parser = argparse.ArgumentParser(description="Start the OmniSVG API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the API server on")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to model directory")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--setup_only", action="store_true", help="Only set up the environment, don't start the API")
    
    args = parser.parse_args()
    
    # Set up the environment
    setup_success = setup_environment()
    if not setup_success:
        logger.error("Failed to set up the environment")
        return 1
    
    if args.setup_only:
        logger.info("Environment setup complete. Exiting without starting the API.")
        return 0
    
    # Set MODEL_DIR environment variable if provided
    if args.model_dir:
        os.environ["MODEL_DIR"] = args.model_dir
    
    # Import the app
    try:
        from api.app import app
    except ImportError as e:
        logger.error(f"Failed to import the API app: {e}")
        logger.error("Make sure you have installed all requirements and that the project structure is correct.")
        return 1
    
    # Run the app
    logger.info(f"Starting the API server on {args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, debug=args.debug)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())