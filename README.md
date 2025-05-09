# OmniSVG: Text-to-SVG Implementation

This project implements a text-to-SVG generation model based on the [OmniSVG paper](https://arxiv.org/abs/2504.06263). It leverages the Qwen2.5-VL 3B model to generate Scalable Vector Graphics (SVG) from text descriptions.

## Features

- Convert text descriptions to high-quality SVGs
- Built on Qwen2.5-VL 3B, a powerful Vision-Language Model
- Efficient SVG parameterization for better token utilization
- Flask web app for easy integration and deployment
- Support for various generation parameters (temperature, top-k, top-p)

## Project Structure

```
omnisvg/
├── data/
│   ├── raw/           # Raw SVG data
│   ├── processed/     # Tokenized SVG data
├── models/           # Saved model checkpoints
├── notebooks/        # Development notebooks
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # SVG tokenization and processing
│   ├── model.py            # OmniSVG model implementation
│   ├── train.py            # Training pipeline
│   ├── utils.py            # Utility functions
│   ├── inference.py        # Inference pipeline
│   ├── config.py           # Configuration settings
├── api/
│   ├── __init__.py
│   ├── app.py              # Flask application
│   ├── templates/          # HTML templates
│       ├── index.html      # Web interface
├── tests/
│   ├── test_data_processing.py
│   ├── test_model.py
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/omnisvg.git
cd omnisvg
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model on your own dataset:

```bash
python -m src.train --config_file config.json
```

### Generating SVGs from Text

Use the inference script to generate SVGs from text:

```bash
python -m src.inference --model_dir models/omnisvg/final_model --prompt "A red heart icon" --output_dir outputs
```

### Running the Web API

Start the Flask web application:

```bash
python -m api.app --model_dir models/omnisvg/final_model --port 5000
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface.

### API Endpoints

- `POST /predict`: Generate SVGs from text
  - Request Body: `{"prompt": "Text prompt", "temperature": 0.7, "top_k": 50, "top_p": 0.95, "num_samples": 1}`
  - Response: `{"svg": ["SVG content"], "generation_time": 1.23}`

- `POST /svg`: Generate and return an SVG directly
  - Request Body: `{"prompt": "Text prompt", "temperature": 0.7, "top_k": 50, "top_p": 0.95}`
  - Response: SVG content with content-type "image/svg+xml"

## Example

```python
from src.inference import OmniSVGGenerator

# Initialize generator
generator = OmniSVGGenerator("models/omnisvg/final_model")

# Generate SVG
svg = generator.generate("A cute cartoon dog with a red collar")[0]

# Save SVG to file
with open("dog.svg", "w") as f:
    f.write(svg)
```

## Model Architecture

This implementation is based on the OmniSVG architecture described in the paper. Key components include:

1. **Qwen2.5-VL Base Model**: A powerful Vision-Language Model with 3B parameters
2. **SVG Tokenization**: Efficient parameterization of SVG commands and coordinates
3. **Token Embedding**: Custom embeddings for SVG-specific tokens
4. **Autoregressive Generation**: Next-token prediction for SVG generation

## Configuration

Model behavior can be configured using a JSON configuration file or environment variables:

- `MODEL_DIR`: Path to the model directory
- `BATCH_SIZE`: Batch size for training and inference
- `MAX_SVG_LEN`: Maximum SVG token length
- `TOP_K`, `TOP_P`, `TEMPERATURE`: Generation parameters

## Testing with cURL

You can test the API endpoints using cURL:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A red heart icon", "temperature": 0.7, "top_k": 50, "top_p": 0.95}'
```

## Credits

This implementation is based on the research paper:

```
@article{yang2025omnisvg,
  title={OmniSVG: A Unified Scalable Vector Graphics Generation Model},
  author={Yang, Yiying and Cheng, Wei and Chen, Sijin and Zeng, Xianfang and Zhang, Jiaxu and Wang, Liao and Yu, Gang and Ma, Xingjun and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2504.06263},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.