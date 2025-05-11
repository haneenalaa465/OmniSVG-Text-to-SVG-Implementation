# OmniSVG: A Unified Framework for SVG Generation

OmniSVG is a framework for generating high-quality Scalable Vector Graphics (SVGs) using vision-language models (VLMs). This implementation is based on the research paper ["OmniSVG: A Unified Scalable Vector Graphics Generation Model"](https://arxiv.org/abs/2504.06263).

## Features

- Generate SVGs from text descriptions
- Convert raster images to vector graphics
- Support for character reference SVG generation
- Handles complex SVG with up to 30k tokens
- Parameterizes SVG commands and coordinates into discrete tokens

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/omnisvg.git
cd omnisvg

# Install dependencies
pip install -e .
```

## Usage

### Command Line Interface

#### Generate an SVG from a text prompt:

```bash
python main.py generate "A red heart with a blue outline" --output heart.svg
```

#### Run multiple test generations:

```bash
python main.py test --output-dir ./generated
```

#### Train a model:

```bash
python main.py train --output-dir ./models/my_model --num-samples 8000
```

### Python API

```python
from omnisvg.modeling import load_model, generate_svg_from_text
from omnisvg.visualization import display_svg

# Load model
model, text_tokenizer, svg_tokenizer = load_model()

# Generate SVG
svg_text = generate_svg_from_text("A green tree icon", model, text_tokenizer, svg_tokenizer)

# Display SVG (in Jupyter notebook)
display_svg(svg_text)

# Save SVG to file
with open("tree.svg", "w") as f:
    f.write(svg_text)
```

## Project Structure

```
omnisvg/
├── __init__.py
├── config.py        # Configuration settings
├── dataset.py       # Data loading and processing
├── tokenizer.py     # SVG tokenization utilities
├── utils.py         # Helper functions
├── visualization.py # Visualization tools
└── modeling/
    ├── __init__.py
    ├── train.py     # Training utilities
    └── predict.py   # Inference utilities
```

## Dataset

The implementation uses the MMSVG-2M dataset, which includes:
- MMSVG-Icon: 1.1 million SVG icons
- MMSVG-Illustration: 0.5 million SVG illustrations
- MMSVG-Character: 0.4 million SVG anime characters

## Model Architecture

OmniSVG is built on top of a pre-trained vision-language model (Qwen2.5-VL) and incorporates an SVG tokenizer. The model tokenizes both text and image inputs as prefix tokens, while the SVG tokenizer encodes vector graphics commands into a unified representation space.

## Citation

If you use this code in your research, please cite the original paper:

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