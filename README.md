# RateMeAI
[![GitHub issues](https://img.shields.io/github/issues/Pop101/RateMeAI)](https://github.com/Pop101/RateMeAI/issues)

# Table of Contents
- [RateMeAI](#ratemeai)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Technologies](#technologies)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Setup](#setup)
  - [Usage](#usage)
- [Project Status](#project-status)

# Overview
RateMeAI is an face evaluation tool that uses data from /r/truerateme to construct a neural network to rate faces. 

This project is ripe for abuse. Please do not abuse it.

# Technologies
This project is created with:
- [PyTorch](https://pytorch.org/): 2.6.0
- [Torchvision](https://pytorch.org/vision/stable/index.html): 0.21.0
- [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch): 0.7.1
- [Polars](https://pola.rs/): 1.27.1
- [Pillow](https://python-pillow.github.io/): 11.2.1
- [Zstandard](https://github.com/indygreg/python-zstandard): 0.23.0
- [Requests](https://requests.readthedocs.io/): 2.32.3
- [tqdm](https://github.com/tqdm/tqdm): 4.67.1

# Getting Started
This project provides tools for image evaluation using deep learning techniques.

## Installation
Clone the repository and ensure poetry is installed
```sh
git clone https://github.com/Pop101/RateMeAI
pip install poetry
```

Install the dependencies using poetry's version management
```sh
poetry install
```

## Setup
Before using the application, you need to download the required datasets and train the model:

1. Download the necessary data:
```sh
poetry run python download_data.py
```

2. Train the model:
```sh
poetry run python train_model.py
```

## Usage
After setting up the project, you can use the trained model to evaluate images. 

```python
# Example usage (to be updated as the project develops)
from ratemeai import evaluate_image

score = evaluate_image("path/to/your/image.jpg")
print(f"Image score: {score}")
```

# Project Status
**Note: This project is currently under development and not yet complete.** Features and documentation will be expanded as development progresses.