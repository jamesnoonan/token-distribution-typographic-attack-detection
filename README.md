# Using Token Distributions to Detect Typographic Attacks on Large Language Models
The code behind the paper *Using Token Distributions to Detect Typographic Attacks on Large Language Models*.

## Installation
You must have LLaVA installed in a sibling directory named "LLaVA" to use this tool.

To install dependencies in the miniconda environment, run `conda install --file requirements.txt`

## Usage
This repository is a set of tools to generate typographic datasets and train typographic detectors.

### Generate Typographic Dataset
Run `main.py generate path/to/images --output <path/to/output>`. You can set the `--train-split` property to a value between 0 and 1 to control the sizes of the train and test sets.

The input folder with images must have subfolders with named with each of the classification classes (e.g. dog, cat, ...).

### Get LLM Prediction Accuracy
Run `main.py llmacc path/to/dataset` to print LLaVA's performance on the test dataset.

### Train Detector Models
Run `main.py train path/to/dataset --image-model-size <model-size>  --text-model-size <model-size>` to train the image and text predictor models. They will be saved to `./data/model`.

### Eval Detector Models
Run `main.py eval path/to/dataset --image-model-size <model-size>  --text-model-size <model-size>` to evaluate the image and text predictor models that are saved to `./data/model` on test set.