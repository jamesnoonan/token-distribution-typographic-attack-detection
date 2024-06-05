# Using Token Distributions to Detect Typographic Attacks on Large Language Models
The code behind the paper *Using Token Distributions to Detect Typographic Attacks on Large Language Models*, which allows for the training of prediction models to detect typographic attacks on Large Vision Language Models (LVLMs). The detection system works by predicting the image and textual content of the image using only the first logit of the LVLM. The architecture of the detection system can be seen below:

![image](https://github.com/jamesnoonan/token-distribution-typographic-attack-detection/assets/43080771/eafb40e8-9891-4288-a1f9-7fd2ab568c1e)

## Installation
You must have LLaVA installed in a sibling directory named "LLaVA" to use this tool.

To install dependencies in the miniconda environment, run `conda install --file requirements.txt`

## Quickstart

Create a folder at "./data/images" containing the subfolders: "cat", "cow", "dog", "elephant", "lion", "owl", "pig", "snake", "swan", "whale", each with the images for that class. Then run the following commands to train the typographic attack detection models.

```python
python main.py generate "./data/images" --output "./data/new-dataset" --train-split 0.8 --font-variation --prompt-variation
python main.py llmacc "./data/new-dataset"
python main.py train "./data/new-dataset" --output "./data/new-model" --image-model-size 1000 --text-model-size 1000 --epochs 60
python main.py eval "./data/new-dataset" --eval-model "./data/new-model" --image-model-size 1000 --text-model-size 1000
```

## Usage
This repository is a set of tools to generate typographic datasets and train typographic detectors. For help, run `python main.py -h`.

### Generate Typographic Dataset

The generation code requires the input images to be structured in a particular way. The folder **must** have subfolders named with each of the classification classes (e.g. dog, cat, ...), each of which contains the images for that class.

<img width="360" alt="photos" src="https://github.com/jamesnoonan/token-distribution-typographic-attack-detection/assets/43080771/c4051888-43e4-4434-b81d-9e3cc4ce8df9">

Run `python main.py generate <path/to/images> --output <path/to/dataset>` to generate a dataset of typographically attacked images. You can set the `--train-split` property to a value between 0 and 1 to control the sizes of the train and test sets.

The optional flags `--font-variation` and `prompt-variation` allows for variation to be added to the data as described in the paper. An example of font variation can be seen below:

![random-attack](https://github.com/jamesnoonan/token-distribution-typographic-attack-detection/assets/43080771/be711a7e-0567-4ff5-ae18-fc365d15d443)


### Get LLM Prediction Accuracy
Run `python main.py llmacc <path/to/dataset>` to print LLaVA's performance on the test dataset.

### Train Detector Models
Run `python main.py train <path/to/dataset> --output <path/to/model>` to train the image and text predictor models. The number of epochs, and size of hidden layers in the image and text models can be adjusted using the `--epochs`, `--image-model-size` and `--text-model-size` arguments respectively.

### Eval Detector Models
Run `python main.py eval <path/to/dataset> --eval-model <path/to/model>` to evaluate the image and text predictor models in the folder given by the `--eval-model` argument on the test set given in the first argument. 

*Note: if the hidden layer size was adjusted in the training command, it must also be set to the same value on the eval command, using `--image-model-size` and `--text-model-size`*
