import argparse
import random
import csv
from pathlib import Path
import matplotlib.pyplot as plt

from image_utils import make_folders, delete_folder_and_contents, save_image_with_text, find_images_in_folder
from llm_utils import init_model_vars, save_tensor, get_first_logit, load_model
from detector import SimpleModel, ImageTensorDataset, TextTensorDataset
from train import train_model, eval_model
from graph import load_csv, compute_stats, graph_results, graph_training

# Import LLaVA Library
import sys
import os
sys.path.append(os.path.abspath('../LLaVA'))

from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

random.seed(1234)

# Constants
query = "In one word, describe what object is in this image"
classes = ["cat", "cow", "dog", "elephant", "lion", "owl", "pig", "snake", "swan", "whale"]
input_size=32000

import torch
import torchvision.transforms as transforms

def init_model():
    global tokenizer
    global model
    global image_processor

    # Check if GPU available for LLM
    torch.manual_seed(12)

    if (not torch.cuda.is_available()):
        print("\nNo GPU is available!")
        print("[LLM attempt cancelled]\n")
        raise SystemExit(0)

    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)

    print("\n Loading LLM model...\n")
    tokenizer, model, image_processor = load_model(model_path, None)
    init_model_vars(tokenizer, model, image_processor, model_path, model_name)
    print("\n LLaVA Model loaded!\n")

    # Retrieve the GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disable_torch_init()


def run_generate(dataset_directory, train_split=0.8, output_folder="./data/datasets"):
    delete_folder_and_contents(output_folder)

    output_image_path = f"{output_folder}/images"
    train_path = f"{output_folder}/train"
    test_path = f"{output_folder}/test"

    make_folders(train_path)
    make_folders(test_path)
    make_folders(output_image_path)

    dataset_metadata = []

    for image_animal in classes:
        image_directory = dataset_directory + image_animal

        for image_path in find_images_in_folder(image_directory):
            filename = (image_path.split("/")[-1]).split(".")[0]

            is_train = random.random() < train_split
            save_folder = train_path if is_train else test_path

            # Loop over classs to add text to image
            for text_animal in classes:
                output_filename = f"{image_animal}_{text_animal}_{filename}"
                annotated_image = save_image_with_text(image_path, text_animal, f"{output_image_path}/{output_filename}", randomise_style=True)

                first_logit, tokens = get_first_logit(query, annotated_image)
                llm_class_name = "".join(tokenizer.batch_decode(torch.tensor(tokens[0:-1]), skip_special_tokens=True))
                
                tensor_path = f"{save_folder}/{output_filename}"
                save_tensor(first_logit, tensor_path)

                dataset_metadata.append([image_animal, text_animal, filename, "train" if is_train else "test", tensor_path, llm_class_name])

    # Write CSV File
    with open(f"{output_folder}/metadata.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(dataset_metadata)


def run_train(dataset_path, image_model_size=100, text_model_size=100):
    print(f"Training with image model size of {image_model_size} and text model size of {text_model_size}")
    delete_folder_and_contents("./data/model")
    make_folders("./data/model")

    num_classes = len(classes)

    # Create models
    image_model = SimpleModel(input_size, image_model_size, image_model_size, image_model_size, num_classes)
    text_model = SimpleModel(input_size, text_model_size, text_model_size, text_model_size, num_classes)

    # Create datasets
    image_dataset = ImageTensorDataset(f"{dataset_path}/train", classes)
    text_dataset = TextTensorDataset(f"{dataset_path}/train", classes)

    # Train both models
    print("--- Training Image Model ---")
    image_model_state, image_loss = train_model(image_model, image_dataset, learning_rate=0.001, num_epochs=40)

    print("--- Training Text Model ---")
    text_model_state, text_loss = train_model(text_model, text_dataset, learning_rate=0.001, num_epochs=40)
    
    # Save model states
    torch.save(text_model_state, './data/model/text_model.pt')
    torch.save(image_model_state, './data/model/image_model.pt')

    with open("./data/model/training_loss.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        train_loss = [image_loss, text_loss]
        writer.writerows(list(map(list, zip(*train_loss))))

    print("Trained and Saved Models!")


def run_eval(dataset_path, image_model_size=200, text_model_size=200):
    num_classes = len(classes)

    # Load Models
    image_model = SimpleModel(input_size, image_model_size, image_model_size, image_model_size, num_classes)
    text_model = SimpleModel(input_size, text_model_size, text_model_size, text_model_size, num_classes)

    image_model.load_state_dict(torch.load("./data/model/image_model.pt"))
    text_model.load_state_dict(torch.load("./data/model/text_model.pt"))

    # Load Test Data
    image_dataset = ImageTensorDataset(f"{dataset_path}/test", classes)
    text_dataset = TextTensorDataset(f"{dataset_path}/test", classes)

    image_model.eval()
    text_model.eval()

    print("--- Eval Image Model ---")
    image_test_acc = eval_model(image_model, image_dataset)
    print("--- Eval Text Model ---")
    text_test_acc = eval_model(text_model, text_dataset)
    
    with open("./data/model/testing_acc.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        test_acc = [image_test_acc, text_test_acc]
        writer.writerows([test_acc])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main.py")

    # Positional Arguments
    parser.add_argument("operation", choices=["generate", "train", "eval", "graph"], help="The operation to perform", type=str)
    parser.add_argument("path", help="The path to the input data", type=str)

    # Named Arguments
    parser.add_argument("-o", "--output", nargs="?", const="./data/datasets", help="The path to save the output to", type=str)
    parser.add_argument("--train-split", nargs="?", const="0.8", help="The proportion of examples to use for training (a value from 0 to 1)", type=float)
    parser.add_argument("--text-model-size", nargs="?", const=100, help="The size of the hidden layers in the text model", type=int)
    parser.add_argument("--image-model-size", nargs="?", const=100, help="The size of the hidden layers in the image model", type=int)

    args = parser.parse_args()


    if (args.operation == "generate"):
        init_model()
        run_generate(args.path, output_folder=args.output, train_split=args.train_split)
    elif (args.operation == "train"):
        run_train(args.path, image_model_size=args.image_model_size, text_model_size=args.text_model_size)
    elif (args.operation == "eval"):
        run_eval(args.path, image_model_size=args.image_model_size, text_model_size=args.text_model_size)
    elif (args.operation == "graph"):
        graph_training(args.path)
        plt.clf()

        # test_results = load_csv(args.path + "/" + )
        # stats = compute_stats(test_results)
        # graph_results(stats)