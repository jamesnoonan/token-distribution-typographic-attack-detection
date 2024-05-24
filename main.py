import argparse
import random
import csv
from pathlib import Path
import matplotlib.pyplot as plt

from image_utils import make_folders, delete_folder_and_contents, save_image_with_text, find_images_in_folder
from models import SimpleModel, ImageTensorDataset, TextTensorDataset
from train import train_model, eval_model
from stats import load_csv, compute_llm_acc

import torch
import torchvision.transforms as transforms

# Handle case where LLaVA isn't installed
llava_installed = Path('../LLaVA').is_dir()
if llava_installed:
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    from llm_utils import init_model_vars, save_tensor, get_first_logit, load_model

    # Import LLaVA Library
    import sys
    import os
    sys.path.append(os.path.abspath('../LLaVA'))

# Set seeds for reproducable results
random.seed(1234)
torch.manual_seed(12)

# Constants for the model
classes = ["cat", "cow", "dog", "elephant", "lion", "owl", "pig", "snake", "swan", "whale"]
queries = [
    "In one word, describe what object is in this image",
    "What is this?",
    "What animal is this?",
    "What animal is in the image? Reply as briefly as possible.",
    "What animal is in the image? Choose one of the following animals: cat, cow, dog, elephant, lion, owl, pig, snake, swan or whale"
]
input_size=32000


def init_model():
    """
    Load LLaVA and save to global variables model, tokenizer and image_processor for use
 
    Args:
        none
 
    Returns:
        void
    """

    global tokenizer
    global model
    global image_processor

    # Check if GPU available for LLM
    if (not torch.cuda.is_available()):
        # Cancel run if LLaVA is attempted to be run without a GPU
        print("\nNo GPU is available!")
        print("[LLM attempt cancelled]\n")
        raise SystemExit(0)

    # Define the particular version of LLaVA to use
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)

    # Load LLaVA and associated variables
    print("\n Loading LLM model...\n")
    tokenizer, model, image_processor = load_model(model_path, None)
    init_model_vars(tokenizer, model, image_processor, model_path, model_name)
    print("\n LLaVA Model loaded!\n")

    disable_torch_init()


def run_generate(dataset_directory, train_split, output_folder, font_variation, prompt_variation):
    """
    Generate a typographically attacked dataset split into train and test folders
 
    Args:
        dataset_directory (string): the path to the original dataset containing the images, sorted into subfolders with their classname
        train_split (float): the proportion of the samples assigned to the training set (the remainder will be assigned to the test set)
        output_folder (string): the folder to save the dataset to, defaults to './data/dataset'
 
    Returns:
        void
    """

    # Clear and make necessary folders
    delete_folder_and_contents(output_folder)

    output_image_path = f"{output_folder}/images"
    train_path = f"{output_folder}/train"
    test_path = f"{output_folder}/test"

    make_folders(train_path)
    make_folders(test_path)
    make_folders(output_image_path)

    # The dataset metadata stores information about each case
    dataset_metadata = []

    # Iterate over every image class
    for image_animal in classes:
        image_directory = dataset_directory + "/" + image_animal

        # Loop over all images in subfolder fro image class
        for image_path in find_images_in_folder(image_directory):
            filename = (image_path.split("/")[-1]).split(".")[0]

            is_train = random.random() < train_split
            save_folder = train_path if is_train else test_path

            # Loop over classes of attack class
            for text_animal in classes:
                output_filename = f"{image_animal}_{text_animal}_{filename}"

                # Create a the typographically attacked iamge
                annotated_image = save_image_with_text(image_path, text_animal, f"{output_image_path}/{output_filename}", randomise_style=font_variation)

                if prompt_variation:
                    query = random.choice(queries) # Randomly sample a prompt to use
                else:
                    query = queries[0]

                # Run the typographically attacked images through LLaVA
                first_logit, tokens = get_first_logit(query, annotated_image)
                llm_class_name = "".join(tokenizer.batch_decode(torch.tensor(tokens[0:-1]), skip_special_tokens=True))
                
                # Save the tensor to the dataset
                tensor_path = f"{save_folder}/{output_filename}"
                save_tensor(first_logit, tensor_path)

                dataset_metadata.append([image_animal, text_animal, filename, "train" if is_train else "test", tensor_path, llm_class_name])

    # Save metadata to csv file
    with open(f"{output_folder}/metadata.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(dataset_metadata)


def run_train(dataset_path, output_folder, image_model_size, text_model_size, epochs):
    """
    Train the image and text predictor models
 
    Args:
        datset_path (string): the path to the generated dataset
        image_model_size (int): the size to use for the hidden layers of the image model
        text_model_size (int): the size to use for the hidden layers of the text model
 
    Returns:
        void
    """

    print(f"Training with image model size of {image_model_size} and text model size of {text_model_size}")
    num_classes = len(classes)

    # Clear and make the necessary folders
    delete_folder_and_contents(output_folder)
    make_folders(output_folder)

    # Create models
    image_model = SimpleModel(input_size, image_model_size, image_model_size, image_model_size, num_classes)
    text_model = SimpleModel(input_size, text_model_size, text_model_size, text_model_size, num_classes)

    # Create datasets
    image_dataset = ImageTensorDataset(f"{dataset_path}/train", classes)
    text_dataset = TextTensorDataset(f"{dataset_path}/train", classes)

    # Train both models
    print("--- Training Image Model ---")
    image_model_state, image_loss = train_model(image_model, image_dataset, learning_rate=0.001, num_epochs=epochs)

    print("--- Training Text Model ---")
    text_model_state, text_loss = train_model(text_model, text_dataset, learning_rate=0.001, num_epochs=epochs)
    
    # Save model states
    torch.save(text_model_state, f"{output_folder}/text_model.pt")
    torch.save(image_model_state, f"{output_folder}/image_model.pt")

    # Save training loss from models to csv file
    with open(f"{output_folder}/training_loss.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        train_loss = [image_loss, text_loss]
        writer.writerows(list(map(list, zip(*train_loss))))

    print("Trained and Saved Models!")


def run_eval(dataset_path, model_path, image_model_size, text_model_size):
    """
    Evaluate the image and text predictor models
 
    Args:
        datset_path (string): the path to the generated dataset
        model_path (string): the path to the folder of the models to evaluate
        image_model_size (int): the size to use for the hidden layers of the image model
        text_model_size (int): the size to use for the hidden layers of the text model
 
    Returns:
        void
    """

    num_classes = len(classes)

    # Load Models
    image_model = SimpleModel(input_size, image_model_size, image_model_size, image_model_size, num_classes)
    text_model = SimpleModel(input_size, text_model_size, text_model_size, text_model_size, num_classes)

    image_model.load_state_dict(torch.load(f"{model_path}/image_model.pt"))
    text_model.load_state_dict(torch.load(f"{model_path}/text_model.pt"))

    image_model.eval()
    text_model.eval()

    # Load Test Data
    image_dataset = ImageTensorDataset(f"{dataset_path}/test", classes)
    text_dataset = TextTensorDataset(f"{dataset_path}/test", classes)
    
    # Evaluate the Model 
    print("--- Eval Image Model ---")
    image_test_acc = eval_model(image_model, image_dataset)
    print("--- Eval Text Model ---")
    text_test_acc = eval_model(text_model, text_dataset)
    
    # Save the testing accuracies to a csv file
    with open(f"{model_path}/testing_acc.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        test_acc = [image_test_acc, text_test_acc]
        writer.writerows([test_acc])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main.py")

    # Positional Arguments
    parser.add_argument("operation", choices=["generate", "llmacc", "train", "eval"], help="The operation to perform", type=str)
    parser.add_argument("path", help="The path to the input dataset", type=str)

    # Named Arguments
    parser.add_argument("--output", help="Set the output directory of this operation", type=str)
    parser.add_argument("--train-split", default="0.8", help="The proportion of examples to use for training (a value from 0 to 1)", type=float)
    parser.add_argument("--font-variation", action='store_true', help="Enable variation in font color, size and position when generating the dataset")
    parser.add_argument("--prompt-variation", action='store_true', help="Enable variation in queries when generating the dataset")

    parser.add_argument("--epochs", default=60, help="The number of epochs to train for", type=int)
    parser.add_argument("--text-model-size", default=1000, help="The size of the hidden layers in the text model", type=int)
    parser.add_argument("--image-model-size", default=1000, help="The size of the hidden layers in the image model", type=int)
    parser.add_argument("--eval-model", default="./data/model", help="Select the containing folder of the model to evaluate", type=str)


    args = parser.parse_args()

    # Handle each operation
    if (args.operation == "generate"): # Generate a dataset
        if (not llava_installed):
            print("LLaVA not found! Please install LLaVA to run this operation")
        else:
            output = args.output if args.output is not None else "./data/dataset"
            init_model()
            run_generate(args.path, output_folder=output, train_split=args.train_split, font_variation=args.font_variation, prompt_variation=args.prompt_variation)

    elif (args.operation == "llmacc"): # Compute the accuracy of LLaVA on a test dataset
        compute_llm_acc(f"{args.path}/metadata.csv")

    elif (args.operation == "train"): # Train prediction models
        output = args.output if args.output is not None else "./data/model"

        run_train(args.path, output_folder=output, image_model_size=args.image_model_size, text_model_size=args.text_model_size, epochs=args.epochs)

    elif (args.operation == "eval"):  # Evaluate prediction models on test set
        run_eval(args.path, model_path=args.eval_model, image_model_size=args.image_model_size, text_model_size=args.text_model_size)

