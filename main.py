import argparse
import random
import csv
from pathlib import Path
from image_utils import make_folders, delete_folder_and_contents, save_image_with_text, find_images_in_folder
from llm_utils import init_model_vars, save_tensor, get_first_logit, load_model

# Import LLaVA Library
import sys
import os
sys.path.append(os.path.abspath('../LLaVA'))

from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

random.seed(1234)

query = "In one word, describe what object is in this image"
classes = ["cat", "cow", "dog", "elephant", "lion", "owl", "pig", "snake", "swan", "whale"]

def run_generate(dataset_directory, train_split=0.8):
    delete_folder_and_contents("./data/datasets")

    output_image_path = "./data/datasets/images"
    train_path = "./data/datasets/train"
    test_path = "./data/datasets/test"

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
                annotated_image = save_image_with_text(image_path, text_animal, f"{output_image_path}/{output_filename}")

                first_logit = get_first_logit(query, annotated_image)
                
                tensor_path = f"{save_folder}/{output_filename}"
                save_tensor(first_logit, tensor_path)

                dataset_metadata.append([image_animal, text_animal, filename, "train" if is_train else "test", tensor_path])

    # Write CSV File
    with open("./data/datasets/metadata.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(dataset_metadata)


def run_train():
    delete_folder_and_contents("./data/model")
    make_folders("./data/model")


def run_eval():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main.py")

    # Positional Arguments
    parser.add_argument("operation", choices=["generate", "train", "eval"], help="The operation to perform", type=str)
    parser.add_argument("path", help="The path to the input data", type=str)

    # Named Arguments
    # parser.add_argument("-o", "--output", help="The path to save the output to", type=str)

    args = parser.parse_args()

    # Check if GPU available for LLM
    import torch
    import torchvision.transforms as transforms

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


    if (args.operation == "generate"):
        run_generate(args.path)
    elif (args.operation == "train"):
        run_train(args.path)
    elif (args.operation == "eval"):
        run_eval(args.path)
