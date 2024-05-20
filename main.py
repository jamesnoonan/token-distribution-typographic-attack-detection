import argparse
import random
from pathlib import Path
from image_utils import make_folders, delete_folder_and_contents, save_image_with_text
from llm_utils import save_tensor, get_first_logit

# Import LLaVA Library
import sys
import os
sys.path.append(os.path.abspath('../LLaVA'))

random.seed(1234)

classes = ["cat", "cow", "dog", "elephant", "lion", "owl", "pig", "snake", "swan", "whale"]

def run_generate(dataset_directory, train_split=0.8):
    delete_folder_and_contents("./data/datasets")

    output_image_path = "./data/datasets/images"
    train_path = "./data/datasets/train"
    test_path = "./data/datasets/test"

    make_folders(train_path)
    make_folders(test_path)
    make_folders(output_image_path)

    for image_animal in classes:
        image_directory = dataset_directory + image_animal

        for image_path in find_images_in_folder(image_directory):
            filename = (image_path.split("/")[-1]).split(".")[0]
            save_folder = train_path if random.random() < train_split else test_path

            # Loop over classs to add text to image
            for text_animal in classes:
                output_filename = f"{image_animal}_{text_animal}_{filename}"
                annotated_image = save_image_with_text(image_path, text_animal, f"{output_image_path}/{output_filename}")

                first_logit = get_first_logit(query, annotated_image)
                save_tensor(first_logit, f"{save_folder}/{output_filename}")


def run_train():
    pass


def run_eval():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main.py")

    # Positional Arguments
    parser.add_argument("operation", choices=["generate", "train", "eval"], help="The operation to perform", type=string)
    parser.add_argument("path", help="The path to the input data", type=string)

    # Named Arguments
    parser.add_argument("-o", "--output", help="The path to save the output to", type=string)


    args = parser.parse_args()
    

    if (args.operation == "generate"):
        run_generate(args.path)


# dataset_directory = "/scratch/kf09/jn1048/.cache/kaggle/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/animals/animals/"