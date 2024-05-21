import csv
import random
import os, shutil
from pathlib import Path

from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont
import textwrap

from datasets import load_dataset

def make_folders(folder_path):
    """
    Creates all folders needed in path
 
    Args:
        folder_path (string): The path to the folder
 
    Returns:
        void
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)

def delete_folder_and_contents(folder_path):
    """
    Delete the contents of a folder and the folder itself
 
    Args:
        folder_path (string): The path to the folder
 
    Returns:
        void
    """
    if Path(folder_path).is_dir():
        delete_folder_contents(folder_path)
        Path(folder_path).rmdir()

def delete_folder_contents(folder_path):
    """
    Deletes all the files in the given folder.
 
    Args:
        folder_path (string): The folder to delete from.
 
    Returns:
        void
    """

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def find_images_in_folder(directory_path):
    pathlist = Path(directory_path).rglob('*.jpg')
    return [str(path) for path in pathlist]


def add_border(image, border_size=0.2): 
    """
    Add a white border to the image
 
    Args:
        image (PIL Image): The image to adjust
        border_size (float): The fractional size for the border
 
    Returns:
        PIL Image: The adjusted image with a border
    """

    image = image.resize((224, 224))
    image_width, image_height = image.size

    # Calculate the width for the white image border
    border_width = int(image_height * border_size)
    size_diff = 2 * border_width

    # Add a border to the image
    bordered_image = Image.new('RGB', (image_width + size_diff, image_height + size_diff), 'white')
    bordered_image.paste(image, (border_width, border_width))

    return bordered_image


def insert_text(image, text, font_size=16, color=(0, 0, 0, 255), position="top"): 
    """
    Insert text into the given image
 
    Args:
        image (PIL Image): The image to adjust
        text (string): The text to insert into the image
        font_size (int): The sizing of the text
        color: (tuple of ints): The color of the text
        position: (string): The position of the text
 
    Returns:
        PIL Image: The adjusted image with text inserted
    """

    image_width, image_height = image.size
    border_width = int(image_height * 1/7)

    try:
	    font = ImageFont.truetype("./fonts/Roboto-Medium.ttf", font_size)
    except IOError:
        print("Could not load font, using default")
        font = ImageFont.load_default()

    lines = textwrap.wrap(text, width=int(image_width / (font_size * 0.45)))
    draw = ImageDraw.Draw(image)

    y_text = border_width*2/3 if position == "top" else int(image_height * 5/7)
    for line in lines:
        _, _, line_width, line_height = font.getbbox(line)
        draw.text((int(image_width/2 - line_width/2), int(y_text - line_height/2)), line, font=font, fill=color)
        y_text += line_height + 5

    return image


def save_image_with_text(image_path, text, output_path, font_size=24, randomise_style=False):
    """
    Load an image and save it with text to a new location
 
    Args:
        image_path (string): The path of the image to modify
        text (string): The text to add to the image
        output_path (string): The path of the output file
        font_size (int): The sizing of the text
        randomise_style (bool): Randomise the styling of the text
 
    Returns:
        string: The path of the saved image
    """

    image = Image.open(image_path)
    border_image = add_border(image)

    text_color=(0, 0, 0, 255)
    text_size = font_size
    text_position = "top"

    if randomise_style:
        colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255), (0, 0, 0, 255)]
        text_color = random.choice(colors)
        text_position = random.choice(["top", "bottom"])
        text_size = random.randint(10, 64)

    text_image = insert_text(border_image, text, font_size=text_size, color=text_color, position=text_position)

    # Save the image
    text_image.save(f"{output_path}.jpg")

    return f"{output_path}.jpg"
