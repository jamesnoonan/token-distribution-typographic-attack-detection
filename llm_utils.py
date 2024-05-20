from llava.eval.run_llava import load_image, load_images
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
# import gc
from PIL import Image
from undecorated import undecorated
from types import MethodType

import torch
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_model_vars(l_tokenizer, l_model, l_image_processor, l_model_path, l_model_name):
    """
    Initialize the model variables as global variables
 
    Args:
        l_tokenizer (Tokenizer): The local tokenizer to set to a global variable
        l_model (Model): The local model to set to a global variable
        l_image_processor (Image Processor): The local tokenizer to set to a global variable
        l_model_path (string): The path to the model
        l_model_name (string): The model name
 
    Returns:
        void
    """
    global tokenizer
    global model
    global image_processor
    global model_path
    global model_name

    tokenizer = l_tokenizer
    model = l_model
    image_processor = l_image_processor
    model_path = l_model_path
    model_name = l_model_name


# Convert a tensor to a PIL Image
tensor_to_pil = transforms.ToPILImage()


def save_tensor_as_image(tensor, filename):
    """
    Preprocess an image before it is given to LLaVA
 
    Args:
        image_path (string): The path to the image to use
 
    Returns:
        torch.Tensor: the preprocessed image tensor
    """

    pil_input_image = tensor_to_pil(tensor)
    pil_input_image.save(f"./data/text-fgsm/{filename}.jpg")


def load_model(model_path, model_base):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )

    return tokenizer, model, image_processor


def preprocess_image(image_path):
    """
    Preprocess an image before it is given to LLaVA
 
    Args:
        image_path (string): The path to the image to use
 
    Returns:
        torch.Tensor: the preprocessed image tensor
    """

    # Load images
    image_files = [image_path]
    images = load_images(image_files)
    image_sizes = [x.size for x in images]

    # Preprocess images
    processed_image_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    return processed_image_tensor


def save_tensor(tensor, path):
    """
    Saves a tensor as a file
 
    Args:
        tensor (torch.Tensor): The tensor to save
        path (string): The path to save the tensor file to
 
    Returns:
        void
    """
    torch.save(tensor, f"{path}.pt")


def preprocess_prompt(qs, model, model_name, tokenizer):
    """
    Preprocess the text prompt for LLaVA
 
    Args:
        qs (string): The input prompt for the LVLM
        model (Model): The model to run
        model_name (string): The name of the model to run
        tokenizer (Tokenizer): The tokenizer to use on the input prompt
 
    Returns:
        torch.Tensor: the input ids for the LVLM
    """

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    # Determine conv mode
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    return input_ids


def llava_generate(prompt, image_tensor_input):
    """
    Run LLaVA on a prompt and image input
 
    Args:
        prompt (string): The input prompt for the LVLM
        image_tensor_input (torch.Tensor): The preprocessed image tensor
 
    Returns:
        dict: dictionary of output scores, attentions and hidden states
    """

    # Tokenize and process the prompt
    input_ids = preprocess_prompt(prompt, model, model_name, tokenizer)

    temperature = 0 # Set temperature to zero for repeatable results

    with torch.inference_mode():
        output_dict = model.generate(
            input_ids,
            images=image_tensor_input,
            image_sizes=[x.size for x in image_tensor_input] if (image_tensor_input is not None) else None,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True
        )

    return output_dict


def get_first_logit(prompt, image_path):
    """
    Get the first logit from LLaVA and token for a given prompt and image
 
    Args:
        prompt (string): The input prompt for the LVLM
        image_path (string): The path to the input image
 
    Returns:
        torch.Tensor, torch.Tensor: the first logit output, the predicted tokens
    """

    preprocessed_image_tensor = preprocess_image(image_path)
    outputs = llava_generate(prompt, preprocessed_image_tensor)

    logits = torch.cat(outputs["scores"], dim=0).cpu().numpy()

    probs = [torch.nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
    output_ids = outputs["sequences"][0][-len(probs):]

    return logits[0], torch.tensor(output_ids)
