import base64
import os
from io import BytesIO
from logging import getLogger
from typing import List

import torch
from fastapi import APIRouter
from PIL import Image
from python_utils.errors import log_and_raise
from python_utils.model_schemas.images import ImageFile

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# import perception_models.core.vision_encoder.pe as pe
# import perception_models.core.vision_encoder.transforms as transforms

logger = getLogger(__name__)

router = APIRouter(prefix="/features")

# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

DEVICE_NAME: str = os.environ["DEVICE_NAME"]
MODEL: pe.CLIP = pe.CLIP.from_config(os.environ["PE_MODEL_NAME"], pretrained=True)
MODEL = MODEL.to(DEVICE_NAME)
PREPROCESS = transforms.get_image_transform(MODEL.image_size)
TOKENISER = transforms.get_text_tokenizer(MODEL.context_length)


def base64_string_to_pil_image(base64_string: str, ensure_rgb: bool = True) -> Image:
    """
    Convert a base64 string to a PIL Image object.

    Args:
        base64_string (str): The base64 string representation of the image.
        ensure_rgb (bool, optional): Ensure the image is in RGB format. Defaults to True.

    Returns:
        Image: The PIL Image object.
    """

    # Decode the base64 string to bytes
    image_bytes: bytes = base64.b64decode(base64_string)

    # Convert the bytes into a BytesIO buffer
    image_buffer = BytesIO(image_bytes)

    # Open the buffer as a PIL Image
    pil_image: Image = Image.open(image_buffer)

    if ensure_rgb:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

    return pil_image


@router.post("/images")
def embed_images(
    images: List[ImageFile],
) -> List[List[float]]:
    """
    Embed a list of images using the Meta Perception Encoder model.

    Args:
        images (List[ImageFile]): A list of ImageFile objects containing base64 encoded image data.

    Returns:
        List[List[float]]: A list of image embeddings as a list of lists of floats.
    """
    try:
        logger.info(f"Image embedding process started for {len(images)} images")

        image_batch: List[torch.Tensor] = []

        image_file: ImageFile
        for image_file in images:
            raw_image: Image = base64_string_to_pil_image(
                base64_string=image_file.file_upload.file_content.base64_string_content
            )

            if raw_image.mode != "RGB":
                raw_image = raw_image.convert("RGB")

            image_batch.append(PREPROCESS(raw_image))

        image_batch: torch.Tensor = torch.stack(image_batch).to(DEVICE_NAME)

        image_embeddings: torch.Tensor = MODEL.encode_image(image_batch)

        image_embeddings_list: List[List[float]] = image_embeddings.cpu().tolist()

        logger.info(
            f"Image embedding process successfully completed for {len(images)} images"
        )

        return image_embeddings_list

    except Exception as e:
        return log_and_raise(error=e)


@router.post("/text")
def embed_text(
    texts: List[str],
) -> List[List[float]]:
    """
    Embed a list of texts using the Meta Perception Encoder model.

    Args:
        texts (List[str]): A list of strings to be embedded.

    Returns:
        List[List[float]]: A list of text embeddings as a list of lists of floats.
    """
    try:
        logger.info(f"Text embedding process started for {len(texts)} texts")

        text_batch: List[torch.Tensor] = TOKENISER(texts).to(DEVICE_NAME)

        text_embeddings: torch.Tensor = MODEL.encode_text(text_batch)

        text_embeddings_list: List[List[float]] = text_embeddings.tolist()

        logger.info(
            f"Text embedding process successfully completed for {len(texts)} texts"
        )

        return text_embeddings_list

    except Exception as e:
        return log_and_raise(error=e)
