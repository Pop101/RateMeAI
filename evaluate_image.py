from modules.image_rating_model import ImageRatingModel
from modules.image_dataset import ImageRatingDataset, pad_to_square

import torchvision.transforms as transforms

from PIL import Image
import torch

from typing import Union
from train_model import BASE_MODEL, IMAGE_SIZE

MODEL = ImageRatingModel.load("models/image_rating_model_final.pth", model_type=BASE_MODEL)

def evaluate_image(image:Union[str, Image.Image, torch.Tensor]) -> float:
    """Evaluates a face's attractiveness based on r/truerateme

    Args:
        image (Union[str, Image.Image, torch.Tensor]): The source image
    Returns:
        float: The attractiveness score
    """
    
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a string, PIL.Image.Image, or torch.Tensor")
    
    # run needed transforms on image
    image = pad_to_square(image)
    image = transforms.Resize(IMAGE_SIZE)(image) # unfortunately hardcoded
    image = ImageRatingDataset.get_transforms(train=False)(image)
    
    # run model
    with torch.no_grad():
        output = MODEL(image.unsqueeze(0))
    return output.item()

if __name__ == "__main__":
    print(evaluate_image("thumbnails/zrasfl.jpg"))