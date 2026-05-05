"""Score and visualize images with the trained model."""
import os
import random
from typing import Union

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

from modules.dino_model import DinoVisionAnalysisModel
from modules.image_dataset import pad_to_square
from train_model import IMAGE_SIZE


def _build_inference_model(checkpoint: str) -> DinoVisionAnalysisModel:
    """Construct the full image→backbone→head model and load saved head weights.

    Training runs on a head-only Lightning module over cached features, so
    the checkpoint state_dict contains only ``head.<...>`` entries. The
    DINOv2 backbone weights are re-fetched by ``__init__`` from HF;
    ``strict=False`` lets the head load while the backbone stays put.
    """
    model = DinoVisionAnalysisModel()
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


MODEL = _build_inference_model("models/image_rating_model_final.ckpt")


def _to_tensor(image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a string, PIL.Image.Image, or torch.Tensor")
    image = pad_to_square(image)
    return transforms.Resize(IMAGE_SIZE)(image)


def evaluate_image(image: Union[str, Image.Image, torch.Tensor]) -> float:
    """Predict the score for a single image."""
    tensor = _to_tensor(image)
    return MODEL.predict(tensor.unsqueeze(0)).item()


def get_focus_map(image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
    """Last-layer CLS-to-patch attention, upsampled to image resolution."""
    tensor = _to_tensor(image)
    with torch.no_grad():
        focus_map = MODEL.get_focus_map(tensor.unsqueeze(0))
    return focus_map[0]


def visualize_image_analysis(image_path: str) -> None:
    original_image = Image.open(image_path)
    image_tensor = transforms.ToTensor()(original_image)
    processed_tensor = pad_to_square(image_tensor)
    processed_tensor = transforms.Resize(IMAGE_SIZE)(processed_tensor)
    processed_image = transforms.ToPILImage()(processed_tensor)

    score = evaluate_image(processed_image)
    focus_map = get_focus_map(processed_image)
    focus_map_np = focus_map.numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(processed_image)
    ax1.set_title('Processed Input Image')
    ax1.axis('off')
    focus_map_plot = ax2.imshow(focus_map_np, cmap='hot')
    ax2.set_title('Focus Map')
    ax2.axis('off')
    plt.colorbar(focus_map_plot, ax=ax2)
    plt.suptitle(f'File: {os.path.basename(image_path)}\nScore: {score:.2f}', y=0.95, fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    jpg_files = []
    for root, _, files in os.walk("thumbnails"):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))

    selected_files = random.sample(jpg_files, min(5, len(jpg_files)))

    for image_path in selected_files:
        print(f"\nAnalyzing {image_path}...")
        try:
            visualize_image_analysis(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
        plt.close('all')
