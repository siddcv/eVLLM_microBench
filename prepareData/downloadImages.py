from datasets import load_dataset
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# Output directory for images
#output_dir = Path("/workspace/datasets/cardiovascular")
output_dir=Path("/workspace/eVLLM_Sidd/eVLLM_microBench/organData/gastrointestinal/images")
output_dir.mkdir(parents=True, exist_ok=True)

# Load the test split of the dataset
dataset = load_dataset("teamasap/gastrointestinal-and-liver-pathology-dataset", split="test")

# Download and save each image
# Download and save each image

for example in dataset:
    image_id = example["image_id"]

    try:
        image = example["image"].convert("RGB")  # The 'image' field is a PIL Image already
        image.save(output_dir / f"{image_id}.png")
        print(f"Saved {image_id}.png")
    except Exception as e:
        print(f"Error saving {image_id}: {e}")


print(f"All images saved in {output_dir}")

