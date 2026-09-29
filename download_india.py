from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

REPO_ID = "dronefreak/RDD2022"

output_images = Path("dataset/images/train")
output_labels = Path("dataset/labels/train")

output_images.mkdir(parents=True, exist_ok=True)
output_labels.mkdir(parents=True, exist_ok=True)

api = HfApi()

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

india_images = [
    f for f in files
    if f.startswith("data/images/train/")
    and "/India_" in f
    and f.endswith(".jpg")
]

print("India images found:", len(india_images))

# First 500 images only — beginner-friendly training size
india_images = india_images[:500]

for i, image_file in enumerate(india_images, start=1):

    label_file = image_file.replace(
        "data/images/train/",
        "data/labels/train/"
    ).replace(".jpg", ".txt")

    hf_hub_download(
        repo_id=REPO_ID,
        filename=image_file,
        repo_type="dataset",
        local_dir="dataset_download"
    )

    hf_hub_download(
        repo_id=REPO_ID,
        filename=label_file,
        repo_type="dataset",
        local_dir="dataset_download"
    )

    print(f"Downloaded {i}/{len(india_images)}")

print("India dataset download completed!")