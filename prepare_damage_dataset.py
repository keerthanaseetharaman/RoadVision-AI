from pathlib import Path
import shutil

source_images = Path("dataset/images/train")
source_labels = Path("dataset/labels/train")

target_images = Path("dataset/damage_train/images")
target_labels = Path("dataset/damage_train/labels")

# Clear old prepared dataset
if target_images.exists():
    shutil.rmtree(target_images)

if target_labels.exists():
    shutil.rmtree(target_labels)

target_images.mkdir(parents=True, exist_ok=True)
target_labels.mkdir(parents=True, exist_ok=True)

copied = 0

for label_file in source_labels.rglob("*.txt"):

    # Skip empty labels
    if label_file.stat().st_size == 0:
        continue

    image_name = label_file.stem + ".jpg"

    matches = list(source_images.rglob(image_name))

    if not matches:
        print("Image not found:", image_name)
        continue

    image_file = matches[0]

    shutil.copy2(
        image_file,
        target_images / image_name
    )

    shutil.copy2(
        label_file,
        target_labels / label_file.name
    )

    copied += 1

print("Damage images copied:", copied)
print("Damage labels copied:", copied)
print("Damage dataset preparation completed!")