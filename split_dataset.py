from pathlib import Path
import shutil

train_images = Path("dataset/images/train")
train_labels = Path("dataset/labels/train")

val_images = Path("dataset/images/val")
val_labels = Path("dataset/labels/val")

val_images.mkdir(parents=True, exist_ok=True)
val_labels.mkdir(parents=True, exist_ok=True)

images = sorted(train_images.rglob("India_*.jpg"))[:100]

print("Moving", len(images), "images to validation set...")

moved = 0

for image in images:

    relative_path = image.relative_to(train_images)

    label = train_labels / relative_path.with_suffix(".txt")

    if not label.exists():
        print("Label not found:", label)
        continue

    target_image = val_images / relative_path
    target_label = val_labels / relative_path.with_suffix(".txt")

    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(image), str(target_image))
    shutil.move(str(label), str(target_label))

    moved += 1

print("Moved images:", moved)
print("Validation split completed!")