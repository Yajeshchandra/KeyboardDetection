import albumentations as A
from albumentations.core.bbox_utils import denormalize_bboxes
import cv2
import os
import json
from glob import glob
from tqdm import tqdm

def create_augmentation_pipeline(image_height, image_width):
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.OneOf(
                [
                    A.RandomRain(p=1),
                    A.RandomSunFlare(p=1),
                    A.RandomShadow(p=1),
                ],
                p=0.3,
            ),
            A.CoarseDropout(
                max_holes=4,
                max_height=min(100, image_height // 4),
                max_width=min(100, image_width // 4),
                min_height=20,
                min_width=20,
                mask_fill_value=0,
                p=0.5,
            ),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
    )

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    return cv2.imread(image_path)

def coco_to_yolo_bbox(bbox, image_width, image_height):
    """Convert COCO bbox [x_min, y_min, width, height] to YOLO format [x_center, y_center, width, height]."""
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / image_width
    y_center = (y_min + height / 2) / image_height
    width /= image_width
    height /= image_height
    return [x_center, y_center, width, height]

def augment_dataset(input_dir, output_dir, annotations_file, num_augmentations=1000):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    coco_data = load_coco_annotation(annotations_file)

    image_annotations = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    for img_info in tqdm(coco_data["images"], desc="Augmenting images"):
        img_path = os.path.join(input_dir, img_info["file_name"])
        image = load_image(img_path)

        if image is None:
            continue

        image_height, image_width = image.shape[:2]
        transform = create_augmentation_pipeline(image_height, image_width)
        anns = image_annotations.get(img_info["id"], [])

        for aug_idx in range(num_augmentations):
            bboxes = [ann["bbox"] for ann in anns]
            category_ids = [ann["category_id"] for ann in anns]

            try:
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids,
                )

                aug_image = transformed["image"]
                aug_bboxes = transformed["bboxes"]
                aug_category_ids = transformed["category_ids"]

                new_filename = f"aug_{img_info['file_name'].split('.')[0]}_{aug_idx}.jpg"
                output_image_path = os.path.join(output_dir, "images", new_filename)
                cv2.imwrite(output_image_path, aug_image)

                # Write YOLO annotations
                label_path = os.path.join(output_dir, "labels", f"{new_filename.split('.')[0]}.txt")
                with open(label_path, "w") as f:
                    for bbox, category_id in zip(aug_bboxes, aug_category_ids):
                        yolo_bbox = coco_to_yolo_bbox(bbox, image_width, image_height)
                        yolo_line = f"{category_id - 1} " + " ".join(map(str, yolo_bbox)) + "\n"
                        f.write(yolo_line)

                # Make sure labels for occluded objects are still saved (even if their bbox is masked)
                # If any bbox was not augmented (like being occluded by the dropout), we still save it
                if len(aug_bboxes) < len(bboxes):
                    for idx in range(len(bboxes)):
                        if bboxes[idx] not in aug_bboxes:
                            # Save the label for this object
                            yolo_bbox = coco_to_yolo_bbox(bboxes[idx], image_width, image_height)
                            with open(label_path, "a") as f:
                                yolo_line = f"{category_ids[idx] - 1} " + " ".join(map(str, yolo_bbox)) + "\n"
                                f.write(yolo_line)

            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

def load_coco_annotation(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    input_dir = "annotated_dataset/train"
    output_dir = "augmented_dataset_yolo_2"
    annotations_file = "annotated_dataset/train/_annotations.coco.json"

    augment_dataset(input_dir, output_dir, annotations_file)
