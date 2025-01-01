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
            # A.CoarseDropout(
            #     max_holes=4,
            #     max_height=10,
            #     max_width=10,
            #     min_height=5,
            #     min_width=5,
            #     fill_value=0,
            #     p=0.5,
            # ),
            # A.RandomBrightnessContrast(p=0.5),
            # A.RandomGamma(p=0.5),
            # A.GaussNoise(p=0.3),
            # A.RandomRotate90(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.3),
            # A.Blur(blur_limit=3, p=0.3),
            # A.OneOf(
            #     [
            #         A.RandomRain(p=1),
            #         A.RandomSunFlare(p=1),
            #         A.RandomShadow(p=1),
            #     ],
            #     p=0.3,
            # ),
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

def augment_dataset(input_dir, output_dir, annotations_file, num_augmentations=200):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    coco_data = load_coco_annotation(annotations_file)

    image_annotations = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    new_images = []
    new_annotations = []
    ann_id = max(ann["id"] for ann in coco_data["annotations"]) + 1

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
                output_path = os.path.join(output_dir, "images", new_filename)
                cv2.imwrite(output_path, aug_image)

                new_img_info = {
                    "id": len(new_images) + len(coco_data["images"]),
                    "file_name": new_filename,
                    "height": aug_image.shape[0],
                    "width": aug_image.shape[1],
                }
                new_images.append(new_img_info)

                for bbox, category_id in zip(aug_bboxes, aug_category_ids):
                    new_ann = {
                        "id": ann_id,
                        "image_id": new_img_info["id"],
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                    new_annotations.append(new_ann)
                    ann_id += 1
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

    augmented_dataset = {
        "images": coco_data["images"] + new_images,
        "annotations": coco_data["annotations"] + new_annotations,
        "categories": coco_data["categories"],
    }

    save_coco_annotation(augmented_dataset, os.path.join(output_dir, "annotations.json"))

def load_coco_annotation(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def save_coco_annotation(annotation, output_path):
    with open(output_path, "w") as f:
        json.dump(annotation, f, indent=2)

if __name__ == "__main__":
    input_dir = "annotated_dataset/train"
    output_dir = "augmented_dataset_3"
    annotations_file = "annotated_dataset/train/_annotations.coco.json"

    augment_dataset(input_dir, output_dir, annotations_file)
