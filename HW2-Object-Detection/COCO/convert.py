import json
from pathlib import Path

def generate_coco_format(mode="train"):
    image_width = 1920
    image_height = 1080

    # Start building the COCO dataset structure.
    coco_format = {
        "images": [],
        "annotations": [],
    }

    coco_format["categories"] = [{
        "id": 0,
        "name": "car"
    }]

    # Assuming you have the same structure of the dataset.
    dataset_path = Path(".")
    images_path = dataset_path / f"{mode}2017"
    annotations_path = dataset_path / f"{mode}_labels"

    # Add image and annotation information
    for image_path in images_path.glob("*.jpg"):
        image_id = int(image_path.stem)  # Assumes filename is the ID

        # Add image information
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": image_width,    # You need to get the image width
            "height": image_height   # You need to get the image height
        })

        # Read the YOLO format annotations
        annotation_path = annotations_path / f"{image_path.stem}.txt"
        with annotation_path.open() as file:
            for i, line in enumerate(file, 1):
                class_id, x_center, y_center, width, height = map(float, line.split())

                # Convert to absolute coordinates and then to COCO format
                x_center_abs = x_center * image_width
                y_center_abs = y_center * image_height
                width_abs = width * image_width
                height_abs = height * image_height

                x_min = x_center_abs - (width_abs / 2)
                y_min = y_center_abs - (height_abs / 2)

                annotation_item = {
                    "id": len(coco_format["annotations"]) + 1,
                    "image_id": int(image_id),
                    "category_id": int(class_id),
                    "bbox": [int(x_min), int(y_min), int(width_abs), int(height_abs)],
                    "area": int(width_abs * height_abs),
                    "iscrowd": 0,
                    "segmentation": []  # If you have segmentation info, include it here
                }
                coco_format["annotations"].append(annotation_item)

    # Save the COCO data to a file
    with open(f"annotations/instances_{mode}2017.json", 'w') as outfile:
        json.dump(coco_format, outfile, indent=4)

if __name__ == "__main__":
    generate_coco_format(mode="train")
    generate_coco_format(mode="val")
