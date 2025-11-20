import os
import shutil
from tqdm import tqdm
from typing import Dict
from PIL import Image

# Ensure the utils module is accessible
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_config,
    create_directory,
    get_image_dimensions,
    load_coco_annotations,
    save_coco_annotations
)


class DataFilter:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_config(config_path)
        self.filter_config = self.config['data_preparation']['filtering']

    def filter_dataset(self, input_path: str, output_path: str) -> Dict:
        """Filter dataset and copy valid data to output directory"""
        print("Filtering dataset...")

        create_directory(output_path)
        stats = {
            'removed_images': 0,
            'removed_annotations': 0,
            'splits': {}
        }

        # Only COCO dataset for detection
        coco_input_path = os.path.join(input_path, "CarDD_COCO")
        coco_output_path = os.path.join(output_path, "CarDD_COCO")
        if os.path.exists(coco_input_path):
            coco_stats = self._filter_coco_dataset(coco_input_path, coco_output_path)
            stats['removed_images'] += coco_stats['removed_images']
            stats['removed_annotations'] += coco_stats['removed_annotations']
            stats['splits']['coco'] = coco_stats['splits']

        return stats

    def _filter_coco_dataset(self, input_path: str, output_path: str) -> Dict:
        stats = {'removed_images': 0, 'removed_annotations': 0, 'splits': {}}
        splits = ['train2017', 'val2017', 'test2017']

        for split in splits:
            try:
                coco_data = load_coco_annotations(input_path, split)
                input_images_dir = os.path.join(input_path, split)
                output_images_dir = os.path.join(output_path, split)
                create_directory(output_images_dir)
                create_directory(os.path.join(output_path, "annotations"))

                valid_images = []
                removed_images = []

                for image_info in tqdm(coco_data['images'], desc=f"Filtering {split}"):
                    input_image_path = os.path.join(input_images_dir, image_info['file_name'])
                    if self._is_valid_image(input_image_path):
                        shutil.copy2(input_image_path, os.path.join(output_images_dir, image_info['file_name']))
                        valid_images.append(image_info)
                    else:
                        removed_images.append(image_info)

                # Filter annotations
                valid_image_ids = {img['id'] for img in valid_images}
                valid_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in valid_image_ids]
                removed_annotations = len(coco_data['annotations']) - len(valid_annotations)

                # Save updated COCO annotations
                coco_data['images'] = valid_images
                coco_data['annotations'] = valid_annotations
                save_coco_annotations(coco_data, output_path, split)

                # Update stats
                stats['splits'][split] = {
                    'removed_images': len(removed_images),
                    'removed_annotations': removed_annotations,
                    'remaining_images': len(valid_images),
                    'remaining_annotations': len(valid_annotations)
                }

                stats['removed_images'] += len(removed_images)
                stats['removed_annotations'] += removed_annotations

                print(f" {split}: Remaining images = {len(valid_images)}, Remaining annotations = {len(valid_annotations)}, Removed images = {len(removed_images)}, Removed annotations = {removed_annotations}")

            except FileNotFoundError:
                print(f" Skipping {split} - annotations not found")
                continue

        return stats

    def _is_valid_image(self, image_path: str) -> bool:
        try:
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.filter_config['allowed_extensions']:
                return False

            if os.path.getsize(image_path) / (1024 * 1024) > self.filter_config['max_file_size_mb']:
                return False

            width, height = get_image_dimensions(image_path)
            min_w, min_h = self.filter_config['min_image_size']
            if width < min_w or height < min_h:
                return False

            with Image.open(image_path) as img:
                img.verify()  # Check for corruption

            return True
        except Exception as e:
            print(f" Skipping invalid image: {image_path} ({e})")
            return False


def main():
    input_path = r"dataset\CarDD_release"
    output_path = r"dataset\Dataset_Prepared\detection"

    data_filter = DataFilter()
    stats = data_filter.filter_dataset(input_path, output_path)

    print("\nFiltering completed!")
    print(f"Removed images: {stats['removed_images']}")
    print(f"Removed annotations: {stats['removed_annotations']}")
    print(f"Filtered dataset saved to: {output_path}")

    # Print summary per split
    for split, split_stats in stats['splits']['coco'].items():
        print(f"{split}: {split_stats}")


if __name__ == "__main__":
    main()
