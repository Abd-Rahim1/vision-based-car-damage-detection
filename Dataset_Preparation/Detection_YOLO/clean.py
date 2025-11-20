import json
from pathlib import Path
import yaml

def valid_bbox(bbox, min_area, min_dim, max_ratio):
    _, _, w, h = bbox
    if w < min_dim or h < min_dim:
        return False
    if w * h < min_area:
        return False
    if max(w/h, h/w) > max_ratio:
        return False
    return True

def clean_mixed_dataset(folder_path, json_folder, params):
    folder_path = Path(folder_path).resolve()
    json_folder = Path(json_folder).resolve()

    stats_total = {"images": 0, "removed_images": 0, "annotations": 0, "removed_annotations": 0}
    splits = ["train2017", "val2017", "test2017"]

    for split in splits:
        json_file = json_folder / f"instances_{split}.json"
        if not json_file.exists():
            continue

        with open(json_file) as f:
            coco = json.load(f)

        # Map image_id → file_name
        img_id_to_name = {img['id']: img['file_name'] for img in coco['images']}
        cat_id_to_index = {cat['id']: i for i, cat in enumerate(coco['categories'])}
        annos_per_image = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in annos_per_image:
                annos_per_image[img_id] = []
            annos_per_image[img_id].append(ann)

        for img_id, file_name in img_id_to_name.items():
            img_path = folder_path / file_name
            if not img_path.exists():
                continue

            txt_lines = []
            anns = annos_per_image.get(img_id, [])
            for ann in anns:
                x, y, w, h = ann['bbox']
                if not valid_bbox((0, 0, w, h), params['min_bbox_area'], params['min_bbox_dimension'], params['max_bbox_aspect_ratio']):
                    stats_total['removed_annotations'] += 1
                    continue

                # Normalize bbox (YOLO format)
                img_w, img_h = ann.get('width', 1), ann.get('height', 1)
                x_c = (x + w / 2) / img_w
                y_c = (y + h / 2) / img_h
                w_n = w / img_w
                h_n = h / img_h
                txt_lines.append(f"{cat_id_to_index[ann['category_id']]} {x_c} {y_c} {w_n} {h_n}")
                stats_total['annotations'] += 1

            if txt_lines:
                # Save YOLO annotation
                out_txt = folder_path / f"{Path(file_name).stem}.txt"
                out_txt.write_text("\n".join(txt_lines))
                stats_total['images'] += 1
            else:
                # Remove image if no valid annotations remain
                img_path.unlink(missing_ok=True)
                stats_total['removed_images'] += 1

    # Print summary
    print("\nCleaning completed!")
    print(f"Removed images: {stats_total['removed_images']}")
    print(f"Removed annotations: {stats_total['removed_annotations']}")
    print(f"Total images kept: {stats_total['images']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to mixed filtered images folder")
    parser.add_argument("--json_folder", required=True, help="Path to COCO JSON annotations folder")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)["data_preparation"]["cleaning"]

    clean_mixed_dataset(args.folder, args.json_folder, params)