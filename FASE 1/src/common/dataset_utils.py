import os
from pathlib import Path

def inspect_dataset_structure(data_dir="../../data"):
    for split in ['train', 'val', 'test']:
        split_dir = Path(data_dir) / split
        if split_dir.exists():
            classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
            total_images = sum(len(list((split_dir / cls).glob('*.*'))) for cls in classes)
            print(f"Split '{split}': {len(classes)} clases | Total imágenes: {total_images}")

if __name__ == "__main__":
    inspect_dataset_structure()
