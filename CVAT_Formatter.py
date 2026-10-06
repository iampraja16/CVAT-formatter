"""
Vehicle plate detection — output format CVAT YOLO 1.1
Siap di-import langsung ke CVAT sebagai pre-annotation.

Struktur output:
  detections/
  ├── obj_train_data/   ← gambar original + label txt (satu folder)
  ├── obj.data
  ├── obj.names
  └── train.txt

Usage: python detect_plates.py [--conf 0.25] [--model best|last]
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

WEIGHTS_DIR = Path(r"D:\PEMROGRAMAN\YOLO-Pipeline\output\vehiclePlateDetection\pretrain\run_001\weights")
IMAGE_DIR   = Path(r"D:\JASAMARGA\Project\Traffic Counter\20260929-08")
OUTPUT_DIR  = IMAGE_DIR / "detections"

# Sesuaikan dengan class di model — cek obj.names dataset asli
CLASS_NAMES = ["Car", "Bus", "Minibus", "Truck", "Plate"]


def run(model_name: str = "best", conf: float = 0.25):
    model = YOLO(WEIGHTS_DIR / f"{model_name}.pt")

    obj_dir = OUTPUT_DIR / "obj_train_data"
    obj_dir.mkdir(parents=True, exist_ok=True)

    images = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
    print(f"Found {len(images)} images. Output: {OUTPUT_DIR}")

    results = model.predict(source=images, conf=conf, save=False, save_txt=False)

    train_paths = []
    for result, img_path in zip(results, images):
        # Salin gambar original (tanpa bounding box)
        shutil.copy2(img_path, obj_dir / img_path.name)

        # Tulis label: class cx cy w h (tanpa conf, sesuai format CVAT)
        label_file = obj_dir / (img_path.stem + ".txt")
        with open(label_file, "w") as f:
            if result.boxes is not None and len(result.boxes):
                for box in result.boxes:
                    cls          = int(box.cls[0])
                    cx, cy, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        train_paths.append(f"data/obj_train_data/{img_path.name}")

    # obj.names
    (OUTPUT_DIR / "obj.names").write_text("\n".join(CLASS_NAMES) + "\n")

    # obj.data
    (OUTPUT_DIR / "obj.data").write_text(
        f"classes = {len(CLASS_NAMES)}\n"
        f"train = data/train.txt\n"
        f"names = data/obj.names\n"
        f"backup = backup/\n"
    )

    # train.txt
    (OUTPUT_DIR / "train.txt").write_text("\n".join(train_paths) + "\n")

    print(f"Done. {len(images)} images. Zip folder '{OUTPUT_DIR.name}' lalu import ke CVAT.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best", choices=["best", "last"])
    parser.add_argument("--conf",  default=0.25, type=float)
    args = parser.parse_args()
    run(args.model, args.conf)
