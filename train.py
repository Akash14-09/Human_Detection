"""
Fine-tune YOLOv8 on an underwater human/diver dataset.

1. Get a dataset in YOLO format (e.g. search "underwater diver" / "swimmer"
   on Roboflow Universe or Kaggle, export as "YOLOv8").
2. Edit DATA_YAML below (or pass --data).
3. Run:  python train.py --data path/to/data.yaml --epochs 50
4. Upload runs/detect/underwater_human/weights/best.pt in the Streamlit sidebar.

Tip: use Google Colab (free GPU) for training.
"""

import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.yaml", help="path to dataset yaml")
    p.add_argument("--model", default="yolov8n.pt", help="starting weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="underwater_human",
        # Augmentations that suit underwater imagery
        hsv_h=0.03, hsv_s=0.6, hsv_v=0.5,
        degrees=10, fliplr=0.5, mosaic=1.0, mixup=0.1,
    )

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
