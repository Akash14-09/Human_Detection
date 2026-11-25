# 🌊 Underwater Human Detection System

Deep-learning (YOLOv8) human / diver / body detection with an underwater
image-enhancement pipeline, built for search-and-rescue submarine/ROV vision.

## Features
- YOLOv8 detection (n / s / m) or your own fine-tuned `best.pt`
- Underwater enhancement: gray-world colour correction, dark-channel dehazing, CLAHE
- "Simulate underwater" mode to test on ordinary photos
- Image, camera snapshot and video processing
- Alert banner, region localisation (e.g. *lower-left*), confidence + FPS metrics
- Session dashboard with CSV export
- `train.py` for fine-tuning on an underwater dataset

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (`app.py`, `requirements.txt`, `packages.txt` at the root).
2. Go to https://share.streamlit.io -> **New app** -> select repo, branch, `app.py`.
3. Deploy (first run downloads the YOLO weights automatically).

## Fine-tune on underwater data
```bash
python train.py --data path/to/data.yaml --epochs 50
```
Upload `runs/detect/underwater_human/weights/best.pt` in the sidebar -> *Custom (.pt upload)*.

## Project structure
```
app.py            Streamlit application
train.py          Fine-tuning script
requirements.txt  Python dependencies
packages.txt      System packages for Streamlit Cloud
```

## Tech stack
Python, PyTorch, Ultralytics YOLOv8, OpenCV, NumPy, Pandas, Streamlit
