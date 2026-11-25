"""
Underwater Human Detection System
=================================
Deep-learning (YOLOv8) based human / diver / body detection for underwater
search-and-rescue imagery, with an underwater image-enhancement pipeline.

Run locally:   streamlit run app.py
"""

import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Underwater Human Detection",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.1rem; font-weight: 700; margin-bottom: 0;}
    .sub-title  {opacity: .75; margin-top: 0; margin-bottom: 1rem;}
    .alert-box  {padding: 14px 18px; border-radius: 10px; font-weight: 600;
                 background: rgba(220, 38, 38, .15); border: 1px solid #dc2626;}
    .ok-box     {padding: 14px 18px; border-radius: 10px; font-weight: 600;
                 background: rgba(22, 163, 74, .15); border: 1px solid #16a34a;}
    </style>
    """,
    unsafe_allow_html=True,
)

HUMAN_KEYWORDS = ("person", "human", "diver", "body", "swimmer", "people", "victim")
ENHANCE_OPTIONS = [
    "None",
    "CLAHE (contrast)",
    "Colour correction (gray-world + CLAHE)",
    "Dehaze (dark channel prior)",
    "Full pipeline (colour + dehaze + CLAHE)",
]


# ----------------------------------------------------------------------------
# Image enhancement (helps with the typical blue-green cast, haze & low contrast)
# ----------------------------------------------------------------------------
def apply_clahe(img_bgr, clip=2.5, grid=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gray_world(img_bgr):
    """Gray-world white balance; restores the red channel lost with depth."""
    f = img_bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    gain = np.clip(gain, 0.6, 3.0)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def dehaze(img_bgr, omega=0.85, t0=0.25, size=15):
    """Simplified dark-channel-prior dehazing."""
    img = img_bgr.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(img.min(axis=2), kernel)
    n = max(int(dark.size * 0.001), 1)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    atmos = img.reshape(-1, 3)[idx].mean(axis=0)
    norm = img / np.maximum(atmos, 1e-6)
    t = 1.0 - omega * cv2.erode(norm.min(axis=2), kernel)
    t = np.clip(t, t0, 1.0)
    out = (img - atmos) / t[..., None] + atmos
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def enhance(img_bgr, method):
    if method == "CLAHE (contrast)":
        return apply_clahe(img_bgr)
    if method == "Colour correction (gray-world + CLAHE)":
        return apply_clahe(gray_world(img_bgr))
    if method == "Dehaze (dark channel prior)":
        return dehaze(img_bgr)
    if method == "Full pipeline (colour + dehaze + CLAHE)":
        return apply_clahe(dehaze(gray_world(img_bgr)))
    return img_bgr


def simulate_underwater(img_bgr, strength=1.0):
    """Make a normal photo look underwater (for testing without real footage)."""
    f = img_bgr.astype(np.float32)
    f[..., 2] *= 1.0 - 0.65 * strength   # red is absorbed first
    f[..., 1] *= 1.0 - 0.20 * strength
    f = cv2.GaussianBlur(f, (0, 0), 1.2 * strength + 0.1)
    haze = np.array([150, 150, 60], dtype=np.float32)  # BGR blue-green water
    alpha = 0.30 * strength
    f = (1 - alpha) * f + alpha * haze
    f += np.random.normal(0, 4 * strength, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Model handling
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return YOLO(path)


def target_class_ids(model, only_humans):
    """Return class ids to keep (None = keep everything)."""
    if not only_humans:
        return None
    ids = [
        i for i, n in model.names.items()
        if any(k in str(n).lower() for k in HUMAN_KEYWORDS)
    ]
    return ids if ids else None


def region_of(box, w, h):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    col = ["left", "centre", "right"][min(int(cx / w * 3), 2)]
    row = ["upper", "middle", "lower"][min(int(cy / h * 3), 2)]
    return f"{row}-{col}"


def draw_detections(img_bgr, result, names, color=(0, 255, 0)):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    dets = []
    if result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        thick = max(2, w // 400)
        scale = max(0.5, w / 1400)
        for box, c, k in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [int(v) for v in box]
            label = f"{names[k]} {c:.0%}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            top = max(y1 - th - 10, 0)
            cv2.rectangle(out, (x1, top), (x1 + tw + 8, top + th + 10), color, -1)
            cv2.putText(out, label, (x1 + 4, top + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
            dets.append({
                "class": names[k],
                "confidence": float(c),
                "region": region_of((x1, y1, x2, y2), w, h),
                "box": (x1, y1, x2, y2),
            })
    return out, dets


def run_pipeline(frame_bgr, model, cfg):
    """simulate -> enhance -> detect -> annotate."""
    base = simulate_underwater(frame_bgr, cfg["sim_strength"]) if cfg["simulate"] else frame_bgr
    enhanced = enhance(base, cfg["enhance"])
    t0 = time.time()
    result = model.predict(
        enhanced,
        conf=cfg["conf"],
        iou=cfg["iou"],
        imgsz=cfg["imgsz"],
        classes=cfg["classes"],
        verbose=False,
    )[0]
    ms = (time.time() - t0) * 1000
    annotated, dets = draw_detections(enhanced, result, model.names)
    return base, enhanced, annotated, dets, ms


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def log_event(source, dets, ms, extra=None):
    confs = [d["confidence"] for d in dets]
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "humans_detected": len(dets),
        "max_conf": round(max(confs), 3) if confs else 0.0,
        "avg_conf": round(float(np.mean(confs)), 3) if confs else 0.0,
        "inference_ms": round(ms, 1),
    }
    if extra:
        entry.update(extra)
    st.session_state.setdefault("log", []).append(entry)


def show_alert(dets):
    if dets:
        regions = ", ".join(sorted({d["region"] for d in dets}))
        best = max(d["confidence"] for d in dets)
        st.markdown(
            f'<div class="alert-box">🚨 ALERT: {len(dets)} possible human(s) detected '
            f"(max confidence {best:.0%}) — region: {regions}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="ok-box">✅ No human detected in this frame.</div>',
                    unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.markdown('<p class="main-title">🌊 Underwater Human Detection System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">YOLOv8 deep-learning detector with underwater image enhancement '
    "for search-and-rescue vehicles</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Model")
    model_choice = st.selectbox(
        "Detection model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "Custom (.pt upload)"],
        help="n = fastest, m = most accurate. Upload your own fine-tuned underwater model "
             "for best results.",
    )
    model_path = model_choice
    if model_choice == "Custom (.pt upload)":
        up = st.file_uploader("Upload YOLOv8 weights", type=["pt"])
        if up is None:
            st.info("Upload your trained `best.pt` to continue.")
            st.stop()
        tmp = os.path.join(tempfile.gettempdir(), up.name)
        with open(tmp, "wb") as f:
            f.write(up.getbuffer())
        model_path = tmp

    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
    iou = st.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05)
    imgsz = st.select_slider("Inference size", options=[320, 480, 640, 800, 960, 1280], value=640)
    only_humans = st.checkbox("Detect humans only", value=True,
                              help="Filters to person/diver/body classes.")

    st.header("🌫️ Underwater processing")
    enhance_method = st.selectbox("Enhancement", ENHANCE_OPTIONS, index=4)
    simulate = st.checkbox("Simulate underwater conditions", value=False,
                           help="Applies blue-green tint, haze, blur and noise to normal photos "
                                "so you can test the pipeline without real footage.")
    sim_strength = st.slider("Simulation strength", 0.2, 1.5, 1.0, 0.1, disabled=not simulate)

model = load_model(model_path)
cfg = {
    "conf": conf,
    "iou": iou,
    "imgsz": imgsz,
    "classes": target_class_ids(model, only_humans),
    "enhance": enhance_method,
    "simulate": simulate,
    "sim_strength": sim_strength,
}

tab_img, tab_vid, tab_dash, tab_about = st.tabs(
    ["🖼️ Image", "🎥 Video", "📊 Dashboard", "ℹ️ About"]
)

# ----------------------------------------------------------------------------
# Tab 1: Image
# ----------------------------------------------------------------------------
with tab_img:
    src = st.radio("Input source", ["Upload image", "Camera snapshot"], horizontal=True)
    file = None
    if src == "Upload image":
        file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                key="img_up")
    else:
        file = st.camera_input("Take a picture")

    if file is not None:
        pil = Image.open(file).convert("RGB")
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        with st.spinner("Running detection..."):
            base, enhanced, annotated, dets, ms = run_pipeline(frame, model, cfg)

        show_alert(dets)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Humans detected", len(dets))
        c2.metric("Max confidence", f"{max([d['confidence'] for d in dets]):.0%}" if dets else "—")
        c3.metric("Inference time", f"{ms:.0f} ms")
        c4.metric("Resolution", f"{frame.shape[1]}×{frame.shape[0]}")

        col_a, col_b, col_c = st.columns(3)
        col_a.image(to_rgb(base), caption="Input" + (" (simulated underwater)" if simulate else ""),
                    use_container_width=True)
        col_b.image(to_rgb(enhanced), caption=f"Enhanced: {enhance_method}",
                    use_container_width=True)
        col_c.image(to_rgb(annotated), caption="Detections", use_container_width=True)

        if dets:
            st.subheader("Detection details")
            df = pd.DataFrame(dets).drop(columns=["box"])
            df["confidence"] = df["confidence"].round(3)
            st.dataframe(df, use_container_width=True)

        ok, buf = cv2.imencode(".png", annotated)
        if ok:
            st.download_button("⬇️ Download annotated image", buf.tobytes(),
                               file_name="detection_result.png", mime="image/png")
        log_event("image", dets, ms)

# ----------------------------------------------------------------------------
# Tab 2: Video
# ----------------------------------------------------------------------------
def reencode_h264(src_path):
    """Browsers can't play mp4v; convert with ffmpeg if available."""
    if shutil.which("ffmpeg") is None:
        return src_path
    dst = src_path.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-loglevel", "error", dst],
            check=True,
        )
        return dst
    except Exception:
        return src_path


with tab_vid:
    vfile = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    v1, v2, v3 = st.columns(3)
    skip = v1.slider("Process every Nth frame", 1, 10, 3)
    max_frames = v2.slider("Max frames to process", 30, 1000, 300, 30)
    max_width = v3.select_slider("Resize width", options=[480, 640, 800, 960, 1280], value=800)

    if vfile is not None and st.button("▶️ Process video", type="primary"):
        suffix = os.path.splitext(vfile.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(vfile.read())
            in_path = tf.name

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
        out_path = os.path.join(tempfile.gettempdir(), f"annotated_{int(time.time())}.mp4")
        writer = None

        preview = st.empty()
        alert_slot = st.empty()
        progress = st.progress(0.0)
        counts, times, confs_all = [], [], []
        idx, processed = 0, 0

        while cap.isOpened() and processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % skip == 0:
                h, w = frame.shape[:2]
                if w > max_width:
                    frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
                _, _, annotated, dets, ms = run_pipeline(frame, model, cfg)
                if writer is None:
                    ph, pw = annotated.shape[:2]
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(fps / skip, 1), (pw, ph))
                cv2.putText(annotated, f"Humans: {len(dets)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                writer.write(annotated)
                preview.image(to_rgb(annotated), use_container_width=True)
                alert_slot.markdown(
                    f'<div class="{"alert-box" if dets else "ok-box"}">'
                    f'{"🚨 Human detected: " + str(len(dets)) if dets else "✅ Clear"} '
                    f"— frame {idx}</div>",
                    unsafe_allow_html=True,
                )
                counts.append(len(dets))
                times.append(ms)
                confs_all.extend([d["confidence"] for d in dets])
                processed += 1
                progress.progress(min(idx / max(total, 1), 1.0))
            idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        progress.progress(1.0)

        if processed == 0:
            st.error("Could not read any frames from this video.")
        else:
            st.success(f"Processed {processed} frames.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Frames with humans", f"{sum(c > 0 for c in counts)}/{processed}")
            m2.metric("Peak humans in a frame", max(counts))
            m3.metric("Avg inference", f"{np.mean(times):.0f} ms")
            m4.metric("Effective FPS", f"{1000 / np.mean(times):.1f}")
            st.line_chart(pd.DataFrame({"humans per processed frame": counts}))

            final = reencode_h264(out_path)
            with open(final, "rb") as f:
                data = f.read()
            if final != out_path:
                st.video(data)
            else:
                st.info("Install ffmpeg to preview the result in-browser; download below.")
            st.download_button("⬇️ Download annotated video", data,
                               file_name="annotated_video.mp4", mime="video/mp4")
            log_event(
                "video", [{"confidence": c} for c in confs_all], float(np.mean(times)),
                extra={"frames": processed, "frames_with_humans": int(sum(c > 0 for c in counts))},
            )

# ----------------------------------------------------------------------------
# Tab 3: Dashboard
# ----------------------------------------------------------------------------
with tab_dash:
    st.subheader("Session detection log")
    log = st.session_state.get("log", [])
    if not log:
        st.info("No detections yet. Run an image or a video first.")
    else:
        df = pd.DataFrame(log)
        d1, d2, d3 = st.columns(3)
        d1.metric("Runs", len(df))
        d2.metric("Total humans detected", int(df["humans_detected"].sum()))
        d3.metric("Avg inference", f"{df['inference_ms'].mean():.0f} ms")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("time")["humans_detected"])
        st.download_button("⬇️ Download log (CSV)", df.to_csv(index=False).encode(),
                           file_name="detection_log.csv", mime="text/csv")
        if st.button("Clear log"):
            st.session_state["log"] = []
            st.rerun()

# ----------------------------------------------------------------------------
# Tab 4: About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### Project overview
Underwater search-and-rescue vehicles need to find people (divers, swimmers, victims)
in poor visibility. This project builds the **vision software module** for such a system.

### Pipeline
1. **Input** – image, camera snapshot or video from the vehicle camera.
2. **Underwater enhancement** – gray-world colour correction (restores lost red),
   dark-channel-prior dehazing and CLAHE contrast enhancement.
3. **Detection** – YOLOv8 (one-stage CNN detector) predicts bounding boxes + confidence.
4. **Post-processing** – class filtering, NMS, region localisation (e.g. *lower-left*).
5. **Alerting & logging** – alert banner, per-run statistics, CSV export.

### Why fine-tuning matters
The default COCO-pretrained weights recognise people in normal scenes. Underwater
colour shifts, turbidity and unusual body poses reduce accuracy, so for real use
**fine-tune on an underwater dataset** with `train.py` and upload `best.pt` via the
sidebar ("Custom (.pt upload)").

### Tech stack
Python · PyTorch · Ultralytics YOLOv8 · OpenCV · NumPy · Pandas · Streamlit

### Limitations
- Detection quality depends heavily on training data and water clarity.
- This is a decision-support tool, not a replacement for trained human operators.
        """
    )
