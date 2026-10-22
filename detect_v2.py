#!/usr/bin/env python3
"""Detect and track objects in a video feed: vehicles, people, animals, buildings,
trees and electronics (cameras, computers, phones).

Uses an open-vocabulary YOLO-World model by default, so any object you can name in
text can be detected (add your own with --extra). Works with a video file, a webcam
index (0, 1, ...), or a stream URL (rtsp://, rtmp://, http://, udp://). Draws labeled
boxes with track IDs, shows live per-category counts, and can save the annotated
video and a CSV log.

Examples:
    python detect.py --source 0 --show
    python detect.py --source flight.mp4 --save out.mp4 --csv log.csv
    python detect.py --source 0 --show --extra "drone,backpack,traffic cone"
    python detect.py --source 0 --show --model yolo11n.pt   # faster, COCO classes only

Live controls (with --show, click the window first):
    1-7         toggle a category on/off (numbers shown in the panel)
    + / -       raise / lower confidence threshold by 0.05
    [ / ]       shrink / enlarge the corner stats panel
    t           toggle tracking on/off
    s           save a snapshot PNG
    space       pause / resume
    h           hide / show the help line
    q           quit
"""
import argparse
import csv
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

# Category -> text prompts. With an open-vocabulary model every prompt is detectable;
# with a COCO model (yolo11*.pt) only prompts matching COCO class names are used.
CATEGORIES = {
    "vehicle": ["car", "truck", "bus", "motorcycle"],
    "person": ["person"],
    "animal": ["dog", "cat", "bird", "horse", "cow", "sheep", "deer"],
    "building": ["building", "house"],
    "tree": ["tree"],
    "electronics": ["camera", "laptop", "computer", "cell phone"],
}
COLORS = {  # BGR
    "vehicle": (255, 140, 0), "person": (0, 200, 0), "animal": (0, 120, 255),
    "building": (180, 105, 255), "tree": (34, 139, 34), "electronics": (0, 215, 255),
    "custom": (255, 0, 255),
}
HELP = "1-7 toggle  +/- conf  [/] panel  t track  s snap  space pause  h help  q quit"


def parse_source(src):
    return int(src) if src.isdigit() else src


def is_open_vocab(model_path):
    name = str(model_path).lower()
    return "world" in name or "yoloe" in name


def setup_model(model_path, categories):
    """Load the model and return (model, {class_id: category})."""
    model = YOLO(model_path)
    if is_open_vocab(model_path):
        prompts, cats = [], []
        for cat, names in categories.items():
            for n in names:
                prompts.append(n)
                cats.append(cat)
        model.set_classes(prompts)
        return model, dict(enumerate(cats))
    by_name = {n: i for i, n in model.names.items()}
    mapping, missing = {}, []
    for cat, names in categories.items():
        for n in names:
            if n in by_name:
                mapping[by_name[n]] = cat
            else:
                missing.append(n)
    if missing:
        print(f"Note: {model_path} can't detect {', '.join(missing)}. "
              "Use a YOLO-World model (default) for those.")
    return model, mapping


def draw_box(frame, xyxy, label, color, scale):
    x1, y1, x2, y2 = map(int, xyxy)
    thick = max(2, round(2 * scale))
    font, fthick = 0.5 * scale, max(1, round(scale))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, fthick)
    pad = round(3 * scale)
    ty = max(y1, th + 2 * pad)
    cv2.rectangle(frame, (x1, ty - th - 2 * pad), (x1 + tw + 2 * pad, ty), color, -1)
    cv2.putText(frame, label, (x1 + pad, ty - pad), cv2.FONT_HERSHEY_SIMPLEX, font,
                (255, 255, 255), fthick, cv2.LINE_AA)


def draw_hud(frame, cat_order, counts, unique, fps, enabled, conf, tracking, show_help, scale):
    lines = [f"FPS {fps:.1f}   conf {conf:.2f}   track {'on' if tracking else 'off'}"] + [
        f"{i}. {cat}: {counts.get(cat, 0)} now / {len(unique[cat])} total"
        + ("" if cat in enabled else "  (off)")
        for i, cat in enumerate(cat_order, 1)
    ]
    if show_help:
        lines.append(HELP)
    font, thick, line_h = 0.55 * scale, max(1, round(scale)), round(22 * scale)
    pad = round(8 * scale)
    panel_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font, thick)[0][0] for t in lines) + 2 * pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, pad + len(lines) * line_h + pad // 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + (i + 1) * line_h - round(6 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, webcam index, or stream URL")
    ap.add_argument("--model", default="yolov8s-worldv2.pt",
                    help="yolov8s/m/l-worldv2.pt (any object by name) or yolo11n/s/m.pt (faster, COCO only)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size; try 1280 for high-altitude drone footage with small objects")
    ap.add_argument("--device", default=None, help="cpu, 0 (CUDA GPU), or mps (Apple Silicon)")
    ap.add_argument("--categories", default=",".join(CATEGORIES),
                    help=f"comma-separated categories to start with on: {', '.join(CATEGORIES)}")
    ap.add_argument("--extra", default="",
                    help='extra objects to detect as a "custom" category, e.g. "drone,backpack"')
    ap.add_argument("--no-track", action="store_true", help="disable tracking (no IDs or unique counts)")
    ap.add_argument("--hud-scale", type=float, default=2.0,
                    help="size of the corner stats panel (1 = small, 2 = default)")
    ap.add_argument("--show", action="store_true", help="show a live window (press q to quit)")
    ap.add_argument("--save", help="path for annotated output video, e.g. out.mp4")
    ap.add_argument("--csv", help="path for a per-detection CSV log")
    args = ap.parse_args()

    categories = dict(CATEGORIES)
    extra = [e.strip() for e in args.extra.split(",") if e.strip()]
    if extra:
        categories["custom"] = extra
    cat_order = list(categories)
    enabled = {c.strip() for c in args.categories.split(",") if c.strip()} | ({"custom"} if extra else set())

    model, class_to_cat = setup_model(args.model, categories)
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    box_scale = max(1.0, h / 720)  # keep box labels readable on 1080p+ feeds

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
    log_file = open(args.csv, "w", newline="") if args.csv else None
    log = csv.writer(log_file) if log_file else None
    if log:
        log.writerow(["frame", "time_s", "track_id", "category", "class", "conf", "x1", "y1", "x2", "y2"])

    conf, tracking, paused, show_help = args.conf, not args.no_track, False, True
    hud_scale = args.hud_scale
    unique = defaultdict(set)
    frame_idx, fps, t_prev = 0, 0.0, time.time()
    try:
        while True:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key == ord("q"):
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            class_ids = sorted(cid for cid, cat in class_to_cat.items() if cat in enabled)
            counts = defaultdict(int)
            if class_ids:
                kwargs = dict(classes=class_ids, conf=conf, imgsz=args.imgsz,
                              device=args.device, verbose=False)
                if tracking:
                    result = model.track(frame, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
                else:
                    result = model.predict(frame, **kwargs)[0]
                boxes = result.boxes
                ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
                for xyxy, cls, score, tid in zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(),
                                                 boxes.conf.tolist(), ids):
                    cat = class_to_cat[cls]
                    name = model.names[cls]
                    counts[cat] += 1
                    if tid is not None:
                        unique[cat].add(tid)
                    label = f"{name} {score:.2f}" + (f" #{tid}" if tid is not None else "")
                    draw_box(frame, xyxy, label, COLORS.get(cat, (255, 255, 255)), box_scale)
                    if log:
                        log.writerow([frame_idx, round(frame_idx / src_fps, 3), tid, cat, name,
                                      round(score, 3), *[round(v) for v in xyxy]])

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(now - t_prev, 1e-6))
            t_prev = now
            draw_hud(frame, cat_order, counts, unique, fps, enabled, conf, tracking,
                     show_help and args.show, hud_scale)

            if writer:
                writer.write(frame)
            if args.show:
                cv2.imshow("detections", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif ord("1") <= key <= ord("9") and key - ord("1") < len(cat_order):
                    enabled ^= {cat_order[key - ord("1")]}
                elif key in (ord("+"), ord("=")):
                    conf = min(0.95, round(conf + 0.05, 2))
                elif key in (ord("-"), ord("_")):
                    conf = max(0.05, round(conf - 0.05, 2))
                elif key == ord("["):
                    hud_scale = max(0.5, hud_scale - 0.25)
                elif key == ord("]"):
                    hud_scale = min(5.0, hud_scale + 0.25)
                elif key == ord("t"):
                    tracking = not tracking
                elif key == ord("h"):
                    show_help = not show_help
                elif key == ord(" "):
                    paused = True
                elif key == ord("s"):
                    snap = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(snap, frame)
                    print(f"Saved {snap}")
            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames.")
    for cat in cat_order:
        print(f"  unique {cat} tracked: {len(unique[cat])}")


if __name__ == "__main__":
    main()
