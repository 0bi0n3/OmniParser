"""Find the time spans of a video where the editor view is on screen.

Samples the video uniformly at --fps (default 4), runs OmniParser on each sampled
frame, and treats a frame as "editor view" when it has more than --min-elements
(default 100) detected UI elements. Consecutive passing frames are grouped into one
segment: start = timestamp of the first passing frame, end = timestamp of the last
passing frame before the count drops back to --min-elements or below.

Output: <out-dir>/<video_stem>_editor_segments.json with the segments plus the
per-frame element counts (so the threshold can be re-tuned without re-running).

Captioning (Florence2) is skipped: only the element count is needed, and captions
do not change the number of boxes.

Must run with PYTHONNOUSERSITE=1 in the `omniparser` env (transformers 4.49).
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def sample_frames(video_path, fps):
    """Yield (timestamp_seconds, PIL.Image) sampled uniformly at `fps`."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = n_frames / src_fps
    t = 0.0
    k = 0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        yield t, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        k += 1
        t = k / fps
    cap.release()


def count_elements(image, som_model, args):
    from util.utils import check_ocr_box, get_som_labeled_img

    (ocr_text, ocr_bbox), _ = check_ocr_box(
        image, display_img=False, output_bb_format="xyxy", goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": args.ocr_text_threshold}, use_paddleocr=True)
    _, _, elements = get_som_labeled_img(
        image, som_model, BOX_TRESHOLD=args.box_threshold, output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox, caption_model_processor=None, ocr_text=ocr_text,
        use_local_semantics=False, iou_threshold=args.iou_threshold, scale_img=False)
    return len(elements)


def find_segments(frames, min_elements):
    """Group consecutive frames with n_elements > min_elements into segments."""
    segments, start, last = [], None, None
    for f in frames:
        if f["n_elements"] > min_elements:
            if start is None:
                start = f["t"]
            last = f["t"]
        elif start is not None:
            segments.append({"start": start, "end": last})
            start = None
    if start is not None:
        segments.append({"start": start, "end": last})
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path, help="input .mp4")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--min-elements", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--box-threshold", type=float, default=0.05)
    ap.add_argument("--iou-threshold", type=float, default=0.7)
    ap.add_argument("--ocr-text-threshold", type=float, default=0.9)
    args = ap.parse_args()

    import torch
    from util.utils import get_yolo_model

    som_model = get_yolo_model(str(REPO / "weights/icon_detect/model.pt")).to(torch.device(args.device))

    frames = []
    t_start = time.time()
    for t, image in sample_frames(args.video, args.fps):
        n = count_elements(image, som_model, args)
        frames.append({"t": round(t, 3), "n_elements": n})
        if len(frames) % 50 == 0:
            print(f"{len(frames)} frames, t={t:.1f}s, {len(frames) / (time.time() - t_start):.2f} frames/s", flush=True)

    segments = find_segments(frames, args.min_elements)
    record = {
        "video": str(args.video.resolve()),
        "sample_fps": args.fps,
        "min_elements": args.min_elements,
        "params": {"box_threshold": args.box_threshold, "iou_threshold": args.iou_threshold,
                   "ocr_text_threshold": args.ocr_text_threshold},
        "segments": segments,
        "frames": frames,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.video.stem}_editor_segments.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.rename(out_path)
    print(f"{len(segments)} segments from {len(frames)} frames -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
