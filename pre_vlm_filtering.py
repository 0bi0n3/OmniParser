"""Find the time spans of a video that show the Godot editor, before any VLM sees it.

Pipeline (from the original pseudo code):
  1. load the video and sample it uniformly at --fps (default 1), one frame held in
     memory at a time;
  2. run OmniParser (YOLO icon detector + PaddleOCR; Florence captioner off) on each frame;
  3. a frame is VALID iff both preconditions pass:
       - OCR gate:   the Godot menu/workspace words ('Scene', 'Project', 'Debug', 'Editor',
                     'Help', '2D', '3D', 'Script', 'AssetLib') are read in the top
                     --top-frac (10%) of the frame; at least --min-keywords of them
                     (default 9 = all) must be found;
       - count gate: more than --min-elements (100) interactable UI elements
                     (OmniParser boxes with interactivity=True);
     if either fails, the frame is invalid and disregarded;
  4. a segment starts at the first valid frame and ends at the last valid frame before
     the preconditions turn negative.

Outputs, per video, in --out-dir:
  <stem>_segments.json   timestamps of the suitable (valid) segments only.
  <stem>_decisions.json  every sampled frame's decision (keywords found, element counts,
                         which gate failed) and every run of consecutive frames labelled
                         valid / ocr_only / count_only / neither, so the near-miss and
                         false-positive sections can be reviewed and the thresholds re-tuned
                         without re-running OmniParser.

Must run with PYTHONNOUSERSITE=1 in the `omniparser` env (transformers 4.49):
    PYTHONNOUSERSITE=1 python pre_vlm_filtering.py VIDEO.mp4 [VIDEO2.mp4 ...]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from video_editor_view_filter import sample_frames  # noqa: E402

KEYWORDS = ("Scene", "Project", "Debug", "Editor", "Help", "2D", "3D", "Script", "AssetLib")


def parse_frame(image, som_model, args):
    """One OmniParser pass: OCR once, reuse it for both gates. Returns the frame's evidence."""
    from util.utils import check_ocr_box, get_som_labeled_img

    (ocr_text, ocr_bbox), _ = check_ocr_box(
        image, display_img=False, output_bb_format="xyxy", goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": args.ocr_text_threshold}, use_paddleocr=True)
    _, _, elements = get_som_labeled_img(
        image, som_model, BOX_TRESHOLD=args.box_threshold, output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox, caption_model_processor=None, ocr_text=ocr_text,
        use_local_semantics=False, iou_threshold=args.iou_threshold, scale_img=False)

    # OCR gate: words from text boxes whose top edge lies in the top band of the frame.
    # Paddle often returns a whole menu bar as one box and glues neighbours together
    # ("SceneProjectDebug", "EditorHelp", "43DScript"), so match substrings of the band's
    # text with spaces removed, not whole words.
    band = args.top_frac * image.size[1]
    top_text = [txt for txt, box in zip(ocr_text, ocr_bbox) if box[1] < band]
    blob = re.sub(r"\s+", "", " ".join(top_text)).lower()
    found = [k for k in KEYWORDS if k.lower() in blob]

    n_interactable = sum(1 for e in elements if e.get("interactivity"))
    return {"keywords_found": found, "top_text": top_text,
            "n_interactable": n_interactable, "n_elements": len(elements)}


def label(ocr_pass: bool, count_pass: bool) -> str:
    return {(True, True): "valid", (True, False): "ocr_only",
            (False, True): "count_only", (False, False): "neither"}[(ocr_pass, count_pass)]


def runs(frames: list[dict]) -> list[dict]:
    """Group consecutive frames with the same label; start/end are the first/last frame's t."""
    out = []
    for f in frames:
        if out and out[-1]["label"] == f["label"]:
            out[-1]["end"] = f["t"]
            out[-1]["n_frames"] += 1
        else:
            out.append({"label": f["label"], "start": f["t"], "end": f["t"], "n_frames": 1})
    return out


def write_json(path: Path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="+", type=Path, help="input .mp4(s)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "pre_vlm_filtering")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--top-frac", type=float, default=0.10, help="OCR band: top fraction of the frame")
    ap.add_argument("--min-keywords", type=int, default=len(KEYWORDS),
                    help=f"keywords needed in the band (default {len(KEYWORDS)} = all)")
    ap.add_argument("--min-elements", type=int, default=100, help="interactable elements must exceed this")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--box-threshold", type=float, default=0.05)
    ap.add_argument("--iou-threshold", type=float, default=0.7)
    ap.add_argument("--ocr-text-threshold", type=float, default=0.9)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from util.utils import get_yolo_model

    som_model = get_yolo_model(str(REPO / "weights/icon_detect/model.pt")).to(torch.device(args.device))
    params = {k: getattr(args, k) for k in ("fps", "top_frac", "min_keywords", "min_elements",
                                            "box_threshold", "iou_threshold", "ocr_text_threshold")}

    for video in args.videos:
        frames, t0 = [], time.time()
        for t, image in sample_frames(video, args.fps):
            ev = parse_frame(image, som_model, args)
            ocr_pass = len(ev["keywords_found"]) >= args.min_keywords
            count_pass = ev["n_interactable"] > args.min_elements
            frames.append({"t": round(t, 3), **ev, "ocr_pass": ocr_pass, "count_pass": count_pass,
                           "label": label(ocr_pass, count_pass)})
            if len(frames) % 50 == 0:
                print(f"  {video.name}: {len(frames)} frames, t={t:.0f}s, "
                      f"{len(frames) / (time.time() - t0):.2f} frames/s", flush=True)

        all_runs = runs(frames)
        segments = [{"start": r["start"], "end": r["end"]} for r in all_runs if r["label"] == "valid"]
        head = {"video": str(video.resolve()), "params": params, "keywords": list(KEYWORDS)}
        write_json(args.out_dir / f"{video.stem}_segments.json", {**head, "segments": segments})
        tally = {}
        for f in frames:
            tally[f["label"]] = tally.get(f["label"], 0) + 1
        write_json(args.out_dir / f"{video.stem}_decisions.json",
                   {**head, "frame_counts": tally, "runs": all_runs, "frames": frames})
        print(f"{video.name}: {len(frames)} frames {tally} -> {len(segments)} valid segments "
              f"({time.time() - t0:.0f}s) -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
