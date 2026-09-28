"""Run OmniParser over every pre/mid/post frame of dataset_v2 pair directories.

Output mirrors the source layout:
  <out>/<tutorial>/<pair_dir>/<role>_labelled.png   OmniParser visual output (one per frame in the pair)
  <out>/<tutorial>/<pair_dir>/<role>.json           boxes, OCR text, interactable flags, captions
  <out>/<tutorial>/<pair_dir>/metadata.json         copy of the source pair metadata

Resumable: a frame is skipped when its JSON already exists.
Sharding: --shard i --num-shards N processes every N-th pair directory.

Must run with PYTHONNOUSERSITE=1 in the `omniparser` env (transformers 4.49; Florence2 breaks on 5.x).
"""
import argparse
import base64
import io
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

ROLES = ("pre", "mid", "post")


def list_pair_dirs(src):
    return sorted(p for p in src.glob("*/*__pair_*") if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="max pair dirs (smoke test)")
    ap.add_argument("--box-threshold", type=float, default=0.05)
    ap.add_argument("--iou-threshold", type=float, default=0.7)
    ap.add_argument("--ocr-text-threshold", type=float, default=0.9)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from util.utils import check_ocr_box, get_caption_model_processor, get_som_labeled_img, get_yolo_model

    device = torch.device(args.device)
    som_model = get_yolo_model(str(REPO / "weights/icon_detect/model.pt")).to(device)
    caption = get_caption_model_processor(
        model_name="florence2", model_name_or_path=str(REPO / "weights/icon_caption_florence"), device=device)

    pairs = list_pair_dirs(args.src)[args.shard::args.num_shards]
    if args.limit:
        pairs = pairs[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / f"_log_shard{args.shard}of{args.num_shards}.jsonl", "a")
    params = {k: v for k, v in vars(args).items() if k in ("box_threshold", "iou_threshold", "ocr_text_threshold")}
    print(f"shard {args.shard}/{args.num_shards}: {len(pairs)} pair dirs on {device}", flush=True)

    t_start, n_done = time.time(), 0
    for i, pair in enumerate(pairs):
        dst = args.out / pair.relative_to(args.src)
        dst.mkdir(parents=True, exist_ok=True)
        if (pair / "metadata.json").exists() and not (dst / "metadata.json").exists():
            shutil.copy2(pair / "metadata.json", dst / "metadata.json")

        for role in ROLES:
            img_path = pair / f"{role}.png"
            json_path = dst / f"{role}.json"
            if not img_path.exists() or json_path.exists():
                continue
            t0 = time.time()
            try:
                image = Image.open(img_path).convert("RGB")
                w, h = image.size
                ratio = max(w, h) / 3200
                draw_cfg = {
                    "text_scale": 0.8 * ratio,
                    "text_thickness": max(int(2 * ratio), 1),
                    "text_padding": max(int(3 * ratio), 1),
                    "thickness": max(int(3 * ratio), 1),
                }
                (ocr_text, ocr_bbox), _ = check_ocr_box(
                    image, display_img=False, output_bb_format="xyxy", goal_filtering=None,
                    easyocr_args={"paragraph": False, "text_threshold": args.ocr_text_threshold}, use_paddleocr=True)
                labelled_b64, _, elements = get_som_labeled_img(
                    image, som_model, BOX_TRESHOLD=args.box_threshold, output_coord_in_ratio=True,
                    ocr_bbox=ocr_bbox, draw_bbox_config=draw_cfg, caption_model_processor=caption,
                    ocr_text=ocr_text, use_local_semantics=True, iou_threshold=args.iou_threshold,
                    scale_img=False, batch_size=128)

                Image.open(io.BytesIO(base64.b64decode(labelled_b64))).save(dst / f"{role}_labelled.png")
                out_elements = []
                for idx, e in enumerate(elements):  # idx == label number drawn on the image
                    x0, y0, x1, y1 = [float(v) for v in e["bbox"]]
                    out_elements.append({
                        "id": idx,
                        "type": e["type"],
                        "interactivity": bool(e["interactivity"]),
                        "content": e["content"],
                        "source": e.get("source"),
                        "bbox_xyxy_ratio": [x0, y0, x1, y1],
                        "bbox_xyxy_px": [round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)],
                    })
                record = {
                    "source_image": str(img_path),
                    "role": role,
                    "width": w,
                    "height": h,
                    "params": params,
                    "ocr": [{"text": t, "bbox_xyxy_px": list(map(int, b))} for t, b in zip(ocr_text, ocr_bbox)],
                    "elements": out_elements,
                    "n_elements": len(out_elements),
                    "n_interactable": sum(e["interactivity"] for e in out_elements),
                    "seconds": round(time.time() - t0, 2),
                }
                tmp = json_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(record, indent=1))
                tmp.rename(json_path)  # JSON written last = frame complete
                log.write(json.dumps({"image": str(img_path), "ok": True, "seconds": record["seconds"]}) + "\n")
                n_done += 1
            except Exception as exc:
                log.write(json.dumps({"image": str(img_path), "ok": False, "error": repr(exc),
                                      "trace": traceback.format_exc()}) + "\n")
                print(f"ERROR {img_path}: {exc!r}", flush=True)
            log.flush()

        if (i + 1) % 25 == 0 or i + 1 == len(pairs):
            rate = n_done / max(time.time() - t_start, 1e-9)
            print(f"[shard {args.shard}] {i + 1}/{len(pairs)} pairs, {n_done} frames, {rate:.2f} frames/s", flush=True)


if __name__ == "__main__":
    main()
