"""Derive the controller's decision thresholds from the frozen validation split (GAMP: thresholds
are a versioned configuration item tied to the engine, never a guess).

For every validation frame the FP16 engine is run through the controller's own GPU letterbox;
each ground-truth box gets the best-matching prediction score of its class (IoU >= 0.5).  The
per-class threshold is the score below which fewer than ``--miss-budget`` of ground-truth boxes
would be lost (recall-driven quantile), floored at ``--min-thr``.  The resulting pack-level
verdict rule is then evaluated on the same split: false-reject rate on nominal packs and escape
rate on defective packs.  Output: models/exported/thresholds.json (loaded by controller.py).

    uv run python src/inference/calibrate_thresholds.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from inference.controller import CLASSES, ENGINE_PATH, FRAME_H, FRAME_W, IMGSZ, N_CAVITIES, Policy, TrtDetector, Frame  # noqa: E402

PROJECT_ROOT = SRC.parent
OUT = PROJECT_ROOT / "models" / "exported" / "thresholds.json"


def gt_boxes_letterbox(label_path: Path, scale: float, pad_top: int, pad_left: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [ln.split() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cls = np.array([int(r[0]) for r in rows])
    xywh = np.array([[float(v) for v in r[1:]] for r in rows]) * np.array([FRAME_W, FRAME_H, FRAME_W, FRAME_H])
    xyxy = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1) * scale
    xyxy[:, [0, 2]] += pad_left
    xyxy[:, [1, 3]] += pad_top
    return cls, xyxy


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss-budget", type=float, default=0.001, help="fraction of GT boxes allowed below the threshold (per class)")
    ap.add_argument("--min-thr", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.0, help="out-of-sample margin subtracted from each in-sample quantile: the 0.1%% quantile of a tightly bunched score distribution sits inside its own tail (asset v2 gate, 2026-09-17: fresh-batch misses at 0.906-0.910 against 0.9107)")
    ap.add_argument("--fp-floor-quantile", type=float, default=0.0, help="unused safeguard placeholder")
    ap.add_argument("--n", type=int, default=0, help="limit frames (0 = all validation frames)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from PIL import Image

    val = [Path(l.strip()) for l in (PROJECT_ROOT / "data" / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n:
        val = val[: args.n]
    det = TrtDetector(ENGINE_PATH, Policy(conf_ok=0.0, conf_defect=0.0))     # raw scores; thresholds decided here
    det.warmup()
    best_scores = {c: [] for c in CLASSES}          # best matched prediction score per GT box
    fp_scores = {c: [] for c in CLASSES}            # unmatched prediction scores (conf >= 0.05)
    per_frame = []                                  # (nominal?, gt_cls, pred rows) for the pack-level evaluation
    t0 = time.perf_counter()
    for p in val:
        img = torch.from_numpy(np.array(Image.open(p).convert("RGB"), dtype=np.uint8)).cuda()
        ev = torch.cuda.Event(); ev.record()
        gt_cls, gt_xyxy = gt_boxes_letterbox(p.parent.parent / "labels" / (p.stem + ".txt"), det.scale, det.pad_top, det.pad_left)
        with torch.cuda.stream(det.stream), torch.inference_mode():
            det.stream.wait_event(ev)
            det.letterbox_gpu(img)
            det.ctx.execute_async_v3(det.stream.cuda_stream)
            raw = det.out[0].T.clone()
        det.stream.synchronize()
        raw = raw.cpu().numpy()
        boxes, scores = raw[:, :4], raw[:, 4:]
        pcls, pconf = scores.argmax(1), scores.max(1)
        keep = pconf >= 0.05
        pb = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)[keep]
        pcls, pconf = pcls[keep], pconf[keep]
        matched = np.zeros(len(pconf), bool)
        for gi in range(len(gt_cls)):
            same = pcls == gt_cls[gi]
            if not same.any():
                best_scores[CLASSES[gt_cls[gi]]].append(0.0)
                continue
            ious = iou_matrix(gt_xyxy[gi:gi + 1], pb[same])[0]
            ok = ious >= 0.5
            if ok.any():
                idx = np.where(same)[0][ok]
                best_scores[CLASSES[gt_cls[gi]]].append(float(pconf[idx].max()))
                matched[idx] = True
            else:
                best_scores[CLASSES[gt_cls[gi]]].append(0.0)
        for c in range(len(CLASSES)):
            fp_scores[CLASSES[c]] += pconf[(~matched) & (pcls == c)].tolist()
        per_frame.append((bool((gt_cls == 0).all() and len(gt_cls) == N_CAVITIES), gt_cls, pb, pcls, pconf))
    elapsed = time.perf_counter() - t0
    # ---- thresholds: recall-driven quantile per class ---------------------------------------
    thr = {}
    stats = {}
    for c in CLASSES:
        s = np.array(best_scores[c])
        q = float(np.quantile(s, args.miss_budget)) if len(s) else 0.0
        thr[c] = round(max(args.min_thr, min(q - args.margin, 0.95)), 4)
        fp = np.array(fp_scores[c]) if fp_scores[c] else np.array([0.0])
        stats[c] = {"gt_boxes": int(len(s)), "matched_score_p0.1": round(float(np.quantile(s, 0.001)), 4) if len(s) else None, "matched_score_p1": round(float(np.quantile(s, 0.01)), 4) if len(s) else None,
                    "matched_score_median": round(float(np.median(s)), 4) if len(s) else None, "matched_score_max": round(float(s.max()), 4) if len(s) else None,
                    "gt_unmatched_at_iou0.5": int((s == 0).sum()), "fp_above_thr": int((fp >= thr[c]).sum()), "fp_max": round(float(fp.max()), 4)}
    # ---- pack-level evaluation with the derived thresholds (same verdict rule as the controller) --
    import torchvision

    pol = Policy(conf_ok=thr["pill_ok"], conf_defect=min(thr[c] for c in CLASSES[1:]))
    frr = esc = n_nom = n_def = 0
    for nominal, gt_cls, pb, pcls, pconf in per_frame:
        t = np.where(pcls == 0, pol.conf_ok, pol.conf_defect)
        k = pconf >= t
        if k.any():
            idx = torchvision.ops.batched_nms(torch.from_numpy(pb[k]).float(), torch.from_numpy(pconf[k]).float(), torch.from_numpy(pcls[k]), pol.nms_iou).numpy()
            b, c, s = pb[k][idx], pcls[k][idx], pconf[k][idx]
        else:
            b, c, s = pb[:0], pcls[:0], pconf[:0]
        ok = b[c == 0]
        if len(ok) > 1:
            keep = torchvision.ops.nms(torch.from_numpy(ok).float(), torch.from_numpy(s[c == 0]).float(), pol.dedupe_iou).numpy()
            ok = ok[keep]
        verdict = "PASS" if (len(ok) == N_CAVITIES and not (c != 0).any()) else "REJECT"
        if nominal:
            n_nom += 1
            frr += verdict == "REJECT"
        else:
            n_def += 1
            esc += verdict == "PASS"
    result = {
        "engine": str(ENGINE_PATH.name), "engine_sha256": json.loads(ENGINE_PATH.with_suffix(".json").read_text(encoding="utf-8")).get("engine_sha256"),
        "derived": dt.datetime.now().astimezone().isoformat(), "frames": len(val), "miss_budget": args.miss_budget, "margin": args.margin, "seconds": round(elapsed, 1),
        "thresholds": {"conf_ok": pol.conf_ok, "conf_defect": pol.conf_defect, "per_class": thr, "nms_iou": pol.nms_iou, "dedupe_iou": pol.dedupe_iou},
        "class_stats": stats,
        "pack_level_on_val": {"nominal_packs": n_nom, "false_rejects": frr, "false_reject_rate": round(frr / max(1, n_nom), 4), "defective_packs": n_def, "escapes": esc, "escape_rate": round(esc / max(1, n_def), 4)},
        "note": "thresholds are recall-driven per-class quantiles on the synthetic validation split minus an out-of-sample margin; re-derive after any retraining or re-export.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print("written", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
