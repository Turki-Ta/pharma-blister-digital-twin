"""Feature-drift gate for an asset change: does the deployed engine still see the pack?

Renders an evaluation batch with the CURRENT factory (the asset version under test) through the
SDG pipeline, runs the deployed engine through the controller's own inference path
(TrtDetector.infer: GPU letterbox, FP16 TensorRT, calibrated thresholds, class-wise NMS, pack
verdict) and scores it against the labels.  The same scoring is run on a reference batch drawn
from the frozen validation split the engine was calibrated on, so every number has a baseline.

Retain criterion (per the visual-hardening directive, read against what the metric can say):
  * per-class recall at the deployed thresholds == 1.0 on every class,
  * pack-level: no escapes and no false rejects,
  * localisation: mean matched IoU within 0.02 of the reference batch.  A detector-vs-ground-truth
    IoU of 0.98 is not reachable by any detector (the reference batch itself is far below it;
    0.98 is the engine-vs-torch parity bound from the export audit), so the gate is
    "no worse than the engine on its own data", and both numbers are reported.
Exit code 0 = retain the engine, 2 = retrain (the directive's authorisation trigger).

    uv run python src/inference/drift_check.py --frames 50
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
from inference.calibrate_thresholds import gt_boxes_letterbox, iou_matrix  # noqa: E402
from inference.controller import CLASSES, ENGINE_PATH, N_CAVITIES, THRESHOLDS_PATH, Frame, Policy, TrtDetector  # noqa: E402
from simulation.assets import blister_factory as bf  # noqa: E402

PROJECT_ROOT = SRC.parent
REPORT = PROJECT_ROOT / "reports" / "drift_check.json"


def evaluate(det: TrtDetector, images: list[Path], label_dir: Path) -> dict:
    from PIL import Image

    gt_n = {c: 0 for c in CLASSES}
    hit_n = {c: 0 for c in CLASSES}
    fp_n = {c: 0 for c in CLASSES}
    ious: list[float] = []
    conf_hit = {c: [] for c in CLASSES}
    packs = {"nominal": 0, "defective": 0, "false_rejects": 0, "escapes": 0}
    lat = []
    for p in images:
        img = torch.from_numpy(np.array(Image.open(p).convert("RGB"), dtype=np.uint8)).cuda()
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        t0 = time.perf_counter()
        verdict, _g, _t = det.infer(Frame(-1, -1, 0, t0, img, ev))
        lat.append((time.perf_counter() - t0) * 1000)
        dets = verdict["scores"]["detections"]                     # [x0,y0,x1,y1,conf,name] in letterbox px
        pb = np.array([d[:4] for d in dets], dtype=np.float64).reshape(-1, 4)
        pc = np.array([CLASSES.index(d[5]) for d in dets], dtype=np.int64)
        pconf = np.array([d[4] for d in dets], dtype=np.float64)
        gt_cls, gt_xyxy = gt_boxes_letterbox(label_dir / (p.stem + ".txt"), det.scale, det.pad_top, det.pad_left)
        matched = np.zeros(len(dets), bool)
        for gi in range(len(gt_cls)):
            c = CLASSES[gt_cls[gi]]
            gt_n[c] += 1
            same = np.where(pc == gt_cls[gi])[0]
            if not len(same):
                continue
            iou = iou_matrix(gt_xyxy[gi:gi + 1], pb[same])[0]
            k = int(iou.argmax())
            if iou[k] >= 0.5 and not matched[same[k]]:
                matched[same[k]] = True
                hit_n[c] += 1
                ious.append(float(iou[k]))
                conf_hit[c].append(float(pconf[same[k]]))
        for j in np.where(~matched)[0]:
            fp_n[CLASSES[pc[j]]] += 1
        nominal = bool(len(gt_cls) == N_CAVITIES and (gt_cls == 0).all())
        packs["nominal" if nominal else "defective"] += 1
        if nominal and verdict["verdict"] == "REJECT":
            packs["false_rejects"] += 1
        if not nominal and verdict["verdict"] == "PASS":
            packs["escapes"] += 1
    per_class = {c: {"gt": gt_n[c], "hit": hit_n[c], "recall": round(hit_n[c] / gt_n[c], 4) if gt_n[c] else None, "false_positives": fp_n[c],
                     "conf_median": round(float(np.median(conf_hit[c])), 4) if conf_hit[c] else None,
                     "conf_min": round(float(min(conf_hit[c])), 4) if conf_hit[c] else None} for c in CLASSES}
    ious_a = np.array(ious) if ious else np.zeros(1)
    return {"frames": len(images), "per_class": per_class, "matched_boxes": len(ious), "iou_mean": round(float(ious_a.mean()), 4),
            "iou_p05": round(float(np.quantile(ious_a, 0.05)), 4), "iou_min": round(float(ious_a.min()), 4),
            "frac_iou_ge_0.98": round(float((ious_a >= 0.98).mean()), 4), "packs": packs, "infer_ms_median": round(float(np.median(lat)), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260918, help="must differ from every dataset seed (v1 20260910, v2 20260917): same seed = the dataset's own first frames")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "drift")
    ap.add_argument("--reference-list", type=Path, default=PROJECT_ROOT / "data" / "val.txt")
    ap.add_argument("--skip-render", action="store_true", help="reuse frames already in --out")
    ap.add_argument("--iou-tolerance", type=float, default=0.02)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.reference_list.exists() or not args.reference_list.read_text(encoding="utf-8").strip():
        print(f"reference list missing or empty: {args.reference_list}")
        return 1
    for used in (20260910, 20260917):
        if args.seed == used:
            print(f"--seed {args.seed} is a dataset seed; the evaluation batch would be the dataset's own frames")
            return 1

    if not args.skip_render:
        from simulation import sdg_pipeline as sdg

        print(f"[1/3] rendering {args.frames} evaluation frames with asset v{bf.ASSET_VERSION} -> {args.out}", flush=True)
        sdg.run(args.out, args.frames, args.seed, 3, 0.0, 1.0, smoke=False, rates=None)
    new_imgs = [Path(l.strip()) for l in (args.out / "train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    ref_imgs = [Path(l.strip()) for l in args.reference_list.read_text(encoding="utf-8").splitlines() if l.strip()][: args.frames]

    print("[2/3] deployed engine + calibrated thresholds", flush=True)
    det = TrtDetector(ENGINE_PATH, Policy.calibrated())
    det.warmup()
    new = evaluate(det, new_imgs, args.out / "labels")
    ref = evaluate(det, ref_imgs, ref_imgs[0].parent.parent / "labels")

    recall_ok = all(v["hit"] == v["gt"] for v in new["per_class"].values() if v["gt"])   # integers: a rounded ratio hides misses on large batches
    packs_ok = new["packs"]["escapes"] == 0 and new["packs"]["false_rejects"] == 0
    iou_ok = new["iou_mean"] >= ref["iou_mean"] - args.iou_tolerance
    retain = recall_ok and packs_ok and iou_ok
    thr = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    import hashlib

    engine_sha = hashlib.sha256(ENGINE_PATH.read_bytes()).hexdigest()
    if thr.get("engine_sha256") != engine_sha:
        print(f"thresholds.json was derived for engine {str(thr.get('engine_sha256'))[:12]} but the deployed engine is {engine_sha[:12]}: re-run calibrate_thresholds.py")
        return 1
    result = {"asset_version": bf.ASSET_VERSION, "engine": ENGINE_PATH.name, "engine_sha256": engine_sha, "thresholds_engine_sha256": thr.get("engine_sha256"),
              "thresholds": thr["thresholds"], "evaluated": dt.datetime.now().astimezone().isoformat(),
              "new_asset": new, "reference_val_split": ref,
              "gate": {"recall_all_classes_1.0": recall_ok, "no_escapes_no_false_rejects": packs_ok,
                       "iou_mean_within_tolerance": iou_ok, "iou_tolerance": args.iou_tolerance, "retain_engine": retain},
              "note": "IoU is detector vs ground truth; 0.98 is the engine-parity bound and not reachable here (see reference)."}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[3/3] result", flush=True)
    print(f"{'class':14s} {'GT':>5s} {'recall v2':>10s} {'recall ref':>11s} {'FP v2':>6s} {'conf med v2':>12s} {'conf med ref':>13s}")
    for c in CLASSES:
        a, b = new["per_class"][c], ref["per_class"][c]
        print(f"{c:14s} {a['gt']:5d} {str(a['recall']):>10s} {str(b['recall']):>11s} {a['false_positives']:6d} {str(a['conf_median']):>12s} {str(b['conf_median']):>13s}")
    print(f"IoU mean v2 {new['iou_mean']} (ref {ref['iou_mean']}) | p05 {new['iou_p05']} (ref {ref['iou_p05']}) | frac>=0.98 {new['frac_iou_ge_0.98']} (ref {ref['frac_iou_ge_0.98']})")
    print(f"packs v2 {new['packs']} | ref {ref['packs']}")
    print("DRIFT GATE:", "RETAIN ENGINE" if retain else "RETRAIN (recall/pack/IoU gate failed)", "| written", REPORT, flush=True)
    return 0 if retain else 2


if __name__ == "__main__":
    raise SystemExit(main())
