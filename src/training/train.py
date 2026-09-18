"""Module 3 - YOLO11n training on the synthetic blister dataset, hardened per blackwell_env_runbook.md.

Runbook mandates enforced here (single-GPU, in-process Ultralytics trainer, so process-level
torch settings made under the ``__main__`` guard apply to training):
* Phase 4.2  ``torch.cuda.set_per_process_memory_fraction(0.94)`` before any CUDA work.
* Phase 4.3  batch-envelope sweep (16/32/64 at 640) measuring peak VRAM and step latency; the
             largest batch with peak <= 30 GB and no superlinear step-time jump is selected.
* Phase 4.4  keep-awake around the run; pending-Windows-Update reboot surfaced before starting.
* Phase 5    TF32 via the fp32_precision API; bf16 autocast without GradScaler (``amp='bf16'``,
             which also skips Ultralytics' fp16 AMP check download).
* Phase 6    ``workers=8``; Ultralytics pins memory for CUDA and keeps a persistent iterator.
* Phase 8    atomic checkpoints (``.pt.tmp`` -> ``os.replace``), rotation (keep 3) plus
             ``latest.pt``, a resume chain, and a non-finite guard that skips the optimizer step
             (zeroing accumulated gradients) and aborts after 50 consecutive skips with
             checkpoints intact.  Smoke runs checkpoint to ``models/ckpt_smoke``.

The detector is wrapped behind ``DetectorBackend`` so the AGPL-licensed Ultralytics dependency
stays confined to this module (Decision 4).

    uv run python src/training/train.py                 # sweep -> select batch -> 5-epoch baseline
    uv run python src/training/train.py --sweep-only
    uv run python src/training/train.py --batch 64 --epochs 5
    uv run python src/training/train.py --smoke         # 1 epoch on data/smoke into models/ckpt_smoke
    uv run python src/training/train.py --resume        # resume chain: latest.pt, rotated, last.pt
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PEAK_LIMIT_GB = 30.0          # runbook 4.3 acceptance rule
NONFINITE_ABORT = 50          # runbook 8.3
KEEP_CHECKPOINTS = 3          # runbook 8.2


# --------------------------------------------------------------------------------------
# Runbook: backend, seatbelt, keep-awake, reboot check
# --------------------------------------------------------------------------------------
def configure_backend() -> dict:
    """Runbook Phase 5 + 4.2.  Returns what was applied for the run manifest."""
    applied = {}
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        applied["tf32"] = "fp32_precision API"
    except Exception:  # noqa: BLE001 - older torch fallback
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        applied["tf32"] = "legacy allow_tf32"
    torch.cuda.set_per_process_memory_fraction(0.94)
    applied["memory_fraction"] = 0.94
    applied["cap_gb"] = round(torch.cuda.get_device_properties(0).total_memory * 0.94 / 1e9, 1)
    try:
        import cv2

        cv2.setNumThreads(0)
        applied["cv2_threads"] = 0
    except Exception:  # noqa: BLE001
        pass
    return applied


def keep_awake(on: bool) -> None:
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def pending_reboot() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg

        winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired")
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------------------
# Hardened Ultralytics trainer (atomic + rotated checkpoints, non-finite guard, timing)
# --------------------------------------------------------------------------------------
def make_trainer_class():
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import LOGGER

    class HardenedTrainer(DetectionTrainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.nonfinite_streak = 0
            self.nonfinite_total = 0
            self.skipped_steps = 0

        # -- runbook 8.3: guard BEFORE optimizer.step, on the loss and on the clipped grad norm ----
        def optimizer_step(self):
            self.scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            loss_ok = bool(torch.isfinite(self.loss).all()) if isinstance(self.loss, torch.Tensor) else True
            if not loss_ok or not bool(torch.isfinite(total_norm)):
                self.nonfinite_streak += 1
                self.nonfinite_total += 1
                self.skipped_steps += 1
                self.optimizer.zero_grad(set_to_none=True)   # discard the poisoned accumulation window
                LOGGER.warning(f"non-finite loss/grad (loss_ok={loss_ok}, grad_norm={float(total_norm):.3g}); optimizer step skipped (streak {self.nonfinite_streak})")
                if self.nonfinite_streak >= NONFINITE_ABORT:
                    raise RuntimeError(f"{NONFINITE_ABORT} consecutive non-finite steps; aborting with checkpoints intact")
                return
            self.nonfinite_streak = 0
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

        # -- runbook 8.1/8.2: atomic last/best + rotated copies + latest.pt ------------------------
        def save_model(self):
            last, best = self.last, self.best
            self.last, self.best = last.with_suffix(".pt.tmp"), best.with_suffix(".pt.tmp")
            try:
                ok = super().save_model()   # writes to the .tmp paths
            finally:
                self.last, self.best = last, best
            if ok:
                for tmp, final in ((last.with_suffix(".pt.tmp"), last), (best.with_suffix(".pt.tmp"), best)):
                    if tmp.exists():
                        os.replace(tmp, final)          # atomic on NTFS
                self._rotate(last)
            return ok

        def _rotate(self, last: Path) -> None:
            ckdir = self.save_dir / "ckpt"
            ckdir.mkdir(parents=True, exist_ok=True)
            for target in (ckdir / f"ckpt_e{self.epoch:04d}.pt", ckdir / "latest.pt"):
                tmp = target.with_suffix(".pt.tmp")
                shutil.copyfile(last, tmp)
                os.replace(tmp, target)
            for old in sorted(ckdir.glob("ckpt_e*.pt"))[:-KEEP_CHECKPOINTS]:
                old.unlink()

    return HardenedTrainer


def resume_candidates(run_dir: Path) -> list[Path]:
    """Runbook 8.2 resume chain: latest.pt, rotated newest-first, then last.pt."""
    ck = run_dir / "ckpt"
    cands = [ck / "latest.pt"] + sorted(ck.glob("ckpt_e*.pt"), reverse=True) + [run_dir / "weights" / "last.pt"]
    good = []
    for c in cands:
        if c.exists():
            try:
                torch.load(c, map_location="cpu", weights_only=False)
                good.append(c)
            except Exception as e:  # noqa: BLE001
                print(f"resume: skipping unreadable checkpoint {c.name}: {type(e).__name__}")
    return good


# --------------------------------------------------------------------------------------
# Detector backend (keeps the Ultralytics API behind one seam)
# --------------------------------------------------------------------------------------
class DetectorBackend:
    def __init__(self, weights: str, trainer_cls):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.trainer_cls = trainer_cls
        self.step_marks: list[float] = []
        self.model.add_callback("on_train_batch_end", lambda tr: self.step_marks.append(time.perf_counter()))

    def train(self, **kw):
        return self.model.train(trainer=self.trainer_cls, **kw)

    def step_stats(self) -> dict:
        d = np.diff(np.array(self.step_marks)) * 1000.0
        d = d[3:] if d.size > 6 else d
        return {"steps": int(len(self.step_marks)), "step_ms_median": round(float(np.median(d)), 1) if d.size else None, "step_ms_p95": round(float(np.percentile(d, 95)), 1) if d.size else None}

    def trainer(self):
        return self.model.trainer


COMMON = dict(imgsz=640, workers=8, amp="bf16", device=0, exist_ok=True, verbose=False, seed=0, pretrained=True, cache=False)


# --------------------------------------------------------------------------------------
# Runbook 4.3 batch-envelope sweep
# --------------------------------------------------------------------------------------
def batch_sweep(data_yaml: Path, weights: str, batches=(16, 32, 64), fraction: float = 0.5) -> tuple[list[dict], int | None]:
    trainer_cls = make_trainer_class()
    rows: list[dict] = []
    for b in batches:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = {"batch": b}
        try:
            det = DetectorBackend(weights, trainer_cls)
            t0 = time.perf_counter()
            det.train(data=str(data_yaml), epochs=1, batch=b, fraction=fraction, val=False, plots=False, save=False,
                      close_mosaic=0, project=str(MODELS_DIR / "sweep"), name=f"b{b}", **COMMON)
            row.update(det.step_stats())
            row["wall_s"] = round(time.perf_counter() - t0, 1)
            row["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            row["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
            row["ms_per_image"] = round(row["step_ms_median"] / b, 2) if row["step_ms_median"] else None
            row["skipped_steps"] = det.trainer().skipped_steps
            del det
        except torch.cuda.OutOfMemoryError:
            row["oom"] = True
            rows.append(row)
            print(f"batch {b}: OOM", flush=True)
            break
        rows.append(row)
        print(f"batch {b:3d}: peak alloc {row['peak_alloc_gb']:.1f} GB, reserved {row['peak_reserved_gb']:.1f} GB, step {row['step_ms_median']} ms, {row['ms_per_image']} ms/img", flush=True)
    # acceptance: peak <= 30 GB and no superlinear jump between adjacent sizes (the WDDM spill signature)
    selected = None
    prev = None
    for r in rows:
        if r.get("oom") or r["peak_alloc_gb"] > PEAK_LIMIT_GB or r["peak_reserved_gb"] > PEAK_LIMIT_GB:
            r["accepted"] = False
            break
        if prev and r["step_ms_median"] and prev["step_ms_median"]:
            ratio = r["step_ms_median"] / prev["step_ms_median"]
            expected = r["batch"] / prev["batch"]
            r["step_ratio_vs_expected"] = round(ratio / expected, 2)
            if ratio > 1.6 * expected:
                r["accepted"] = False
                r["note"] = "superlinear step-time jump (spill signature)"
                break
        r["accepted"] = True
        selected = r["batch"]
        prev = r
    return rows, selected


# --------------------------------------------------------------------------------------
# Baseline training
# --------------------------------------------------------------------------------------
def train_baseline(data_yaml: Path, weights: str, batch: int, epochs: int, run_dir: Path, resume_from: Path | None) -> dict:
    trainer_cls = make_trainer_class()
    det = DetectorBackend(str(resume_from) if resume_from else weights, trainer_cls)
    kw = dict(data=str(data_yaml), epochs=epochs, batch=batch, close_mosaic=max(1, min(10, epochs // 2)), plots=True,
              project=str(run_dir.parent), name=run_dir.name, **COMMON)
    if resume_from:
        kw["resume"] = True
    t0 = time.perf_counter()
    metrics = det.train(**kw)
    wall = time.perf_counter() - t0
    tr = det.trainer()
    out = {"batch": batch, "epochs": epochs, "wall_s": round(wall, 1), **det.step_stats(),
           "nonfinite_total": tr.nonfinite_total, "skipped_steps": tr.skipped_steps,
           "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "run_dir": str(tr.save_dir)}
    if metrics is not None:
        box = metrics.box
        names = metrics.names
        per_class = {}
        for k, ci in enumerate(box.ap_class_index.tolist()):
            per_class[names[ci]] = {"precision": round(float(box.p[k]), 4), "recall": round(float(box.r[k]), 4), "ap50": round(float(box.ap50[k]), 4), "ap50_95": round(float(box.ap[k]), 4)}
        out["val"] = {"map50": round(float(box.map50), 4), "map50_95": round(float(box.map), 4), "per_class": per_class}
    csv = Path(tr.save_dir) / "results.csv"
    if csv.exists():
        out["results_csv"] = csv.read_text(encoding="utf-8").strip().splitlines()[-epochs:]
    ck = Path(tr.save_dir) / "ckpt"
    out["checkpoints"] = sorted(p.name for p in ck.glob("*.pt")) if ck.exists() else []
    out["weights"] = sorted(p.name for p in (Path(tr.save_dir) / "weights").glob("*.pt"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Hardened YOLO11n training (runbook-compliant).")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "data.yaml")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=None, help="skip the sweep and use this batch")
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--sweep-fraction", type=float, default=0.5)
    ap.add_argument("--sweep-batches", type=str, default="16,32,64", help="comma-separated batch sizes for the envelope sweep")
    ap.add_argument("--name", default="yolo11n_baseline")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on data/smoke, checkpoints in models/ckpt_smoke")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if pending_reboot():
        print("WARNING: Windows Update reboot pending - reboot before an unattended run (runbook 4.4)", flush=True)
    applied = configure_backend()
    print("backend:", applied, "| torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0), flush=True)

    if args.smoke:
        data, run_dir, epochs = PROJECT_ROOT / "data" / "smoke" / "data.yaml", MODELS_DIR / "ckpt_smoke" / "smoke", 1
        batch = args.batch or 8
    else:
        data, run_dir, epochs = args.data, MODELS_DIR / "runs" / args.name, args.epochs
        batch = args.batch
    if not data.exists():
        print(f"dataset yaml not found: {data}")
        return 1

    summary: dict = {"backend": applied, "data": str(data)}
    keep_awake(True)
    try:
        if batch is None or args.sweep_only:
            rows, selected = batch_sweep(data, args.model, batches=tuple(int(b) for b in args.sweep_batches.split(",")), fraction=args.sweep_fraction)
            summary["sweep"] = rows
            summary["selected_batch"] = selected
            print("sweep:", json.dumps(rows, indent=1), "\nselected batch:", selected, flush=True)
            if args.sweep_only:
                (MODELS_DIR / f"summary_{args.name}_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                return 0
            if selected is None:
                print("no batch size passed the envelope; aborting")
                return 1
            batch = selected
        resume_from = None
        if args.resume:
            cands = resume_candidates(run_dir)
            if not cands:
                print("resume requested but no readable checkpoint found")
                return 1
            resume_from = cands[0]
            print("resuming from", resume_from, flush=True)
        summary["baseline"] = train_baseline(data, args.model, batch, epochs, run_dir, resume_from)
    finally:
        keep_awake(False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"summary_{'smoke' if args.smoke else args.name}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["baseline"], indent=1), flush=True)
    print("summary written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
