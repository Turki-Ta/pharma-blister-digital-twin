"""Module 5 - closed-loop inspection controller (host RTX 5090 today, Jetson Orin profile).

Three workers, one ledger, one clock:

  W1 ingest      FrameSource -> Frame(pack_id, trigger_ticks, CUDA uint8 HWC image, ready_event)
  W2 inference   owns its own TensorRT runtime/engine/context and CUDA stream (created inside the
                 thread); GPU letterbox (720x1280 -> 640x640) + normalisation on that stream; raw
                 [1, 8, 8400] decode + class-wise NMS on the GPU; 10/10 occupancy verdict
  W3 actuation   encoder-clocked shift register: every pack is latched at its trigger tick and
                 actioned when the belt has travelled d_nozzle - lead (lead = v_measured *
                 t_valve); a pack without a valid PASS verdict at that tick is REJECTED (fail-closed)
  supervisor     heartbeat watchdog (> 20 ms stale -> degraded, in-flight verdicts forced to REJECT)

Concurrency rules honoured (Windows, Python threads):
* the TensorRT execution context and its stream are created and used by W2 only;
* every frame carries a CUDA event recorded by its producer; W2 waits on it on its own stream
  before touching the memory, so producer/consumer streams never race;
* queues are bounded; overflow never drops a PACK - the pack is registered with an OVERFLOW
  verdict (REJECT) and only the frame is discarded;
* time is never used to place the rejector: encoder ticks are (belt speed only enters through the
  valve lead term, computed from the measured tick rate, so ramps and stops stay correct).

Audit trail: append-only JSONL, one record per pack, written at actuation time.

    uv run python src/inference/controller.py --test
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import queue
import socket
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = PROJECT_ROOT / "models" / "exported" / "yolo11n_rtx5090_fp16.engine"
AUDIT_PATH = PROJECT_ROOT / "reports" / "audit_trail.jsonl"
CONTROLLER_VERSION = "0.1.0"

CLASSES = ("pill_ok", "pill_damaged", "cavity_empty", "foil_damaged")
N_CAVITIES = 10
IMGSZ = 640
FRAME_H, FRAME_W = 720, 1280
PAD_VALUE = 114.0 / 255.0


# --------------------------------------------------------------------------------------
# Line configuration (Decision 5) and controller policy
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LineConfig:
    v_belt_mps: float = 1.6              # design speed (800 packs/min at 120 mm pitch)
    pack_pitch_m: float = 0.120
    d_nozzle_m: float = 0.300            # camera trigger plane -> rejector nozzle plane
    encoder_pitch_m: float = 0.0005      # belt travel per encoder tick (0.5 mm)
    t_valve_s: float = 0.010             # calibrated signal-to-jet reaction time
    hit_window_m: float = 0.0375         # (L_pack - w_jet) / 2, from the Q2 budget

    @property
    def nozzle_ticks(self) -> int:
        return int(round(self.d_nozzle_m / self.encoder_pitch_m))

    @property
    def pitch_ticks(self) -> int:
        return int(round(self.pack_pitch_m / self.encoder_pitch_m))


THRESHOLDS_PATH = PROJECT_ROOT / "models" / "exported" / "thresholds.json"


@dataclass(frozen=True)
class Policy:
    conf_ok: float = 0.60                # bar for "pill_ok"; replaced by the calibrated value (thresholds.json)
    conf_defect: float = 0.15            # low bar for any defect class (recall-biased)
    nms_iou: float = 0.70
    dedupe_iou: float = 0.45             # two pill_ok boxes closer than this are one cavity
    inference_budget_s: float = 0.025    # capture -> verdict; slower verdicts are not trusted
    heartbeat_stale_s: float = 0.020
    frame_queue_size: int = 4
    source: str = "defaults"

    @staticmethod
    def calibrated(path: Path = THRESHOLDS_PATH, **overrides) -> "Policy":
        """Thresholds derived from the frozen validation split by calibrate_thresholds.py; they are
        tied to the engine hash and re-derived after any retraining or re-export."""
        if not path.exists():
            return Policy(**overrides)
        t = json.loads(path.read_text(encoding="utf-8"))
        kw = {"conf_ok": t["thresholds"]["conf_ok"], "conf_defect": t["thresholds"]["conf_defect"], "nms_iou": t["thresholds"]["nms_iou"], "dedupe_iou": t["thresholds"]["dedupe_iou"], "source": f"{path.name} (engine {str(t.get('engine_sha256'))[:12]})"}
        kw.update(overrides)
        return Policy(**kw)


# --------------------------------------------------------------------------------------
# Frames and sources
# --------------------------------------------------------------------------------------
@dataclass
class Frame:
    frame_id: int
    pack_id: int
    trigger_ticks: int
    t_capture: float                       # perf_counter seconds
    image: torch.Tensor | None             # CUDA uint8 HWC RGB (FRAME_H, FRAME_W, 3); None = dropped
    ready_event: torch.cuda.Event | None   # recorded by the producer after the image was written


class FrameSource(ABC):
    """Produces frames keyed by the hardware trigger.  ``next`` returns None when nothing is due."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def next(self, encoder_ticks: int) -> Frame | None: ...

    @abstractmethod
    def close(self) -> None: ...


class DLPackFrameSource(FrameSource):
    """Zero-copy ingestion from any producer exposing DLPack (torch tensors, ovrtx render vars
    mapped to CUDA, CuPy arrays).  ``producer(encoder_ticks)`` returns None when no pack is due,
    else (dlpack_capable_object, pack_id, trigger_ticks)."""

    def __init__(self, producer: Callable[[int], tuple | None]):
        self.producer = producer
        self._n = 0

    def open(self) -> None:
        pass

    def next(self, encoder_ticks: int) -> Frame | None:
        item = self.producer(encoder_ticks)
        if item is None:
            return None
        obj, pack_id, trigger_ticks = item
        img = obj if isinstance(obj, torch.Tensor) else torch.from_dlpack(obj)
        assert img.is_cuda, "DLPackFrameSource expects CUDA-resident frames"
        if img.shape[-1] == 4:                   # RGBA from a renderer -> RGB view (no copy)
            img = img[..., :3]
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        self._n += 1
        return Frame(self._n, pack_id, trigger_ticks, time.perf_counter(), img, ev)

    def close(self) -> None:
        pass


class ZeroMQFrameSource(FrameSource):
    """Networked ingestion stub for the Jetson deployment: PULL socket, multipart
    [header JSON][uint8 HWC RGB bytes]; header = {pack_id, trigger_ticks, h, w, c}.
    The payload lands in pinned host memory and is copied to the GPU on the default stream."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555"):
        self.endpoint = endpoint
        self.sock = None
        self._n = 0

    def open(self) -> None:
        import zmq  # optional dependency (pyzmq)

        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVHWM, 4)
        self.sock.connect(self.endpoint)

    def next(self, encoder_ticks: int) -> Frame | None:
        import zmq

        try:
            header, payload = self.sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        h = json.loads(header)
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(h["h"], h["w"], h["c"])
        img = torch.from_numpy(arr).pin_memory().cuda(non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        self._n += 1
        return Frame(self._n, int(h["pack_id"]), int(h["trigger_ticks"]), time.perf_counter(), img, ev)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(0)


# --------------------------------------------------------------------------------------
# Encoder (distance clock) and actuator
# --------------------------------------------------------------------------------------
class Encoder(ABC):
    @abstractmethod
    def ticks(self) -> int: ...

    @abstractmethod
    def tick_rate(self) -> float:
        """ticks per second, measured (used only for the valve lead term)."""


class SimEncoder(Encoder):
    """Belt travel integrated from a speed profile; ticks are DISTANCE, so a stopped belt stops
    the count and the shift register with it."""

    def __init__(self, cfg: LineConfig, speed_fn: Callable[[float], float] | None = None):
        self.cfg = cfg
        self.speed_fn = speed_fn or (lambda t: cfg.v_belt_mps)
        self.t0 = None                       # belt released by release(); until then no distance accrues
        self._dist = 0.0
        self._t_last = None
        self._lock = threading.Lock()
        self._hist: list[tuple[float, int]] = []

    def release(self) -> None:
        """Start the belt (line interlock: only once the inspection pipeline reports healthy)."""
        with self._lock:
            self.t0 = self._t_last = time.perf_counter()

    def _advance(self) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._t_last is None:
                return
            dt_ = now - self._t_last
            if dt_ > 0:
                self._dist += self.speed_fn(now - self.t0) * dt_
                self._t_last = now
                self._hist.append((now, int(self._dist / self.cfg.encoder_pitch_m)))
                if len(self._hist) > 200:
                    del self._hist[:100]

    def ticks(self) -> int:
        self._advance()
        with self._lock:
            return int(self._dist / self.cfg.encoder_pitch_m)

    def tick_rate(self) -> float:
        with self._lock:
            if len(self._hist) < 2:
                return self.cfg.v_belt_mps / self.cfg.encoder_pitch_m
            (t0, k0), (t1, k1) = self._hist[max(0, len(self._hist) - 40)], self._hist[-1]
            return (k1 - k0) / (t1 - t0) if t1 > t0 else 0.0


class Actuator(ABC):
    @abstractmethod
    def fire(self, pack_id: int, ticks: int) -> None: ...

    @abstractmethod
    def confirm(self, pack_id: int) -> bool:
        """True when the reject was physically confirmed (bin sensor / twin overlap query)."""


class SimActuator(Actuator):
    def __init__(self):
        self.fired: list[tuple[int, int, float]] = []
        self._lock = threading.Lock()

    def fire(self, pack_id: int, ticks: int) -> None:
        with self._lock:
            self.fired.append((pack_id, ticks, time.perf_counter()))

    def confirm(self, pack_id: int) -> bool:
        with self._lock:
            return any(p == pack_id for p, _, _ in self.fired)


# --------------------------------------------------------------------------------------
# Ledger and audit trail
# --------------------------------------------------------------------------------------
@dataclass
class PackRecord:
    pack_id: int
    trigger_ticks: int
    t_trigger: float
    frame_id: int | None = None
    verdict: str | None = None            # PASS | REJECT
    reason: str | None = None
    scores: dict = field(default_factory=dict)
    t_verdict: float | None = None
    latency_ms: float | None = None
    action: str | None = None             # pass | reject
    action_ticks: int | None = None
    t_action: float | None = None
    confirmed: bool | None = None
    late_verdict: bool = False


class PackLedger:
    """Every triggered pack gets exactly one verdict and exactly one action."""

    def __init__(self):
        self._lock = threading.Lock()
        self.packs: dict[int, PackRecord] = {}
        self.order: list[int] = []

    def trigger(self, pack_id: int, ticks: int, frame_id: int | None) -> PackRecord:
        with self._lock:
            assert pack_id not in self.packs, f"pack {pack_id} triggered twice"
            rec = PackRecord(pack_id, ticks, time.perf_counter(), frame_id)
            self.packs[pack_id] = rec
            self.order.append(pack_id)
            return rec

    def verdict(self, pack_id: int, verdict: str, reason: str, scores: dict, t_capture: float) -> bool:
        """Returns False when the pack was already actioned (late verdict: recorded, not applied)."""
        with self._lock:
            rec = self.packs[pack_id]
            if rec.action is not None:
                rec.late_verdict = True
                rec.reason = (rec.reason or "") + f"|late_verdict({verdict}:{reason})"
                return False
            if rec.verdict is None or verdict == "REJECT":   # a REJECT always wins
                rec.verdict, rec.reason, rec.scores = verdict, reason, scores
                rec.t_verdict = time.perf_counter()
                rec.latency_ms = round((rec.t_verdict - t_capture) * 1000, 2)
            return True

    def force_reject_pending(self, reason: str) -> list[int]:
        """Watchdog: packs whose verdict is still PENDING are rejected (their inference may be
        late or stale).  Verdicts already produced within budget by a then-healthy worker stand."""
        with self._lock:
            hit = []
            for pid in self.order:
                rec = self.packs[pid]
                if rec.action is None and rec.verdict is None:
                    rec.verdict, rec.reason = "REJECT", reason
                    hit.append(pid)
            return hit

    def due(self, ticks: int, lead_ticks: int, nozzle_ticks: int) -> list[PackRecord]:
        with self._lock:
            return [self.packs[p] for p in self.order if self.packs[p].action is None and ticks >= self.packs[p].trigger_ticks + nozzle_ticks - lead_ticks]

    def action(self, pack_id: int, action: str, ticks: int, confirmed: bool | None, reason_if_forced: str | None) -> PackRecord:
        with self._lock:
            rec = self.packs[pack_id]
            if rec.verdict is None:                       # fail-closed: no verdict at the fire tick
                rec.verdict, rec.reason = "REJECT", reason_if_forced or "no_verdict"
            rec.action, rec.action_ticks, rec.t_action, rec.confirmed = action, ticks, time.perf_counter(), confirmed
            return rec

    def invariant(self) -> dict:
        with self._lock:
            n_trig = len(self.packs)
            n_verd = sum(1 for r in self.packs.values() if r.verdict is not None)
            n_act = sum(1 for r in self.packs.values() if r.action is not None)
            return {"triggers": n_trig, "verdicts": n_verd, "actions": n_act, "ok": n_trig == n_verd == n_act}

    def pending(self) -> int:
        with self._lock:
            return sum(1 for r in self.packs.values() if r.action is None)


class AuditTrail:
    """Append-only JSONL; one record per pack at actuation; fsync per record."""

    def __init__(self, path: Path, run_meta: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.meta = run_meta
        self._lock = threading.Lock()
        self._f = open(path, "a", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, rec: PackRecord) -> None:
        record = {
            "schema": "blister.audit/1", "run_id": self.meta["run_id"], "controller_version": CONTROLLER_VERSION, "engine_sha256": self.meta["engine_sha256"], "host": self.meta["host"],
            "pack_id": rec.pack_id, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
            "encoder_ticks": {"trigger": rec.trigger_ticks, "action": rec.action_ticks}, "frame_id": rec.frame_id,
            "class_confidences": rec.scores, "verdict": rec.verdict, "reason": rec.reason, "verdict_latency_ms": rec.latency_ms, "late_verdict": rec.late_verdict,
            "actuation": {"action": rec.action, "confirmed": rec.confirmed, "status": ("fired+confirmed" if rec.action == "reject" and rec.confirmed else "fired+UNCONFIRMED" if rec.action == "reject" else "passed")},
            "thresholds": self.meta["thresholds"],
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            self._f.write(line)
            self._f.flush()
            os.fsync(self._f.fileno())
            self.count += 1

    def close(self) -> None:
        with self._lock:
            self._f.close()


# --------------------------------------------------------------------------------------
# W2: TensorRT inference on its own stream, GPU letterbox, occupancy verdict
# --------------------------------------------------------------------------------------
class TrtDetector:
    """Created and used by the inference thread only."""

    def __init__(self, engine_path: Path, policy: Policy):
        import tensorrt as trt
        import torchvision  # noqa: F401 - NMS

        self.policy = policy
        self.stream = torch.cuda.Stream()
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.inp_name = next(self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT)
        self.out_name = next(self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT)
        self.inp = torch.empty(tuple(self.engine.get_tensor_shape(self.inp_name)), dtype=torch.float32, device="cuda")
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(self.out_name)), dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.inp_name, self.inp.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        assert tuple(self.inp.shape) == (1, 3, IMGSZ, IMGSZ) and tuple(self.out.shape) == (1, 4 + len(CLASSES), 8400)
        # letterbox geometry for the fixed 720x1280 camera frame
        self.scale = IMGSZ / max(FRAME_H, FRAME_W)
        self.new_h, self.new_w = int(round(FRAME_H * self.scale)), int(round(FRAME_W * self.scale))
        self.pad_top, self.pad_left = (IMGSZ - self.new_h) // 2, (IMGSZ - self.new_w) // 2

    def warmup(self, n: int = 20) -> None:
        """First calls pay for lazy kernel loading (interpolate, TensorRT, torchvision NMS ~75 ms).
        Run the whole path, including a NON-empty NMS, before the belt starts."""
        import torchvision

        with torch.cuda.stream(self.stream), torch.inference_mode():
            b = torch.tensor([[10.0, 10, 50, 50], [12, 12, 52, 52], [100, 100, 140, 140]], device="cuda")
            s = torch.tensor([0.9, 0.8, 0.7], device="cuda")
            torchvision.ops.batched_nms(b, s, torch.tensor([0, 0, 1], device="cuda"), 0.7)
            torchvision.ops.nms(b, s, 0.45)
        self.stream.synchronize()
        # full path (letterbox, engine, decode, NMS, verdict) on noise and on a flat frame, as a real frame would go
        for i in range(n):
            dummy = torch.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=torch.uint8, device="cuda") if i % 2 == 0 else torch.full((FRAME_H, FRAME_W, 3), 128, dtype=torch.uint8, device="cuda")
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            self.infer(Frame(-1, -1, 0, time.perf_counter(), dummy, ev))

    def letterbox_gpu(self, img_hwc_u8: torch.Tensor) -> None:
        """uint8 HWC RGB (720, 1280, 3) on CUDA -> self.inp (1, 3, 640, 640) float32, on self.stream."""
        x = img_hwc_u8.permute(2, 0, 1).unsqueeze(0).float()                       # 1x3xHxW
        x = torch.nn.functional.interpolate(x, size=(self.new_h, self.new_w), mode="bilinear", align_corners=False, antialias=False)
        self.inp.fill_(PAD_VALUE)
        self.inp[:, :, self.pad_top:self.pad_top + self.new_h, self.pad_left:self.pad_left + self.new_w] = x / 255.0

    @torch.inference_mode()
    def infer(self, frame: Frame) -> tuple[dict, float, float]:
        import torchvision

        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(frame.ready_event)          # producer's writes are visible on our stream
            self.letterbox_gpu(frame.image)
            ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
            assert ok, "execute_async_v3 failed"
            raw = self.out[0].T                                 # [8400, 8] (view on our stream)
            boxes, scores = raw[:, :4], raw[:, 4:]
            conf, cls = scores.max(1)
            thr = torch.where(cls == 0, torch.full_like(conf, self.policy.conf_ok), torch.full_like(conf, self.policy.conf_defect))
            keep = conf >= thr
            boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
            xyxy = torch.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
            idx = torchvision.ops.batched_nms(xyxy, conf, cls, self.policy.nms_iou)
            dets = torch.cat([xyxy[idx], conf[idx, None], cls[idx, None].float()], 1).cpu()   # sync point on our stream
        self.stream.synchronize()
        t_gpu = time.perf_counter() - t0
        return self.verdict(dets), t_gpu, time.perf_counter() - t0

    def verdict(self, dets: torch.Tensor) -> dict:
        import torchvision

        ok = dets[dets[:, 5] == 0]
        # distinct cavities: collapse pill_ok boxes that overlap each other beyond dedupe_iou
        if len(ok) > 1:
            keep = torchvision.ops.nms(ok[:, :4], ok[:, 4], self.policy.dedupe_iou)
            ok = ok[keep]
        n_ok = int(len(ok))
        defects = dets[dets[:, 5] != 0]
        per_class = {c: {"count": int((dets[:, 5] == i).sum()), "max_conf": round(float(dets[dets[:, 5] == i][:, 4].max()), 4) if int((dets[:, 5] == i).sum()) else 0.0} for i, c in enumerate(CLASSES)}
        reasons = []
        if len(defects):
            reasons.append("defect:" + ",".join(sorted({CLASSES[int(c)] for c in defects[:, 5].tolist()})))
        if n_ok < N_CAVITIES:
            reasons.append(f"under_occupancy:{n_ok}/{N_CAVITIES}")
        if n_ok > N_CAVITIES:
            reasons.append(f"over_occupancy:{n_ok}/{N_CAVITIES}")
        verdict = "PASS" if not reasons else "REJECT"
        return {"verdict": verdict, "reason": ";".join(reasons) if reasons else "10/10_pill_ok",
                "scores": {"per_class": per_class, "n_pill_ok_distinct": n_ok, "detections": [[round(v, 2) for v in d[:5].tolist()] + [CLASSES[int(d[5])]] for d in dets]}}


# --------------------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------------------
class Controller:
    def __init__(self, source: FrameSource, encoder: Encoder, actuator: Actuator, *, cfg: LineConfig = LineConfig(), policy: Policy = Policy(), engine_path: Path = ENGINE_PATH, audit_path: Path = AUDIT_PATH, faults: dict | None = None):
        self.source, self.encoder, self.actuator, self.cfg, self.policy, self.engine_path = source, encoder, actuator, cfg, policy, engine_path
        self.faults = faults or {}
        self.ledger = PackLedger()
        self.frames: queue.Queue[Frame] = queue.Queue(maxsize=policy.frame_queue_size)
        self.stop = threading.Event()
        self.errors: list[str] = []
        self.heartbeat: dict[str, float] = {}
        self.hb_lock = threading.Lock()
        self.watchdog_events: list[dict] = []
        self.alarms: list[dict] = []          # line-stop conditions (late actuation = possible escape)
        self.stats = {"frames": 0, "inferences": 0, "overflow": 0, "dropped_frames": 0, "timeouts": 0, "late_actuations": 0, "gpu_ms": [], "verdict_ms": []}
        engine_meta = json.loads(engine_path.with_suffix(".json").read_text(encoding="utf-8")) if engine_path.with_suffix(".json").exists() else {}
        self.audit = AuditTrail(audit_path, {"run_id": uuid.uuid4().hex, "engine_sha256": engine_meta.get("engine_sha256") or hashlib.sha256(engine_path.read_bytes()).hexdigest(), "host": socket.gethostname(),
                                             "thresholds": {"conf_ok": policy.conf_ok, "conf_defect": policy.conf_defect, "nms_iou": policy.nms_iou, "dedupe_iou": policy.dedupe_iou, "inference_budget_s": policy.inference_budget_s, "source": policy.source}})
        self.detector_ready = threading.Event()
        self.threads = [threading.Thread(target=t, name=n, daemon=True) for n, t in (("W1-ingest", self._w1), ("W2-inference", self._w2), ("W3-actuation", self._w3), ("supervisor", self._supervisor))]

    def beat(self, name: str) -> None:
        with self.hb_lock:
            self.heartbeat[name] = time.perf_counter()

    def _fail(self, where: str, e: BaseException) -> None:
        self.errors.append(f"{where}: {type(e).__name__}: {e}")
        self.stop.set()

    # -- W1 ----------------------------------------------------------------------------
    def _w1(self) -> None:
        try:
            self.source.open()
            while not self.stop.is_set() and not self.detector_ready.wait(timeout=0.005):   # interlock: no ingestion before W2 is healthy
                self.beat("W1")
            while not self.stop.is_set():
                self.beat("W1")
                fr = self.source.next(self.encoder.ticks())
                if fr is None:
                    time.sleep(0.0005)
                    continue
                self.stats["frames"] += 1
                self.ledger.trigger(fr.pack_id, fr.trigger_ticks, fr.frame_id)          # ledger first: the pack exists even if the frame is lost
                if fr.pack_id in self.faults.get("drop_frames", ()):                   # injected fault: frame lost after the trigger
                    self.stats["dropped_frames"] += 1
                    continue
                try:
                    self.frames.put(fr, timeout=0.002)
                except queue.Full:                                                      # never drop a pack: mark it, drop the frame
                    self.stats["overflow"] += 1
                    self.ledger.verdict(fr.pack_id, "REJECT", "queue_overflow", {}, fr.t_capture)
        except BaseException as e:  # noqa: BLE001
            self._fail("W1", e)
        finally:
            self.source.close()

    # -- W2 ----------------------------------------------------------------------------
    def _w2(self) -> None:
        try:
            det = TrtDetector(self.engine_path, self.policy)                             # context + stream owned by this thread
            det.warmup()
            self.detector_ready.set()
            while not self.stop.is_set():
                self.beat("W2")
                try:
                    fr = self.frames.get_nowait()     # Queue.get(timeout) rides the 15.6 ms Windows timer; poll with the high-res sleep instead
                except queue.Empty:
                    time.sleep(0.0005)
                    continue
                stall = self.faults.get("stall_ms", {}).get(fr.pack_id)
                if stall:                                                               # injected fault: pipeline stall
                    time.sleep(stall / 1000.0)
                result, t_gpu, t_total = det.infer(fr)
                self.stats["inferences"] += 1
                self.stats["gpu_ms"].append(t_gpu * 1000)
                latency = time.perf_counter() - fr.t_capture
                self.stats["verdict_ms"].append(latency * 1000)
                verdict, reason = result["verdict"], result["reason"]
                if latency > self.policy.inference_budget_s:                            # a slow verdict is not trusted
                    self.stats["timeouts"] += 1
                    verdict, reason = "REJECT", f"inference_timeout:{latency * 1000:.1f}ms>{self.policy.inference_budget_s * 1000:.0f}ms|{reason}"
                self.ledger.verdict(fr.pack_id, verdict, reason, result["scores"], fr.t_capture)
                self.beat("W2")
        except BaseException as e:  # noqa: BLE001
            self.detector_ready.set()
            self._fail("W2", e)

    # -- W3 ----------------------------------------------------------------------------
    def _w3(self) -> None:
        try:
            nozzle = self.cfg.nozzle_ticks
            window_ticks = self.cfg.hit_window_m / self.cfg.encoder_pitch_m
            while not self.stop.is_set():
                self.beat("W3")
                ticks = self.encoder.ticks()
                lead_ticks = int(round(self.encoder.tick_rate() * self.cfg.t_valve_s))   # the only speed-dependent term
                for rec in self.ledger.due(ticks, lead_ticks, nozzle):
                    fire_tick = rec.trigger_ticks + nozzle - lead_ticks
                    if rec.verdict == "PASS":
                        rec = self.ledger.action(rec.pack_id, "pass", ticks, None, None)
                    else:                                                               # REJECT or no verdict -> fire (fail-closed)
                        self.actuator.fire(rec.pack_id, ticks)
                        rec = self.ledger.action(rec.pack_id, "reject", ticks, self.actuator.confirm(rec.pack_id), "no_verdict_at_fire_tick")
                        if ticks - fire_tick > window_ticks:                           # jet fired after the pack left the hit window: a structural escape -> alarm
                            rec.confirmed = False
                            rec.reason = (rec.reason or "") + f"|LATE_ACTUATION:{ticks - fire_tick}ticks"
                            self.stats["late_actuations"] += 1
                            self.alarms.append({"pack_id": rec.pack_id, "late_ticks": ticks - fire_tick, "t": round(time.perf_counter(), 4)})
                    self.audit.write(rec)
                time.sleep(0.0005)
        except BaseException as e:  # noqa: BLE001
            self._fail("W3", e)

    # -- supervisor ----------------------------------------------------------------------
    def _supervisor(self) -> None:
        try:
            self.detector_ready.wait()                       # the line does not run before W2 is healthy; no packs to protect yet
            while not self.stop.is_set():
                now = time.perf_counter()
                with self.hb_lock:
                    stale = {k: round((now - v) * 1000, 1) for k, v in self.heartbeat.items() if now - v > self.policy.heartbeat_stale_s}
                if stale:
                    forced = self.ledger.force_reject_pending("watchdog:" + ",".join(f"{k}={v}ms" for k, v in stale.items()))
                    self.watchdog_events.append({"t": round(now, 4), "stale_ms": stale, "forced_reject": forced})
                    with self.hb_lock:                       # one event per stall, not one per poll
                        for k in stale:
                            self.heartbeat[k] = now
                time.sleep(0.005)
        except BaseException as e:  # noqa: BLE001
            self._fail("supervisor", e)

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        sys.setswitchinterval(0.001)        # pollers must not hold the GIL for the default 5 ms while W2 decodes
        for t in self.threads:
            t.start()
        self.detector_ready.wait(timeout=60)

    def shutdown(self) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(timeout=5)
        self.audit.close()

    def summary(self) -> dict:
        g, v = np.array(self.stats["gpu_ms"] or [0]), np.array(self.stats["verdict_ms"] or [0])
        return {**{k: val for k, val in self.stats.items() if k not in ("gpu_ms", "verdict_ms")},
                "gpu_ms_median": round(float(np.median(g)), 2), "gpu_ms_p99": round(float(np.percentile(g, 99)), 2),
                "verdict_ms_median": round(float(np.median(v)), 2), "verdict_ms_p99": round(float(np.percentile(v, 99)), 2),
                "watchdog_events": len(self.watchdog_events), "alarms": list(self.alarms), "errors": list(self.errors), "ledger": self.ledger.invariant(), "audit_records": self.audit.count}


# --------------------------------------------------------------------------------------
# Self-test: replay validation frames on a simulated belt with injected faults
# --------------------------------------------------------------------------------------
class ReplayFrameSource(DLPackFrameSource):
    """Frames preloaded to the GPU; pack k is 'triggered' when the encoder passes k * pitch."""

    def __init__(self, images: list[torch.Tensor], pitch_ticks: int):
        self.images, self.pitch_ticks = images, pitch_ticks
        self._next = 0
        super().__init__(self._produce)

    def _produce(self, ticks: int):
        if self._next >= len(self.images) or ticks < self._next * self.pitch_ticks:
            return None
        k = self._next
        self._next += 1
        return self.images[k], k, k * self.pitch_ticks

    @property
    def done(self) -> bool:
        return self._next >= len(self.images)


def load_val_frames(n: int) -> tuple[list[torch.Tensor], list[str], list[Path]]:
    from PIL import Image

    paths = [Path(l.strip()) for l in (PROJECT_ROOT / "data" / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    imgs, expected = [], []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        assert arr.shape == (FRAME_H, FRAME_W, 3), arr.shape
        imgs.append(torch.from_numpy(arr).cuda())
        rows = [ln.split() for ln in (p.parent.parent / "labels" / (p.stem + ".txt")).read_text(encoding="utf-8").splitlines() if ln.strip()]
        expected.append("PASS" if all(r[0] == "0" for r in rows) and len(rows) == N_CAVITIES else "REJECT")
    torch.cuda.synchronize()
    return imgs, expected, paths


def self_test(n_packs: int = 20, drop_pack: int = 7, stall_pack: int = 12, stall_ms: float = 60.0) -> int:
    cfg, policy = LineConfig(), Policy.calibrated()
    print(f"policy: conf_ok={policy.conf_ok} conf_defect={policy.conf_defect} nms={policy.nms_iou} dedupe={policy.dedupe_iou} budget={policy.inference_budget_s * 1000:.0f} ms ({policy.source})", flush=True)
    imgs, expected, paths = load_val_frames(n_packs)
    source = ReplayFrameSource(imgs, cfg.pitch_ticks)
    encoder = SimEncoder(cfg)
    actuator = SimActuator()
    ctl = Controller(source, encoder, actuator, cfg=cfg, policy=policy, faults={"drop_frames": {drop_pack}, "stall_ms": {stall_pack: stall_ms}})
    print(f"self-test: {n_packs} packs at {cfg.v_belt_mps} m/s ({60 * cfg.v_belt_mps / cfg.pack_pitch_m:.0f} packs/min), nozzle {cfg.nozzle_ticks} ticks, pitch {cfg.pitch_ticks} ticks, faults: drop frame of pack {drop_pack}, {stall_ms:.0f} ms stall on pack {stall_pack}", flush=True)
    ctl.start()                               # returns once W2 has loaded and warmed the engine
    if ctl.errors:
        print("controller failed to start:", ctl.errors)
        return 1
    encoder.release()                         # line interlock: the belt runs only with a healthy inspection
    deadline = time.perf_counter() + (n_packs * cfg.pack_pitch_m + cfg.d_nozzle_m) / cfg.v_belt_mps + 3.0
    while time.perf_counter() < deadline and not ctl.stop.is_set():
        if source.done and len(ctl.ledger.packs) == n_packs and ctl.ledger.pending() == 0:
            break
        time.sleep(0.01)
    ctl.shutdown()
    s = ctl.summary()
    # ---- assertions ------------------------------------------------------------------
    problems = []
    if s["errors"]:
        problems.append(f"unhandled errors: {s['errors']}")
    if not s["ledger"]["ok"] or s["ledger"]["triggers"] != n_packs:
        problems.append(f"ledger invariant violated: {s['ledger']}")
    if s["audit_records"] != n_packs:
        problems.append(f"audit records {s['audit_records']} != {n_packs}")
    if s["alarms"]:
        problems.append(f"late actuations (possible escapes): {s['alarms']}")
    mismatches = []
    for k, exp in enumerate(expected):
        rec = ctl.ledger.packs.get(k)
        if rec is None:
            problems.append(f"pack {k} missing from ledger")
            continue
        if k == drop_pack:
            if not (rec.verdict == "REJECT" and rec.action == "reject" and ("no_verdict" in (rec.reason or "") or "watchdog" in (rec.reason or ""))):
                problems.append(f"dropped-frame pack {k} not fail-closed: {rec.verdict}/{rec.reason}/{rec.action}")
        elif k == stall_pack:
            if not (rec.verdict == "REJECT" and rec.action == "reject" and ("inference_timeout" in (rec.reason or "") or "watchdog" in (rec.reason or ""))):
                problems.append(f"stalled pack {k} not fail-closed: {rec.verdict}/{rec.reason}/{rec.action}")
        elif rec.verdict != exp and "watchdog" not in (rec.reason or ""):
            mismatches.append((k, exp, rec.verdict, rec.reason))
        if (rec.action == "reject") != (rec.verdict == "REJECT"):
            problems.append(f"pack {k}: action {rec.action} inconsistent with verdict {rec.verdict}")
        if rec.action == "reject" and not rec.confirmed:
            problems.append(f"pack {k}: reject not confirmed")
        window_ticks = cfg.hit_window_m / cfg.encoder_pitch_m
        if rec.action_ticks is not None and abs(rec.action_ticks - (rec.trigger_ticks + cfg.nozzle_ticks - round(cfg.v_belt_mps * cfg.t_valve_s / cfg.encoder_pitch_m))) > window_ticks:
            problems.append(f"pack {k}: actuation tick outside the hit window")
    if mismatches:
        problems.append(f"verdict != label for {len(mismatches)} packs: {mismatches}")
    # ---- audit file check ----------------------------------------------------------------
    run_id = ctl.audit.meta["run_id"]
    lines = [json.loads(l) for l in AUDIT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [r for r in lines if r["run_id"] == run_id]
    req = {"pack_id", "timestamp_utc", "encoder_ticks", "class_confidences", "verdict", "actuation", "engine_sha256"}
    if len(mine) != n_packs or any(not req <= set(r) for r in mine) or sorted(r["pack_id"] for r in mine) != list(range(n_packs)):
        problems.append(f"audit trail records for this run malformed or incomplete ({len(mine)})")
    # ---- report ----------------------------------------------------------------------
    print(json.dumps(s, indent=1), flush=True)
    print(f"{'pack':>4} {'expected':>8} {'verdict':>8} {'action':>7} {'lat ms':>7} {'trig':>6} {'act':>6}  reason")
    for k in range(n_packs):
        r = ctl.ledger.packs[k]
        print(f"{k:4d} {expected[k]:>8} {r.verdict:>8} {r.action:>7} {r.latency_ms if r.latency_ms is not None else '-':>7} {r.trigger_ticks:6d} {r.action_ticks:6d}  {r.reason}")
    print("watchdog events:", json.dumps(ctl.watchdog_events))
    print("audit trail:", AUDIT_PATH, f"(+{len(mine)} records this run, run_id {run_id})")
    print("sample record:", json.dumps(mine[0], sort_keys=True)[:600] if mine else "-")
    print("SELF-TEST:", "PASS" if not problems else f"FAIL {problems}", flush=True)
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Closed-loop inspection controller (W1 ingest, W2 TensorRT, W3 encoder shift register).")
    ap.add_argument("--test", action="store_true", help="replay 20 validation frames on a simulated 1.6 m/s belt with injected faults")
    ap.add_argument("--packs", type=int, default=20)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.test:
        return self_test(args.packs)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
