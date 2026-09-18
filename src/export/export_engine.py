"""Module 4 - static ONNX export, host TensorRT engine build, Jetson packaging, parity + latency audit.

Audited export mechanics (TensorRT 11.3.0.99 / tensorrt-cu13 on Windows Blackwell, verified 2026-09-10):
* The ONNX is exported ONCE, static [1, 3, 640, 640], opset 17, FP32 weights, raw head output
  [1, 8, 8400] (4 box + 4 class rows over 8,400 anchors), Ultralytics metadata stripped into a
  sidecar JSON.  It is the portable artifact: Jetson engines are built on the target from it.
* TensorRT 11 has NO BuilderFlag.FP16 / INT8 and no calibrator classes: precision is carried by
  the network's tensor types.  The host FP16 engine is therefore built STRONGLY TYPED from an
  FP16 copy of the ONNX (float32 I/O kept via Cast nodes; the head's box-decode math is kept in
  FP32 so coordinate rounding cannot erode box parity).  A weakly typed FP32 engine is built from
  the portable ONNX as well, mirroring what the Jetson's TensorRT 10.x produces.
* Only current APIs: build_serialized_network, set_memory_pool_limit(WORKSPACE, 2 GiB),
  execute_async_v3 with torch CUDA tensors as device buffers (no pycuda), timing cache on disk.
* Parity is measured on identical pre-normalised float32 NCHW tensors, on the raw [1, 8, 8400]
  outputs (PyTorch fp32 with cuDNN/matmul forced to IEEE so TF32 cannot leak into the reference,
  ONNX Runtime, TensorRT), then on decoded boxes after a deterministic NMS.

    uv run python src/export/export_engine.py            # everything
    uv run python src/export/export_engine.py --onnx-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"
BEST_PT = PROJECT_ROOT / "models" / "runs" / "yolo11n_baseline" / "weights" / "best.pt"
ONNX_FP32 = EXPORT_DIR / "yolo11n_blister.onnx"
ONNX_FP16 = EXPORT_DIR / "yolo11n_blister_fp16io32.onnx"
ENGINE_FP16 = EXPORT_DIR / "yolo11n_rtx5090_fp16.engine"
ENGINE_FP32 = EXPORT_DIR / "yolo11n_rtx5090_fp32.engine"
TIMING_CACHE = EXPORT_DIR / "rtx5090_timing.cache"
IMGSZ = 640
N_OUT_ROWS = 8          # 4 box + 4 classes
N_ANCHORS = 8400        # (80*80 + 40*40 + 20*20) at 640
WORKSPACE_BYTES = 2 << 30


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# 1. Static FP32 ONNX
# --------------------------------------------------------------------------------------
def export_onnx(best_pt: Path, out: Path) -> dict:
    import onnx
    from ultralytics import YOLO

    out.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(best_pt))
    names = dict(model.names)
    produced = Path(model.export(format="onnx", imgsz=IMGSZ, batch=1, dynamic=False, simplify=True, opset=17, half=False, nms=False, device="cpu"))
    shutil.move(str(produced), str(out))
    m = onnx.load(str(out))
    # strip training metadata (Ultralytics writes names/args/date/version/... as metadata props)
    stripped = {p.key: p.value for p in m.metadata_props}
    del m.metadata_props[:]
    m.doc_string = ""
    m.producer_name, m.producer_version = "blister-export", "1"
    onnx.checker.check_model(m)
    opsets = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    assert opsets.get("ai.onnx") == 17, f"opset {opsets}"
    inp, outp = m.graph.input[0], m.graph.output[0]
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    assert dims(inp) == [1, 3, IMGSZ, IMGSZ], dims(inp)
    assert dims(outp) == [1, N_OUT_ROWS, N_ANCHORS], dims(outp)
    assert inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and outp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    init_types = {onnx.TensorProto.DataType.Name(t.data_type) for t in m.graph.initializer}
    assert "FLOAT16" not in init_types and "BFLOAT16" not in init_types, init_types
    onnx.save(m, str(out))
    info = {
        "path": str(out), "sha256": sha256(out), "bytes": out.stat().st_size, "opset": 17,
        "input": {"name": inp.name, "shape": dims(inp), "dtype": "float32", "layout": "NCHW RGB, /255, letterbox 640x640 pad 114"},
        "output": {"name": outp.name, "shape": dims(outp), "dtype": "float32", "rows": ["cx", "cy", "w", "h"] + [names[i] for i in sorted(names)]},
        "initializer_dtypes": sorted(init_types), "nodes": len(m.graph.node), "names": names,
        "stripped_metadata_keys": sorted(stripped),
    }
    (out.with_suffix(".json")).write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out.parent / (out.name + ".sha256")).write_text(f"{info['sha256']}  {out.name}\n", encoding="utf-8")
    return info


# --------------------------------------------------------------------------------------
# 2a. FP16 ONNX for the strongly typed host engine (float32 I/O, FP32 box decode)
# --------------------------------------------------------------------------------------
def make_fp16_onnx(src: Path, dst: Path) -> dict:
    """Region-aware FP32 -> FP16 conversion with float32 graph I/O.

    onnxconverter-common's converter asserts when the graph output's producer is kept in FP32
    (exactly what the decode-precision rule needs), so the conversion is done here: nodes outside
    the block list get FP16 initializers and FP16 outputs; Cast nodes are inserted only where an
    edge crosses an FP16/FP32 boundary; per-op type rules are respected (Resize roi/scales stay
    float32).  Blocked = the Detect head minus its cv2/cv3 conv stacks (DFL, anchor arithmetic,
    sigmoid, final concat), so pixel coordinates are never rounded to half precision.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper, shape_inference

    m = shape_inference.infer_shapes(onnx.load(str(src)))
    g = m.graph
    out_name = g.output[0].name
    producer = next(n for n in g.node if out_name in n.output)
    head_prefix = producer.name.rsplit("/", 1)[0] + "/"          # e.g. "/model.23/"
    blocked = {n.name for n in g.node if n.name.startswith(head_prefix) and "/cv2" not in n.name and "/cv3" not in n.name}
    # tensors that are float (activations or initializers); everything else (INT64 shapes/indices) is untouched
    tensor_dtype = {vi.name: vi.type.tensor_type.elem_type for vi in list(g.value_info) + list(g.input) + list(g.output)}
    inits = {t.name: t for t in g.initializer}
    for t in g.initializer:
        tensor_dtype[t.name] = t.data_type
    is_float = lambda name: tensor_dtype.get(name) == TensorProto.FLOAT  # noqa: E731
    # inputs that must keep float32 even on converted nodes (ONNX type constraints)
    keep32_inputs = {"Resize": {1, 2}, "NonMaxSuppression": {2, 3, 4}, "TopK": set()}
    producers = {o: n for n in g.node for o in n.output}
    # 1) convert initializers consumed by unblocked nodes (an initializer shared by an unblocked
    #    and a blocked consumer gets an FP16 copy for the unblocked one)
    fp16_tensors: set[str] = set()           # activations/initializers that are FP16 after conversion
    converted_inits: dict[str, str] = {}
    for n in g.node:
        if n.name in blocked:
            continue
        for i, name in enumerate(n.input):
            if name in inits and is_float(name) and i not in keep32_inputs.get(n.op_type, set()):
                if name not in converted_inits:
                    arr = numpy_helper.to_array(inits[name]).astype(np.float16)
                    new_name = name + "_fp16"
                    g.initializer.append(numpy_helper.from_array(arr, new_name))
                    converted_inits[name] = new_name
                    fp16_tensors.add(new_name)
                n.input[i] = converted_inits[name]
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float16), a.t.name))
        for o in n.output:
            if is_float(o):
                fp16_tensors.add(o)
    # drop initializers no longer referenced
    used = {name for n in g.node for name in n.input}
    for t in [t for t in g.initializer if t.name not in used]:
        g.initializer.remove(t)
    # 2) insert casts on boundary edges
    new_nodes = []
    cast_count = 0
    cast_cache: dict[tuple[str, int], str] = {}

    def casted(name: str, to: int) -> str:
        nonlocal cast_count
        key = (name, to)
        if key not in cast_cache:
            out = f"{name}_cast{'16' if to == TensorProto.FLOAT16 else '32'}"
            new_nodes.append(helper.make_node("Cast", [name], [out], to=to, name=f"Cast_{cast_count}"))
            cast_count += 1
            cast_cache[key] = out
        return cast_cache[key]

    graph_inputs = {i.name for i in g.input}
    for n in g.node:
        for i, name in enumerate(n.input):
            if not name or name in inits or name in converted_inits.values() or not (is_float(name) or name in fp16_tensors):
                continue
            src_fp16 = name in fp16_tensors
            want_fp16 = n.name not in blocked and i not in keep32_inputs.get(n.op_type, set())
            if name in graph_inputs and want_fp16:
                n.input[i] = casted(name, TensorProto.FLOAT16)
            elif src_fp16 and not want_fp16:
                n.input[i] = casted(name, TensorProto.FLOAT)
            elif not src_fp16 and want_fp16 and name not in graph_inputs:
                n.input[i] = casted(name, TensorProto.FLOAT16)
    # graph outputs stay float32: if a producer became FP16, cast back under the original output name
    for out in g.output:
        if out.name in fp16_tensors:
            p = producers[out.name]
            inner = out.name + "_fp16"
            p.output[list(p.output).index(out.name)] = inner
            fp16_tensors.add(inner)
            new_nodes.append(helper.make_node("Cast", [inner], [out.name], to=TensorProto.FLOAT, name=f"Cast_out_{cast_count}"))
            cast_count += 1
    # Cast nodes must precede their consumers: rebuild node list in topological order
    all_nodes = list(g.node) + new_nodes
    del g.node[:]
    avail = set(graph_inputs) | {t.name for t in g.initializer}
    pending = all_nodes
    while pending:
        progressed = False
        rest = []
        for n in pending:
            if all((not i) or i in avail for i in n.input):
                g.node.append(n)
                avail.update(n.output)
                progressed = True
            else:
                rest.append(n)
        pending = rest
        if not progressed:
            raise RuntimeError(f"could not topologically order {len(pending)} nodes: {[n.name for n in pending[:5]]}")
    del g.value_info[:]
    m = shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    n16 = sum(1 for t in m.graph.initializer if t.data_type == TensorProto.FLOAT16)
    n32 = sum(1 for t in m.graph.initializer if t.data_type == TensorProto.FLOAT)
    return {"path": str(dst), "sha256": sha256(dst), "head_prefix": head_prefix, "fp32_blocked_nodes": len(blocked), "fp16_initializers": n16, "fp32_initializers": n32, "casts": cast_count, "nodes": len(m.graph.node)}


# --------------------------------------------------------------------------------------
# 2b. TensorRT 11 engine build
# --------------------------------------------------------------------------------------
def build_engine(onnx_path: Path, engine_path: Path, *, strongly_typed: bool, precision_label: str) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed for {onnx_path.name}: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    if not strongly_typed:
        config.set_flag(trt.BuilderFlag.TF32)          # FP32 network; TF32 tensor cores allowed (host only)
    cache = config.create_timing_cache(TIMING_CACHE.read_bytes() if TIMING_CACHE.exists() else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    tmp = engine_path.with_suffix(engine_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(serialized)
    tmp.replace(engine_path)
    TIMING_CACHE.write_bytes(bytes(cache.serialize()))
    # deserialize check (fresh runtime) + I/O description
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    assert engine is not None, "deserialize_cuda_engine returned None"
    io = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        io.append({"name": name, "mode": str(engine.get_tensor_mode(name)).split(".")[-1], "shape": list(engine.get_tensor_shape(name)), "dtype": str(engine.get_tensor_dtype(name)).split(".")[-1]})
    ctx = engine.create_execution_context()
    assert ctx is not None
    del ctx, engine
    props = torch.cuda.get_device_properties(0)
    meta = {
        "engine": str(engine_path), "engine_sha256": sha256(engine_path), "bytes": engine_path.stat().st_size, "build_s": round(build_s, 1),
        "onnx": onnx_path.name, "onnx_sha256": sha256(onnx_path), "precision": precision_label, "strongly_typed": strongly_typed,
        "workspace_bytes": WORKSPACE_BYTES, "tensorrt": trt.__version__, "cuda": torch.version.cuda, "torch": torch.__version__,
        "gpu": props.name, "compute_capability": f"{props.major}.{props.minor}", "io": io, "built": dt.datetime.now().astimezone().isoformat(),
        "portable": False, "note": "TensorRT engines are bound to this GPU architecture and TensorRT version; build Jetson engines on the Jetson from the ONNX.",
    }
    engine_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------------------
# 3. Runners
# --------------------------------------------------------------------------------------
class TrtRunner:
    """Static-shape TensorRT runner; buffers are torch CUDA tensors addressed by pointer."""

    def __init__(self, engine_path: Path):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        dmap = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.INT32: torch.int32, trt.DataType.INT8: torch.int8, trt.DataType.BOOL: torch.bool}
        if hasattr(trt.DataType, "BF16"):
            dmap[trt.DataType.BF16] = torch.bfloat16
        self.bufs: dict[str, torch.Tensor] = {}
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            t = torch.empty(shape, dtype=dmap[self.engine.get_tensor_dtype(name)], device="cuda")
            self.bufs[name] = t
            self.ctx.set_tensor_address(name, t.data_ptr())
            (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(name)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.stream(self.stream):
            self.bufs[self.inputs[0]].copy_(x.to(self.bufs[self.inputs[0]].dtype))
            ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        assert ok, "execute_async_v3 failed"
        return self.bufs[self.outputs[0]].float().clone()

    def bench(self, x: torch.Tensor, warmup: int, iters: int) -> dict:
        self.bufs[self.inputs[0]].copy_(x)
        torch.cuda.synchronize()
        for _ in range(warmup):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        times = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(self.stream)
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            e.record(self.stream)
            e.synchronize()
            times.append(s.elapsed_time(e))
        a = np.array(times)
        return {"warmup": warmup, "iters": iters, "median_ms": round(float(np.median(a)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3), "mean_ms": round(float(a.mean()), 3), "min_ms": round(float(a.min()), 3)}


class TorchRunner:
    """FP32 IEEE reference (no TF32) from best.pt, fused like the exporter does."""

    def __init__(self, best_pt: Path):
        from ultralytics import YOLO

        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        m = YOLO(str(best_pt)).model
        m = m.fuse() if hasattr(m, "fuse") else m
        self.model = m.float().eval().cuda()

    @torch.inference_mode()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x.cuda().float())
        y = y[0] if isinstance(y, (tuple, list)) else y
        return y.float()

    def bench(self, x: torch.Tensor, warmup: int, iters: int) -> dict:
        x = x.cuda()
        with torch.inference_mode():
            for _ in range(warmup):
                self.model(x)
            torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                self.model(x)
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e))
        a = np.array(times)
        return {"warmup": warmup, "iters": iters, "median_ms": round(float(np.median(a)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3)}


class OrtRunner:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = None
        self.provider = None
        # CUDA EP defaults to TF32 for conv/matmul; pin IEEE so ORT is a true FP32 reference
        for providers in ([("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"], ["CPUExecutionProvider"]):
            try:
                self.sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
                self.provider = self.sess.get_providers()[0]
                break
            except Exception as e:  # noqa: BLE001
                print(f"ORT providers {providers} unavailable: {type(e).__name__}: {str(e)[:120]}")
        assert self.sess is not None
        self.inp = self.sess.get_inputs()[0].name

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sess.run(None, {self.inp: x.cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(np.ascontiguousarray(y)).float()


# --------------------------------------------------------------------------------------
# 4. Preprocess, decode, parity
# --------------------------------------------------------------------------------------
def preprocess(path: Path) -> torch.Tensor:
    """Identical letterbox for every backend: RGB, resize long side to 640, pad 114, /255, NCHW."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    r = IMGSZ / max(w, h)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    canvas.paste(im, ((IMGSZ - nw) // 2, (IMGSZ - nh) // 2))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None]


def decode(raw: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.7) -> torch.Tensor:
    """[1, 8, 8400] -> [N, 6] (x1, y1, x2, y2, conf, cls) after deterministic class-wise NMS."""
    import torchvision

    x = raw[0].T.float().cpu()                     # [8400, 8]
    boxes, scores = x[:, :4], x[:, 4:]
    conf, cls = scores.max(1)
    keep = conf > conf_thres
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
    xyxy = torch.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
    idx = torchvision.ops.batched_nms(xyxy, conf, cls, iou_thres)
    return torch.cat([xyxy[idx], conf[idx, None], cls[idx, None].float()], 1)


def box_iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    import torchvision

    return torchvision.ops.box_iou(a[:, :4], b[:, :4]) if len(a) and len(b) else torch.zeros((len(a), len(b)))


def tensor_metrics(ref: torch.Tensor, other: torch.Tensor, conf_thres: float = 0.25) -> dict:
    """Raw [1, 8, 8400] comparison.  The unmasked box error is dominated by anchors with ~zero
    confidence (flat DFL distributions whose expected value swings with rounding), so the box
    error is also reported over the anchors a decoder would keep (reference max score > thres)."""
    r, o = ref.flatten().double(), other.flatten().double()
    cos = float(torch.nn.functional.cosine_similarity(r[None], o[None]).item())
    d = (ref - other).abs()
    keep = ref[0, 4:, :].max(0).values > conf_thres
    box_conf = float(d[0, :4, :][:, keep].max()) if bool(keep.any()) else 0.0
    return {"max_abs_err": round(float(d.max()), 5), "max_abs_err_boxes_px": round(float(d[:, :4, :].max()), 4), "max_abs_err_boxes_px_conf": round(box_conf, 4),
            "max_abs_err_scores": round(float(d[:, 4:, :].max()), 6), "mean_abs_err": round(float(d.mean()), 6), "cosine": round(cos, 8), "cosine_conf": round(float(torch.nn.functional.cosine_similarity(ref[0][:, keep].T.flatten().double()[None], other[0][:, keep].T.flatten().double()[None]).item()), 8) if bool(keep.any()) else 1.0}


def box_parity(ref_raw: torch.Tensor, other_raw: torch.Tensor) -> dict:
    a, b = decode(ref_raw), decode(other_raw)
    if len(a) == 0 and len(b) == 0:
        return {"n_ref": 0, "n_other": 0, "min_iou": 1.0, "mean_iou": 1.0, "frac_iou_ge_098": 1.0, "class_mismatch": 0}
    ious = box_iou_matrix(a, b)
    same_cls = a[:, 5][:, None] == b[:, 5][None, :]
    ious = torch.where(same_cls, ious, torch.zeros_like(ious))
    best = ious.max(1).values if len(b) else torch.zeros(len(a))
    return {"n_ref": int(len(a)), "n_other": int(len(b)), "min_iou": round(float(best.min()), 4) if len(a) else 1.0, "mean_iou": round(float(best.mean()), 4) if len(a) else 1.0,
            "frac_iou_ge_098": round(float((best >= 0.98).float().mean()), 4) if len(a) else 1.0, "class_mismatch": int((best == 0).sum()) if len(a) else 0}


def aggregate(per_image: list[dict]) -> dict:
    out = {}
    for k in per_image[0]:
        vals = [d[k] for d in per_image]
        if k in ("cosine", "cosine_conf", "min_iou", "frac_iou_ge_098"):
            out[k + "_min"] = round(min(vals), 6)
            out[k + "_mean"] = round(float(np.mean(vals)), 6)
        elif k.startswith("max_") or k in ("class_mismatch",):
            out[k + "_max"] = round(max(vals), 6)
        elif k in ("n_ref", "n_other"):
            out[k + "_total"] = int(sum(vals))
        else:
            out[k + "_mean"] = round(float(np.mean(vals)), 6)
    return out


def run_parity(best_pt: Path, onnx_fp32: Path, engines: dict[str, Path], images: list[Path]) -> dict:
    torch_r = TorchRunner(best_pt)
    ort_r = OrtRunner(onnx_fp32)
    trt_r = {k: TrtRunner(p) for k, p in engines.items()}
    pairs = {f"torch_vs_ort[{ort_r.provider}]": [], **{f"torch_vs_trt_{k}": [] for k in trt_r}, **{f"ort_vs_trt_{k}": [] for k in trt_r}}
    boxes = {f"torch_vs_trt_{k}": [] for k in trt_r}
    boxes["torch_vs_ort"] = []
    n_boxes = 0
    for p in images:
        x = preprocess(p)
        y_t = torch_r.infer(x).cpu()
        y_o = ort_r.infer(x).cpu()
        y_e = {k: r.infer(x.cuda()).cpu() for k, r in trt_r.items()}
        assert y_t.shape == y_o.shape == (1, N_OUT_ROWS, N_ANCHORS), (y_t.shape, y_o.shape)
        pairs[f"torch_vs_ort[{ort_r.provider}]"].append(tensor_metrics(y_t, y_o))
        boxes["torch_vs_ort"].append(box_parity(y_t, y_o))
        for k, y in y_e.items():
            assert y.shape == (1, N_OUT_ROWS, N_ANCHORS), y.shape
            pairs[f"torch_vs_trt_{k}"].append(tensor_metrics(y_t, y))
            pairs[f"ort_vs_trt_{k}"].append(tensor_metrics(y_o, y))
            boxes[f"torch_vs_trt_{k}"].append(box_parity(y_t, y))
        n_boxes += len(decode(y_t))
    return {"images": len(images), "ort_provider": ort_r.provider, "boxes_in_reference": n_boxes,
            "raw_tensor": {k: aggregate(v) for k, v in pairs.items()}, "decoded_boxes": {k: aggregate(v) for k, v in boxes.items()}}


# --------------------------------------------------------------------------------------
# 5. Jetson packaging
# --------------------------------------------------------------------------------------
ORIN_SCRIPT = """#!/usr/bin/env bash
# Build and benchmark the Jetson Orin TensorRT engine ON THE TARGET.
# TensorRT engines are bound to the GPU architecture and TensorRT version: the RTX 5090 engine in
# this folder (sm_120, TensorRT 11.3) will not deserialize on Orin (sm_87, TensorRT 10.x).
# Portable artifact: yolo11n_blister.onnx (opset 17, static [1,3,640,640], FP32 weights, raw
# [1,8,8400] output). JetPack 6.2 = TensorRT 10.3 / CUDA 12.6; JetPack 7.2 = TensorRT 10.16 / CUDA 13.2.
set -euo pipefail
cd "$(dirname "$0")"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX=yolo11n_blister.onnx

sha256sum -c "${ONNX}.sha256"                       # refuse to build from a modified graph
sudo nvpmodel -m 0 2>/dev/null || true               # MAXN (MAXN_SUPER on NX / Nano Super)
sudo jetson_clocks 2>/dev/null || true

# --- FP16 engine (the deployment default) --------------------------------------------------
"$TRTEXEC" --onnx="$ONNX" \\
           --saveEngine=yolo11n_orin_fp16.engine \\
           --fp16 \\
           --memPoolSize=workspace:2048 \\
           --timingCacheFile=orin_timing.cache \\
           --skipInference

# --- benchmark: 500 warm-up + 500 timed iterations, CUDA graph, p99 -----------------------
"$TRTEXEC" --loadEngine=yolo11n_orin_fp16.engine --useCudaGraph --warmUp=500 --iterations=500 --percentile=99 \\
           | tee orin_fp16_benchmark.txt

# --- optional INT8 (calibrated on synthetic validation frames) ---------------------------
# 1. Copy the validation frames listed in data/val.txt to this folder (e.g. ./calib_frames/).
# 2. Produce the calibration cache ON THE JETSON with the entropy calibrator (needs the
#    JetPack TensorRT python bindings and torch):
#       python3 int8_calibrate_orin.py --onnx "$ONNX" --images-dir ./calib_frames --n 500 --cache orin_int8.cache
# 3. Build the mixed INT8/FP16 engine from the SAME ONNX with that cache:
#       "$TRTEXEC" --onnx="$ONNX" --saveEngine=yolo11n_orin_int8.engine --int8 --fp16 \\
#                  --calib=orin_int8.cache --memPoolSize=workspace:2048 --timingCacheFile=orin_timing.cache --skipInference
# 4. Accept INT8 only if per-class recall on the frozen validation set is unchanged at the
#    deployed thresholds (expect < 1 ms gain on a nano model; INT8 is worth it for s-models).
# Note: TensorRT 11.x (not on JetPack) removed the --fp16/--int8 builder flags and the calibrator
# API; on TensorRT 11 hosts precision comes from typed ONNX graphs (see export_engine.py).
"""

ORIN_CALIB = '''"""INT8 entropy calibration for the Jetson build - run ON THE JETSON (TensorRT 10.x bindings).

    python3 int8_calibrate_orin.py --onnx yolo11n_blister.onnx --images-dir ./calib_frames --n 500 --cache orin_int8.cache

Frames are preprocessed exactly like the deployment path (RGB, letterbox 640, pad 114, /255).
"""
import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from PIL import Image

IMGSZ = 640


def preprocess(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    r = IMGSZ / max(w, h)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    canvas.paste(im, ((IMGSZ - nw) // 2, (IMGSZ - nh) // 2))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, files, cache):
        super().__init__()
        self.files, self.cache, self.i = files, Path(cache), 0
        self.buf = torch.empty((1, 3, IMGSZ, IMGSZ), dtype=torch.float32, device="cuda")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.i >= len(self.files):
            return None
        self.buf.copy_(torch.from_numpy(preprocess(self.files[self.i]))[None])
        self.i += 1
        return [int(self.buf.data_ptr())]

    def read_calibration_cache(self):
        return self.cache.read_bytes() if self.cache.exists() else None

    def write_calibration_cache(self, cache):
        self.cache.write_bytes(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--cache", default="orin_int8.cache")
    a = ap.parse_args()
    files = sorted(Path(a.images_dir).glob("*.png"))[: a.n]
    assert files, "no calibration frames"
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(Path(a.onnx).read_bytes()), [parser.get_error(i) for i in range(parser.num_errors)]
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = Calibrator(files, a.cache)
    plan = builder.build_serialized_network(network, config)   # runs calibration, writes the cache
    assert plan is not None
    print("calibration cache written:", a.cache, "(engine plan discarded; build the deployment engine with trtexec --calib)")


if __name__ == "__main__":
    main()
'''


def write_orin_package(onnx_fp32: Path) -> list[Path]:
    sh = EXPORT_DIR / "build_orin_engine.sh"
    sh.write_text(ORIN_SCRIPT, encoding="utf-8", newline="\n")
    calib = EXPORT_DIR / "int8_calibrate_orin.py"
    calib.write_text(ORIN_CALIB, encoding="utf-8", newline="\n")
    return [sh, calib, onnx_fp32.parent / (onnx_fp32.name + ".sha256")]


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Static ONNX export, TensorRT 11 host engines, Jetson package, parity + latency audit.")
    ap.add_argument("--best", type=Path, default=BEST_PT)
    ap.add_argument("--onnx-only", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--n-parity", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    print("[1/5] static FP32 ONNX export", flush=True)
    summary["onnx_fp32"] = export_onnx(args.best, ONNX_FP32)
    print(json.dumps({k: summary["onnx_fp32"][k] for k in ("sha256", "bytes", "opset", "input", "output", "initializer_dtypes", "nodes")}, indent=1), flush=True)
    summary["onnx_fp16"] = make_fp16_onnx(ONNX_FP32, ONNX_FP16)
    print("FP16 ONNX:", json.dumps(summary["onnx_fp16"]), flush=True)
    summary["orin_package"] = [str(p) for p in write_orin_package(ONNX_FP32)]
    if args.onnx_only:
        (EXPORT_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0

    if not args.skip_build:
        print("[2/5] TensorRT 11 host engines (FP16 strongly typed, FP32 weakly typed)", flush=True)
        summary["engine_fp16"] = build_engine(ONNX_FP16, ENGINE_FP16, strongly_typed=True, precision_label="fp16 (typed graph, fp32 I/O, fp32 box decode)")
        print(json.dumps({k: summary["engine_fp16"][k] for k in ("engine", "bytes", "build_s", "io", "tensorrt", "gpu")}, indent=1), flush=True)
        summary["engine_fp32"] = build_engine(ONNX_FP32, ENGINE_FP32, strongly_typed=False, precision_label="fp32 (+TF32)")
        print(json.dumps({k: summary["engine_fp32"][k] for k in ("engine", "bytes", "build_s")}, indent=1), flush=True)

    val_list = PROJECT_ROOT / "data" / "val.txt"
    images = [Path(l.strip()) for l in val_list.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_parity]
    if not args.skip_parity:
        print(f"[3/5] parity on {len(images)} validation images", flush=True)
        summary["parity"] = run_parity(args.best, ONNX_FP32, {"fp16": ENGINE_FP16, "fp32": ENGINE_FP32}, images)
        print(json.dumps(summary["parity"], indent=1), flush=True)
    if not args.skip_bench:
        print("[4/5] latency", flush=True)
        x = preprocess(images[0]).cuda()
        summary["latency"] = {"trt_fp16": TrtRunner(ENGINE_FP16).bench(x, args.warmup, args.iters), "trt_fp32": TrtRunner(ENGINE_FP32).bench(x, args.warmup, args.iters), "torch_fp32_eager": TorchRunner(args.best).bench(x, args.warmup, 200)}
        print(json.dumps(summary["latency"], indent=1), flush=True)
    print("[5/5] summary", flush=True)
    (EXPORT_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("written", EXPORT_DIR / "export_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
