"""INT8 entropy calibration for the Jetson build - run ON THE JETSON (TensorRT 10.x bindings).

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
