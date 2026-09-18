"""Module 2 - headless synthetic data generation (Mode 1).

Renders the inspection cell authored by ``blister_factory`` with the NVIDIA Omniverse
libraries (Set A) and writes an Ultralytics-style YOLO dataset:

    <out>/images/<stem>.png
    <out>/labels/<stem>.txt        "<class_id> <xc> <yc> <w> <h>"  normalised to [0, 1]
    <out>/data.yaml, manifest.jsonl, run_manifest.json

Per frame (deterministic in (seed, frame_index)):
* pack state from ``PackScheduler`` -> one visible variant per cavity (visibility tokens);
* pack pose on the belt, camera pose (cone +/-8 deg, distance +/-15 %, roll), lights
  (intensity, colour temperature, key-light direction), material roughness and pill colour;
* one batched ordinal, warm-up steps, then RGB + semantic capture with every handle released;
* boxes from the semantic mask extents, cross-checked against the projected pocket footprint
  (camera model), exactly ten boxes per pack with strict integer class ids;
* simulated strobe motion blur along the belt direction (v_belt * t_exposure / GSD px).

    uv run python src/simulation/sdg_pipeline.py --smoke-test 10
    uv run python src/simulation/sdg_pipeline.py --frames 20000 --out data --val-fraction 0.1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from simulation.assets import blister_factory as bf  # noqa: E402
from simulation.ovx_runtime import (  # noqa: E402
    Frame,
    Scene,
    check_versions,
    kelvin_to_rgb,
    parse_label,
    rot_z,
    usd_matrix,
)

PRODUCT = "/Render/Camera"
N_CAV = bf.N_COLS * bf.N_ROWS


# --------------------------------------------------------------------------------------
# Camera model (pinhole; identity orientation looks down -Z with +Y up, +X right)
# --------------------------------------------------------------------------------------
@dataclass
class CameraPose:
    R: np.ndarray        # 3x3 column-vector rotation, world <- camera
    t: np.ndarray        # camera position in world

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixel coordinates (N, 2), x right, y down."""
        pc = (pts_world - self.t) @ self.R          # = R^T (p - t) for each row
        z = -pc[:, 2]
        z = np.where(z < 1e-6, 1e-6, z)
        u = bf.IMAGE_W / 2 + bf.F_PX * pc[:, 0] / z
        v = bf.IMAGE_H / 2 - bf.F_PX * pc[:, 1] / z
        return np.stack([u, v], axis=1)

    def plane_depth(self, z_world: float) -> np.ndarray:
        """Camera-axis depth (H, W) at which each pixel's ray meets the horizontal plane z=z_world.
        Matches ovrtx's DistanceToImagePlaneSD for that plane, so (depth < plane_depth - h) is a
        purely geometric 'raised above the foil by at least h' mask."""
        u = (np.arange(bf.IMAGE_W) + 0.5 - bf.IMAGE_W / 2) / bf.F_PX
        v = -(np.arange(bf.IMAGE_H) + 0.5 - bf.IMAGE_H / 2) / bf.F_PX
        dz = self.R[2, 0] * u[None, :] + self.R[2, 1] * v[:, None] - self.R[2, 2]   # world z of R @ (u, v, -1)
        dz = np.where(np.abs(dz) < 1e-9, -1e-9, dz)
        return (z_world - self.t[2]) / dz


def rot_axis_angle(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


# --------------------------------------------------------------------------------------
# Randomisation
# --------------------------------------------------------------------------------------
PILL_PALETTE = np.array([
    [0.95, 0.95, 0.92], [0.90, 0.87, 0.80], [0.75, 0.12, 0.10], [0.15, 0.25, 0.70],
    [0.90, 0.75, 0.15], [0.90, 0.55, 0.60], [0.20, 0.55, 0.30],
], dtype=np.float32)


@dataclass
class FrameParams:
    frame: int
    pack_id: int
    states: list[str]
    pack_xy_yaw: tuple[float, float, float]
    cam_tilt_axis_deg: float
    cam_tilt_deg: float
    cam_roll_deg: float
    cam_dist_factor: float
    cam_lookat_xy: tuple[float, float]
    dome_intensity: float
    dome_kelvin: float
    key_intensity: float
    key_kelvin: float
    key_dir_deg: tuple[float, float]
    rough_foil: float
    rough_pill: float
    rough_belt: float
    frost_glass: float
    pill_rgb: tuple[float, float, float]
    foil_rgb: tuple[float, float, float]
    exposure_s: float
    ring_intensity: float = bf.RING_INTENSITY      # asset v2 ring light (DiskLight child of the camera)
    blur_px: float = 0.0
    box_sources: list[str] | None = None


class Randomizer:
    """Draws all per-frame parameters from ``default_rng([seed, frame])``."""

    def __init__(self, seed: int, scheduler: bf.PackScheduler, cone_deg: float = 8.0, dist_pct: float = 15.0):
        self.seed, self.scheduler, self.cone_deg, self.dist_pct = seed, scheduler, cone_deg, dist_pct

    def sample(self, frame: int) -> FrameParams:
        rng = np.random.default_rng([self.seed, frame, 7])
        state = self.scheduler.sample(frame)
        x_margin = (bf.FOV_ALONG_M - bf.PACK_L_M) / 2 - 0.012          # keep the pack fully in view
        pack_xy_yaw = (float(rng.uniform(-x_margin, x_margin)), float(rng.uniform(-0.010, 0.010)), float(rng.uniform(-4.0, 4.0)))
        foil_grey = float(rng.uniform(0.76, 0.90))
        return FrameParams(
            frame=frame, pack_id=frame, states=list(state.states), pack_xy_yaw=pack_xy_yaw,
            cam_tilt_axis_deg=float(rng.uniform(0, 360)), cam_tilt_deg=float(rng.uniform(0, self.cone_deg)),
            cam_roll_deg=float(rng.uniform(-5, 5)), cam_dist_factor=float(rng.uniform(1 - self.dist_pct / 100, 1 + self.dist_pct / 100)),
            cam_lookat_xy=(pack_xy_yaw[0] + float(rng.uniform(-0.008, 0.008)), pack_xy_yaw[1] + float(rng.uniform(-0.008, 0.008))),
            dome_intensity=float(rng.uniform(350, 1400)), dome_kelvin=float(rng.uniform(3000, 6500)),
            key_intensity=float(rng.uniform(0, 1200)), key_kelvin=float(rng.uniform(3000, 6500)),
            key_dir_deg=(float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50))),
            rough_foil=float(rng.uniform(0.12, 0.6)), rough_pill=float(rng.uniform(0.3, 0.9)),
            rough_belt=float(rng.uniform(0.5, 0.95)), frost_glass=float(rng.uniform(0.0, 0.25)),
            pill_rgb=tuple(float(v) for v in PILL_PALETTE[int(rng.integers(len(PILL_PALETTE)))]),
            foil_rgb=(foil_grey, foil_grey + float(rng.uniform(-0.02, 0.02)), min(1.0, foil_grey + float(rng.uniform(0.0, 0.05)))),
            exposure_s=float(rng.uniform(30e-6, 120e-6)),
            ring_intensity=float(rng.uniform(0.5, 1.5)) * bf.RING_INTENSITY,
        )


def camera_pose(p: FrameParams) -> CameraPose:
    """Tilted within a cone about the look-at point, distance-scaled, rolled about its own axis."""
    axis = (math.cos(math.radians(p.cam_tilt_axis_deg)), math.sin(math.radians(p.cam_tilt_axis_deg)), 0.0)
    R = rot_axis_angle(axis, p.cam_tilt_deg) @ rot_z(p.cam_roll_deg)
    target = np.array([p.cam_lookat_xy[0], p.cam_lookat_xy[1], bf.PACK_T_M], dtype=np.float64)
    t = target + R @ np.array([0.0, 0.0, bf.CAM_HEIGHT_M * p.cam_dist_factor])
    return CameraPose(R=R, t=t)


def key_light_matrix(p: FrameParams) -> np.ndarray:
    from simulation.ovx_runtime import rot_x, rot_y

    return usd_matrix(rot_x(p.key_dir_deg[0]) @ rot_y(p.key_dir_deg[1]), (0.0, 0.0, 0.0))


class SceneDriver:
    """Applies FrameParams to the loaded stage with a minimum of writes (visibility diffs)."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.prev_states: list[str] | None = None
        self.leaves = {(i, s): bf.variant_leaf_paths(i, s) for i in range(N_CAV) for s in bf.STATES}

    def prime(self, p: FrameParams, cam: CameraPose) -> None:
        """Apply once and render a throw-away step: on this build the first transform write after
        population is not visible in the very next frame (verified 2026-09-10)."""
        o = self.apply(p, cam)
        self.scene.render(PRODUCT, steps=1, want_seg=False, ordinal=o)

    def apply(self, p: FrameParams, cam: CameraPose) -> int:
        """Write order matters on this build: visibility, lights and materials FIRST, transforms
        LAST within the ordinal.  Transforms followed by scalar writes in the same ordinal render
        at a stale pose from the third frame on (isolated 2026-09-10, recipe matrix a-f)."""
        sc = self.scene
        ordinal = sc.begin()
        # 1) cavity state variants: hide everything not active, show the active one (diff vs previous frame)
        for i in range(N_CAV):
            for s in bf.STATES:
                active = p.states[i] == s
                was_active = (self.prev_states[i] == s) if self.prev_states else (s == "pill_ok")
                if active != was_active:
                    sc.write_token(self.leaves[(i, s)], "visibility", "inherited" if active else "invisible", ordinal=ordinal)
        self.prev_states = list(p.states)
        # 2) lights
        sc.write_float(["/World/DomeLight"], "inputs:intensity", [p.dome_intensity], ordinal=ordinal)
        sc.write_color(["/World/DomeLight"], "inputs:color", np.array([kelvin_to_rgb(p.dome_kelvin)]), ordinal=ordinal)
        sc.write_float(["/World/KeyLight"], "inputs:intensity", [p.key_intensity], ordinal=ordinal)
        sc.write_color(["/World/KeyLight"], "inputs:color", np.array([kelvin_to_rgb(p.key_kelvin)]), ordinal=ordinal)
        sc.write_float(["/World/Camera/RingLight"], "inputs:intensity", [p.ring_intensity], ordinal=ordinal)
        # 3) materials
        sc.write_float(["/World/Looks/Foil/Shader"], "inputs:reflection_roughness_constant", [p.rough_foil], ordinal=ordinal)
        sc.write_color(["/World/Looks/Foil/Shader"], "inputs:diffuse_color_constant", np.array([p.foil_rgb]), ordinal=ordinal)
        sc.write_float(["/World/Looks/Seal/Shader"], "inputs:reflection_roughness_constant", [min(0.95, p.rough_foil + 0.12)], ordinal=ordinal)
        sc.write_color(["/World/Looks/Seal/Shader"], "inputs:diffuse_color_constant", np.array([p.foil_rgb]), ordinal=ordinal)
        sc.write_float(["/World/Looks/Pill/Shader"], "inputs:reflection_roughness_constant", [p.rough_pill], ordinal=ordinal)
        sc.write_color(["/World/Looks/Pill/Shader"], "inputs:diffuse_color_constant", np.array([p.pill_rgb]), ordinal=ordinal)
        sc.write_float(["/World/Looks/Belt/Shader"], "inputs:reflection_roughness_constant", [p.rough_belt], ordinal=ordinal)
        if bf.GLASS_MODE == "omniglass":
            sc.write_float(["/World/Looks/Glass/Shader"], "inputs:frosting_roughness", [p.frost_glass], ordinal=ordinal)
        # 4) transforms LAST: pack pose on the belt, camera, key light
        x, y, yaw = p.pack_xy_yaw
        sc.write_xforms(["/World/Pack"], usd_matrix(rot_z(yaw), (x, y, 0.0))[None], ordinal=ordinal)
        sc.write_xforms(["/World/Camera"], usd_matrix(cam.R, cam.t)[None], ordinal=ordinal)
        sc.write_xforms(["/World/KeyLight"], key_light_matrix(p)[None], ordinal=ordinal)
        sc.seal(ordinal)
        return ordinal


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
def pack_local_to_world(pts: np.ndarray, pack_xy_yaw) -> np.ndarray:
    x, y, yaw = pack_xy_yaw
    return pts @ rot_z(yaw).T + np.array([x, y, 0.0])


def projected_box(cam: CameraPose, idx: int, pack_xy_yaw) -> tuple[float, float, float, float]:
    px = cam.project(pack_local_to_world(bf.pocket_footprint_corners(idx), pack_xy_yaw))
    return float(px[:, 0].min()), float(px[:, 1].min()), float(px[:, 0].max()), float(px[:, 1].max())


def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def yolo_labels(frame: Frame, p: FrameParams, cam: CameraPose) -> tuple[list[tuple[int, float, float, float, float]], list[str], list[float], int]:
    """Exactly ten (class_id, xc, yc, w, h) rows, one per cavity, normalised to [0, 1].

    Identity and class come from what we scheduled (the cavity index and its state) anchored by
    the camera projection of the pocket footprint; the semantic mask only tightens the box.  Each
    segmentation id is assigned to the nearest projected cavity centre, so the labels never depend
    on the renderer's id->label map, which lags the segmentation by a frame after visibility
    toggles on this build (observed 2026-09-10: cavity 8's pixels carried cavity 9's label).  The
    map is decoded anyway and disagreements are counted (fourth return value) for monitoring.
    """
    seg = frame.seg
    proj_boxes = [projected_box(cam, i, p.pack_xy_yaw) for i in range(N_CAV)]
    proj_areas = np.array([max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in proj_boxes])
    # Foreground = every segment that is neither background/belt nor the pack body (those are far
    # larger than a pocket).  Segment ids are deliberately NOT matched to labels: after visibility
    # toggles the renderer reassigns ids and its id->label map lags, and a single id can even span
    # two pockets.  Cavity identity comes from the projected footprint window instead.
    if frame.depth is not None:
        # Primary foreground: rendered depth closer than the foil plane by >= 0.8 mm (domes rise
        # 4.5 mm, crushed domes 1.6 mm, pills 3 mm).  Independent of segmentation ids entirely.
        fg = np.isfinite(frame.depth) & (frame.depth < cam.plane_depth(bf.PACK_T_M) - 0.0008)
    else:
        ids, counts = np.unique(seg, return_counts=True)
        big = {int(s) for s, c in zip(ids.tolist(), counts.tolist()) if c > 3.0 * proj_areas.max()}
        fg = ~np.isin(seg, list(big)) if big else np.ones(seg.shape, bool)
    # id-map cross-check for monitoring: how many cavity windows contain pixels whose mapped label
    # disagrees with the scheduled cavity/state (expected 0 when the map is fresh)
    label_of = {sid: bf.parse_cavity_label(parse_label(raw)) for sid, raw in frame.labels.items()}
    rows, sources, ious = [], [], []
    mismatches = 0
    for i in range(N_CAV):
        state = p.states[i]
        proj = proj_boxes[i]
        pw, ph = proj[2] - proj[0], proj[3] - proj[1]
        cx, cy = (proj[0] + proj[2]) / 2, (proj[1] + proj[3]) / 2
        # window half-size 0.65 x footprint: tolerates ~10 px projection error, cannot reach the
        # neighbouring pocket (pockets are ~1.3 footprints apart centre to centre)
        wx0, wx1 = int(max(0, cx - 0.65 * pw)), int(min(bf.IMAGE_W, cx + 0.65 * pw))
        wy0, wy1 = int(max(0, cy - 0.65 * ph)), int(min(bf.IMAGE_H, cy + 0.65 * ph))
        sub = fg[wy0:wy1, wx0:wx1]
        n_fg = int(sub.sum())
        if n_fg >= 0.35 * proj_areas[i]:
            ys, xs = np.where(sub)
            x0, y0, x1, y1 = wx0 + int(xs.min()), wy0 + int(ys.min()), wx0 + int(xs.max()) + 1, wy0 + int(ys.max()) + 1
            src = "mask"
            win_ids = {int(s) for s in np.unique(seg[wy0:wy1, wx0:wx1][sub]).tolist()}
            if any(label_of.get(s) not in (None, (i, state)) for s in win_ids):
                mismatches += 1
        else:
            x0, y0, x1, y1 = proj
            src = "projection" if n_fg == 0 else "projection(mask-too-small)"
        x0, x1 = max(0.0, min(bf.IMAGE_W, x0)), max(0.0, min(bf.IMAGE_W, x1))
        y0, y1 = max(0.0, min(bf.IMAGE_H, y0)), max(0.0, min(bf.IMAGE_H, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise RuntimeError(f"frame {p.frame}: cavity {i} ({state}) has a degenerate box {x0, y0, x1, y1}")
        rows.append((bf.CLASS_IDS[state], (x0 + x1) / 2 / bf.IMAGE_W, (y0 + y1) / 2 / bf.IMAGE_H, (x1 - x0) / bf.IMAGE_W, (y1 - y0) / bf.IMAGE_H))
        sources.append(src)
        ious.append(iou((x0, y0, x1, y1), proj))
    assert len(rows) == N_CAV
    return rows, sources, ious, mismatches


def format_labels(rows) -> str:
    return "".join(f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in rows)


# --------------------------------------------------------------------------------------
# Image post-processing: strobe motion blur + sensor noise
# --------------------------------------------------------------------------------------
def _shift_bilinear(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    ix, iy = math.floor(dx), math.floor(dy)
    fx, fy = dx - ix, dy - iy
    def sh(a, sx, sy):
        return np.roll(np.roll(a, sx, axis=1), sy, axis=0)
    return ((1 - fx) * (1 - fy) * sh(img, ix, iy) + fx * (1 - fy) * sh(img, ix + 1, iy)
            + (1 - fx) * fy * sh(img, ix, iy + 1) + fx * fy * sh(img, ix + 1, iy + 1))


def motion_blur(rgb: np.ndarray, vec_px: np.ndarray, taps: int = 7) -> np.ndarray:
    """Average ``taps`` copies shifted along ``vec_px`` (total length |vec|), bilinear sub-pixel."""
    length = float(np.hypot(*vec_px))
    if length < 0.05:
        return rgb
    acc = np.zeros(rgb.shape, np.float32)
    for k in range(taps):
        f = -0.5 + (k + 0.5) / taps
        acc += _shift_bilinear(rgb.astype(np.float32), f * vec_px[0], f * vec_px[1])
    return acc / taps


def blur_vector_px(cam: CameraPose, p: FrameParams) -> np.ndarray:
    """Pixel displacement of a point on the pack during the exposure (belt moves along +X)."""
    z = bf.PACK_T_M + bf.POCKET_H_M / 2
    a = np.array([[p.pack_xy_yaw[0], p.pack_xy_yaw[1], z], [p.pack_xy_yaw[0] + bf.V_BELT_MPS * p.exposure_s, p.pack_xy_yaw[1], z]])
    px = cam.project(a)
    return px[1] - px[0]


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def run(out_dir: Path, n_frames: int, seed: int, warmup: int, val_fraction: float, noise_std: float, smoke: bool, rates: dict | None = None) -> int:
    from PIL import Image

    versions = check_versions()
    out_dir = Path(out_dir)
    img_dir, lab_dir = out_dir / "images", out_dir / "labels"
    if smoke:  # smoke artefacts are disposable: start from a clean data/smoke (never touch real outputs)
        for d in (img_dir, lab_dir):
            if d.exists():
                for fp in d.glob("smoke_*"):
                    fp.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    scheduler = bf.PackScheduler(seed, **(rates or {}))
    # exact split: every k-th frame is validation (k = round(1 / val_fraction)); deterministic and
    # independent of the defect draw, so any frame can be regenerated in isolation
    val_every = int(round(1.0 / val_fraction)) if (val_fraction > 0 and not smoke) else 0
    split_paths: dict[str, list[str]] = {"train": [], "val": [], "smoke": []}
    randomizer = Randomizer(seed, scheduler)
    usda = bf.scene_usda(bf.PackState.nominal())
    t_start = time.perf_counter()
    frame_times, box_sources, ious_all = [], [], []
    idmap_mismatches = 0
    with Scene("blister.sdg") as scene:
        t0 = time.perf_counter()
        scene.load_usda(usda)
        print(f"scene loaded in {time.perf_counter() - t0:.1f} s ({versions})", flush=True)
        driver = SceneDriver(scene)
        p0 = randomizer.sample(0)
        driver.prime(p0, camera_pose(p0))
        manifest = open(out_dir / "manifest.jsonl", "w", encoding="utf-8")
        try:
            for f in range(n_frames):
                t = time.perf_counter()
                p = randomizer.sample(f)
                cam = camera_pose(p)
                ordinal = driver.apply(p, cam)
                fr = scene.render(PRODUCT, steps=warmup, want_depth=True, width=bf.IMAGE_W, height=bf.IMAGE_H, ordinal=ordinal)
                rows, sources, ious, mism = yolo_labels(fr, p, cam)
                idmap_mismatches += mism
                vec = blur_vector_px(cam, p)
                p.blur_px = float(np.hypot(*vec))
                p.box_sources = sources
                rgb = motion_blur(fr.rgb, vec)
                if noise_std > 0:
                    rgb = rgb + np.random.default_rng([seed, f, 11]).normal(0.0, noise_std, rgb.shape)
                rgb8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
                split = "smoke" if smoke else ("val" if (val_every and f % val_every == 0) else "train")
                stem = f"{split}_{f:06d}"
                Image.fromarray(rgb8).save(img_dir / f"{stem}.png")
                split_paths[split].append((img_dir / f"{stem}.png").resolve().as_posix())
                (lab_dir / f"{stem}.txt").write_text(format_labels(rows), encoding="utf-8")
                rec = asdict(p)
                rec.update({"stem": stem, "split": split, "iou_mask_vs_projection": [round(v, 3) for v in ious], "idmap_mismatches": mism, "ms": round((time.perf_counter() - t) * 1000, 1)})
                manifest.write(json.dumps(rec) + "\n")
                frame_times.append(time.perf_counter() - t)
                box_sources += sources
                ious_all += ious
                if smoke or f % 100 == 0:
                    print(f"frame {f:6d} {stem}: {frame_times[-1] * 1000:6.0f} ms  states={''.join(str(bf.CLASS_IDS[s]) for s in p.states)}  blur={p.blur_px:.2f}px  sources={sorted(set(sources))}  min IoU(mask,proj)={min(ious):.2f}", flush=True)
        finally:
            manifest.close()
    names = {v: k for k, v in bf.CLASS_IDS.items()}
    # Ultralytics dataset yaml with explicit image lists per split (labels resolve via images->labels)
    for sp, paths in split_paths.items():
        if paths:
            (out_dir / f"{sp}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    train_list = "smoke.txt" if smoke else "train.txt"
    val_list = "smoke.txt" if smoke else ("val.txt" if split_paths["val"] else "train.txt")
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\ntrain: {train_list}\nval: {val_list}\nnames:\n" + "".join(f"  {i}: {names[i]}\n" for i in sorted(names)), encoding="utf-8")
    # per-split class instance counts (the sizing check for the 300-instance recall bound)
    class_counts = {}
    for sp, paths in split_paths.items():
        if not paths:
            continue
        cnt = {n: 0 for n in names.values()}
        for ip in paths:
            for ln in (lab_dir / (Path(ip).stem + ".txt")).read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    cnt[names[int(ln.split()[0])]] += 1
        class_counts[sp] = {"frames": len(paths), **cnt}
    src_counts = {s: box_sources.count(s) for s in sorted(set(box_sources))}
    summary = {
        "frames": n_frames, "seed": seed, "warmup_steps": warmup, "out": str(out_dir), "versions": versions,
        "scheduler": scheduler.rates, "expected_boxes_per_pack": scheduler.expected_boxes_per_pack(),
        "split": {"val_every": val_every, "counts": class_counts},
        "ms_per_frame_median": round(float(np.median(frame_times)) * 1000, 1), "total_s": round(time.perf_counter() - t_start, 1),
        "box_sources": src_counts, "iou_mask_vs_projection_median": round(float(np.median(ious_all)), 3), "iou_min": round(float(min(ious_all)), 3),
        "idmap_label_mismatches": idmap_mismatches,
        "line": {"v_belt_mps": bf.V_BELT_MPS, "gsd_m": bf.GSD_M, "camera_height_m": bf.CAM_HEIGHT_M},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return validate_dataset(out_dir, n_frames) if smoke else 0


def validate_dataset(out_dir: Path, n_frames: int) -> int:
    """Smoke-test contract: every label file has exactly 10 rows, integer class in 0..3, coords in [0, 1]."""
    lab_dir = out_dir / "labels"
    files = sorted(lab_dir.glob("*.txt"))
    problems = []
    if len(files) != n_frames:
        problems.append(f"expected {n_frames} label files, found {len(files)}")
    for fp in files:
        rows = [ln.split() for ln in fp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(rows) != N_CAV:
            problems.append(f"{fp.name}: {len(rows)} rows")
        for r in rows:
            if len(r) != 5 or not r[0].isdigit() or int(r[0]) not in range(len(bf.STATES)):
                problems.append(f"{fp.name}: bad class token {r[:1]}")
                continue
            vals = [float(v) for v in r[1:]]
            if not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                problems.append(f"{fp.name}: coords out of range {vals}")
        if not (out_dir / "images" / f"{fp.stem}.png").exists():
            problems.append(f"{fp.name}: missing image")
    if files:
        print(f"\n--- {files[0].name} ---\n{files[0].read_text(encoding='utf-8')}", flush=True)
    print("SMOKE RESULT:", "PASS" if not problems else f"FAIL {problems[:8]}", flush=True)
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless synthetic data generation for the blister inspection cell.")
    ap.add_argument("--smoke-test", type=int, default=0, metavar="N", help="render N frames to data/smoke and validate the labels")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--warmup-steps", type=int, default=3, help="render steps per frame (temporal warm-up)")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--noise-std", type=float, default=1.0, help="sensor noise in 8-bit units (0 disables)")
    ap.add_argument("--p-nominal", type=float, default=None, help="probability of an all-good pack (default from PackScheduler)")
    ap.add_argument("--p-empty", type=float, default=None)
    ap.add_argument("--p-pill-damaged", type=float, default=None)
    ap.add_argument("--p-foil", type=float, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rates = {k: v for k, v in {"p_nominal": args.p_nominal, "p_empty": args.p_empty, "p_pill_damaged": args.p_pill_damaged, "p_foil": args.p_foil}.items() if v is not None}
    if args.smoke_test:
        return run(PROJECT_ROOT / "data" / "smoke", args.smoke_test, args.seed, args.warmup_steps, 0.0, args.noise_std, smoke=True, rates=rates)
    return run(args.out, args.frames, args.seed, args.warmup_steps, args.val_fraction, args.noise_std, smoke=False, rates=rates)


if __name__ == "__main__":
    raise SystemExit(main())
