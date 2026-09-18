"""Visual monitoring layer for the closed-loop twin: dual-camera SCADA dashboard (v2).

Draws a 1920x1080 operator view from two render products of the SAME frame of simulation:

  left  pane  /Render/Camera   - the triggered inspection exposure, exactly the CUDA frame the
                                 controller inspected (ROI window included): reticle at d = 0,
                                 the detector's own boxes, cavity badge, verdict pill;
  right pane  /Render/Overview - the cell: infeed, inspection station, reject nozzle (d = 300 mm),
                                 chute and tote, outfeed ramp and stacking tote (d = 620 mm) with
                                 the accepted-product counter, verdict tags on the physics poses,
                                 the pneumatic kick impulse while the valve is open;
  telemetry   line throughput and encoder odometer, ledger, per-class detections and the run's
              defect class distribution (legend integrated), verdict latency against the 25 ms
              GAMP ceiling, pack history.

Nothing here reads or writes simulation state: the twin hands over a snapshot (``HudFrame``) and
gets back a BGR canvas.  The inspection pane is a HOLD of the last strobe exposure (the camera is
trigger-driven, so there is no 30 fps inspection video to show) - the age of that exposure is on
screen so the hold is never mistaken for a live feed.

Typography: TrueType (Segoe UI, falling back to Bahnschrift / Arial / Pillow's default) rendered
by Pillow's FreeType into small anti-aliased patches that are alpha-blended straight into the
BGR canvas.  Fonts are loaded once per (weight, size) and every rendered patch is cached by
(text, size, weight, colour), so a frame's ~120 strings cost well under the display budget on
the dashboard thread; the canvas never round-trips through PIL.

Colour (validated, not eyeballed - dataviz six checks, dark surface #1a1a19, --pairs all):
  detector classes   #3987e5 blue / #d95926 orange / #1baf7a aqua / #eda100 yellow
                     -> CVD worst pair dE 9.1 (target >= 8), normal-vision worst pair dE 17.5
                        (floor >= 15), contrast all >= 3:1.  Slot 4 (#eda100, OKLCH L 0.76) sits
                        above the dark lightness band: these marks are drawn over a rendered
                        frame, each with a 2 px dark casing and a direct class label.
  status (reserved)  good #0ca30c (PASS), warning #fab219, critical #d03b3b (REJECT) - always
                     with a text label.
Single (dark) theme by design: this is a video overlay, not a themed document.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------------------
def bgr(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


SURFACE = bgr("#1a1a19")          # card surface
PLANE = bgr("#0d0d0d")            # page plane
INK = bgr("#ffffff")              # primary ink
INK2 = bgr("#c3c2b7")             # secondary ink
MUTED = bgr("#898781")            # axis / labels
GRID = bgr("#2c2c2a")             # hairline
BASELINE = bgr("#383835")
GOOD = bgr("#0ca30c")
WARNING = bgr("#fab219")
CRITICAL = bgr("#d03b3b")
CASING = bgr("#0b0b0b")           # dark ring under marks drawn on imagery

CLASS_COLORS = {                  # fixed order = the detector's class ids 0..3, never cycled
    "pill_ok": bgr("#3987e5"),
    "pill_damaged": bgr("#d95926"),
    "cavity_empty": bgr("#1baf7a"),
    "foil_damaged": bgr("#eda100"),
}
CLASS_ORDER = list(CLASS_COLORS)
CLASS_SHORT = {"pill_ok": "OK", "pill_damaged": "DMG", "cavity_empty": "EMPTY", "foil_damaged": "FOIL"}

CANVAS_W, CANVAS_H = 1920, 1080
PANE_SCALE = 0.725                                     # 1280x720 -> 928x522
PANE_W, PANE_H = 928, 522
PANE_X = (24, 968)
PANE_Y = 108
HEAD_H = 72
CARD_Y0, CARD_Y1 = 648, 1044
CARD_X = ((24, 480), (496, 952), (968, 1424), (1440, 1896))


# --------------------------------------------------------------------------------------
# TrueType typography: preloaded fonts, cached anti-aliased patches, direct BGR blits
# --------------------------------------------------------------------------------------
class Type:
    FONT_DIRS = (Path(r"C:\Windows\Fonts"), Path("/usr/share/fonts/truetype/dejavu"))
    FILES = {"regular": ("segoeui.ttf", "bahnschrift.ttf", "arial.ttf", "DejaVuSans.ttf"),
             "bold": ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"),
             "light": ("segoeuil.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")}
    CACHE_MAX = 6000

    def __init__(self):
        self.paths: dict[str, Path | None] = {}
        for weight, names in self.FILES.items():
            self.paths[weight] = next((d / n for d in self.FONT_DIRS for n in names if (d / n).exists()), None)
        self.family = self.paths["regular"].name if self.paths["regular"] else "PIL default"
        self._fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._cache: dict[tuple, tuple[np.ndarray, int, int]] = {}

    def font(self, weight: str, size: int):
        key = (weight, size)
        f = self._fonts.get(key)
        if f is None:
            p = self.paths.get(weight) or self.paths.get("regular")
            f = ImageFont.truetype(str(p), size) if p else ImageFont.load_default(size)
            self._fonts[key] = f
        return f

    def warm(self, sizes=(11, 12, 13, 14, 15, 16, 18, 20, 22, 26, 30, 34)) -> None:
        for w in self.FILES:
            for s in sizes:
                self.font(w, s)

    def _patch(self, text: str, size: int, weight: str, color) -> tuple[np.ndarray, int, int]:
        key = (text, size, weight, color)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        f = self.font(weight, size)
        l, t, r, b = f.getbbox(text, anchor="ls")
        w, h = max(1, r - l), max(1, b - t)
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).text((-l, -t), text, font=f, fill=255, anchor="ls")
        a = np.asarray(mask, dtype=np.float32)[:, :, None] / 255.0
        col = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        pre = (a * col).astype(np.float32)                     # premultiplied colour, alpha kept
        val = (np.concatenate([pre, a], 2), l, t)
        if len(self._cache) >= self.CACHE_MAX:
            self._cache.clear()
        self._cache[key] = val
        return val

    def width(self, text: str, size: int = 15, weight: str = "regular") -> int:
        l, _t, r, _b = self.font(weight, size).getbbox(text, anchor="ls")
        return r - l

    def draw(self, img: np.ndarray, text: str, x: int, y: int, *, size: int = 15, weight: str = "regular", color=INK2, align: str = "left") -> int:
        """Blend ``text`` with its baseline at (x, y).  Returns the advance width."""
        if not text:
            return 0
        patch, l, t = self._patch(text, size, weight, tuple(int(c) for c in color))
        h, w = patch.shape[:2]
        if align == "right":
            x -= w
        elif align == "center":
            x -= w // 2
        x0, y0 = int(x + l), int(y + t)
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
        if X1 <= X0 or Y1 <= Y0:
            return w
        src = patch[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        reg = img[Y0:Y1, X0:X1]
        a = src[:, :, 3:4]
        reg[:] = (reg.astype(np.float32) * (1.0 - a) + src[:, :, :3]).astype(np.uint8)
        return w


TYPE = Type()


def text(img, s, x, y, *, size=15, color=INK2, weight="regular", right=False, center=False) -> int:
    return TYPE.draw(img, s, x, y, size=size, weight=weight, color=color, align="right" if right else "center" if center else "left")


def plate(img, s, x, y, *, size=13, color=INK2, right=False, pad=6, fill=CASING) -> None:
    """Text on a dark plate - legible wherever it lands on a rendered frame."""
    w = TYPE.width(s, size)
    x0 = x - w - pad if right else x - pad
    cv2.rectangle(img, (x0, y - size - pad + 2), (x0 + w + 2 * pad, y + pad), fill, -1)
    text(img, s, x0 + pad, y, size=size, color=color)


def pill(img, x, y, w, h, color, *, filled=True, thickness=1) -> None:
    r = h // 2
    if filled:
        cv2.circle(img, (x + r, y + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x + w - r, y + r), r, color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    else:
        cv2.ellipse(img, (x + r, y + r), (r, r), 0, 90, 270, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 0, -90, 90, color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)


def status_pill(img, x, y, label: str, color, *, size=15, h=26, pad=12) -> int:
    w = TYPE.width(label, size, "bold") + 2 * pad
    pill(img, x, y, w, h, color)
    text(img, label, x + pad, y + h - 7, size=size, weight="bold", color=PLANE)
    return x + w


def card(img, x0, y0, x1, y1, title: str | None = None) -> None:
    cv2.rectangle(img, (x0, y0), (x1, y1), SURFACE, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), GRID, 1)
    if title:
        text(img, title, x0 + 16, y0 + 24, size=12, color=MUTED)
        cv2.line(img, (x0 + 16, y0 + 34), (x1 - 16, y0 + 34), GRID, 1)


def dashed(img, p, q, color, *, dash=7, gap=6, thickness=1, casing=False) -> None:
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.hypot(*(q - p)))
    if L < 1:
        return
    d = (q - p) / L
    s = 0.0
    while s < L:
        a = p + d * s
        b = p + d * min(L, s + dash)
        if casing:
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), CASING, thickness + 2, cv2.LINE_AA)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)
        s += dash + gap


# --------------------------------------------------------------------------------------
# Overview camera model (mirrors the USDA prim: translate then rotateXYZ(rx, 0, 0))
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OverviewCam:
    position: tuple
    rot_x_deg: float
    focal: float
    aperture_h: float
    aperture_v: float
    width: int = 1280
    height: int = 720

    def project(self, p) -> tuple[float, float, float]:
        """World point -> (pixel x, pixel y, depth along the optical axis, metres).

        USD camera convention: looks down -Z of its own frame, +X right, +Y up.  The prim carries
        one rotation about world X, so R^T is applied to (p - C) directly."""
        a = np.radians(self.rot_x_deg)
        c, s = np.cos(a), np.sin(a)
        d = np.asarray(p, dtype=np.float64) - np.asarray(self.position, dtype=np.float64)
        xc = d[0]
        yc = c * d[1] + s * d[2]
        zc = -s * d[1] + c * d[2]
        depth = -zc
        if depth <= 1e-6:
            return (float("nan"), float("nan"), depth)
        ndc_x = (xc / depth) * (self.focal / (self.aperture_h / 2))
        ndc_y = (yc / depth) * (self.focal / (self.aperture_v / 2))
        return ((1 + ndc_x) / 2 * self.width, (1 - ndc_y) / 2 * self.height, depth)

    def px(self, p, scale: float = PANE_SCALE) -> tuple[int, int]:
        u, v, _ = self.project(p)
        return (int(round(u * scale)), int(round(v * scale)))


# --------------------------------------------------------------------------------------
# What the twin hands over for one display frame
# --------------------------------------------------------------------------------------
@dataclass
class PackTag:
    pack_id: int
    xyz: tuple
    state: str                     # belt | kicked | falling | outfeed
    verdict: str | None            # PASS | REJECT | None (still pending)
    actioned: bool = False
    confirmed: bool = False


@dataclass
class HudFrame:
    sim_time: float
    wall_time: float
    rtf: float
    mode: str
    speed: float                   # sim seconds per playback second
    overview_rgb: np.ndarray
    inspection_rgb: np.ndarray | None = None
    inspection_age_s: float = 0.0
    last_pack: dict | None = None  # pack_id, verdict, reason, n_ok, latency_ms, per_class, dets
    tags: list[PackTag] = field(default_factory=list)
    ledger: dict = field(default_factory=dict)
    controller: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    marks: list[tuple] = field(default_factory=list)   # (world_xyz, label[, "near"]) station markers; "near" labels the operator-side end
    confirm_flash: tuple | None = None                 # (pack_id, world_xyz) of a just-confirmed ejection
    footer: str = ""
    packs_total: int = 0
    # v2 telemetry
    kick_active: bool = False
    kick_pack: int | None = None
    kick_xyz: tuple | None = None                      # nozzle position (world)
    tote_xyz: tuple | None = None                      # tote rim (world), for the tote status label
    tote_count: int = 0
    encoder_m: float = 0.0
    encoder_ticks: int = 0
    throughput_ppm: float | None = None
    design_ppm: float = 800.0
    belt_mps: float = 1.6
    class_totals: dict = field(default_factory=dict)   # boxes per class over every inspected pack
    # optical trigger sensor at d = 0 (visual only: the trigger itself stays encoder-clocked)
    beam_a: tuple | None = None                        # emitter (world)
    beam_b: tuple | None = None                        # receiver (world)
    beam_broken: bool = False                          # a pack straddles the trigger plane
    strobe: bool = False                               # a strobe exposure fired within the last few ms
    # outfeed stacking tote (nominal product): counter tag over the tote
    outfeed_xyz: tuple | None = None                   # far rim, middle (world)
    outfeed_count: int = 0                             # accepted packs that left the belt into the outfeed
    outfeed_in_tote: int = 0                           # settled packs in the rolling FIFO
    outfeed_capacity: int = 0


# --------------------------------------------------------------------------------------
# Inspection pane
# --------------------------------------------------------------------------------------
def letterbox_to_frame(det_xyxy, frame_w: int, frame_h: int, imgsz: int):
    """Inverse of TrtDetector.letterbox_gpu: detector boxes are in letterbox pixels."""
    s = imgsz / max(frame_h, frame_w)
    pad_top, pad_left = (imgsz - int(round(frame_h * s))) // 2, (imgsz - int(round(frame_w * s))) // 2
    x0, y0, x1, y1 = det_xyxy
    return ((x0 - pad_left) / s, (y0 - pad_top) / s, (x1 - pad_left) / s, (y1 - pad_top) / s)


def _missing_cavity(dets_frame, frame_w: int, frame_h: int, *, fov_m: float = 0.200, pitch_m: float = 0.016, row_m: float = 0.011, tol_px: int = 45) -> tuple[int, int]:
    """Expected cavity centre with no pill_ok box near it (under-occupancy rejects carry no defect
    box).  Twin geometry: the pack is centred at d = 0 when the strobe fires, so the 2x5 grid sits
    on the frame centre at the line's 0.156 mm/px."""
    ppm = frame_w / fov_m
    oks = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, _c, n in dets_frame if n == "pill_ok"]
    for row in (-1, 1):
        for col in range(5):
            ex, ey = frame_w / 2 + (col - 2) * pitch_m * ppm, frame_h / 2 - row * row_m * ppm
            if not any(abs(ox - ex) < tol_px and abs(oy - ey) < tol_px for ox, oy in oks):
                return int(ex), int(ey)
    return frame_w // 2, frame_h // 2


def _defect_inset(pane, hf: HudFrame, lp: dict, *, frame_w: int, frame_h: int, imgsz: int, n_cavities: int) -> None:
    """Picture-in-picture at the upper right of the inspection pane: on REJECT the primary defect
    cavity (highest-confidence defect box, else the cavity with no pill_ok box) magnified 2x at
    full frame resolution with its class tag; on PASS the nominal grid badge."""
    W, H, half = 256, 176, 64                        # 128 px crop = 20 mm at 0.156 mm/px, shown 2x
    x0, y0 = PANE_W - 16 - W, 62
    verdict = lp.get("verdict")
    dets = [(letterbox_to_frame(d[:4], frame_w, frame_h, imgsz), float(d[4]), d[5]) for d in lp.get("dets", [])]
    if verdict == "REJECT" and hf.inspection_rgb is not None:
        defects = [d for d in dets if d[2] != "pill_ok"]
        if defects:
            (bx0, by0, bx1, by1), conf, name = max(defects, key=lambda d: d[1])
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            tag, box = f"{name}  {conf:.2f}", (bx0, by0, bx1, by1)
        else:
            cx, cy = _missing_cavity(dets, frame_w, frame_h)
            name, tag, box = "cavity_missing", "no pill_ok box in this cavity", None
        cx, cy = int(np.clip(cx, half, frame_w - half)), int(np.clip(cy, half, frame_h - half))
        crop = cv2.cvtColor(hf.inspection_rgb[cy - half:cy + half, cx - half:cx + half], cv2.COLOR_RGB2BGR)
        zoom = cv2.resize(crop, (W, W), interpolation=cv2.INTER_CUBIC)
        top = (W - H) // 2
        zoom = np.ascontiguousarray(zoom[top:top + H])
        col = CLASS_COLORS.get(name, WARNING)
        s = W / (2 * half)
        if box is not None:
            p0 = (int((box[0] - (cx - half)) * s), int((box[1] - (cy - half)) * s - top))
            p1 = (int((box[2] - (cx - half)) * s), int((box[3] - (cy - half)) * s - top))
            cv2.rectangle(zoom, p0, p1, CASING, 4, cv2.LINE_AA)
            cv2.rectangle(zoom, p0, p1, col, 2, cv2.LINE_AA)
        else:
            c = (W // 2, H // 2)
            cv2.circle(zoom, c, int(0.006 * (frame_w / 0.2) * s), CASING, 4, cv2.LINE_AA)
            cv2.circle(zoom, c, int(0.006 * (frame_w / 0.2) * s), col, 2, cv2.LINE_AA)
        pane[y0:y0 + H, x0:x0 + W] = zoom
        cv2.rectangle(pane, (x0 - 2, y0 - 2), (x0 + W + 1, y0 + H + 1), CRITICAL, 3)
        plate(pane, "DEFECT  2x", x0 + 6, y0 + 18, size=12, color=CRITICAL)
        plate(pane, tag, x0 + W - 6, y0 + H - 8, size=12, color=col, right=True)
    else:
        cv2.rectangle(pane, (x0, y0), (x0 + W, y0 + H), CASING, -1)
        cv2.rectangle(pane, (x0 - 2, y0 - 2), (x0 + W + 1, y0 + H + 1), GOOD if verdict == "PASS" else MUTED, 3)
        n_ok = lp.get("n_ok")
        text(pane, "NOMINAL" if verdict == "PASS" else "PENDING", x0 + W // 2, y0 + 40, size=20, weight="bold", color=GOOD if verdict == "PASS" else MUTED, center=True)
        text(pane, f"{n_ok if n_ok is not None else '-'} / {n_cavities} cavities pill_ok", x0 + W // 2, y0 + 64, size=13, color=INK2, center=True)
        for row in range(2):                                   # the nominal 2x5 grid, one dot per cavity
            for col in range(5):
                cv2.circle(pane, (x0 + 58 + col * 35, y0 + 100 + row * 34), 10, GOOD if verdict == "PASS" else MUTED, -1, cv2.LINE_AA)
                cv2.circle(pane, (x0 + 58 + col * 35, y0 + 100 + row * 34), 10, CASING, 1, cv2.LINE_AA)


def _reticle(pane) -> None:
    cx, cy = PANE_W // 2, PANE_H // 2
    for col, th in ((CASING, 3), (INK2, 1)):
        cv2.circle(pane, (cx, cy), 46, col, th, cv2.LINE_AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(pane, (cx + dx * 14, cy + dy * 14), (cx + dx * 70, cy + dy * 70), col, th, cv2.LINE_AA)
    plate(pane, "reticle  d = 0 mm trigger plane", 14, 78, size=12, color=INK2)   # caption clear of the pack


def inspection_pane(hf: HudFrame, *, frame_w: int, frame_h: int, imgsz: int, roi_half_px: int, n_cavities: int) -> np.ndarray:
    pane = np.full((PANE_H, PANE_W, 3), PLANE, np.uint8)
    if hf.inspection_rgb is None:
        text(pane, "WAITING FOR FIRST TRIGGER", PANE_W // 2, PANE_H // 2, size=22, color=MUTED, center=True)
        _reticle(pane)
        return pane
    pane[:] = cv2.resize(cv2.cvtColor(hf.inspection_rgb, cv2.COLOR_RGB2BGR), (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)
    lp = hf.last_pack or {}
    # inspection window (pitch/FOV gate): the belt-filled margins are already in the pixels
    for xr in (frame_w // 2 - roi_half_px, frame_w // 2 + roi_half_px):
        xp = int(round(xr * PANE_SCALE))
        dashed(pane, (xp, 0), (xp, PANE_H), MUTED, dash=8, gap=8)
    plate(pane, "INSPECTION WINDOW", int(round((frame_w // 2 - roi_half_px) * PANE_SCALE)) + 10, PANE_H - 14, size=12, color=MUTED)
    _reticle(pane)
    # detections: 2 px class-colour box on a dark casing, short label, full names in the legend card
    for d in lp.get("dets", []):
        x0, y0, x1, y1, conf, name = d
        fx0, fy0, fx1, fy1 = letterbox_to_frame((x0, y0, x1, y1), frame_w, frame_h, imgsz)
        p0 = (int(round(fx0 * PANE_SCALE)), int(round(fy0 * PANE_SCALE)))
        p1 = (int(round(fx1 * PANE_SCALE)), int(round(fy1 * PANE_SCALE)))
        col = CLASS_COLORS.get(name, MUTED)
        cv2.rectangle(pane, (p0[0] - 1, p0[1] - 1), (p1[0] + 1, p1[1] + 1), CASING, 3, cv2.LINE_AA)
        cv2.rectangle(pane, p0, p1, col, 2, cv2.LINE_AA)
        label = f"{CLASS_SHORT.get(name, name)} {conf:.2f}"
        tw = TYPE.width(label, 12)
        ty = p0[1] - 6 if p0[1] > 22 else p1[1] + 16
        cv2.rectangle(pane, (p0[0] - 1, ty - 13), (p0[0] + tw + 7, ty + 4), CASING, -1)
        text(pane, label, p0[0] + 3, ty, size=12, color=col)
    # banner: verdict pill, pack id, reason, latency; cavity badge on the right
    verdict = lp.get("verdict")
    col = GOOD if verdict == "PASS" else CRITICAL if verdict == "REJECT" else MUTED
    cv2.rectangle(pane, (0, 0), (PANE_W, 54), CASING, -1)
    cv2.line(pane, (0, 54), (PANE_W, 54), col, 2)
    xe = status_pill(pane, 14, 13, verdict or "PENDING", col, size=17, h=28, pad=14)
    if lp:
        text(pane, f"PACK {lp['pack_id']:03d}", xe + 16, 34, size=19, weight="bold", color=INK)
        text(pane, (lp.get("reason") or "")[:60], xe + 130, 34, size=14, color=INK2)
        n_ok = lp.get("n_ok")
        badge = f"{n_ok if n_ok is not None else '-'} / {n_cavities}"
        bw = TYPE.width(badge, 16, "bold") + 28
        pill(pane, PANE_W - 16 - bw, 13, bw, 28, GOOD if n_ok == n_cavities else WARNING)
        text(pane, badge, PANE_W - 16 - bw + 14, 34, size=16, weight="bold", color=PLANE)
        text(pane, "CAVITIES", PANE_W - 16 - bw - 8, 34, size=11, color=MUTED, right=True)
        if lp.get("latency_ms") is not None:                  # bottom-left, clear of the defect inset
            plate(pane, f"verdict {lp['latency_ms']:.1f} ms", 16, PANE_H - 14, size=12, color=INK2)
    # defect zoom inset (picture-in-picture): the primary defect cavity magnified, or the nominal grid
    if lp:
        _defect_inset(pane, hf, lp, frame_w=frame_w, frame_h=frame_h, imgsz=imgsz, n_cavities=n_cavities)
    # strobe hold indicator
    age_ms = hf.inspection_age_s * 1000
    if age_ms < 12:
        cv2.rectangle(pane, (1, 1), (PANE_W - 2, PANE_H - 2), INK, 2)
        plate(pane, "STROBE", PANE_W - 16, PANE_H - 14, size=13, color=INK, right=True)
    else:
        plate(pane, f"HOLD  +{age_ms:5.1f} ms", PANE_W - 16, PANE_H - 14, size=12, color=INK2, right=True)
    return pane


# --------------------------------------------------------------------------------------
# Overview pane
# --------------------------------------------------------------------------------------
def overview_pane(hf: HudFrame, cam: OverviewCam) -> np.ndarray:
    pane = cv2.resize(cv2.cvtColor(hf.overview_rgb, cv2.COLOR_RGB2BGR), (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)
    # station rules across the belt, label at the far end (clear of the pack lane)
    for i, mark in enumerate(hf.marks):
        (x, y, z), label = mark[0], mark[1]
        near = len(mark) > 2 and mark[2] == "near"               # label at the operator-side end (the outfeed rule: its far end is the reject corner)
        a, b = cam.px((x, y - 0.17, z)), cam.px((x, y + (0.13 if near else 0.17), z))   # the outfeed rule stops at the ramp's far rail
        dashed(pane, a, b, INK2, dash=9, gap=7, casing=True)
        anchor = a if near else b
        tw = TYPE.width(label, 12)
        tx = min(max(anchor[0] - tw // 2, 4), PANE_W - tw - 8)
        ty = max(anchor[1] - 12 - (22 if i % 2 else 0), 20)      # staggered: neighbouring stations' labels never touch
        cv2.line(pane, anchor, (anchor[0], ty + 4), MUTED, 1)
        plate(pane, label, tx, ty, size=12, color=INK2)
    # optical trigger beam at d = 0: emitter -> receiver across the belt, state from the physics poses
    if hf.beam_a is not None and hf.beam_b is not None:
        a, b = cam.px(hf.beam_a), cam.px(hf.beam_b)
        col = CRITICAL if hf.beam_broken else bgr("#6b2a2a")
        cv2.line(pane, a, b, CASING, 4, cv2.LINE_AA)
        cv2.line(pane, a, b, col, 2 if hf.beam_broken else 1, cv2.LINE_AA)
        for p in (a, b):
            cv2.circle(pane, p, 4, CASING, -1, cv2.LINE_AA)
            cv2.circle(pane, p, 3, col, -1, cv2.LINE_AA)
        state = "BEAM  BROKEN" if hf.beam_broken else "BEAM  CLEAR"
        if hf.strobe:
            state += "   STROBE"
            m = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            cv2.circle(pane, m, 11, CASING, 3, cv2.LINE_AA)
            cv2.circle(pane, m, 11, INK, 2, cv2.LINE_AA)
        plate(pane, state, a[0] + 8, a[1] + 20, size=12, color=INK if hf.strobe else col)
    # tote status
    if hf.tote_xyz is not None:
        u, v = cam.px(hf.tote_xyz)
        plate(pane, f"TOTE  {hf.tote_count} rejected", min(max(u - 40, 4), PANE_W - 150), v, size=12, color=INK2)
    # outfeed stacking tote: accepted-product counter above the far rim, FIFO fill below it
    if hf.outfeed_xyz is not None:
        u, v = cam.px(hf.outfeed_xyz)
        lab = f"ACCEPTED: {hf.outfeed_count} packs"
        tw = TYPE.width(lab, 13, "bold")
        bx = min(max(u - tw // 2 - 8, 4), PANE_W - tw - 20)
        cv2.rectangle(pane, (bx, v - 50), (bx + tw + 16, v - 26), CASING, -1)
        cv2.rectangle(pane, (bx, v - 50), (bx + tw + 16, v - 26), GOOD, 1)
        text(pane, lab, bx + 8, v - 32, size=13, weight="bold", color=GOOD)
        plate(pane, f"stacking tote  {hf.outfeed_in_tote} / {hf.outfeed_capacity} FIFO", bx, v - 8, size=11, color=INK2)
    # pneumatic kick impulse: chevrons leaving the nozzle across the belt
    if hf.kick_active and hf.kick_xyz is not None:
        x, y, z = hf.kick_xyz
        n0 = np.array(cam.px((x, y, z)), float)
        n1 = np.array(cam.px((x, y + 0.30, z)), float)
        d = n1 - n0
        d /= np.hypot(*d) + 1e-9
        perp = np.array([-d[1], d[0]])
        for k, dist in enumerate((22, 40, 58)):
            c = n0 + d * dist
            for sgn in (-1, 1):
                p = c - d * 10 + perp * sgn * (9 + 2 * k)
                cv2.line(pane, tuple(p.astype(int)), tuple(c.astype(int)), CASING, 4, cv2.LINE_AA)
                cv2.line(pane, tuple(p.astype(int)), tuple(c.astype(int)), WARNING, 2, cv2.LINE_AA)
        lab = "KICK" + (f"  pack {hf.kick_pack:02d}" if hf.kick_pack is not None else "")
        plate(pane, lab, int(n0[0]) + 12, int(n0[1]) + 22, size=12, color=WARNING)
    # per-pack verdict tags on the live physics poses, leader line to the pack
    for t in hf.tags:
        u, v = cam.px(t.xyz)
        if not (0 <= u < PANE_W and 0 <= v < PANE_H):
            continue
        col = GOOD if t.verdict == "PASS" else CRITICAL if t.verdict == "REJECT" else MUTED
        lab = f"{t.pack_id:02d}  " + (t.verdict or "PENDING") + ("  EJECTED" if t.confirmed else "")
        tw = TYPE.width(lab, 12)
        bx, by = int(u - tw / 2 - 6), int(v - 38)
        cv2.line(pane, (u, v - 8), (u, by + 20), col, 1, cv2.LINE_AA)
        cv2.circle(pane, (u, v - 8), 2, col, -1, cv2.LINE_AA)
        cv2.rectangle(pane, (bx, by), (bx + tw + 12, by + 20), CASING, -1)
        cv2.rectangle(pane, (bx, by), (bx + tw + 12, by + 20), col, 1)
        text(pane, lab, bx + 6, by + 15, size=12, color=col)
    if hf.confirm_flash:                               # the pack leaves the pool the instant the
        pid, xyz = hf.confirm_flash                    # overlap confirms it, so mark where it went
        u, v = cam.px(xyz)
        cv2.circle(pane, (u, v), 16, CASING, 4, cv2.LINE_AA)
        cv2.circle(pane, (u, v), 16, GOOD, 2, cv2.LINE_AA)
        lab = f"PACK {pid:02d}  EJECTION CONFIRMED"
        tw = TYPE.width(lab, 13)
        bx = min(max(u - tw // 2 - 6, 4), PANE_W - tw - 16)
        cv2.rectangle(pane, (bx, v + 22), (bx + tw + 12, v + 44), CASING, -1)
        cv2.rectangle(pane, (bx, v + 22), (bx + tw + 12, v + 44), GOOD, 1)
        text(pane, lab, bx + 6, v + 38, size=13, color=GOOD)
    return pane


# --------------------------------------------------------------------------------------
# Telemetry cards
# --------------------------------------------------------------------------------------
def _kpi(img, x, y, label: str, value: str, color=INK, sub: str = "") -> None:
    text(img, label, x, y, size=12, color=MUTED)
    text(img, value, x, y + 34, size=30, weight="bold", color=color)
    if sub:
        text(img, sub, x, y + 54, size=12, color=MUTED)


def card_line(img, hf: HudFrame, x0, x1, y0, y1) -> None:
    card(img, x0, y0, x1, y1, "LINE   throughput and encoder")
    ppm = hf.throughput_ppm
    text(img, "THROUGHPUT", x0 + 16, y0 + 62, size=12, color=MUTED)
    text(img, f"{ppm:.0f}" if ppm is not None else "-", x0 + 16, y0 + 96, size=30, weight="bold", color=INK)
    text(img, "packs / min", x0 + 16 + TYPE.width(f"{ppm:.0f}" if ppm is not None else "-", 30, "bold") + 8, y0 + 96, size=12, color=MUTED)
    gx0, gx1, gy = x0 + 16, x1 - 16, y0 + 118
    top = max(1000.0, hf.design_ppm * 1.25)
    cv2.rectangle(img, (gx0, gy), (gx1, gy + 10), BASELINE, -1)
    if ppm:
        cv2.rectangle(img, (gx0, gy), (gx0 + int((gx1 - gx0) * min(ppm, top) / top), gy + 10), INK2, -1)
    dx = gx0 + int((gx1 - gx0) * hf.design_ppm / top)
    cv2.line(img, (dx, gy - 4), (dx, gy + 14), WARNING, 2)
    text(img, f"design {hf.design_ppm:.0f}", dx, gy + 30, size=11, color=WARNING, center=True)
    text(img, "0", gx0, gy + 30, size=11, color=MUTED)
    text(img, f"{top:.0f}", gx1, gy + 30, size=11, color=MUTED, right=True)
    cv2.line(img, (x0 + 16, y0 + 168), (x1 - 16, y0 + 168), GRID, 1)
    _kpi(img, x0 + 16, y0 + 194, "ENCODER ODOMETER", f"{hf.encoder_m:8.3f} m", sub=f"{hf.encoder_ticks:,} ticks at 0.5 mm")
    _kpi(img, x0 + 236, y0 + 194, "BELT", f"{hf.belt_mps:.2f} m/s", sub=f"pitch {60 * hf.belt_mps / max(hf.design_ppm, 1) * 1000:.0f} mm")
    _kpi(img, x0 + 16, y0 + 292, "SIM TIME", f"{hf.sim_time:7.3f} s", sub=f"wall {hf.wall_time:.1f} s")
    _kpi(img, x0 + 236, y0 + 292, "REAL-TIME FACTOR", f"{hf.rtf:.2f}", sub=("real time" if abs(hf.speed - 1) < 1e-6 else f"{1 / hf.speed:.0f}x slow-motion playback"))


def card_ledger(img, hf: HudFrame, x0, x1, y0, y1) -> None:
    card(img, x0, y0, x1, y1, "LEDGER   triggers = verdicts = actions")
    led, ctl = hf.ledger, hf.controller
    rej = sum(1 for h in hf.history if h["verdict"] == "REJECT")
    pas = sum(1 for h in hf.history if h["verdict"] == "PASS")
    conf = sum(1 for h in hf.history if h.get("confirmed"))
    _kpi(img, x0 + 16, y0 + 62, "INSPECTED", f"{led.get('triggers', 0):d}", sub=f"of {hf.packs_total} scheduled")
    _kpi(img, x0 + 166, y0 + 62, "PASSED", f"{pas:d}", color=GOOD)
    _kpi(img, x0 + 316, y0 + 62, "REJECTED", f"{rej:d}", color=CRITICAL)
    _kpi(img, x0 + 16, y0 + 160, "EJECTIONS CONFIRMED", f"{conf:d}/{rej:d}", sub="PhysX overlap at the chute")
    _kpi(img, x0 + 236, y0 + 160, "PENDING", f"{led.get('triggers', 0) - led.get('actions', 0):d}", sub=f"verdicts {led.get('verdicts', 0)}  actions {led.get('actions', 0)}")
    cv2.line(img, (x0 + 16, y0 + 240), (x1 - 16, y0 + 240), GRID, 1)
    faults = (("dropped frames", ctl.get("dropped_frames", 0)), ("queue overflow", ctl.get("overflow", 0)),
              ("inference timeouts", ctl.get("timeouts", 0)), ("late actuations", ctl.get("late_actuations", 0)),
              ("watchdog events", ctl.get("watchdog_events", 0)), ("worker errors", len(ctl.get("errors", []) or [])))
    for i, (k, v) in enumerate(faults):
        xx, yy = x0 + 16 + (i % 2) * 220, y0 + 268 + (i // 2) * 30
        cv2.circle(img, (xx + 5, yy - 5), 5, CRITICAL if v else GOOD, -1, cv2.LINE_AA)
        text(img, k, xx + 18, yy, size=13, color=INK2)
        text(img, str(v), xx + 200, yy, size=13, color=CRITICAL if v else INK2, right=True)


def card_detections(img, hf: HudFrame, x0, x1, y0, y1) -> None:
    card(img, x0, y0, x1, y1, "DETECTIONS   last pack, and the run's class distribution")
    lp = hf.last_pack or {}
    if lp:
        v = lp.get("verdict")
        xe = status_pill(img, x0 + 16, y0 + 50, v or "PENDING", GOOD if v == "PASS" else CRITICAL if v == "REJECT" else MUTED, size=14, h=24)
        text(img, f"PACK {lp['pack_id']:03d}", xe + 12, y0 + 68, size=17, weight="bold", color=INK)
        if lp.get("latency_ms") is not None:
            text(img, f"{lp['latency_ms']:.1f} ms", x1 - 16, y0 + 68, size=14, color=INK2, right=True)
        text(img, (lp.get("reason") or "")[:54], x0 + 16, y0 + 94, size=12, color=INK2)
    else:
        text(img, "no pack inspected yet", x0 + 16, y0 + 68, size=14, color=MUTED)
    text(img, "CLASS", x0 + 40, y0 + 122, size=11, color=MUTED)
    text(img, "LAST", x0 + 250, y0 + 122, size=11, color=MUTED, right=True)
    text(img, "MAX CONF", x0 + 330, y0 + 122, size=11, color=MUTED, right=True)
    text(img, "RUN TOTAL", x1 - 16, y0 + 122, size=11, color=MUTED, right=True)
    per = lp.get("per_class") or {}
    totals = hf.class_totals or {}
    tmax = max([totals.get(c, 0) for c in CLASS_ORDER] + [1])
    for i, name in enumerate(CLASS_ORDER):                 # legend, last-pack row and run bar in one line
        yy = y0 + 152 + i * 46
        cv2.rectangle(img, (x0 + 16, yy - 11), (x0 + 30, yy + 1), CLASS_COLORS[name], -1)
        row = per.get(name, {})
        text(img, name, x0 + 40, yy, size=14, color=INK2)
        text(img, str(row.get("count", 0)), x0 + 250, yy, size=14, weight="bold", color=INK, right=True)
        text(img, f"{row.get('max_conf', 0.0):.3f}", x0 + 330, yy, size=13, color=INK2, right=True)
        text(img, f"{totals.get(name, 0)}", x1 - 16, yy, size=14, weight="bold", color=INK, right=True)
        bw = int((x1 - 16 - (x0 + 40)) * totals.get(name, 0) / tmax)
        cv2.rectangle(img, (x0 + 40, yy + 8), (x0 + 40 + max(bw, 1 if totals.get(name, 0) else 0), yy + 14), CLASS_COLORS[name], -1)
    n_ok = lp.get("n_ok")
    text(img, "distinct pill_ok cavities in the last pack", x0 + 16, y1 - 18, size=12, color=MUTED)
    text(img, f"{n_ok if n_ok is not None else '-'}/{hf.last_pack.get('n_cavities', 10) if hf.last_pack else 10}", x1 - 16, y1 - 18, size=16, weight="bold",
         color=GOOD if lp and n_ok == lp.get("n_cavities", 10) else CRITICAL if lp else MUTED, right=True)


def card_latency(img, hf: HudFrame, x0, x1, y0, y1, budget_ms: float) -> None:
    card(img, x0, y0, x1, y1, "VERDICT LATENCY   capture -> verdict, per pack")
    cx0, cx1, cy0, cy1 = x0 + 52, x1 - 16, y0 + 56, y0 + 214
    lat = [h.get("latency_ms") or 0.0 for h in hf.history]
    top = max(budget_ms * 1.25, (max(lat) if lat else 0) * 1.15, 1.0)
    cv2.line(img, (cx0, cy1), (cx1, cy1), BASELINE, 1)
    for val, lab, col in ((0.0, "0", MUTED), (budget_ms, f"{budget_ms:.0f} ms", WARNING)):
        yy = int(cy1 - (val / top) * (cy1 - cy0))
        if val:
            dashed(img, (cx0, yy), (cx1, yy), col, dash=6, gap=6)
            text(img, "GAMP ceiling", cx1, yy - 5, size=11, color=col, right=True)
        text(img, lab, cx0 - 8, yy + 4, size=11, color=col, right=True)
    n = max(1, hf.packs_total)
    plot_w = cx1 - cx0
    shown = min(len(lat), max(1, plot_w // 5))
    bars = lat[-shown:]
    pitch = plot_w / max(len(bars), n if n * 5 <= plot_w else len(bars))
    bw = max(2, int(pitch) - 3)
    for i, val in enumerate(bars):
        bx = int(cx0 + i * pitch)
        by = int(cy1 - min(val / top, 1.0) * (cy1 - cy0))
        cv2.rectangle(img, (bx, by), (bx + bw, cy1 - 1), CRITICAL if val > budget_ms else INK2, -1)
    if lat:
        med = float(np.median(lat))
        text(img, f"median {med:.1f} ms   p99 {float(np.quantile(lat, 0.99)):.1f} ms   max {max(lat):.1f} ms" + (f"   (last {shown} of {len(lat)})" if shown < len(lat) else ""),
             cx0, cy1 + 18, size=12, color=INK2)
    # history tiles, bounded by the card: most recent packs, caption says what is left out
    ty0 = cy1 + 34
    avail = x1 - x0 - 32
    cols = max(1, min(20, n, avail // 26))
    tw = int((avail - (cols - 1) * 3) / cols)
    rows_fit = max(1, (y1 - 34 - ty0) // 26)
    tiles = hf.history[-(cols * rows_fit):]
    for i, h in enumerate(tiles):
        r, c = divmod(i, cols)
        tx, ty = x0 + 16 + c * (tw + 3), ty0 + r * 26
        col = GOOD if h["verdict"] == "PASS" else CRITICAL
        cv2.rectangle(img, (tx, ty), (tx + tw, ty + 18), col, -1)
        if h.get("confirmed"):
            cv2.rectangle(img, (tx, ty + 19), (tx + tw, ty + 21), INK2, -1)
        text(img, f"{h['pack_id']:02d}", tx + tw // 2, ty + 14, size=11, color=PLANE, center=True)
    hidden = len(hf.history) - len(tiles)
    text(img, "tile = verdict, underline = ejection confirmed" + (f"   ({hidden} earlier not shown)" if hidden else ""), x0 + 16, y1 - 14, size=11, color=MUTED)


# --------------------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------------------
def compose(hf: HudFrame, cam: OverviewCam, *, frame_w: int, frame_h: int, imgsz: int, roi_half_px: int,
            n_cavities: int, budget_ms: float) -> np.ndarray:
    img = np.full((CANVAS_H, CANVAS_W, 3), PLANE, np.uint8)
    cv2.rectangle(img, (0, 0), (CANVAS_W, HEAD_H), SURFACE, -1)
    cv2.line(img, (0, HEAD_H), (CANVAS_W, HEAD_H), GRID, 1)
    text(img, "BLISTER LINE DIGITAL TWIN", 24, 46, size=26, weight="bold", color=INK)
    text(img, "closed loop:  strobe exposure  ->  TensorRT verdict  ->  encoder shift register  ->  pneumatic reject  ->  chute", 420, 45, size=13, color=MUTED)
    speed_lab = "real time" if abs(hf.speed - 1.0) < 1e-6 else f"{1 / hf.speed:.0f}x slow motion"
    for i, (k, v) in enumerate((("MODE", hf.mode.upper()), ("PLAYBACK", speed_lab), ("SIM t", f"{hf.sim_time:7.3f} s"), ("RTF", f"{hf.rtf:.2f}"))):
        xx = 1250 + i * 170
        text(img, k, xx, 28, size=11, color=MUTED)
        text(img, v, xx, 52, size=17, weight="bold", color=INK2)
    for x, title in zip(PANE_X, ("INSPECTION STATION   d = 0 mm   1280x720 strobe exposure, 50 us",
                                 "CELL OVERVIEW   infeed  ->  inspection  ->  reject nozzle (d = 300 mm)  ->  chute and tote  ->  outfeed (d = 620 mm)")):
        text(img, title, x, PANE_Y - 12, size=13, color=INK2)
    left = inspection_pane(hf, frame_w=frame_w, frame_h=frame_h, imgsz=imgsz, roi_half_px=roi_half_px, n_cavities=n_cavities)
    right = overview_pane(hf, cam)
    img[PANE_Y:PANE_Y + PANE_H, PANE_X[0]:PANE_X[0] + PANE_W] = left
    img[PANE_Y:PANE_Y + PANE_H, PANE_X[1]:PANE_X[1] + PANE_W] = right
    for x in PANE_X:
        cv2.rectangle(img, (x - 1, PANE_Y - 1), (x + PANE_W, PANE_Y + PANE_H), GRID, 1)
    (a0, a1), (b0, b1), (c0, c1), (d0, d1) = CARD_X
    card_line(img, hf, a0, a1, CARD_Y0, CARD_Y1)
    card_ledger(img, hf, b0, b1, CARD_Y0, CARD_Y1)
    card_detections(img, hf, c0, c1, CARD_Y0, CARD_Y1)
    card_latency(img, hf, d0, d1, CARD_Y0, CARD_Y1, budget_ms)
    text(img, hf.footer, 24, CANVAS_H - 12, size=12, color=MUTED)
    return img


# --------------------------------------------------------------------------------------
# Encoder: ffmpeg pipe (NVENC on the RTX 5090 when it answers, else libx264), OpenCV fallback
# --------------------------------------------------------------------------------------
def _nvenc_available(exe: str) -> bool:
    """One 256x256 yuv420p test encode (NVENC rejects frames below its minimum dimension, so a
    tiny probe would report a healthy encoder as missing)."""
    try:
        r = subprocess.run([exe, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2:r=30", "-pix_fmt", "yuv420p",
                            "-c:v", "h264_nvenc", "-preset", "p5", "-f", "null", "-"], capture_output=True, timeout=30)
        return r.returncode == 0 and b"Error" not in r.stderr
    except Exception:  # noqa: BLE001
        return False


class VideoSink:
    """Frames in BGR at a fixed size; H.264/yuv420p MP4 out.

    OpenCV's bundled FFmpeg on this host cannot initialise libopenh264 (avc1/H264 report
    isOpened() but write no video), so a real ffmpeg binary is preferred - h264_nvenc on the GPU
    when the encoder probe succeeds, libx264 otherwise - and cv2's mp4v is only the fallback.
    The chosen path is recorded in ``self.backend``."""

    def __init__(self, path: Path, fps: int, size: tuple[int, int], crf: int = 18):
        self.path, self.fps, self.size, self.frames = Path(path), fps, size, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exe = shutil.which("ffmpeg")
        self.proc = self.writer = self._err = None
        if exe:
            if _nvenc_available(exe):
                codec = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-profile:v", "high"]
                self.backend = f"ffmpeg h264_nvenc cq {crf}"
            else:
                codec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
                self.backend = f"ffmpeg libx264 crf {crf}"
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an", *codec,
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
            # stderr goes to a file, never a pipe: nothing reads a pipe while the run is in
            # progress, so a chatty encoder would fill the buffer, stop draining stdin and
            # deadlock the whole capture chain.
            self._err = tempfile.TemporaryFile()
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err)
        else:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            assert self.writer.isOpened(), f"cannot open a video writer for {self.path}"
            self.backend = "opencv mp4v"

    def _stderr(self) -> str:
        if self._err is None:
            return ""
        try:
            self._err.seek(0)
            return self._err.read().decode(errors="replace")
        except (OSError, ValueError):
            return ""

    def write(self, frame_bgr: np.ndarray) -> None:
        assert frame_bgr.shape[1::-1] == self.size, f"frame {frame_bgr.shape[1::-1]} != {self.size}"
        buf = np.ascontiguousarray(frame_bgr)
        if self.proc is not None:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ffmpeg exited early ({self.proc.returncode}): {self._stderr()[:500]}")
            self.proc.stdin.write(buf.tobytes())
        else:
            self.writer.write(buf)
        self.frames += 1

    def close(self) -> dict:
        try:
            if self.proc is not None:
                try:
                    self.proc.stdin.close()
                except OSError:                        # already broken: wait() below reports why
                    pass
                rc = self.proc.wait(timeout=120)
                if rc != 0:
                    raise RuntimeError(f"ffmpeg failed ({rc}): {self._stderr()[:800]}")
            elif self.writer is not None:
                self.writer.release()
        finally:
            self.proc = self.writer = None
            if self._err is not None:
                self._err.close()
                self._err = None
        return self.info()

    def kill(self) -> None:
        """Last resort when the producer could not be stopped cleanly: no half-written MP4 is
        left looking like a finished recording."""
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        if self.writer is not None:
            self.writer.release()
        self.proc = self.writer = None
        if self._err is not None:
            self._err.close()
            self._err = None

    def info(self) -> dict:
        return {"path": str(self.path), "backend": self.backend, "frames": self.frames, "fps": self.fps,
                "size": list(self.size), "bytes": self.path.stat().st_size if self.path.exists() else 0,
                "duration_s": round(self.frames / self.fps, 2)}


def gui_available() -> bool:
    """True when this OpenCV build can open a window (headless wheels cannot)."""
    try:
        return bool(getattr(cv2, "imshow", None) and getattr(cv2, "namedWindow", None)) and "GUI" in cv2.getBuildInformation()
    except Exception:  # noqa: BLE001
        return False


def window_closed(name: str) -> bool:
    """True once the operator has closed the live window with the title-bar button (waitKey then
    returns -1 for ever, so a pause loop that only watches keys would never end)."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True
