"""Module 1 - procedural blister-pack inspection cell (USDA authoring).

Builds the complete inspection cell as multi-line USDA text (the OpenUSD text parser rejects
single-line prim bodies) and writes it to ``assets/pack.usda``:

* conveyor belt primitive (static collider), diffuse dome light, key light, camera at the
  frozen line geometry (Decision 5), render product with RGB + semantic outputs;
* a 10-cavity (2x5) blister pack: aluminium backing foil, PVC domes, pills;
* per cavity, four self-contained state variants (pill_ok, pill_damaged, cavity_empty,
  foil_damaged).  Every leaf of a variant carries the SAME ``SemanticsAPI:class`` label
  ``c{idx:02d}_{state}`` so the renderer's semantic id map yields one id per cavity per
  state.  Exactly one variant per cavity is visible at a time (visibility token toggled at
  runtime by the SDG pipeline / twin), which gives exactly ten labelled regions per pack.

The same file is the single source of truth for both simulation modes (SDG and live twin).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSET = PROJECT_ROOT / "assets" / "pack.usda"

# --------------------------------------------------------------------------------------
# Frozen line geometry (Decision 5) and camera model
# --------------------------------------------------------------------------------------
PITCH_M = 0.120            # pack pitch along the belt
PACK_L_M = 0.090           # pack length along the belt (X)
PACK_W_M = 0.045           # pack width across the belt (Y)
PACK_T_M = 0.0015          # backing card + lidding foil thickness
FOV_ALONG_M = 0.200        # camera field of view along the belt
STANDOFF_M = 0.300         # camera -> rejector distance
V_BELT_MPS = 1.6           # design belt velocity (800 packs/min at 120 mm pitch)
STROBE_S = 50e-6           # nominal simulated strobe exposure

IMAGE_W, IMAGE_H = 1280, 720
FOCAL_MM = 50.0
APERTURE_H_MM = 36.0
APERTURE_V_MM = APERTURE_H_MM * IMAGE_H / IMAGE_W          # 20.25 mm -> 16:9, no letterbox
CAM_HEIGHT_M = FOV_ALONG_M * FOCAL_MM / APERTURE_H_MM       # 0.27778 m gives 200 mm along the belt
GSD_M = FOV_ALONG_M / IMAGE_W                               # 0.156 mm/px at the belt plane
F_PX = FOCAL_MM / APERTURE_H_MM * IMAGE_W                   # 1777.8 px focal length

# --------------------------------------------------------------------------------------
# Cavity layout and labels
# --------------------------------------------------------------------------------------
N_COLS, N_ROWS = 5, 2
CAVITY_PITCH_X_M = 0.016
CAVITY_ROW_Y_M = 0.011
POCKET_R_M = 0.006          # dome footprint radius (12 mm pocket)
POCKET_H_M = 0.0045         # dome height above the foil
PILL_R_M = 0.004            # 8 mm round tablet
PILL_H_M = 0.003
PILL_BAND_M = 0.0012        # cylindrical band of the biconvex tablet; the two caps share the rest
CARD_CORNER_R_M = 0.003     # rounded card corners
CARD_CHAMFER_M = 0.0003     # edge chamfer of the card slab
SEAL_PITCH_M = 0.00125      # knurl pitch of the sealing tool (waffle seal)
SEAL_AMP_M = 0.00008        # 80 um knurl height (0.5 px at the 0.156 mm GSD: a specular texture, not geometry, to the detector)
SEAL_CLEAR_M = 0.0007       # flat ring around each pocket
RING_INTENSITY = 2000.0     # nominal ring light (DiskLight around the lens), randomised by the SDG
ASSET_VERSION = 2           # 1 = boxy card, cylinder tablets, dome+key only

STATES = ("pill_ok", "pill_damaged", "cavity_empty", "foil_damaged")
CLASS_IDS = {s: i for i, s in enumerate(STATES)}   # 0..3, strict integers for YOLO
PACK_LABEL = "pack"
GLASS_MODE = "omniglass"    # "omniglass" (refractive PVC) or "opacity" (OmniPBR cutout fallback)


def cavity_centers() -> list[tuple[float, float]]:
    """Pack-local (x, y) centres, index c = row * 5 + col; row 0 is +Y (image top)."""
    out = []
    for row in range(N_ROWS):
        y = CAVITY_ROW_Y_M if row == 0 else -CAVITY_ROW_Y_M
        for col in range(N_COLS):
            out.append(((col - (N_COLS - 1) / 2) * CAVITY_PITCH_X_M, y))
    return out


def cavity_label(idx: int, state: str) -> str:
    return f"c{idx:02d}_{state}"


def parse_cavity_label(label: str) -> tuple[int, str] | None:
    """``"c03_pill_ok"`` -> (3, "pill_ok"); None for non-cavity labels."""
    if len(label) < 5 or label[0] != "c" or not label[1:3].isdigit() or label[3] != "_":
        return None
    state = label[4:]
    return (int(label[1:3]), state) if state in STATES else None


def cavity_path(idx: int) -> str:
    return f"/World/Pack/Cavity_{idx:02d}"


def variant_path(idx: int, state: str) -> str:
    return f"{cavity_path(idx)}/{state}"


_VARIANT_LEAVES = {
    "pill_ok": ("Dome", "Floor", "Pill"),
    "pill_damaged": ("Dome", "Floor", "PillChunk", "PillFrag"),
    "cavity_empty": ("Dome", "Floor"),
    "foil_damaged": ("DomeCrushed", "Patch0", "Patch1", "PillFlat"),
}


def variant_leaf_paths(idx: int, state: str) -> list[str]:
    return [f"{variant_path(idx, state)}/{leaf}" for leaf in _VARIANT_LEAVES[state]]


def pocket_footprint_corners(idx: int) -> np.ndarray:
    """8 pack-local corners of the pocket's bounding box (used for projection labels)."""
    cx, cy = cavity_centers()[idx]
    r, z0, z1 = POCKET_R_M, PACK_T_M, PACK_T_M + POCKET_H_M
    return np.array([[cx + sx * r, cy + sy * r, z] for z in (z0, z1) for sx in (-1, 1) for sy in (-1, 1)], dtype=np.float64)


# --------------------------------------------------------------------------------------
# Pack state and deterministic scheduler
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PackState:
    states: tuple[str, ...]   # length 10, entries from STATES

    def __post_init__(self):
        assert len(self.states) == N_COLS * N_ROWS and all(s in STATES for s in self.states)

    @property
    def is_nominal(self) -> bool:
        return all(s == "pill_ok" for s in self.states)

    @property
    def class_ids(self) -> list[int]:
        return [CLASS_IDS[s] for s in self.states]

    @staticmethod
    def nominal() -> "PackState":
        return PackState(("pill_ok",) * (N_COLS * N_ROWS))


class PackScheduler:
    """Seeded, order-independent defect assignment: ``sample(pack_id)`` is a pure function of
    (seed, pack_id), so any frame can be regenerated in isolation."""

    # Training mix (2026-09-10 sizing): 35 % nominal packs; in a defective pack each cavity is
    # i.i.d. with P(ok, pill_damaged, empty, foil) = (0.62, 0.12, 0.14, 0.12), all-ok draws rejected.
    # Expected boxes per pack: pill_ok 7.51, cavity_empty 0.92, pill_damaged 0.79, foil_damaged 0.79.
    # The production defect prior is far lower; it is imposed later by thresholds on real data.
    def __init__(self, seed: int, p_nominal: float = 0.35, p_empty: float = 0.14, p_pill_damaged: float = 0.12, p_foil: float = 0.12):
        self.seed = int(seed)
        self.p_nominal = p_nominal
        self.p = np.array([1.0 - p_empty - p_pill_damaged - p_foil, p_pill_damaged, p_empty, p_foil], dtype=np.float64)
        assert self.p.min() >= 0.0

    @property
    def rates(self) -> dict:
        return {"p_nominal": self.p_nominal, "p_ok": float(self.p[0]), "p_pill_damaged": float(self.p[1]), "p_empty": float(self.p[2]), "p_foil": float(self.p[3])}

    def expected_boxes_per_pack(self) -> dict:
        """Closed-form expectation used for dataset sizing (conditional on >= 1 defect in the
        defective branch)."""
        p_all_ok = float(self.p[0]) ** (N_COLS * N_ROWS)
        cond = 1.0 - p_all_ok
        n = N_COLS * N_ROWS
        e_def = {s: (1.0 - self.p_nominal) * n * float(self.p[i]) / cond for i, s in enumerate(STATES) if s != "pill_ok"}
        e_ok = self.p_nominal * n + (1.0 - self.p_nominal) * (n * float(self.p[0]) - n * p_all_ok) / cond
        return {"pill_ok": e_ok, **e_def}

    def sample(self, pack_id: int) -> PackState:
        rng = np.random.default_rng([self.seed, int(pack_id)])
        if rng.random() < self.p_nominal:
            return PackState.nominal()
        for _ in range(16):
            draw = rng.choice(len(STATES), size=N_COLS * N_ROWS, p=self.p)
            if (draw != 0).any():
                return PackState(tuple(STATES[i] for i in draw))
        draw = [0] * (N_COLS * N_ROWS)
        draw[int(rng.integers(N_COLS * N_ROWS))] = 2
        return PackState(tuple(STATES[i] for i in draw))


# --------------------------------------------------------------------------------------
# USDA authoring helpers (always multi-line bodies)
# --------------------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return f"{v:.6g}"


def _v3(v) -> str:
    return "(" + ", ".join(_fmt(float(x)) for x in v) + ")"


def _prim(kind: str, name: str, body: list[str], schemas: tuple[str, ...] = (), indent: int = 4) -> str:
    pad = " " * indent
    head = f'{pad}def {kind} "{name}"'
    if schemas:
        head += " (\n" + pad + "    prepend apiSchemas = [" + ", ".join(f'"{s}"' for s in schemas) + "]\n" + pad + ")"
    lines = [head, pad + "{"]
    lines += [pad + "    " + b for b in body]
    lines.append(pad + "}")
    return "\n".join(lines)


def _xform_ops(translate=None, rotate_xyz=None, scale=None) -> list[str]:
    body, order = [], []
    if translate is not None:
        body.append(f"double3 xformOp:translate = {_v3(translate)}")
        order.append('"xformOp:translate"')
    if rotate_xyz is not None:
        body.append(f"double3 xformOp:rotateXYZ = {_v3(rotate_xyz)}")
        order.append('"xformOp:rotateXYZ"')
    if scale is not None:
        body.append(f"float3 xformOp:scale = {_v3(scale)}")
        order.append('"xformOp:scale"')
    if order:
        body.append("uniform token[] xformOpOrder = [" + ", ".join(order) + "]")
    return body


def _semantic(label: str) -> list[str]:
    return [f'string semantic:class:params:semanticData = "{label}"', 'string semantic:class:params:semanticType = "class"']


def _material_binding(material: str) -> list[str]:
    return [f"rel material:binding = </World/Looks/{material}>"]


def _leaf(kind: str, name: str, label: str, material: str, translate, scale, rotate_xyz=None, visible=True, extra: list[str] | None = None, indent=12) -> str:
    body = list(extra or [])
    body += _semantic(label) + _material_binding(material)
    if not visible:
        body.append('token visibility = "invisible"')
    body += _xform_ops(translate=translate, rotate_xyz=rotate_xyz, scale=scale)
    return _prim(kind, name, body, ("SemanticsAPI:class", "MaterialBindingAPI"), indent=indent)


def _mesh(kind_name: str, label: str | None, material: str, geom: tuple, translate=None, rotate_xyz=None, scale=None, visible=True, indent=12, schemas: tuple[str, ...] = ()) -> str:
    """UsdGeomMesh prim from (points, normals | None, faceVertexCounts, faceVertexIndices).
    Per-vertex normals (when given) make lathe surfaces shade smoothly; the slab and the seal
    grid ship their own.  Textures do not load on this build (probed 2026-09-17: diffuse and
    normal-map inputs are ignored from a string-populated stage), so every micro-detail here is
    real geometry."""
    pts, nrm, counts, idx = geom
    body = [
        "int[] faceVertexCounts = [" + ", ".join(str(c) for c in counts) + "]",
        "int[] faceVertexIndices = [" + ", ".join(str(i) for i in idx) + "]",
        "point3f[] points = [" + ", ".join(_v3(p) for p in pts) + "]",
        'uniform token subdivisionScheme = "none"',
    ]
    if nrm is not None:
        body += ["normal3f[] normals = [" + ", ".join(_v3(n) for n in nrm) + "] (", '    interpolation = "vertex"', ")"]
    if label is not None:
        body += _semantic(label)
    body += _material_binding(material)
    if not visible:
        body.append('token visibility = "invisible"')
    body += _xform_ops(translate=translate, rotate_xyz=rotate_xyz, scale=scale)
    sch = (("SemanticsAPI:class",) if label is not None else ()) + ("MaterialBindingAPI",) + schemas
    return _prim("Mesh", kind_name, body, sch, indent=indent)


def _lathe(profile: list[tuple[float, float]], segments: int = 32) -> tuple:
    """Revolve an (r, z) profile about Z.  The first and last profile points must have r = 0
    (poles).  Normals come from the profile's local slope, so caps and bands shade smoothly."""
    rz = np.asarray(profile, dtype=np.float64)
    assert rz[0, 0] == 0.0 and rz[-1, 0] == 0.0 and len(rz) >= 3
    # profile normals (2D): perpendicular to the tangent, pointing outward (+r)
    tang = np.gradient(rz, axis=0)
    nrm2 = np.stack([tang[:, 1], -tang[:, 0]], 1)
    nrm2 /= np.linalg.norm(nrm2, axis=1, keepdims=True) + 1e-12
    nrm2[nrm2[:, 0] < 0] *= -1
    nrm2[0] = (0.0, -1.0)
    nrm2[-1] = (0.0, 1.0)
    pts, nrms = [(0.0, 0.0, rz[0, 1])], [(0.0, 0.0, -1.0)]
    rings = []
    for k in range(1, len(rz) - 1):
        ring = []
        for s in range(segments):
            a = 2 * math.pi * s / segments
            c, sn = math.cos(a), math.sin(a)
            ring.append(len(pts))
            pts.append((rz[k, 0] * c, rz[k, 0] * sn, rz[k, 1]))
            nrms.append((nrm2[k, 0] * c, nrm2[k, 0] * sn, nrm2[k, 1]))
        rings.append(ring)
    top = len(pts)
    pts.append((0.0, 0.0, rz[-1, 1]))
    nrms.append((0.0, 0.0, 1.0))
    counts, idx = [], []
    for s in range(segments):                       # bottom fan
        counts.append(3)
        idx += [0, rings[0][(s + 1) % segments], rings[0][s]]
    for a, b in zip(rings[:-1], rings[1:]):          # bands
        for s in range(segments):
            counts.append(4)
            idx += [a[s], a[(s + 1) % segments], b[(s + 1) % segments], b[s]]
    for s in range(segments):                       # top fan
        counts.append(3)
        idx += [top, rings[-1][s], rings[-1][(s + 1) % segments]]
    return pts, nrms, counts, idx


def tablet_geometry(radius: float = PILL_R_M, height: float = PILL_H_M, band: float = PILL_BAND_M, cap_samples: int = 7, segments: int = 32) -> tuple:
    """Biconvex tablet: cylindrical band plus two spherical caps, 8 mm footprint kept exactly.
    Cap sphere radius Rs = (h_c^2 + R^2) / (2 h_c) so the cap meets the band tangent-free at R."""
    h_c = (height - band) / 2
    rs = (h_c * h_c + radius * radius) / (2 * h_c)
    prof = [(0.0, -height / 2)]
    for k in range(1, cap_samples + 1):                                   # bottom cap, pole -> band
        z = -height / 2 + h_c * k / cap_samples
        zc = -height / 2 + rs                                             # centre above the pole
        prof.append((math.sqrt(max(rs * rs - (z - zc) ** 2, 0.0)), z))
    prof.append((radius, band / 2))                                       # band top edge
    for k in range(1, cap_samples + 1):                                   # top cap, band -> pole
        z = band / 2 + h_c * k / cap_samples
        zc = height / 2 - rs
        r = math.sqrt(max(rs * rs - (z - zc) ** 2, 0.0)) if k < cap_samples else 0.0
        prof.append((r, z))
    return _lathe(prof, segments)


def rounded_slab_geometry(length: float, width: float, thickness: float, corner_r: float, chamfer: float, corner_segments: int = 6) -> tuple:
    """Closed slab with rounded corners and a chamfered top and bottom edge: four loops (bottom
    inset, lower edge, upper edge, top inset) joined by quads, capped by two n-gons.  Flat shaded."""
    def loop(inset: float) -> list[tuple[float, float]]:
        pts = []
        r = corner_r - inset
        for cx, cy, a0 in ((length / 2 - corner_r, width / 2 - corner_r, 0), (-length / 2 + corner_r, width / 2 - corner_r, 90),
                           (-length / 2 + corner_r, -width / 2 + corner_r, 180), (length / 2 - corner_r, -width / 2 + corner_r, 270)):
            for k in range(corner_segments + 1):
                a = math.radians(a0 + 90 * k / corner_segments)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts
    loops = [(loop(chamfer), 0.0), (loop(0.0), chamfer), (loop(0.0), thickness - chamfer), (loop(chamfer), thickness)]
    m = len(loops[0][0])
    pts = [(x, y, z) for lp, z in loops for (x, y) in lp]
    counts, idx = [], []
    for li in range(3):
        for k in range(m):
            a, b = li * m + k, li * m + (k + 1) % m
            counts.append(4)
            idx += [a, b, b + m, a + m]
    counts.append(m)
    idx += [3 * m + k for k in range(m)]                                   # top cap
    counts.append(m)
    idx += [m - 1 - k for k in range(m)]                                   # bottom cap, reversed winding
    return pts, None, counts, idx


def seal_geometry(length: float, width: float, z0: float, pitch: float, amp: float, clear_r: float, holes: list[tuple[float, float]], inset: float = 0.001, samples_per_pitch: int = 2) -> tuple:
    """Knurled seal surface: a grid over the card top, displaced by a waffle pattern, flat within
    ``clear_r`` of each pocket.  Per-vertex normals from the analytic gradient.  The grid must be
    sampled at least twice per pitch: at one sample per pitch every vertex lands on a zero of
    sin^2 and the surface authored is flat (the first v2 render had no knurl for that reason)."""
    assert samples_per_pitch >= 2
    nx, ny = int(round((length - 2 * inset) / pitch * samples_per_pitch)), int(round((width - 2 * inset) / pitch * samples_per_pitch))
    xs = np.linspace(-length / 2 + inset, length / 2 - inset, nx + 1)
    ys = np.linspace(-width / 2 + inset, width / 2 - inset, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    sx, sy = np.sin(math.pi * X / pitch) ** 2, np.sin(math.pi * Y / pitch) ** 2
    H = amp * sx * sy
    dHdx = amp * (2 * math.pi / pitch) * np.sin(math.pi * X / pitch) * np.cos(math.pi * X / pitch) * sy
    dHdy = amp * (2 * math.pi / pitch) * np.sin(math.pi * Y / pitch) * np.cos(math.pi * Y / pitch) * sx
    mask = np.ones_like(H, dtype=bool)
    for hx, hy in holes:
        mask &= (X - hx) ** 2 + (Y - hy) ** 2 > clear_r * clear_r
    H, dHdx, dHdy = np.where(mask, H, 0.0), np.where(mask, dHdx, 0.0), np.where(mask, dHdy, 0.0)
    N = np.stack([-dHdx, -dHdy, np.ones_like(H)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    pts = [(float(X[j, i]), float(Y[j, i]), float(z0 + H[j, i])) for j in range(ny + 1) for i in range(nx + 1)]
    nrm = [tuple(float(v) for v in N[j, i]) for j in range(ny + 1) for i in range(nx + 1)]
    counts, idx = [], []
    w = nx + 1
    for j in range(ny):
        for i in range(nx):
            counts.append(4)
            idx += [j * w + i, j * w + i + 1, (j + 1) * w + i + 1, (j + 1) * w + i]
    return pts, nrm, counts, idx


def chamfered_box_geometry(sx: float, sy: float, sz: float, c: float) -> tuple:
    """Box (sx, sy, sz) centred at the origin with all twelve edges chamfered by ``c``: 24
    vertices, 6 face rectangles + 12 edge quads + 8 corner triangles.  Flat shaded (every face is
    planar and the chamfer facets are meant to read as facets).  Winding is fixed numerically so
    every face points outward regardless of how the corner loops are enumerated."""
    c = min(c, min(sx, sy, sz) / 2 - 1e-6)
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    pts, idx_of = [], {}
    for ex in (-1, 1):
        for ey in (-1, 1):
            for ez in (-1, 1):
                for kind, p in (("x", (ex * hx, ey * (hy - c), ez * (hz - c))), ("y", (ex * (hx - c), ey * hy, ez * (hz - c))), ("z", (ex * (hx - c), ey * (hy - c), ez * hz))):
                    idx_of[(ex, ey, ez, kind)] = len(pts)
                    pts.append(p)
    faces = []
    for axis, kind in (("x", "x"), ("y", "y"), ("z", "z")):                       # 6 face rectangles
        for s in (-1, 1):
            corners = [(ex, ey, ez) for ex in (-1, 1) for ey in (-1, 1) for ez in (-1, 1) if {"x": ex, "y": ey, "z": ez}[axis] == s]
            faces.append([idx_of[(*cn, kind)] for cn in corners])
    for a, b in (("x", "y"), ("y", "z"), ("x", "z")):                              # 12 edge quads
        third = ({"x", "y", "z"} - {a, b}).pop()
        for sa in (-1, 1):
            for sb in (-1, 1):
                quad = []
                for st in (-1, 1):
                    cn = {a: sa, b: sb, third: st}
                    key = (cn["x"], cn["y"], cn["z"])
                    quad += [idx_of[(*key, a)], idx_of[(*key, b)]]
                faces.append(quad)
    for ex in (-1, 1):                                                              # 8 corner triangles
        for ey in (-1, 1):
            for ez in (-1, 1):
                faces.append([idx_of[(ex, ey, ez, k)] for k in ("x", "y", "z")])
    P = np.asarray(pts)
    counts, idx = [], []
    for f in faces:
        poly = P[f]
        centre = poly.mean(0)
        # order the polygon's vertices around its centroid so quads/rects are non-self-intersecting
        n0 = centre / (np.linalg.norm(centre) + 1e-12)
        u = np.cross(n0, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(n0, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n0, u)
        ang = [math.atan2(float(np.dot(p - centre, v)), float(np.dot(p - centre, u))) for p in poly]
        order = [f[i] for i in np.argsort(ang)]
        q = P[order]
        nrm = np.cross(q[1] - q[0], q[2] - q[0])
        if np.dot(nrm, centre) < 0:                                                  # outward winding
            order = order[::-1]
        counts.append(len(order))
        idx += [int(i) for i in order]
    return [tuple(float(v) for v in p) for p in pts], None, counts, idx


def proxy_pack_prim(name: str, translate, rotate_xyz, has_pill: tuple, indent: int = 8, visible: bool = False) -> tuple[str, list[str]]:
    """Render-only stand-in for a pack (card, domes, lidding floors, tablets where ``has_pill``):
    no physics schema, no semantic label.  Used by the twin for packs that have already left the
    simulation (sliding down the chute, resting in the tote) so the dynamic pool is never held.
    Returns the USDA text and the leaf paths under /World/Props/<name> for visibility writes."""
    pad = " " * indent
    leaves, paths = [], []
    leaves.append(_mesh("Card", None, "Foil", rounded_slab_geometry(PACK_L_M, PACK_W_M, PACK_T_M, CARD_CORNER_R_M, CARD_CHAMFER_M), visible=visible, indent=indent + 4))
    paths.append(f"/World/Props/{name}/Card")
    for i, (cx, cy) in enumerate(cavity_centers()):
        leaves.append(_mesh(f"Dome_{i:02d}", None, "Glass", _DOME, translate=(cx, cy, PACK_T_M), visible=visible, indent=indent + 4))
        leaves.append(_prim("Cylinder", f"Floor_{i:02d}", _material_binding("Lidding") + ["double radius = 1", "double height = 1", 'uniform token axis = "Z"']
                            + ([] if visible else ['token visibility = "invisible"']) + _xform_ops(translate=(cx, cy, PACK_T_M + 0.0001), scale=(POCKET_R_M - 0.0005, POCKET_R_M - 0.0005, 0.0001)), ("MaterialBindingAPI",), indent=indent + 4))
        paths += [f"/World/Props/{name}/Dome_{i:02d}", f"/World/Props/{name}/Floor_{i:02d}"]
        if has_pill[i]:
            leaves.append(_mesh(f"Pill_{i:02d}", None, "Pill", _TABLET, translate=(cx, cy, PACK_T_M + 0.0002 + PILL_H_M / 2), visible=visible, indent=indent + 4))
            paths.append(f"/World/Props/{name}/Pill_{i:02d}")
    body = _xform_ops(translate=translate, rotate_xyz=rotate_xyz) + [lf.strip() for lf in leaves]
    text = _prim("Xform", name, body, indent=indent)
    # nested leaves were authored at their own indent; the text parser ignores indentation, but keep
    # bodies multi-line (already guaranteed by _prim)
    return text + "\n", paths


def annulus_geometry(r_in: float, r_out: float, z: float, segments: int = 48) -> tuple:
    """Flat ring in the XY plane facing -Z (the visible glow of the ring light)."""
    pts, nrm = [], []
    for s in range(segments):
        a = 2 * math.pi * s / segments
        pts += [(r_in * math.cos(a), r_in * math.sin(a), z), (r_out * math.cos(a), r_out * math.sin(a), z)]
        nrm += [(0.0, 0.0, -1.0), (0.0, 0.0, -1.0)]
    counts, idx = [], []
    for s in range(segments):
        i0, o0, i1, o1 = 2 * s, 2 * s + 1, 2 * ((s + 1) % segments), 2 * ((s + 1) % segments) + 1
        counts.append(4)
        idx += [i0, i1, o1, o0]
    return pts, nrm, counts, idx


def _mdl_material(name: str, module: str, sub_identifier: str, inputs: list[str]) -> str:
    shader = _prim("Shader", "Shader", [
        'uniform token info:implementationSource = "sourceAsset"',
        f"uniform asset info:mdl:sourceAsset = @{module}@",
        f'uniform token info:mdl:sourceAsset:subIdentifier = "{sub_identifier}"',
        *inputs,
        "token outputs:out",
    ], indent=12)
    return _prim("Material", name, [f"token outputs:mdl:surface.connect = </World/Looks/{name}/Shader.outputs:out>", shader.strip()], indent=8).replace("\n            def Shader", "\n            def Shader")


def _materials() -> str:
    # Thermoformed PVC: OmniGlass is a physically based dielectric, so the Fresnel falloff is
    # inherent (grazing reflectance rises towards 1); the visible tuning is a faint blue-white
    # tint, a PVC-like frosting and a thin-wall depth so the domes read as formed film.
    glass_inputs = (
        ["color3f inputs:glass_color = (0.97, 0.985, 1)", "float inputs:glass_ior = 1.53", "float inputs:frosting_roughness = 0.07", "bool inputs:thin_walled = 1", "float inputs:depth = 0.0008"]
        if GLASS_MODE == "omniglass"
        else ["color3f inputs:diffuse_color_constant = (0.85, 0.9, 0.95)", "bool inputs:enable_opacity = 1", "float inputs:opacity_constant = 0.35", "float inputs:reflection_roughness_constant = 0.1"]
    )
    glass = _mdl_material("Glass", "OmniGlass.mdl" if GLASS_MODE == "omniglass" else "OmniPBR.mdl", "OmniGlass" if GLASS_MODE == "omniglass" else "OmniPBR", glass_inputs)
    mats = [
        _mdl_material("Foil", "OmniPBR.mdl", "OmniPBR", ["color3f inputs:diffuse_color_constant = (0.82, 0.83, 0.86)", "float inputs:metallic_constant = 0.9", "float inputs:reflection_roughness_constant = 0.35"]),
        _mdl_material("Seal", "OmniPBR.mdl", "OmniPBR", ["color3f inputs:diffuse_color_constant = (0.80, 0.81, 0.84)", "float inputs:metallic_constant = 0.9", "float inputs:reflection_roughness_constant = 0.45"]),
        glass,
        _mdl_material("Pill", "OmniPBR.mdl", "OmniPBR", ["color3f inputs:diffuse_color_constant = (0.95, 0.95, 0.92)", "float inputs:metallic_constant = 0.0", "float inputs:reflection_roughness_constant = 0.55", "float inputs:specular_level = 0.35"]),
        _mdl_material("Lidding", "OmniPBR.mdl", "OmniPBR", ["color3f inputs:diffuse_color_constant = (0.50, 0.50, 0.54)", "float inputs:metallic_constant = 0.8", "float inputs:reflection_roughness_constant = 0.45"]),
        _mdl_material("Belt", "OmniPBR.mdl", "OmniPBR", ["color3f inputs:diffuse_color_constant = (0.13, 0.14, 0.15)", "float inputs:metallic_constant = 0.0", "float inputs:reflection_roughness_constant = 0.85"]),
        _mdl_material("Damage", "OmniPBR.mdl", "OmniPBR", ["color3f inputs:diffuse_color_constant = (0.05, 0.045, 0.04)", "float inputs:metallic_constant = 0.2", "float inputs:reflection_roughness_constant = 0.9"]),
        _mdl_material("Station", "OmniPBR.mdl", "OmniPBR", ["color3f inputs:diffuse_color_constant = (0.10, 0.105, 0.11)", "float inputs:metallic_constant = 0.3", "float inputs:reflection_roughness_constant = 0.6"]),
        _mdl_material("Diffuser", "OmniPBR.mdl", "OmniPBR", ["color3f inputs:diffuse_color_constant = (0.9, 0.9, 0.9)", "bool inputs:enable_emission = 1", "color3f inputs:emissive_color = (1, 0.97, 0.92)", "float inputs:emissive_intensity = 20"]),
    ]
    # re-indent material blocks under /World/Looks (they were authored at indent 8 with shader at 12)
    return _prim("Scope", "Looks", [m.strip() for m in mats], indent=4)


def dome_geometry(radius: float = POCKET_R_M, height: float = POCKET_H_M, samples: int = 10, segments: int = 32, power: float = 2.6) -> tuple:
    """Thermoformed pocket: superellipse profile r(z) = R (1 - (z/H)^p)^(1/p) - a flatter top and
    steeper wall than a sphere - revolved with per-vertex normals (the Sphere primitive is
    tessellated coarsely on this build, which the ring light exposes as a scalloped rim)."""
    prof = [(0.0, height)]                                            # pole on top; lathe wants poles first/last
    zs = [height * (1 - k / samples) for k in range(1, samples + 1)]  # top -> base
    prof = [(0.0, 0.0)] + [(radius * (1 - (z / height) ** power) ** (1 / power), z) for z in reversed(zs)] + [(0.0, height)]
    prof[1] = (radius, 0.0)                                           # base rim exactly at R
    return _lathe(prof, segments)


_TABLET = tablet_geometry()
_DOME = dome_geometry()


def _cavity(idx: int, state_visible: str) -> str:
    cx, cy = cavity_centers()[idx]
    z_foil = PACK_T_M
    dome_scale = (POCKET_R_M, POCKET_R_M, POCKET_H_M)
    floor_scale = (POCKET_R_M - 0.0005, POCKET_R_M - 0.0005, 0.0001)
    pill_scale = (PILL_R_M, PILL_R_M, PILL_H_M / 2)
    z_pill = z_foil + 0.0002 + PILL_H_M / 2
    variants = []
    for state in STATES:
        vis = state == state_visible
        lab = cavity_label(idx, state)
        leaves = []
        if state in ("pill_ok", "pill_damaged", "cavity_empty"):
            leaves.append(_mesh("Dome", lab, "Glass", _DOME, translate=(0, 0, z_foil), visible=vis))
            leaves.append(_leaf("Cylinder", "Floor", lab, "Lidding", (0, 0, z_foil + 0.0001), floor_scale, visible=vis, extra=["double radius = 1", "double height = 1", 'uniform token axis = "Z"']))
        if state == "pill_ok":
            leaves.append(_mesh("Pill", lab, "Pill", _TABLET, translate=(0, 0, z_pill), visible=vis))
        elif state == "pill_damaged":
            leaves.append(_leaf("Cylinder", "PillChunk", lab, "Pill", (-0.0015, 0.0005, z_pill), (PILL_R_M * 0.55, PILL_R_M, PILL_H_M / 2), rotate_xyz=(0, 0, 20), visible=vis, extra=["double radius = 1", "double height = 1", 'uniform token axis = "Z"']))
            leaves.append(_leaf("Cube", "PillFrag", lab, "Pill", (0.0025, -0.002, z_foil + 0.0002 + 0.0008), (0.0018, 0.0012, 0.0008), rotate_xyz=(0, 0, -35), visible=vis, extra=["double size = 1"]))
        elif state == "foil_damaged":
            leaves.append(_mesh("DomeCrushed", lab, "Glass", _DOME, translate=(0, 0, z_foil), rotate_xyz=(12, 0, 0), scale=(1.0, 0.9, 0.35), visible=vis))
            leaves.append(_leaf("Cube", "Patch0", lab, "Damage", (-0.002, 0.0015, z_foil + 0.0002), (0.005, 0.0025, 0.0003), rotate_xyz=(0, 0, 25), visible=vis, extra=["double size = 1"]))
            leaves.append(_leaf("Cube", "Patch1", lab, "Damage", (0.0025, -0.002, z_foil + 0.0002), (0.0035, 0.002, 0.0003), rotate_xyz=(0, 0, -40), visible=vis, extra=["double size = 1"]))
            leaves.append(_leaf("Cylinder", "PillFlat", lab, "Pill", (0.0005, 0, z_foil + 0.0002 + PILL_H_M * 0.2), (PILL_R_M, PILL_R_M, PILL_H_M * 0.2), visible=vis, extra=["double radius = 1", "double height = 1", 'uniform token axis = "Z"']))
        variants.append(_prim("Xform", state, [lf.strip() for lf in leaves], indent=8))
    return _prim("Xform", f"Cavity_{idx:02d}", _xform_ops(translate=(cx, cy, 0.0)) + [v.strip() for v in variants], indent=4)


def build_scene_usda(state: PackState | None = None, width: int = IMAGE_W, height: int = IMAGE_H) -> str:
    """Complete inspection cell.  ``state`` selects the visible variant per cavity."""
    state = state or PackState.nominal()
    # Physics invariant: the collider stays a plain box (PhysicsCollisionAPI on ``Body``), inset
    # 1 mm in x/y and 0.1 mm in z so it is fully enclosed by the visual card and never shows.
    # Mass and the centre of mass (z = T/2) are unchanged from asset v1.
    body = _prim("Cube", "Body", _material_binding("Foil") + ["double size = 1"] + _xform_ops(translate=(0, 0, PACK_T_M / 2), scale=(PACK_L_M - 0.002, PACK_W_M - 0.002, PACK_T_M - 0.0002)), ("MaterialBindingAPI", "PhysicsCollisionAPI"), indent=4)
    card = _mesh("Card", PACK_LABEL, "Foil", rounded_slab_geometry(PACK_L_M, PACK_W_M, PACK_T_M, CARD_CORNER_R_M, CARD_CHAMFER_M), indent=4)
    seal = _mesh("Seal", PACK_LABEL, "Seal", seal_geometry(PACK_L_M, PACK_W_M, PACK_T_M + 0.00002, SEAL_PITCH_M, SEAL_AMP_M, POCKET_R_M + SEAL_CLEAR_M, cavity_centers()), indent=4)
    cavities = [_cavity(i, state.states[i]) for i in range(N_COLS * N_ROWS)]
    pack = _prim("Xform", "Pack", ["float physics:mass = 0.012", body.strip(), card.strip(), seal.strip()] + [c.strip() for c in cavities], ("PhysicsRigidBodyAPI", "PhysicsMassAPI"), indent=0)
    # Re-indent the pack block (authored at indent 0) under /World.
    pack = "\n".join("    " + ln if ln else ln for ln in pack.splitlines())
    belt = _prim("Cube", "Belt", _material_binding("Belt") + ["double size = 1"] + _xform_ops(translate=(0, 0, -0.01), scale=(1.2, 0.30, 0.02)), ("MaterialBindingAPI", "PhysicsCollisionAPI"), indent=4)
    # Lighting lives in the asset only (the twin adds none), so the training frames and the
    # live inspection see the same illumination: dome + key rebalanced, plus the ring light.
    dome = _prim("DomeLight", "DomeLight", ["float inputs:intensity = 600", "color3f inputs:color = (1, 1, 1)"], indent=4)
    key = _prim("DistantLight", "KeyLight", ["float inputs:intensity = 550", "float inputs:angle = 1.5", "color3f inputs:color = (1, 1, 1)"] + _xform_ops(rotate_xyz=(25, 18, 0)), indent=4)
    # Inspection station as children of the camera (camera-local frame looks down -Z, so every
    # part sits at local z > 0, behind the sensor, and follows the SDG camera randomisation):
    # ring light around the lens (UsdLux DiskLight emits along its -Z), a visible diffuser ring,
    # the lens barrel that turns the disk into a ring, hood, housing and the mounting arm.
    station = [
        _prim("DiskLight", "RingLight", [f"float inputs:intensity = {_fmt(RING_INTENSITY)}", "float inputs:radius = 0.05", "color3f inputs:color = (1, 0.98, 0.95)"] + _xform_ops(translate=(0, 0, 0.002)), indent=8),
        _mesh("Diffuser", None, "Diffuser", annulus_geometry(0.022, 0.050, 0.0), translate=(0, 0, 0.0035), indent=8),
        _prim("Cylinder", "LensBarrel", _material_binding("Station") + ["double radius = 1", "double height = 1", 'uniform token axis = "Z"'] + _xform_ops(translate=(0, 0, 0.006), scale=(0.016, 0.016, 0.010)), ("MaterialBindingAPI",), indent=8),
        _prim("Cylinder", "Hood", _material_binding("Station") + ["double radius = 1", "double height = 1", 'uniform token axis = "Z"'] + _xform_ops(translate=(0, 0, 0.014), scale=(0.062, 0.062, 0.018)), ("MaterialBindingAPI",), indent=8),
        _prim("Cube", "Housing", _material_binding("Station") + ["double size = 1"] + _xform_ops(translate=(0, 0, 0.058), scale=(0.062, 0.062, 0.070)), ("MaterialBindingAPI",), indent=8),
        _prim("Cube", "Arm", _material_binding("Station") + ["double size = 1"] + _xform_ops(translate=(0, 0.13, 0.075), scale=(0.030, 0.200, 0.030)), ("MaterialBindingAPI",), indent=8),
    ]
    cam = _prim("Camera", "Camera", [
        f"float focalLength = {_fmt(FOCAL_MM / 100.0)}",
        f"float horizontalAperture = {_fmt(APERTURE_H_MM / 100.0)}",
        f"float verticalAperture = {_fmt(APERTURE_V_MM / 100.0)}",
        "float2 clippingRange = (0.01, 100)",
        'token projection = "perspective"',
    ] + _xform_ops(translate=(0, 0, CAM_HEIGHT_M)) + [s.strip() for s in station], indent=4)
    physics_scene = _prim("PhysicsScene", "physicsScene", ["vector3f physics:gravityDirection = (0, 0, -1)", "float physics:gravityMagnitude = 9.81"], indent=4)
    render = _prim("Scope", "Render", [_prim("RenderProduct", "Camera", [
        "rel camera = </World/Camera>",
        "rel orderedVars = [</Render/Camera/LdrColor>, </Render/Camera/SemanticSegmentation>, </Render/Camera/SemanticIdMap>, </Render/Camera/DistanceToImagePlaneSD>]",
        f"int2 resolution = ({width}, {height})",
        _prim("RenderVar", "LdrColor", ['string sourceName = "LdrColor"'], indent=8).strip(),
        _prim("RenderVar", "SemanticSegmentation", ['string sourceName = "SemanticSegmentation"'], indent=8).strip(),
        _prim("RenderVar", "SemanticIdMap", ['string sourceName = "SemanticIdMap"'], indent=8).strip(),
        _prim("RenderVar", "DistanceToImagePlaneSD", ['string sourceName = "DistanceToImagePlaneSD"'], indent=8).strip(),
    ], indent=4).strip()], indent=0)
    world = "\n".join([
        'def Xform "World"', "{",
        physics_scene, _materials(), dome, key, cam, belt, pack,
        "}",
    ])
    header = '#usda 1.0\n(\n    upAxis = "Z"\n    metersPerUnit = 1\n    defaultPrim = "World"\n    doc = "Blister inspection cell - generated by src/simulation/assets/blister_factory.py"\n)\n'
    return header + world + "\n" + render + "\n"


def _reindent_nested(text: str) -> str:
    """Normalise indentation of nested prim blocks by re-parsing braces (keeps bodies multi-line)."""
    out, depth = [], 0
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            out.append("")
            continue
        if s.startswith("}") or s.startswith(")"):
            depth = max(0, depth - 1)
        out.append("    " * depth + s)
        if s.endswith("{") or s.endswith("(") or s == "(":
            depth += 1
    return "\n".join(out) + "\n"


def scene_usda(state: PackState | None = None, width: int = IMAGE_W, height: int = IMAGE_H) -> str:
    return _reindent_nested(build_scene_usda(state, width, height))


def write_pack_asset(path: Path = DEFAULT_ASSET, state: PackState | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = scene_usda(state)
    path.write_text(text, encoding="utf-8")
    meta = {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "line": {"pitch_m": PITCH_M, "pack_l_m": PACK_L_M, "pack_w_m": PACK_W_M, "fov_along_m": FOV_ALONG_M, "standoff_m": STANDOFF_M, "v_belt_mps": V_BELT_MPS, "strobe_s": STROBE_S},
        "camera": {"width": IMAGE_W, "height": IMAGE_H, "focal_mm": FOCAL_MM, "aperture_h_mm": APERTURE_H_MM, "aperture_v_mm": APERTURE_V_MM, "height_m": CAM_HEIGHT_M, "gsd_m": GSD_M, "f_px": F_PX},
        "classes": CLASS_IDS,
        "glass_mode": GLASS_MODE,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def validate_with_pxr(path: Path) -> str:
    try:
        from pxr import Usd  # type: ignore
    except Exception:  # noqa: BLE001
        return "pxr not installed (skip)"
    stage = Usd.Stage.Open(str(path))
    if not stage:
        return "FAIL: pxr could not open the layer"
    n = sum(1 for _ in stage.Traverse())
    return f"OK: pxr opened the layer, {n} prims"


def render_check(asset: Path, out_dir: Path, steps: int = 4) -> dict:
    """Load the asset in ovrtx, render, and verify the ten nominal cavity labels are present."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from PIL import Image

    from simulation.ovx_runtime import Scene, masks_by_label, mask_bbox_xyxy, check_versions

    check_versions()
    out_dir.mkdir(parents=True, exist_ok=True)
    text = Path(asset).read_text(encoding="utf-8")
    report: dict = {}
    with Scene("blister.factory.check") as sc:
        sc.load_usda(text)
        fr = sc.render("/Render/Camera", steps=steps, width=IMAGE_W, height=IMAGE_H)
        Image.fromarray(fr.rgb).save(out_dir / "pack_preview.png")
        masks = masks_by_label(fr.seg, fr.labels)
        rng = np.random.default_rng(0)
        pal = rng.integers(40, 255, (int(fr.seg.max()) + 1, 3), dtype=np.uint8)
        pal[0] = 0
        Image.fromarray(pal[fr.seg.astype(np.int64)]).save(out_dir / "pack_semantic.png")
        boxes = {}
        for lab, m in masks.items():
            bb = mask_bbox_xyxy(m)
            boxes[lab] = (int(m.sum()), bb)
        expected = {cavity_label(i, "pill_ok") for i in range(N_COLS * N_ROWS)}
        found = set(boxes)
        report["mean_rgb"] = float(fr.rgb.mean())
        report["labels_found"] = sorted(found)
        report["missing_nominal"] = sorted(expected - found)
        report["unexpected"] = sorted(found - expected - {PACK_LABEL})
        report["boxes"] = boxes
        report["ok"] = not report["missing_nominal"] and not report["unexpected"] and PACK_LABEL in found
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Author the blister inspection cell as USDA.")
    ap.add_argument("--out", type=Path, default=DEFAULT_ASSET)
    ap.add_argument("--render-check", action="store_true", help="load in ovrtx, render, verify labels")
    ap.add_argument("--validate", action="store_true", help="open with pxr (usd-core) if installed")
    args = ap.parse_args()
    path = write_pack_asset(args.out)
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    if args.validate:
        print("pxr:", validate_with_pxr(path))
    if args.render_check:
        rep = render_check(path, path.parent)
        print(json.dumps({k: v for k, v in rep.items() if k != "boxes"}, indent=2))
        for lab, (area, bb) in sorted(rep["boxes"].items()):
            print(f"  {lab:>16s}: {area:7d} px  bbox xyxy={bb}")
        print("RENDER CHECK:", "PASS" if rep["ok"] else "FAIL")
        return 0 if rep["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
