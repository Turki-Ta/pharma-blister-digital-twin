"""Thin, leak-safe wrappers around the NVIDIA Omniverse libraries (Set A).

Pinned in pyproject.toml: ovrtx==0.4.1.364340, ovstage==0.1.1.355824, ovphysx==0.5.11.

Lessons encoded here (from the 2026-09-10 smoke tests on this machine):
* ovrtx logs "Leaking step result outputs" unless every mapped render var is unmapped and
  every frame/product/products handle is deleted before the next step -> ``StepOutputs``.
* Runtime transforms are written to the ``omni:xform`` 4x4 attribute (row-vector USD
  convention: translation in the last row), not to ``xformOp:*`` -> ``usd_matrix``.
* SemanticIdMap is a packed table (24-byte entries on this build) -> ``decode_idmap``.
* Render var keys differ between ovrtx releases (source name vs full path) -> suffix match.
* Write ORDER inside one ordinal matters: scalar/colour/token writes first, ``omni:xform``
  writes last.  Transforms followed by scalar writes in the same ordinal render at a stale pose
  from the third frame on.  The first transform write after population needs one priming step
  before it is visible.  (Both isolated on 2026-09-10 with a recipe matrix; see sdg_pipeline.)
* Population domains: PHYSICS|RENDERING in ONE stage drops SemanticsAPI labels on this build.
  Use a RENDERING stage for the renderer and a separate PHYSICS stage for ovphysx, and mirror
  poses explicitly (the renderer never consumes physics poses by itself).
"""
from __future__ import annotations

import math
import re
import struct
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import ovrtx
import ovstage

# --------------------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------------------
PINNED = {"ovrtx": "0.4.1", "ovstage": "0.1.1.355824", "ovphysx": "0.5.11"}


def check_versions() -> dict[str, str]:
    """Return installed versions; raise if they differ from the Set A pins."""
    import ovphysx

    found = {"ovrtx": ovrtx.__version__, "ovstage": ovstage.__version__, "ovphysx": ovphysx.__version__}
    bad = {k: (found[k], v) for k, v in PINNED.items() if not found[k].startswith(v)}
    if bad:
        raise RuntimeError(f"Omniverse library versions differ from Set A pins: {bad}")
    return found


# --------------------------------------------------------------------------------------
# Transforms (USD row-vector convention, as consumed by omni:xform)
# --------------------------------------------------------------------------------------
def usd_matrix(rotation: np.ndarray, translation, scale=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Build a 4x4 USD matrix (row-vector convention) from a column-vector rotation ``R``,
    a translation and a per-axis scale.  Points transform as ``p' = p @ M`` so the upper-left
    block is ``(S @ R)^T`` and the translation sits in the last row.  Verified on this build
    by the camera-tilt probe (a +8 deg tilt about world X moves the pack DOWN in the image).
    """
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    S = np.diag(np.asarray(scale, dtype=np.float64))
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = (R @ S).T
    M[3, :3] = np.asarray(translation, dtype=np.float64)
    return M


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def quat_to_rot(q_xyzw) -> np.ndarray:
    """Column-vector rotation matrix from an (x, y, z, w) quaternion (ovphysx pose layout)."""
    x, y, z, w = (float(v) for v in q_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------------------
# Semantic id map
# --------------------------------------------------------------------------------------
def decode_idmap(buf: bytes) -> dict[int, str]:
    """Decode ovrtx's SemanticIdMap buffer into {semantic_id: label}.

    Layout observed on this build: N entries of (id u32, pad u32 x3, label_len u32,
    label_off u32) followed by UTF-8 label bytes and a trailing u32 entry count.  A 12-byte
    layout is tried second.  Labels look like ``"class: pill_ok_c03; "`` -> use ``parse_label``.
    """
    if len(buf) < 4:
        return {}
    count = struct.unpack_from("<I", buf, len(buf) - 4)[0]
    for esize, fmt, idx in ((24, "<IIIIII", (0, 4, 5)), (12, "<III", (0, 1, 2))):
        out: dict[int, str] = {}
        ok = True
        for k in range(count):
            if k * esize + esize > len(buf):
                ok = False
                break
            vals = struct.unpack_from(fmt, buf, k * esize)
            sid, ln, off = vals[idx[0]], vals[idx[1]], vals[idx[2]]
            if ln == 0 or ln > 512 or off + ln > len(buf):
                ok = False
                break
            s = buf[off : off + ln].decode("utf-8", "replace")
            if not s.isprintable():
                ok = False
                break
            out[sid] = s
        if ok and out:
            return out
    return {}


def parse_label(raw: str) -> str:
    """``"class: pill_ok_c03; "`` -> ``"pill_ok_c03"``."""
    s = raw.strip()
    if ":" in s:
        s = s.split(":", 1)[1]
    return s.strip().rstrip(";").strip()


# --------------------------------------------------------------------------------------
# Renderer / stage lifecycle
# --------------------------------------------------------------------------------------
@dataclass
class Frame:
    """CPU copies of the render vars captured from one step (handles already released)."""

    rgba: np.ndarray | None = None
    seg: np.ndarray | None = None
    labels: dict[int, str] = field(default_factory=dict)
    depth: np.ndarray | None = None   # DistanceToImagePlaneSD, metres along the camera axis (H, W)

    @property
    def rgb(self) -> np.ndarray:
        assert self.rgba is not None
        return self.rgba[..., :3]


def _fetch(frame, suffix: str) -> np.ndarray:
    keys = [k for k in frame.render_vars.keys() if k.endswith(suffix)]
    if not keys:
        raise KeyError(f"render var {suffix!r} not among {list(frame.render_vars.keys())}")
    var = frame.render_vars[keys[0]].map(device=ovrtx.Device.CPU)
    try:
        return np.from_dlpack(var).copy()
    finally:
        var.unmap()
        del var


@contextmanager
def StepOutputs(renderer, product, ordinal: int, delta_time: float):
    """Step the renderer and guarantee that every output handle is released afterwards.

    ``product`` is one render product path or an iterable of them (both products of a dual-camera
    view come from ONE step, so the two panes show the same instant of simulation)."""
    wanted = {product} if isinstance(product, str) else set(product)
    products = renderer.step(render_products=wanted, delta_time=delta_time, ordinal=ordinal)
    try:
        yield products
    finally:
        # Two ways a caller leaks: it never touches ``prod.frames`` (ovrtx then reports "Leaking
        # step result outputs"), or it keeps the ``with ... as p`` name alive past the block - the
        # binding outlives the statement.  Drain and empty the mapping here so neither can happen.
        # Nothing in this block may raise: it runs in a finally, so an exception here would both
        # abort the drain and replace whatever the caller was already failing with.
        try:
            items = list(products.items())
        except Exception:  # noqa: BLE001
            items = []
        for _name, prod in items:
            try:
                for frame in prod.frames:
                    del frame
            except Exception:  # noqa: BLE001
                pass
            del prod
        del items
        try:
            products.clear()
        except Exception:  # noqa: BLE001
            pass
        del products


def render_frame(
    renderer,
    product: str,
    ordinal: int,
    *,
    steps: int = 1,
    delta_time: float = 1.0 / 60.0,
    want_seg: bool = True,
    want_depth: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> Frame:
    """Step ``steps`` times at ``ordinal`` (temporal warm-up) and return CPU copies of the last
    delivered frame.  Raises if no frame was delivered."""
    out = Frame()
    delivered = False
    for i in range(steps):
        with StepOutputs(renderer, product, ordinal, delta_time) as products:
            for _name, prod in products.items():
                for frame in prod.frames:
                    delivered = True
                    if i == steps - 1:
                        out.rgba = _fetch(frame, "LdrColor")
                        if want_seg:
                            seg = _fetch(frame, "SemanticSegmentation")
                            if width and height and seg.size == width * height:
                                seg = seg.reshape(height, width)
                            elif seg.ndim == 3:
                                seg = seg[..., 0]
                            out.seg = seg
                            out.labels = decode_idmap(_fetch(frame, "SemanticIdMap").tobytes())
                        if want_depth:
                            d = _fetch(frame, "DistanceToImagePlaneSD")
                            out.depth = d.reshape(height, width) if (width and height and d.size == width * height) else (d[..., 0] if d.ndim == 3 else d)
                    del frame
                del prod
    if not delivered or out.rgba is None:
        raise RuntimeError("renderer.step delivered no frame")
    return out


class Scene:
    """Owns one renderer + one attached ovstage stage and a monotonically increasing ordinal."""

    def __init__(self, name: str, config: ovrtx.RendererConfig | None = None):
        self.renderer = ovrtx.Renderer(config) if config is not None else ovrtx.Renderer()
        self.stage = ovstage.Stage(name)
        self.renderer.attach_ovstage(self.stage)
        self.ordinal = 0
        self.paths = None
        self._queries: dict[tuple[str, ...], tuple[int, object]] = {}
        self._roles: dict[str, int] = {}

    # -- loading -----------------------------------------------------------------------
    def load_usda(self, usda: str, domains=None) -> int:
        self.ordinal += 1
        if domains is None:
            ovstage.population.open_usd_from_string(self.stage, usda, ordinal=self.ordinal)
        else:
            ovstage.population.open_usd_from_string(self.stage, usda, ordinal=self.ordinal, domains=domains)
        self.stage.advance_write_floor(self.ordinal, ovstage.Scope.ALL).wait()
        self.paths = ovstage.PathDictionary(self.stage)
        return self.ordinal

    # -- writes ------------------------------------------------------------------------
    def _query(self, prim_paths: list[str]):
        key = tuple(prim_paths)
        if key not in self._queries:
            pl = self.paths.create_path_list_from_strings(list(prim_paths))
            self._queries[key] = (pl, self.stage.query_from_path_list(pl))
        return self._queries[key][1]

    def write(self, prim_paths: list[str], attr: str, values, *, semantic: int = 0, lanes: int | None = None, dtype=None, ordinal: int | None = None) -> int:
        """Write one scalar/vector value per prim at a new ordinal and seal it.  ``values`` has
        shape (N,) or (N, lanes).  Returns the ordinal to pass to ``render_frame``."""
        q = self._query(prim_paths)
        arr = np.ascontiguousarray(values, dtype=dtype) if dtype is not None else np.ascontiguousarray(values)
        if lanes:
            dl = ovstage.numpy_to_dldatatype(arr.dtype, lanes=lanes)
            tensor = ovstage.make_dltensor(arr, dtype=dl, shape=[len(prim_paths)], ndim=1)
        else:
            tensor = arr
        if ordinal is None:
            self.ordinal += 1
            ordinal = self.ordinal
        # Fabric fixes an attribute's role at population time (e.g. a light's inputs:color is
        # VECTOR, a shader's diffuse_color_constant is COLOR).  A mismatched role is rejected, so
        # remember the role that worked per attribute and, on a mismatch, retry with the role
        # the runtime reports for the existing column.
        semantic = self._roles.get(attr, semantic)
        try:
            self.stage.write_attribute(q, attr, ordinal=ordinal, tensors=tensor, is_array=False, semantic=semantic).wait()
        except ovstage.OvstageError as e:
            m = re.search(r"existing Fabric column is [A-Z_]+\((\d+)\)", str(e))
            if not m or "different semantic" not in str(e):
                raise
            semantic = int(m.group(1))
            self.stage.write_attribute(q, attr, ordinal=ordinal, tensors=tensor, is_array=False, semantic=semantic).wait()
        self._roles[attr] = semantic
        return ordinal

    def begin(self) -> int:
        """Open a new ordinal so several writes can be batched and sealed together."""
        self.ordinal += 1
        return self.ordinal

    def seal(self, ordinal: int | None = None) -> int:
        ordinal = self.ordinal if ordinal is None else ordinal
        self.stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
        return ordinal

    def write_xforms(self, prim_paths: list[str], matrices: np.ndarray, ordinal: int | None = None) -> int:
        m = np.ascontiguousarray(matrices, dtype=np.float64).reshape(len(prim_paths), 4, 4)
        return self.write(prim_paths, "omni:xform", m, lanes=16, dtype=np.float64, ordinal=ordinal)

    def write_float(self, prim_paths: list[str], attr: str, values, ordinal: int | None = None) -> int:
        return self.write(prim_paths, attr, np.asarray(values, dtype=np.float32).reshape(len(prim_paths)), ordinal=ordinal)

    def write_color(self, prim_paths: list[str], attr: str, rgb: np.ndarray, ordinal: int | None = None) -> int:
        c = np.asarray(rgb, dtype=np.float32).reshape(len(prim_paths), 3)
        return self.write(prim_paths, attr, c, semantic=ovstage.AttributeSemantic.COLOR, lanes=3, dtype=np.float32, ordinal=ordinal)

    def write_token(self, prim_paths: list[str], attr: str, token: str, ordinal: int | None = None) -> int:
        tid = self.paths.intern_token(token)
        ids = np.full(len(prim_paths), tid, dtype=np.uint64)
        return self.write(prim_paths, attr, ids, semantic=ovstage.AttributeSemantic.TOKEN_ID, ordinal=ordinal)

    # -- rendering ---------------------------------------------------------------------
    def render(self, product: str, *, steps: int = 1, want_seg: bool = True, want_depth: bool = False, width: int | None = None, height: int | None = None, ordinal: int | None = None) -> Frame:
        return render_frame(self.renderer, product, self.ordinal if ordinal is None else ordinal, steps=steps, want_seg=want_seg, want_depth=want_depth, width=width, height=height)

    # -- teardown ----------------------------------------------------------------------
    def close(self) -> None:
        for pl, q in self._queries.values():
            try:
                q.release().wait()
                self.paths.destroy_path_list(pl)
            except Exception:  # noqa: BLE001
                pass
        self._queries.clear()
        if self.paths is not None:
            try:
                self.paths.destroy()
            except Exception:  # noqa: BLE001
                pass
            self.paths = None
        try:
            self.renderer.detach_ovstage()
        finally:
            self.stage.destroy()
            self.renderer.destroy()

    def __enter__(self) -> "Scene":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------------------
# Small helpers shared by SDG and the twin
# --------------------------------------------------------------------------------------
def masks_by_label(seg: np.ndarray, labels: dict[int, str]) -> dict[str, np.ndarray]:
    """{clean_label: boolean mask} for every labelled id present in the segmentation."""
    out: dict[str, np.ndarray] = {}
    present = set(np.unique(seg).tolist())
    for sid, raw in labels.items():
        if sid in present:
            out[parse_label(raw)] = seg == sid
    return out


def mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """Approximate blackbody colour (Tanner Helland fit), normalised to [0, 1]."""
    t = max(1000.0, min(40000.0, kelvin)) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
        b = 255.0
    clamp = lambda v: max(0.0, min(255.0, v)) / 255.0  # noqa: E731
    return clamp(r), clamp(g), clamp(b)


def iter_chunks(n: int, size: int) -> Iterator[tuple[int, int]]:
    for s in range(0, n, size):
        yield s, min(n, s + size)
