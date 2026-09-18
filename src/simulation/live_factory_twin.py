"""Module 6 - closed-loop live factory digital twin (Mode 2).

Consumes the verified artifacts unchanged: assets/pack.usda (cell + pack), the FP16 TensorRT
engine with its calibrated thresholds, the Module 5 controller, and the Set A runtime helpers.

Cell composition (no prims are created or destroyed while running):
* the pack subtree of assets/pack.usda is instanced ``--pool`` times (Pack_00..) and parked
  upstream, off-camera; packs are recycled by pose writes (object pool);
* belt collider with a zero-friction physics material, a second "overview" camera over the
  nozzle/bin for visual evidence; the inspection camera keeps the asset's geometry (d = 0);
* an outfeed past the belt end (invisible static colliders: 10 deg stainless ramp, stacking
  tote) where accepted packs leave the belt under gravity and settle by PhysX contact; a
  rolling FIFO keeps a few settled packs in the tote and recycles the oldest into the pool.
Two stages from one composed USDA (Gate 0.5 rule): PHYSICS-domain stage for ovphysx, RENDERING-
domain stage for ovrtx; physics poses are mirrored with write_xforms LAST in each ordinal.

Transport and handoff: ovphysx exposes no kinematic toggle, so packs are dynamic rigid bodies
whose velocity is re-driven to (v_belt, 0, 0) every physics step while on the belt.  At the
nozzle the controller's actuation enqueues a kick: the lateral constraint is released (the body
keeps its 1.6 m/s forward velocity) and the calibrated wrench acts at the side face for a few
steps.  A PhysX overlap query on the bin volume confirms the physical ejection and appends a
confirmation event to the audit trail (the controller's own record at actuation time reflects
the valve command).

Modes: --lockstep (the belt advances one physics step at a time; W3 must observe every tick and
verdicts are awaited at each trigger -> deterministic) or --realtime (wall-clock pacer, RTF ~1).

Visual monitoring (src/simulation/twin_hud.py) is opt-in and never in the control path: --gui
opens a live window, --record-video writes the dual-camera dashboard to an MP4.  Both add
visual-only cell furniture (bin, nozzle, inspection mast, floor) to the composed USDA - prims
with no physics schemas, all clear of the inspection illumination - and render the overview
product at the display cadence.  The triggered inspection exposure itself is reused, not
re-rendered, so the frames the detector sees are the frames on screen.

    uv run python src/simulation/live_factory_twin.py --demo --packs 30 --lockstep
    uv run python src/simulation/live_factory_twin.py --demo --packs 20 --lockstep --record-video
    uv run python src/simulation/live_factory_twin.py --demo --packs 12 --realtime --gui
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import ovrtx  # noqa: E402
import ovstage  # noqa: E402
from ovphysx import PhysX, SceneQueryGeometryType, SceneQueryMode, TensorType  # noqa: E402

from inference.controller import (  # noqa: E402
    AUDIT_PATH,
    ENGINE_PATH,
    IMGSZ,
    Actuator,
    Controller,
    DLPackFrameSource,
    Encoder,
    LineConfig,
    Policy,
)
from simulation.assets import blister_factory as bf  # noqa: E402
from simulation.ovx_runtime import Scene, StepOutputs, check_versions, quat_to_rot, usd_matrix  # noqa: E402

def _rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """USD rotateXYZ (X first, then Y, then Z) as a column-vector rotation matrix."""
    from simulation.ovx_runtime import rot_x, rot_y, rot_z

    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k])) * 2
    q = np.zeros(4)
    q[i], q[j], q[k], q[3] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s, (m[k, j] - m[j, k]) / s
    return q


def _slerp_rot(R0: np.ndarray, R1: np.ndarray, u: float) -> np.ndarray:
    q0, q1 = _rot_to_quat(R0), _rot_to_quat(R1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + (q1 - q0) * u
    else:
        th = math.acos(min(1.0, d))
        q = (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
    return quat_to_rot(q / (np.linalg.norm(q) or 1.0))


ASSET = PROJECT_ROOT / "assets" / "pack.usda"
REPORT_DIR = PROJECT_ROOT / "reports" / "twin"
DEFAULT_VIDEO = REPORT_DIR / "digital_twin_demo.mp4"
PRODUCT_INSPECT, PRODUCT_OVERVIEW = "/Render/Camera", "/Render/Overview"
# pitch-aware inspection window, half width in pixels (see _apply_pack_roi); the dashboard
# draws the same rule, so both read it from here
ROI_HALF_PX = int(round((bf.PACK_L_M / 2 + 0.010) * (bf.IMAGE_W / bf.FOV_ALONG_M)))


# --------------------------------------------------------------------------------------
# Twin geometry (metres) - Decision 5 line, plus what the twin adds
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TwinConfig:
    physics_hz: int = 240
    x_spawn: float = -0.10           # packs enter the belt here (outside the camera FOV of +/-0.1 m)
    x_camera: float = 0.0            # trigger plane (d = 0)
    x_exit: float = 0.42             # past the nozzle: deviation bookkeeping point (v1 recycled nominal packs here)
    belt_half_width: float = 0.15
    bin_center: tuple = (0.50, 0.32, -0.20)
    bin_half: tuple = (0.25, 0.16, 0.18)
    kick_force_n: float = 1.2        # 5 steps at 240 Hz -> J = 25 mN.s -> ~2.1 m/s lateral on a 12 g pack
    kick_steps: int = 5
    z_lost: float = -0.45            # a kicked pack below this never reached the bin volume
    capture_steps: int = 3           # temporal warm-up steps per triggered frame (same as SDG)
    nominal_dev_limit: float = 0.005
    # outfeed (nominal product): past the belt collider a pack is a free body; it slides down
    # the outfeed ramp into a stacking tote.  A rolling FIFO keeps ``outfeed_capacity`` settled
    # packs; the oldest (bottom of the stack) recycles into the pool when the next one settles.
    x_belt_end: float = 0.60         # end of the belt collider (asset: Belt cube x = +/-0.6)
    outfeed_capacity: int = 5        # settled packs kept in the tote
    outfeed_settle_steps: int = 24   # consecutive still steps (0.1 s) before a pack counts as settled
    outfeed_v_still: float = 0.05    # m/s
    outfeed_w_still: float = 1.0     # rad/s
    outfeed_dwell_max_s: float = 2.0 # a pack still creeping after this counts as settled (never stalls the run)


@dataclass(frozen=True)
class VizConfig:
    """Visual monitoring (--record-video / --gui).  Nothing here is read by the closed loop."""
    fps: int = 30                    # container rate of the recording and of the live window
    speed: float = 0.25              # simulated seconds per playback second (0.25 = 4x slow motion)
    steps: int = 2                   # renderer steps per display frame (temporal convergence)
    park_sink_z: float = -5.0        # parked pool packs are drawn below the cell, not on the belt
    flash_s: float = 0.10            # how long a bin confirmation is flagged at the bin
    crf: int = 18


# --------------------------------------------------------------------------------------
# Cell composition from the immutable asset
# --------------------------------------------------------------------------------------
# Off-camera, non-overlapping park slots on the infeed end of the belt.  The first ten are the
# original layout (a --pool 10 cell composes exactly as before); rows at y = +/-0.055 (10 mm from
# the neighbouring rows, 45 mm wide packs) and the lane itself at x <= -0.25 (60 mm clear of the
# spawn point at -0.10) extend the pool to 24 for the outfeed FIFO.
PARK_SLOTS = ([(x, y) for y in (-0.11, 0.11) for x in (-0.55, -0.45, -0.35, -0.25, -0.15)]
              + [(x, y) for y in (-0.055, 0.055) for x in (-0.55, -0.45, -0.35, -0.25, -0.15)]
              + [(x, 0.0) for x in (-0.55, -0.45, -0.35, -0.25)])
BELT_OCCUPANCY = 6            # packs between the spawn point (-0.10) and the belt end (0.60) at 120 mm pitch
REJECT_TRANSIT = 2            # kicked packs not yet confirmed (parked ~0.1 s after the kick)
MIN_POOL = BELT_OCCUPANCY + REJECT_TRANSIT + 2   # + at least one outfeed body + one spare: below this the pool can exhaust

# Cell overview: a 3/4 view from the operator side that holds the whole line - infeed, inspection
# station, nozzle at d = 300 mm and the bin - in one frame.  The numbers are the single source of
# truth for both the USDA prim and the HUD's world->pixel projection (twin_hud.OverviewCam).
# Reframed for the outfeed (audit 2026-09-17, twin_hud.OverviewCam projection of the cell's
# extreme points): from (0.36, -0.70, 0.86) f 0.35 the outfeed tote (x 0.80..1.04) fell 150 px
# past the right edge.  0.12 m downstream, 0.06 m back and up, focal 0.33 keeps the infeed mark
# (u 5..120), the mast top at the frame edge (v 5..10), both totes and the outfeed label (u <= 1176) inside 1280x720;
# a pack at d = 0 is 89 px long (was 101).
OV_POS, OV_RX, OV_FOCAL, OV_AP_H, OV_AP_V = (0.48, -0.76, 0.92), 44.0, 0.33, 0.36, 0.2025
OVERVIEW_CAMERA = f'''    def Camera "OverviewCamera"
    {{
        float focalLength = {OV_FOCAL}
        float horizontalAperture = {OV_AP_H}
        float verticalAperture = {OV_AP_V}
        float2 clippingRange = (0.01, 100)
        token projection = "perspective"
        double3 xformOp:translate = ({OV_POS[0]}, {OV_POS[1]}, {OV_POS[2]})
        double3 xformOp:rotateXYZ = ({OV_RX}, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}
'''
OVERVIEW_PRODUCT = '''    def RenderProduct "Overview"
    {
        rel camera = </World/OverviewCamera>
        rel orderedVars = </Render/Overview/LdrColor>
        int2 resolution = (1280, 720)
        def RenderVar "LdrColor"
        {
            string sourceName = "LdrColor"
        }
    }
'''
BELT_PHYSICS_MATERIAL = '''        def Material "BeltPhysics" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {
            float physics:dynamicFriction = 0
            float physics:staticFriction = 0
            float physics:restitution = 0
        }
'''


def _phys_material(name: str, static: float, dynamic: float, restitution: float) -> str:
    return (f'        def Material "{name}" (\n            prepend apiSchemas = ["PhysicsMaterialAPI"]\n        )\n        {{\n'
            f'            float physics:dynamicFriction = {dynamic}\n            float physics:staticFriction = {static}\n'
            f'            float physics:restitution = {restitution}\n        }}\n')


# stainless ramp (PVC/foil on 304: low friction) and the tote (packs on packs / powder floor)
OUTFEED_PHYSICS_MATERIALS = _phys_material("RampPhysics", 0.30, 0.25, 0.05) + _phys_material("TotePhysics", 0.60, 0.50, 0.05)


def _viz_material(name: str, diffuse: str, metallic: float, roughness: float) -> str:
    return f'''        def Material "{name}"
        {{
            token outputs:mdl:surface.connect = </World/Looks/{name}/Shader.outputs:out>
            def Shader "Shader"
            {{
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = {diffuse}
                float inputs:metallic_constant = {metallic}
                float inputs:reflection_roughness_constant = {roughness}
                token outputs:out
            }}
        }}
'''


VIZ_MATERIALS = (_viz_material("VizSteel", "(0.30, 0.31, 0.33)", 0.6, 0.35)
                 + _viz_material("VizDark", "(0.07, 0.075, 0.08)", 0.0, 0.9)
                 + _viz_material("VizAccent", "(0.72, 0.36, 0.06)", 0.0, 0.5)
                 + _viz_material("VizAlu", "(0.70, 0.71, 0.73)", 0.9, 0.42)        # anodised extrusion
                 + _viz_material("VizBlack", "(0.03, 0.03, 0.03)", 0.0, 0.7)        # T-slot grooves, hose
                 + _viz_material("VizWear", "(0.92, 0.92, 0.90)", 0.0, 0.5)         # UHMW wear strips
                 + _viz_material("VizBrass", "(0.78, 0.60, 0.25)", 1.0, 0.35)
                 + _viz_material("VizStainless", "(0.74, 0.75, 0.77)", 1.0, 0.30)   # brushed 304, brighter base so the slide reads against the floor
                 + _viz_material("VizPowder", "(0.78, 0.79, 0.81)", 0.0, 0.55)      # RAL 7035 powder coat: dielectric, lit by the dome from any angle
                 + _viz_material("VizBrushed", "(0.72, 0.73, 0.75)", 0.85, 0.50)    # brushed 304 (tote): rough enough to hold the dome light
                 + _viz_material("VizLED", "(0.9, 0.9, 0.9)", 0.0, 0.4).replace("                token outputs:out", "                bool inputs:enable_emission = 1\n                color3f inputs:emissive_color = (1, 0.97, 0.92)\n                float inputs:emissive_intensity = 60\n                token outputs:out")
                 + _viz_material("VizYellow", "(0.95, 0.75, 0.05)", 0.0, 0.6)
                 + """        def Material "VizAir"
        {
            token outputs:mdl:surface.connect = </World/Looks/VizAir/Shader.outputs:out>
            def Shader "Shader"
            {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = (0.92, 0.95, 1)
                bool inputs:enable_opacity = 1
                float inputs:opacity_constant = 0.28
                float inputs:reflection_roughness_constant = 0.9
                token outputs:out
            }
        }
""")


def _viz_prop(kind: str, name: str, material: str, translate, scale, rotate=None, extra: str = "") -> str:
    ops = ['double3 xformOp:translate = (%g, %g, %g)' % tuple(translate)]
    order = ['"xformOp:translate"']
    if rotate is not None:
        ops.append('double3 xformOp:rotateXYZ = (%g, %g, %g)' % tuple(rotate))
        order.append('"xformOp:rotateXYZ"')
    ops.append('float3 xformOp:scale = (%g, %g, %g)' % tuple(scale))
    order.append('"xformOp:scale"')
    body = "\n".join(f"            {o}" for o in ops)
    return (f'        def {kind} "{name}" (\n'
            f'            prepend apiSchemas = ["MaterialBindingAPI"]\n'
            f'        )\n'
            f'        {{\n'
            f'            rel material:binding = </World/Looks/{material}>\n'
            f'{extra}{body}\n'
            f'            uniform token[] xformOpOrder = [{", ".join(order)}]\n'
            f'        }}\n')


# Visual-only cell furniture (no physics schemas -> ovphysx never sees these prims, so the
# closed loop is untouched).  Placement rules that matter, both learned the hard way:
#  * NOTHING spans the belt near the inspection zone.  The KeyLight travels (-0.28, +0.42, -0.86),
#    so any structure on the -Y side casts its shadow onto the belt; the inspection mast stands
#    on the +Y side and only meets the arm the asset hangs from the camera housing.
#  * NOTHING sits in the ejection lane: rejected packs cross y = +0.165 at x = 0.26..0.41, so the
#    far-side extrusion, wear strip and slots stop at x = 0.24 and the chute mouth begins there.
_CUBE = "            double size = 1\n"
_CYL = "            double radius = 1\n            double height = 1\n"
_HIDDEN = '            token visibility = "invisible"\n'
AIR_BURST_PATH = "/World/Props/AirBurst"


def _hazard_stripes(x0: float, x1: float, y0: float, y1: float, z: float, w: float = 0.04, seg: float = 0.08, prefix: str = "Stripe") -> str:
    """Yellow/black safety perimeter (reject zone, outfeed zone): flat segments on the floor."""
    out, k = [], 0

    def run(ax, ay, bx, by, along_x):
        nonlocal k
        n = max(1, int(round((abs(bx - ax) if along_x else abs(by - ay)) / seg)))
        for i in range(n):
            t = (i + 0.5) / n
            cx = ax + (bx - ax) * t if along_x else ax
            cy = ay + (by - ay) * t if not along_x else ay
            sx = abs(bx - ax) / n if along_x else w
            sy = abs(by - ay) / n if not along_x else w
            out.append(_viz_prop("Cube", f"{prefix}_{k:02d}", "VizYellow" if i % 2 == 0 else "VizBlack", (cx, cy, z), (sx, sy, 0.002), extra=_CUBE))
            k += 1

    run(x0, y0 + w / 2, x1, y0 + w / 2, True)
    run(x0, y1 - w / 2, x1, y1 - w / 2, True)
    run(x0 + w / 2, y0 + w, x0 + w / 2, y1 - w, False)
    run(x1 - w / 2, y0 + w, x1 - w / 2, y1 - w, False)
    return "".join(out)


def _mesh_prop(name: str, material: str, geom: tuple, translate, rotate=None) -> str:
    """CAD-grade prop: a generated mesh (chamfered box, lathe, slab) with no physics schema."""
    return bf._mesh(name, None, material, geom, translate=translate, rotate_xyz=rotate, indent=8) + "\n"


def _light_prop(kind: str, name: str, intensity: float, radius: float, translate, rotate, color="(1, 0.97, 0.92)") -> str:
    return (f'        def {kind} "{name}"\n        {{\n            float inputs:intensity = {intensity:g}\n            float inputs:radius = {radius:g}\n'
            f'            color3f inputs:color = {color}\n            double3 xformOp:translate = ({translate[0]:g}, {translate[1]:g}, {translate[2]:g})\n'
            f'            double3 xformOp:rotateXYZ = ({rotate[0]:g}, {rotate[1]:g}, {rotate[2]:g})\n'
            f'            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]\n        }}\n')


def _sensor_head_geometry(radius: float, length: float, chamfer: float) -> tuple:
    """Cylindrical photoelectric sensor body with chamfered ends, revolved about Z with normals."""
    prof = [(0.0, 0.0), (radius - chamfer, 0.0), (radius, chamfer), (radius, length - chamfer), (radius - chamfer, length), (0.0, length)]
    return bf._lathe(prof, 32)


# Reject chute: the kicked pack keeps its belt velocity, so the slide runs DOWNSTREAM beside the belt
# (x from the landing zone to the belt end) and drops into a tote past the end roller.  A slide that
# descends away from the overview camera shows only its back and edges (probed 2026-09-17); this
# one tilts toward the camera like the belt does.
CHUTE_X0, CHUTE_X1 = 0.26, 0.60                                   # upstream lip -> lower edge (along x)
CHUTE_YA, CHUTE_YB = 0.165, 0.42                                  # near edge at the belt, far edge
CHUTE_Z0, CHUTE_Z1 = -0.006, -0.050                                # shallow tray: stays above the belt horizon from the overview
TOTE_RIM_Z, TOTE_FLOOR_Z = -0.070, -0.240                          # tote on a stand at working height, rim just under the tray end
CHUTE_LEN = math.hypot(CHUTE_X1 - CHUTE_X0, CHUTE_Z1 - CHUTE_Z0)
CHUTE_DEG = math.degrees(math.atan2(CHUTE_Z0 - CHUTE_Z1, CHUTE_X1 - CHUTE_X0))   # rotateY: local +x descends along world +x
CHUTE_C = ((CHUTE_X0 + CHUTE_X1) / 2, (CHUTE_YA + CHUTE_YB) / 2, (CHUTE_Z0 + CHUTE_Z1) / 2)
TOTE_C = (0.72, 0.29)                                             # tote centre (x, y), 0.28 x 0.28 on a stand


def chute_z(x: float) -> float:
    return CHUTE_Z0 - (x - CHUTE_X0) * (CHUTE_Z0 - CHUTE_Z1) / (CHUTE_X1 - CHUTE_X0)


BEAM_A, BEAM_B = (0.0, -0.200, 0.012), (0.0, 0.198, 0.012)      # emitter -> reflector, HUD beam indicator
N_REST, N_SLIDE = 8, 4
SLIDE_S = 0.30                                                   # render-only slide down the chute after confirmation


def _slope_local(deg: float, dx: float, dz: float) -> tuple[float, float, float]:
    """Offset in a rotateY(deg) frame (dx down the slope, dz normal to the slide) -> world."""
    t = math.radians(deg)
    return (dx * math.cos(t) + dz * math.sin(t), 0.0, -dx * math.sin(t) + dz * math.cos(t))


def _chute_local(dx: float, dz: float) -> tuple[float, float, float]:
    return _slope_local(CHUTE_DEG, dx, dz)


def _out_local(dx: float, dz: float) -> tuple[float, float, float]:
    return _slope_local(OUT_DEG, dx, dz)


def _at(base, off) -> tuple[float, float, float]:
    return (base[0] + off[0], base[1] + off[1], base[2] + off[2])


# Outfeed (nominal product), downstream of the belt end.  Geometry from the physics probe of
# 2026-09-17 (probe_outfeed.py): at 1.6 m/s a pack leaving the collider end is ballistic for
# ~11 cm, meets a 10 deg ramp at x ~ 0.73 and slides in contact (tilt locked to the ramp) to the
# lower edge, then drops into the tote and stops against its far wall within 0.5 s; six
# consecutive passes stack flat (6 mm pile) and a bottom-pack recycle re-settles the pile in
# < 0.3 s.  Steeper ramps (12-16 deg) are overflown - the pack only touches at the ramp end.
OUT_X0, OUT_Z0 = 0.62, -0.006                                     # upper lip (top surface), 5 mm past the roller nose
OUT_DEG, OUT_LEN, OUT_W = 10.0, 0.20, 0.26
OUT_X1 = OUT_X0 + OUT_LEN * math.cos(math.radians(OUT_DEG))
OUT_Z1 = OUT_Z0 - OUT_LEN * math.sin(math.radians(OUT_DEG))
OUT_C = ((OUT_X0 + OUT_X1) / 2, 0.0, (OUT_Z0 + OUT_Z1) / 2)
OUT_TOTE_X0 = OUT_X1 - 0.015                                      # tote inner faces (the ramp end overhangs the rim)
OUT_TOTE_X1 = OUT_TOTE_X0 + 0.24
OUT_TOTE_HALF_Y = 0.125                                           # 250 mm wide: rim and tape stay clear of the reject tote's near wall (y = 0.147)
OUT_TOTE_RIM_Z, OUT_TOTE_FLOOR_Z = -0.10, -0.22                    # 120 mm walls on a stand: the pile is visible from the overview
OUT_COL_T = 0.03                                                  # collider thickness: a 1.3 mm pack at 3 m/s must not tunnel


def _collider(name: str, translate, scale, rotate=None, material: str = "TotePhysics") -> str:
    """Invisible static box collider (PhysicsCollisionAPI, physics material) - the physics side
    of the outfeed.  Verified on this build: invisible + rotateXYZ static cubes collide."""
    return _viz_prop("Cube", name, "Belt", translate, scale, rotate=rotate,
                     extra=f"            rel material:binding:physics = </World/Looks/{material}>\n" + _CUBE + _HIDDEN).replace(
        'prepend apiSchemas = ["MaterialBindingAPI"]', 'prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI"]')


def _outfeed_colliders() -> str:
    """Physics proxies of the outfeed: ramp slab, tote floor and four walls (30 mm boxes whose
    inner faces are the visual surfaces).  Always in the cell, dashboard or not."""
    out = _collider("OutfeedRampCol", _at(OUT_C, _out_local(0.0, -OUT_COL_T / 2)), (OUT_LEN, OUT_W, OUT_COL_T), (0, OUT_DEG, 0), "RampPhysics")
    wall_h = OUT_TOTE_RIM_Z - OUT_TOTE_FLOOR_Z
    wz = (OUT_TOTE_RIM_Z + OUT_TOTE_FLOOR_Z) / 2
    tx, tl, tw = (OUT_TOTE_X0 + OUT_TOTE_X1) / 2, OUT_TOTE_X1 - OUT_TOTE_X0, 2 * OUT_TOTE_HALF_Y
    out += _collider("OutfeedFloorCol", (tx, 0.0, OUT_TOTE_FLOOR_Z - OUT_COL_T / 2), (tl + 2 * OUT_COL_T, tw + 2 * OUT_COL_T, OUT_COL_T))
    out += _collider("OutfeedWallUpCol", (OUT_TOTE_X0 - OUT_COL_T / 2, 0.0, wz), (OUT_COL_T, tw + 2 * OUT_COL_T, wall_h))
    out += _collider("OutfeedWallDownCol", (OUT_TOTE_X1 + OUT_COL_T / 2, 0.0, wz), (OUT_COL_T, tw + 2 * OUT_COL_T, wall_h))
    out += _collider("OutfeedWallNearCol", (tx, -OUT_TOTE_HALF_Y - OUT_COL_T / 2, wz), (tl, OUT_COL_T, wall_h))
    out += _collider("OutfeedWallFarCol", (tx, OUT_TOTE_HALF_Y + OUT_COL_T / 2, wz), (tl, OUT_COL_T, wall_h))
    return out.replace("\n        ", "\n    ").replace("        def Cube", "    def Cube", 1)   # authored at indent 8 by _viz_prop; these live directly under /World


OUTFEED_COLLIDERS = _outfeed_colliders()


def _tube_geometry(radius: float, length: float, chamfer: float) -> tuple:
    """Round tube/rod from z = 0 to z = length, chamfered ends, per-vertex normals (lathe)."""
    return _sensor_head_geometry(radius, length, chamfer)


def _ball_geometry(radius: float, samples: int = 10) -> tuple:
    """Sphere (lathe with per-vertex normals; the Sphere prim tessellates coarsely on this build)."""
    prof = [(radius * math.sin(math.pi * k / samples), -radius * math.cos(math.pi * k / samples)) for k in range(samples + 1)]
    prof[0], prof[-1] = (0.0, -radius), (0.0, radius)
    return bf._lathe(prof, 24)


def _tube_rot(d) -> tuple[float, float, float]:
    """rotateXYZ that points a +Z tube along world direction ``d`` (checked against _rot_xyz)."""
    d = np.asarray(d, dtype=np.float64)
    d = d / np.linalg.norm(d)
    rx = math.degrees(math.acos(max(-1.0, min(1.0, float(d[2])))))
    rz = math.degrees(math.atan2(float(d[0]), -float(d[1]))) if abs(d[2]) < 0.999999 else 0.0
    assert np.allclose(_rot_xyz(rx, 0.0, rz) @ np.array([0.0, 0.0, 1.0]), d, atol=1e-6)
    return (rx, 0.0, rz)


def _tube(name: str, material: str, radius: float, a, b, chamfer: float = 0.0015) -> str:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return _mesh_prop(name, material, _tube_geometry(radius, float(np.linalg.norm(b - a)), chamfer), tuple(a), _tube_rot(b - a))


# Work-light assembly: anchored to the downstream end of ExtrusionFar (the 40 mm profile stops at
# x = 0.24, clear of the ejection lane), which stands on the conveyor frame legs - nothing floats.
LAMP_ANCHOR = (0.21, 0.176)                                       # on the extrusion's top face (z = 0.04)
LAMP_TOP_Z = 0.36                                                 # elbow height
LAMP_PIVOT = (0.50, 0.31, 0.27)                                   # arm end / fixture swivel over the chute end
LAMP_TILT_DEG = -23.0                                             # rotateY: the fixture's -Z (emission) -> (+0.39, 0, -0.92), into the tote
WORK_LIGHT_PATH = "/World/Props/WorkLight"
WORK_LIGHT_INTENSITY = 1500.0


def _chute_assembly() -> str:
    M, rot = _mesh_prop, (0, CHUTE_DEG, 0)
    box = bf.chamfered_box_geometry
    W = CHUTE_YB - CHUTE_YA
    out = ""
    # slide plate (brushed stainless), safety-yellow chamfered flanges (18 mm), powder-coated frame
    out += M("ChutePlate", "VizStainless", bf.rounded_slab_geometry(CHUTE_LEN, W, 0.004, 0.004, 0.001), _at(CHUTE_C, _chute_local(0.0, -0.004)), rot)
    out += M("ChuteFlangeFar", "VizYellow", box(CHUTE_LEN, 0.006, 0.018, 0.0012), _at((CHUTE_C[0], CHUTE_YB - 0.003, CHUTE_C[2]), _chute_local(0.0, 0.009)), rot)
    # near flange only on the lower half: the upper half is the open mouth the belt feeds
    out += M("ChuteFlangeNear", "VizYellow", box(CHUTE_LEN / 2, 0.006, 0.018, 0.0012), _at((CHUTE_C[0], CHUTE_YA + 0.003, CHUTE_C[2]), _chute_local(CHUTE_LEN / 4, 0.009)), rot)
    out += M("ChuteLipUp", "VizYellow", box(0.006, W, 0.03, 0.0012), _at(CHUTE_C, _chute_local(-CHUTE_LEN / 2 + 0.003, 0.013)), rot)
    for name, y in (("ChuteRailNear", CHUTE_YA + 0.015), ("ChuteRailFar", CHUTE_YB - 0.015)):
        out += M(name, "VizPowder", box(CHUTE_LEN, 0.03, 0.03, 0.003), _at((CHUTE_C[0], y, CHUTE_C[2]), _chute_local(0.0, -0.021)), rot)
    out += M("ChuteCross", "VizPowder", box(0.03, W + 0.03, 0.03, 0.003), _at(CHUTE_C, _chute_local(CHUTE_LEN / 2 - 0.03, -0.021)), rot)
    out += M("ChuteApron", "VizStainless", box(0.20, 0.03, 0.004, 0.001), (0.36, 0.163, -0.004))
    for name, y in (("ChuteLegNear", CHUTE_YA + 0.015), ("ChuteLegFar", CHUTE_YB - 0.015)):
        out += M(name, "VizPowder", box(0.03, 0.03, 0.47, 0.003), (CHUTE_X1 - 0.015, y, -0.325))
    out += _work_light_assembly()
    # tote past the belt end: powder-coated body, stainless rims, tape on the near rim
    tx, ty = TOTE_C
    wall_h = TOTE_RIM_Z - TOTE_FLOOR_Z
    wz = (TOTE_RIM_Z + TOTE_FLOOR_Z) / 2
    out += M("ToteStand", "VizDark", box(0.24, 0.24, TOTE_FLOOR_Z + 0.565, 0.003), (tx, ty, (TOTE_FLOOR_Z - 0.565) / 2))
    out += M("ToteFloor", "VizPowder", box(0.28, 0.28, 0.006, 0.001), (tx, ty, TOTE_FLOOR_Z - 0.003))
    for name, t, sz in (("ToteWallUp", (tx - 0.14, ty, wz), (0.006, 0.28, wall_h)), ("ToteWallDown", (tx + 0.14, ty, wz), (0.006, 0.28, wall_h)),
                        ("ToteWallNear", (tx, ty - 0.14, wz), (0.28, 0.006, wall_h)), ("ToteWallFar", (tx, ty + 0.14, wz), (0.28, 0.006, wall_h))):
        out += M(name, "VizPowder", box(*sz, 0.0015), t)
    for name, t, sz in (("ToteRimUp", (tx - 0.14, ty, TOTE_RIM_Z + 0.003), (0.012, 0.29, 0.006)), ("ToteRimDown", (tx + 0.14, ty, TOTE_RIM_Z + 0.003), (0.012, 0.29, 0.006)),
                        ("ToteRimFar", (tx, ty + 0.14, TOTE_RIM_Z + 0.003), (0.29, 0.012, 0.006))):
        out += M(name, "VizStainless", box(*sz, 0.0012), t)
    out += "".join(_viz_prop("Cube", f"RimTape_{i}", "VizYellow" if i % 2 == 0 else "VizBlack", (tx - 0.14 + 0.02 + i * 0.04, ty - 0.14, TOTE_RIM_Z + 0.003), (0.04, 0.012, 0.006), extra=_CUBE) for i in range(7))
    return out


def _work_light_assembly() -> str:
    """Industrial LED work light on a cantilever: T-slot clamp plate and boss on the extrusion end,
    25 mm anodised mast, cast elbow, 20 mm cross-arm angled down over the chute end, swivel
    knuckle, stem and a slim powder-coated LED bar with its lens strip; two baffle skirts on the
    upstream (inspection) and belt sides so the DiskLight only sees the chute end and the tote."""
    M, box = _mesh_prop, bf.chamfered_box_geometry
    ax, ay = LAMP_ANCHOR
    top = np.array([ax, ay, LAMP_TOP_Z])
    pivot = np.array(LAMP_PIVOT)
    R = _rot_xyz(0.0, LAMP_TILT_DEG, 0.0)
    loc = lambda dx, dy, dz: tuple(pivot + R @ np.array([dx, dy, dz]))  # noqa: E731 - fixture-local offsets
    out = ""
    out += M("LampClamp", "VizPowder", box(0.06, 0.05, 0.012, 0.0015), (ax, ay, 0.046))
    out += M("LampBoss", "VizPowder", _tube_geometry(0.018, 0.024, 0.002), (ax, ay, 0.052))
    out += M("LampBolt0", "VizBlack", _tube_geometry(0.004, 0.004, 0.0008), (ax - 0.021, ay, 0.052))
    out += M("LampBolt1", "VizBlack", _tube_geometry(0.004, 0.004, 0.0008), (ax + 0.021, ay, 0.052))
    out += _tube("LampMast", "VizAlu", 0.0125, (ax, ay, 0.074), top)
    out += M("LampElbow", "VizPowder", _ball_geometry(0.021), tuple(top))
    out += _tube("LampArm", "VizAlu", 0.010, top, pivot)
    out += M("LampKnuckle", "VizPowder", _ball_geometry(0.017), tuple(pivot))
    out += M("LampStem", "VizAlu", _tube_geometry(0.007, 0.030, 0.001), loc(0.0, 0.0, -0.034), (0, LAMP_TILT_DEG, 0))
    out += M("LampBar", "VizPowder", box(0.24, 0.05, 0.024, 0.002), loc(0.0, 0.0, -0.046), (0, LAMP_TILT_DEG, 0))
    out += M("LampLens", "VizLED", box(0.20, 0.016, 0.004, 0.0008), loc(0.0, 0.0, -0.059), (0, LAMP_TILT_DEG, 0))
    out += M("LampSkirtUp", "VizPowder", box(0.006, 0.05, 0.05, 0.001), loc(-0.123, 0.0, -0.071), (0, LAMP_TILT_DEG, 0))   # louver toward the inspection station
    out += M("LampSkirtBelt", "VizPowder", box(0.24, 0.006, 0.035, 0.001), loc(0.0, -0.028, -0.0635), (0, LAMP_TILT_DEG, 0))  # louver toward the belt
    out += _light_prop("DiskLight", "WorkLight", WORK_LIGHT_INTENSITY, 0.025, loc(0.0, 0.0, -0.062), (0, LAMP_TILT_DEG, 0))
    return out


def _outfeed_assembly() -> str:
    """Visual outfeed on top of the invisible colliders: brushed-stainless ramp plate (dead plate
    under the roller nose), 15 mm safety-yellow chamfered guide rails, powder-coated frame rails,
    cross member, brackets off the conveyor frame and legs to the floor; stainless stacking tote
    on a stand with rim lips and hazard tape on all four rims."""
    M, box, rot = _mesh_prop, bf.chamfered_box_geometry, (0, OUT_DEG, 0)
    out = ""
    out += M("OutfeedPlate", "VizStainless", bf.rounded_slab_geometry(OUT_LEN + 0.003, OUT_W, 0.004, 0.004, 0.001), _at(OUT_C, _out_local(-0.0015, -0.004)), rot)
    for name, y in (("OutfeedRailNear", -OUT_W / 2 + 0.003), ("OutfeedRailFar", OUT_W / 2 - 0.003)):
        out += M(name, "VizYellow", box(OUT_LEN, 0.006, 0.015, 0.0012), _at((OUT_C[0], y, OUT_C[2]), _out_local(0.0, 0.0075)), rot)
    for name, y in (("OutfeedFrameNear", -OUT_W / 2 + 0.015), ("OutfeedFrameFar", OUT_W / 2 - 0.015)):
        out += M(name, "VizPowder", box(OUT_LEN, 0.03, 0.03, 0.003), _at((OUT_C[0], y, OUT_C[2]), _out_local(0.0, -0.019)), rot)
    out += M("OutfeedCross", "VizPowder", box(0.03, OUT_W - 0.06, 0.03, 0.003), _at(OUT_C, _out_local(OUT_LEN / 2 - 0.06, -0.019)), rot)
    for name, y in (("OutfeedBracketNear", -0.10), ("OutfeedBracketFar", 0.10)):                 # off the conveyor frame's end face (x = 0.60)
        out += M(name, "VizPowder", box(0.07, 0.03, 0.02, 0.002), (0.63, y, -0.035))
    # legs 75 mm up the slope (clear of the tote rim): the frame rails' bottom face slopes with the
    # ramp (higher upstream), so the leg top sits 3 mm above that face at the leg's UPSTREAM edge
    # and is buried 3..8 mm inside the 30 mm rail across the footprint (never through its top)
    leg_x = OUT_X1 - 0.075
    pb = _at(OUT_C, _out_local(0.0, -0.034))                       # a point on the rails' bottom face
    leg_top = pb[2] - (leg_x - 0.015 - pb[0]) * math.tan(math.radians(OUT_DEG)) + 0.003
    for name, y in (("OutfeedLegNear", -OUT_W / 2 + 0.015), ("OutfeedLegFar", OUT_W / 2 - 0.015)):
        out += M(name, "VizPowder", box(0.03, 0.03, leg_top + 0.56, 0.003), (leg_x, y, (leg_top - 0.56) / 2))
    # tote: brushed stainless body, lips and tape on all four rims, stand to the floor
    tx, tl, tw = (OUT_TOTE_X0 + OUT_TOTE_X1) / 2, OUT_TOTE_X1 - OUT_TOTE_X0, 2 * OUT_TOTE_HALF_Y
    wall_h = OUT_TOTE_RIM_Z - OUT_TOTE_FLOOR_Z
    wz = (OUT_TOTE_RIM_Z + OUT_TOTE_FLOOR_Z) / 2
    out += M("OutToteStand", "VizDark", box(tl - 0.06, tw - 0.06, OUT_TOTE_FLOOR_Z - 0.006 + 0.56, 0.003), (tx, 0.0, (OUT_TOTE_FLOOR_Z - 0.006 - 0.56) / 2))
    out += M("OutToteFloor", "VizBrushed", box(tl + 0.012, tw + 0.012, 0.006, 0.001), (tx, 0.0, OUT_TOTE_FLOOR_Z - 0.003))
    for name, t, sz in (("OutToteWallUp", (OUT_TOTE_X0 - 0.003, 0.0, wz), (0.006, tw + 0.012, wall_h)), ("OutToteWallDown", (OUT_TOTE_X1 + 0.003, 0.0, wz), (0.006, tw + 0.012, wall_h)),
                        ("OutToteWallNear", (tx, -OUT_TOTE_HALF_Y - 0.003, wz), (tl, 0.006, wall_h)), ("OutToteWallFar", (tx, OUT_TOTE_HALF_Y + 0.003, wz), (tl, 0.006, wall_h))):
        out += M(name, "VizBrushed", box(*sz, 0.0015), t)
    for name, t, sz in (("OutToteRimUp", (OUT_TOTE_X0 - 0.003, 0.0, OUT_TOTE_RIM_Z + 0.003), (0.014, tw + 0.026, 0.006)), ("OutToteRimDown", (OUT_TOTE_X1 + 0.003, 0.0, OUT_TOTE_RIM_Z + 0.003), (0.014, tw + 0.026, 0.006)),
                        ("OutToteRimNear", (tx, -OUT_TOTE_HALF_Y - 0.003, OUT_TOTE_RIM_Z + 0.003), (tl, 0.014, 0.006)), ("OutToteRimFar", (tx, OUT_TOTE_HALF_Y + 0.003, OUT_TOTE_RIM_Z + 0.003), (tl, 0.014, 0.006))):
        out += M(name, "VizStainless", box(*sz, 0.0012), t)
    k = 0
    for i in range(6):                                                         # near and far rims, along x
        cx = OUT_TOTE_X0 + 0.02 + i * 0.04
        for y in (-OUT_TOTE_HALF_Y - 0.003, OUT_TOTE_HALF_Y + 0.003):
            out += _viz_prop("Cube", f"OutTape_{k:02d}", "VizYellow" if i % 2 == 0 else "VizBlack", (cx, y, OUT_TOTE_RIM_Z + 0.0065), (0.04, 0.014, 0.001), extra=_CUBE)
            k += 1
    for i in range(7):                                                         # up and down rims, along y
        cy = -OUT_TOTE_HALF_Y + 0.02 + i * 0.04
        for x in (OUT_TOTE_X0 - 0.003, OUT_TOTE_X1 + 0.003):
            out += _viz_prop("Cube", f"OutTape_{k:02d}", "VizYellow" if i % 2 == 0 else "VizBlack", (x, cy, OUT_TOTE_RIM_Z + 0.0065), (0.014, 0.04, 0.001), extra=_CUBE)
            k += 1
    return out


def _sensor_brackets() -> str:
    """Retro-reflective photoelectric trigger sensor at d = 0: emitter on the operator side,
    reflector on the far side, L-brackets on the extrusions, apertures through the extrusion faces.
    Visual only - the trigger itself is encoder-clocked (a 240 Hz raycast would add 6.7 mm of
    jitter to the calibrated inspection window)."""
    M, box = _mesh_prop, bf.chamfered_box_geometry
    out = ""
    for side, y_out, y_in in (("Near", -0.196, -0.156), ("Far", 0.196, 0.156)):
        sgn = -1 if side == "Near" else 1
        out += M(f"SensorArm{side}", "VizAlu", box(0.03, 0.036, 0.006, 0.001), (0.0, y_out + sgn * 0.018, 0.036))
        out += M(f"SensorPost{side}", "VizAlu", box(0.03, 0.006, 0.05, 0.001), (0.0, y_out + sgn * 0.033, 0.014))
        out += M(f"SensorSlot{side}Out", "VizBlack", box(0.012, 0.002, 0.008, 0.0005), (0.0, y_out, 0.012))
        out += M(f"SensorSlot{side}In", "VizBlack", box(0.012, 0.002, 0.008, 0.0005), (0.0, y_in, 0.012))
    out += M("SensorHead", "VizDark", _sensor_head_geometry(0.008, 0.028, 0.0015), (0.0, -0.200, 0.012), (90, 0, 0))
    out += M("SensorLens", "Glass", _sensor_head_geometry(0.005, 0.003, 0.0005), (0.0, -0.2005, 0.012), (90, 0, 0))
    out += M("SensorReflector", "VizAccent", box(0.016, 0.003, 0.014, 0.0006), (0.0, 0.199, 0.012))
    return out


def _rest_poses(seed: int = 7) -> list[tuple]:
    """Scattered resting poses inside the tote (three on the floor, then a tilted pile)."""
    rng = np.random.default_rng(seed)
    poses = []
    for k in range(N_REST):
        layer = 0 if k < 3 else 1 if k < 6 else 2
        x = float(rng.uniform(TOTE_C[0] - 0.09, TOTE_C[0] + 0.09)); y = float(rng.uniform(TOTE_C[1] + 0.00, TOTE_C[1] + 0.10))   # far half: the near half is behind the tote's own wall from the overview
        z = TOTE_FLOOR_Z + 0.003 + layer * 0.0075
        yaw = float(rng.uniform(-40, 40)); tilt = (float(rng.uniform(-10, 10)), float(rng.uniform(-8, 8))) if layer else (0.0, 0.0)
        poses.append(((x, y, z), (tilt[0], tilt[1], yaw)))
    return poses


REST_POSES = _rest_poses()


def _proxies() -> tuple[str, dict]:
    """Render-only packs for the part of the story that happens after PhysX confirmation: N_SLIDE
    animated slide proxies (authored visible, parked below the cell until used) and N_REST resting
    proxies in the tote (authored invisible, revealed as ejections land).  The dynamic pool is
    never held: bodies are recycled the instant the overlap query confirms them, exactly as before."""
    rng = np.random.default_rng(11)
    text, leaves = "", {"slide": [], "rest": [], "slide_pills": []}
    for k in range(N_SLIDE):
        t, paths = bf.proxy_pack_prim(f"SlidePack_{k}", (-1.0 - 0.2 * k, 0.0, -5.0), (0, 0, 0), (True,) * 10, indent=8, visible=True)
        text += t
        leaves["slide"].append(paths)
        leaves["slide_pills"].append([p for p in paths if "/Pill_" in p])
    for k, (pos, rot) in enumerate(REST_POSES):
        has = tuple(bool(rng.uniform() < 0.72) for _ in range(10))
        t, paths = bf.proxy_pack_prim(f"TotePack_{k}", pos, rot, has, indent=8, visible=False)
        text += t
        leaves["rest"].append(paths)
    return text, leaves


PROXY_USDA, PROXY_LEAVES = _proxies()


def _viz_props() -> str:
    P = _viz_prop
    legs = "".join(P("Cube", f"Leg_{i}", "VizAlu", (x, y, -0.36), (0.04, 0.04, 0.40), extra=_CUBE)
                   for i, (x, y) in enumerate(((-0.5, -0.13), (-0.5, 0.13), (0.5, -0.13), (0.5, 0.13))))
    props = (
        # --- floor and conveyor bed ------------------------------------------------------
        P("Cube", "Floor", "VizDark", (0.20, 0.35, -0.57), (4.2, 3.2, 0.02), extra=_CUBE)
        + P("Cube", "ConveyorFrame", "VizDark", (0.0, 0.0, -0.09), (1.2, 0.26, 0.13), extra=_CUBE)
        + legs
        # 80/20 extrusions: 40 mm profile with two T-slot grooves on the inner face, wear strips at belt level
        + P("Cube", "ExtrusionNear", "VizAlu", (0.0, -0.176, 0.02), (1.2, 0.04, 0.04), extra=_CUBE)
        + P("Cube", "SlotNearLo", "VizBlack", (0.0, -0.1565, 0.011), (1.2, 0.002, 0.007), extra=_CUBE)
        + P("Cube", "SlotNearHi", "VizBlack", (0.0, -0.1565, 0.029), (1.2, 0.002, 0.007), extra=_CUBE)
        + P("Cube", "WearNear", "VizWear", (0.0, -0.1535, 0.006), (1.2, 0.005, 0.012), extra=_CUBE)
        + P("Cube", "ExtrusionFar", "VizAlu", (-0.18, 0.176, 0.02), (0.84, 0.04, 0.04), extra=_CUBE)
        + P("Cube", "SlotFarLo", "VizBlack", (-0.18, 0.1565, 0.011), (0.84, 0.002, 0.007), extra=_CUBE)
        + P("Cube", "SlotFarHi", "VizBlack", (-0.18, 0.1565, 0.029), (0.84, 0.002, 0.007), extra=_CUBE)
        + P("Cube", "WearFar", "VizWear", (-0.18, 0.1535, 0.006), (0.84, 0.005, 0.012), extra=_CUBE)
        # 20 mm knife-edge nose rollers tangent to the belt top: the belt wraps them, nothing stands proud of the surface
        + P("Cylinder", "RollerIn", "VizSteel", (-0.605, 0.0, -0.010), (0.010, 0.010, 0.34), rotate=(90, 0, 0), extra=_CYL)
        + P("Cylinder", "RollerOut", "VizSteel", (0.605, 0.0, -0.010), (0.010, 0.010, 0.34), rotate=(90, 0, 0), extra=_CYL)
        # --- inspection mast (the camera housing, ring light, hood and arm live in the asset) ---
        + P("Cube", "MastBase", "VizAlu", (0.0, 0.235, -0.565), (0.14, 0.14, 0.01), extra=_CUBE)
        + P("Cube", "Mast", "VizAlu", (0.0, 0.235, -0.10), (0.04, 0.04, 0.93), extra=_CUBE)
        # --- reject station at d = 300 mm: valve block, brass nozzle and fitting, hose, air burst
        + P("Cube", "ValveBlock", "VizSteel", (0.30, -0.265, 0.035), (0.06, 0.08, 0.07), extra=_CUBE)
        + P("Cylinder", "Fitting", "VizBrass", (0.30, -0.224, 0.018), (0.013, 0.013, 0.016), rotate=(90, 0, 0), extra=_CYL)
        + P("Cylinder", "Nozzle", "VizBrass", (0.30, -0.19, 0.018), (0.007, 0.007, 0.06), rotate=(90, 0, 0), extra=_CYL)
        + P("Cylinder", "HoseUp", "VizBlack", (0.30, -0.30, 0.10), (0.006, 0.006, 0.12), extra=_CYL)
        + P("Cylinder", "HoseBack", "VizBlack", (0.19, -0.30, 0.16), (0.006, 0.006, 0.22), rotate=(0, 90, 0), extra=_CYL)
        + P("Cone", "AirBurst", "VizAir", (0.30, -0.10, 0.02), (0.02, 0.02, 0.12), rotate=(90, 0, 0), extra=_CYL + _HIDDEN)
        # --- stainless chute from the belt edge into the tote --------------------------------
        + _chute_assembly()
        + _outfeed_assembly()
        + _sensor_brackets()
        + PROXY_USDA
        # --- floor safety perimeters: reject zone, outfeed zone --------------------------------
        + _hazard_stripes(0.20, 0.94, 0.13, 0.66, -0.5595)
        + _hazard_stripes(OUT_TOTE_X0 - 0.06, OUT_TOTE_X1 + 0.06, -OUT_TOTE_HALF_Y - 0.06, 0.11, -0.5595, prefix="OutStripe")
    )
    return '    def Scope "Props"\n    {\n' + props + "    }\n"


VIZ_PROPS = _viz_props()

def compose_cell(asset_text: str, pool: int, viz: bool = False) -> tuple[str, list[str]]:
    """Instance the asset's pack subtree ``pool`` times (parked), keep RGB-only rendering, add the
    belt physics material, the outfeed colliders (physics side of the outfeed: always present, so
    a recorded run and a headless run simulate the same cell) and the overview camera.  ``viz``
    adds visual-only cell furniture for the dashboard.  Every rewrite is asserted so an asset
    change fails loudly instead of silently producing a different cell."""
    assert MIN_POOL <= pool <= len(PARK_SLOTS), f"--pool {pool}: need {MIN_POOL}..{len(PARK_SLOTS)} (belt {BELT_OCCUPANCY} + reject transit {REJECT_TRANSIT} + 1 outfeed body + 1 spare; {len(PARK_SLOTS)} park slots)"
    lines = asset_text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('    def Xform "Pack" ('))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "    }")
    block = lines[start : end + 1]
    mass_i = next(i for i, l in enumerate(block) if "physics:mass" in l)
    copies: list[str] = []
    paths = []
    for k in range(pool):
        b = [l.replace('def Xform "Pack" (', f'def Xform "Pack_{k:02d}" (', 1) for l in block]
        x, y = PARK_SLOTS[k]
        b[mass_i + 1 : mass_i + 1] = [f"        double3 xformOp:translate = ({x}, {y}, 0)", '        uniform token[] xformOpOrder = ["xformOp:translate"]']
        copies += b
        paths.append(f"/World/Pack_{k:02d}")
    lines[start : end + 1] = copies
    text = "\n".join(lines) + "\n"
    rewrites = [
        ("rel orderedVars = [</Render/Camera/LdrColor>, </Render/Camera/SemanticSegmentation>, </Render/Camera/SemanticIdMap>, </Render/Camera/DistanceToImagePlaneSD>]", "rel orderedVars = </Render/Camera/LdrColor>"),
        ("        rel material:binding = </World/Looks/Belt>\n", "        rel material:binding = </World/Looks/Belt>\n        rel material:binding:physics = </World/Looks/BeltPhysics>\n"),
        ('    def Scope "Looks"\n    {\n', '    def Scope "Looks"\n    {\n' + BELT_PHYSICS_MATERIAL + OUTFEED_PHYSICS_MATERIALS + (VIZ_MATERIALS if viz else "")),
        ('    def Cube "Belt" (', OVERVIEW_CAMERA + OUTFEED_COLLIDERS + (VIZ_PROPS if viz else "") + '    def Cube "Belt" ('),
    ]
    for old, new in rewrites:
        assert text.count(old) == 1, f"asset rewrite anchor not found exactly once: {old[:60]!r}"
        text = text.replace(old, new)
    head, sep, tail = text.rpartition("}\n")
    assert sep and tail.strip() == "", "unexpected asset tail"
    text = head + OVERVIEW_PRODUCT + "}\n"
    return text, paths


# --------------------------------------------------------------------------------------
# Controller adapters
# --------------------------------------------------------------------------------------
class TwinEncoder(Encoder):
    """Distance clock driven by the simulation; W3's reads are tracked for the lockstep barrier."""

    def __init__(self, cfg: LineConfig):
        self.cfg = cfg
        self._ticks = 0
        self.last_observed = -1
        self._lock = threading.Lock()

    def set(self, ticks: int) -> None:
        with self._lock:
            self._ticks = ticks

    def ticks(self) -> int:
        with self._lock:
            t = self._ticks
            if t > self.last_observed:
                self.last_observed = t
            return t

    def tick_rate(self) -> float:
        return self.cfg.v_belt_mps / self.cfg.encoder_pitch_m


class TwinActuator(Actuator):
    """fire() enqueues a kick for the next physics step; confirm() reports the bin overlap state
    known so far (physical confirmation arrives later as its own audit event)."""

    def __init__(self):
        self.pending: queue.Queue[tuple[int, int]] = queue.Queue()
        self.confirmed: set[int] = set()
        self.fired: dict[int, int] = {}

    def fire(self, pack_id: int, ticks: int) -> None:
        self.fired[pack_id] = ticks
        self.pending.put((pack_id, ticks))

    def confirm(self, pack_id: int) -> bool:
        return pack_id in self.confirmed


class TwinFrameSource(DLPackFrameSource):
    def __init__(self):
        self.q: deque = deque()
        self._lock = threading.Lock()
        super().__init__(self._produce)

    def push(self, img: torch.Tensor, pack_id: int, trigger_ticks: int) -> None:
        with self._lock:
            self.q.append((img, pack_id, trigger_ticks))

    def _produce(self, ticks: int):
        with self._lock:
            return self.q.popleft() if self.q else None


# --------------------------------------------------------------------------------------
# The twin
# --------------------------------------------------------------------------------------
@dataclass
class PoolPack:
    slot: int
    path: str
    idx: int                                  # row in the physics bindings
    state: str = "parked"                     # parked | belt | kicked | falling
    pack_id: int | None = None
    spawn_ticks: int = 0
    triggered: bool = False
    scheduled: bf.PackState | None = None
    kick_steps_left: int = 0
    kick_torque: tuple = (0.0, 0.0, 0.0)
    max_abs_y: float = 0.0
    landed: bool = False
    exited: bool = False                      # passed the x_exit bookkeeping point (deviation recorded)
    t_outfeed: float = 0.0                    # sim time the centre of mass left the belt collider
    still_steps: int = 0
    settled: bool = False
    visible_states: tuple = ("pill_ok",) * bf.N_COLS * bf.N_ROWS


@dataclass
class TwinStats:
    physics_steps: int = 0
    renders: int = 0
    display_frames: int = 0
    frames_pushed: int = 0
    kicks: int = 0
    confirmed: int = 0
    lost: int = 0
    outfeed_entered: int = 0
    outfeed_settled: int = 0
    outfeed_recycled: int = 0                 # rolling FIFO: oldest settled pack returned to the pool
    outfeed_forced: int = 0                   # pool guard with every outfeed body still moving: the oldest recycled mid-flight
    outfeed_lost: int = 0
    outfeed_outside: int = 0                  # settled, but not inside the tote volume
    outfeed_timeouts: int = 0                 # counted settled by dwell time, still creeping
    outfeed_max_in_tote: int = 0
    outfeed_settle_s: list = field(default_factory=list)
    barrier_waits_ms: list = field(default_factory=list)
    trig_render_ms: list = field(default_factory=list)
    trig_copy_ms: list = field(default_factory=list)
    disp_mirror_ms: list = field(default_factory=list)
    disp_render_ms: list = field(default_factory=list)
    disp_compose_ms: list = field(default_factory=list)
    disp_write_ms: list = field(default_factory=list)
    trigger_wall_ms: list = field(default_factory=list)
    verdict_wait_ms: list = field(default_factory=list)
    capture_path: str = "?"


class LiveFactoryTwin:
    def __init__(self, n_packs: int, seed: int, mode: str, pool: int = 16, line: LineConfig = LineConfig(), twin: TwinConfig = TwinConfig(), save_frames: int = 3,
                 record: Path | None = None, gui: bool = False, vc: VizConfig = VizConfig()):
        assert mode in ("lockstep", "realtime")
        self.n_packs, self.seed, self.mode, self.line, self.tc, self.save_frames = n_packs, seed, mode, line, twin, save_frames
        self.dt = 1.0 / twin.physics_hz
        self.record, self.gui, self.vc = record, gui, vc
        self.viz = bool(record or gui)
        # with the dashboard on, every step renders BOTH products (see _render_multi: switching
        # products between steps costs ~130 ms on this build)
        self.render_set = {PRODUCT_INSPECT, PRODUCT_OVERVIEW} if self.viz else PRODUCT_INSPECT
        # one display frame every N physics steps; N is what makes the playback speed exact
        self.capture_every = max(1, int(round((vc.speed / vc.fps) / self.dt)))
        self.play_speed = self.capture_every * self.dt * vc.fps
        self.scheduler = bf.PackScheduler(seed)
        self.usda, self.pack_paths = compose_cell(ASSET.read_text(encoding="utf-8"), pool, viz=self.viz)
        self.stats = TwinStats()
        self.sim_time = 0.0
        self.belt_dist = 0.0
        self.ticks = 0
        self.next_pack = 0
        self.confirm_events: list[dict] = []
        self.deviation: dict[int, float] = {}
        self.outcome: dict[int, dict] = {}
        self.shot: dict | None = None            # last strobe exposure held on the dashboard
        self.burst_until = -1                    # physics step until which the air-burst prop stays visible
        self.burst_visible = False
        self.last_kick_pack: int | None = None
        self.outfeed: list[PoolPack] = []        # accepted packs past the belt end, oldest first (FIFO)
        # pool budget: belt occupancy + reject transit + 1 spare are always reserved; the rest may
        # sit in the outfeed (moving or stacked).  The FIFO keeps up to outfeed_capacity settled;
        # when packs in transit press against the cap, the bottom of the stack simply goes early.
        self.outfeed_max = max(1, pool - BELT_OCCUPANCY - REJECT_TRANSIT - 1)
        self.outfeed_capacity = max(0, min(twin.outfeed_capacity, self.outfeed_max - 2))
        self.slides: list[dict] = []             # render-only slides down the chute (proxy, t0, start pose, rest index)
        self.slide_free: list[int] = list(range(N_SLIDE))
        self.rest_visible: set[int] = set()
        self.next_rest = 0
        self.last_trigger_time = -1.0
        self.trigger_times: list[float] = []     # sim time of each trigger (throughput meter)
        self.class_totals: dict[str, int] = {}
        self.class_counted: set[int] = set()
        self.flash: tuple | None = None          # (pack_id, world xyz, sim_time) of the last confirmation
        self.sink = None
        self.window = None
        self.paused_s = 0.0
        self.video_info: dict | None = None
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # -- setup -------------------------------------------------------------------------
    def setup(self) -> None:
        check_versions()
        # render stage
        self.scene = Scene("twin.render")
        self.scene.load_usda(self.usda, domains=ovstage.PopulationDomain.RENDERING)
        # physics stage
        self.pstage = ovstage.Stage("twin.physics")
        ovstage.population.open_usd_from_string(self.pstage, self.usda, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
        self.pstage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        self.physx = PhysX()
        self.physx.attach_ovstage(self.pstage, read_ordinal=1)
        self.pose_b = self.physx.create_tensor_binding(pattern="/World/Pack_*", tensor_type=TensorType.RIGID_BODY_POSE)
        self.vel_b = self.physx.create_tensor_binding(pattern="/World/Pack_*", tensor_type=TensorType.RIGID_BODY_VELOCITY)
        self.wrench_b = self.physx.create_tensor_binding(pattern="/World/Pack_*", tensor_type=TensorType.RIGID_BODY_WRENCH)
        order = list(self.pose_b.prim_paths)
        assert sorted(order) == sorted(self.pack_paths), (order, self.pack_paths)
        self.pool = [PoolPack(slot=int(p[-2:]), path=p, idx=order.index(p)) for p in self.pack_paths]
        self.n = len(self.pool)
        self.pose = np.zeros((self.n, 7), np.float32)
        self.vel = np.zeros((self.n, 6), np.float32)
        self.wrench = np.zeros((self.n, 9), np.float32)
        self.pose_b.read(self.pose)
        # scene-query hits carry PhysX object handles, not prim paths: learn each pack's rigid-body
        # handle with a thin overlap box inside its parked body (above the belt surface, so only
        # the pack is hit)
        self.handle_to_pack: dict[int, PoolPack] = {}
        for pk in self.pool:
            x, y = PARK_SLOTS[pk.slot]
            hits = self.physx.overlap(SceneQueryGeometryType.BOX, mode=SceneQueryMode.ALL, half_extent=(0.03, 0.015, 0.0004), position=(x, y, 0.0009))
            ids = {int(h["rigid_body"]) for h in hits if int(h.get("rigid_body", 0))}
            assert len(ids) == 1, f"expected exactly one body in the park slot of {pk.path}, got {ids}"
            self.handle_to_pack[ids.pop()] = pk
        # controller
        self.encoder = TwinEncoder(self.line)
        self.actuator = TwinActuator()
        self.source = TwinFrameSource()
        self.ctl = Controller(self.source, self.encoder, self.actuator, cfg=self.line, policy=Policy.calibrated(), engine_path=ENGINE_PATH, audit_path=AUDIT_PATH)
        self.ctl.start()
        assert not self.ctl.errors, self.ctl.errors
        # prime the render stage (first transform write needs a throw-away step) and warm the cameras
        self._mirror_poses(prime=True)
        self._render_multi({PRODUCT_INSPECT, PRODUCT_OVERVIEW}, steps=3)
        self._render_multi(self.render_set, steps=3)
        self.leaf_cache: dict[tuple[int, int, str], list[str]] = {}
        if self.viz:
            self._setup_viz()

    # -- visual monitoring -------------------------------------------------------------
    def _setup_viz(self) -> None:
        from simulation import twin_hud as hud

        self.hud = hud
        self.ov_cam = hud.OverviewCam(OV_POS, OV_RX, OV_FOCAL, OV_AP_H, OV_AP_V, width=bf.IMAGE_W, height=bf.IMAGE_H)
        pol = self.ctl.policy
        self.hud_footer = (f"engine {ENGINE_PATH.name} | thresholds {pol.source} conf_ok {pol.conf_ok} conf_defect {pol.conf_defect} | "
                           f"run {self.ctl.audit.meta['run_id'][:12]} | physics {self.tc.physics_hz} Hz, belt {self.line.v_belt_mps} m/s, "
                           f"pitch {self.line.pack_pitch_m * 1000:.0f} mm, nozzle d={self.line.d_nozzle_m * 1000:.0f} mm")
        self._disp_q: queue.Queue = queue.Queue(maxsize=8)
        self._disp_out: queue.Queue = queue.Queue(maxsize=2)
        self._disp_err: list[str] = []
        self._disp_done = 0
        if self.record:
            self.sink = hud.VideoSink(self.record, self.vc.fps, (hud.CANVAS_W, hud.CANVAS_H), crf=self.vc.crf)
        if self.gui:
            import cv2

            if not hud.gui_available():
                raise RuntimeError("--gui needs an OpenCV build with HighGUI (this one is headless); use --record-video")
            self.window = "Blister line digital twin"
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, 1600, 900)
        # pay the one-time HUD costs (font tables, canvas allocation, first cv2 calls) before the
        # belt starts, so the first dashboard frames are not delayed by them.  Note this does NOT
        # remove the first-inference spike seen in --realtime: that one is GPU contention between
        # the overview renders and W2 (see the report).
        hud.compose(self._hud_frame(np.zeros((bf.IMAGE_H, bf.IMAGE_W, 3), np.uint8), 0.0), self.ov_cam,
                    frame_w=bf.IMAGE_W, frame_h=bf.IMAGE_H, imgsz=IMGSZ, roi_half_px=ROI_HALF_PX,
                    n_cavities=bf.N_COLS * bf.N_ROWS, budget_ms=self.ctl.policy.inference_budget_s * 1000)
        self._disp_thread = threading.Thread(target=self._disp_worker, name="dashboard", daemon=True)
        self._disp_thread.start()

    def leaves(self, slot: int, cav: int, state: str) -> list[str]:
        key = (slot, cav, state)
        if key not in self.leaf_cache:
            self.leaf_cache[key] = [p.replace("/World/Pack/", f"/World/Pack_{slot:02d}/", 1) for p in bf.variant_leaf_paths(cav, state)]
        return self.leaf_cache[key]

    def _render_xyz(self, pk: PoolPack):
        """Render-stage position of a pool pack.  The physics pose, except that with the dashboard
        on, packs waiting in the pool are drawn far below the cell instead of queued on the belt
        (they are physics bodies resting off-camera; on the overview they would read as a stalled
        line).  Physics is untouched - this is a render-stage write only."""
        if self.viz and pk.state == "parked":
            return (float(self.pose[pk.idx, 0]), float(self.pose[pk.idx, 1]), self.vc.park_sink_z)
        return self.pose[pk.idx, 0:3]

    # -- render-only continuation of a rejected pack: slide down the chute, rest in the tote ---
    def _start_slide(self, pk: PoolPack) -> None:
        if not self.slide_free:
            return                                    # more than N_SLIDE packs in the chute at once: skip the animation, keep the count
        k = self.slide_free.pop(0)
        rest = self.next_rest % N_REST
        self.next_rest += 1
        p0 = np.array(self.pose[pk.idx, 0:3], dtype=np.float64)
        q0 = np.array(self.pose[pk.idx, 3:7], dtype=np.float64)
        pills = [self.viz_pill_visible(pk.scheduled.states[i]) for i in range(10)]
        self.slides.append({"proxy": k, "t0": self.sim_time, "p0": p0, "q0": q0, "rest": rest, "pills": pills, "pills_written": False})

    @staticmethod
    def viz_pill_visible(state: str) -> bool:
        return state != "cavity_empty"

    def _slide_frames(self, o: int) -> list[tuple[int, np.ndarray]]:
        """Poses of the slide proxies for this ordinal.  Path: along the chute (offset above the
        plate) to its lower edge, then a short drop onto the resting pose; orientation slerps from
        the confirmation pose to the resting pose so the kick's tumble carries into the slide."""
        out = []
        active = {sl["proxy"] for sl in self.slides}
        for sl in list(self.slides):
            k = sl["proxy"]
            if not sl["pills_written"]:                # match the proxy's tablets to the pack's own state (scalar writes first)
                show = [p for p, v in zip(PROXY_LEAVES["slide_pills"][k], sl["pills"]) if v]
                hide = [p for p, v in zip(PROXY_LEAVES["slide_pills"][k], sl["pills"]) if not v]
                if show:
                    self.scene.write_token(show, "visibility", "inherited", ordinal=o)
                if hide:
                    self.scene.write_token(hide, "visibility", "invisible", ordinal=o)
                sl["pills_written"] = True
            u = min(1.0, max(0.0, (self.sim_time - sl["t0"]) / SLIDE_S))
            rest_pos, rest_rot = REST_POSES[sl["rest"]]
            x0 = float(np.clip(sl["p0"][0], CHUTE_X0 + 0.05, CHUTE_X1 - 0.08))
            y = float(np.clip(sl["p0"][1] + 0.06, CHUTE_YA + 0.05, CHUTE_YB - 0.05))
            top = np.array([x0, y, chute_z(x0) + 0.012])
            bottom = np.array([CHUTE_X1 + 0.02, y, CHUTE_Z1 + 0.012])
            landing = np.array(rest_pos)
            if u < 0.7:
                e = (u / 0.7) ** 2                     # accelerating down the slide
                if e < 0.05:                           # blend from the confirmation pose onto the slide path
                    pos = sl["p0"] + (top - sl["p0"]) * (e / 0.05)
                else:
                    pos = top + (bottom - top) * ((e - 0.05) / 0.95)
            else:
                e = (u - 0.7) / 0.3
                pos = bottom + (landing - bottom) * e
            R = _slerp_rot(quat_to_rot(sl["q0"]), _rot_xyz(*rest_rot), u)
            out.append((k, usd_matrix(R, pos)))
            if u >= 1.0:
                if sl["rest"] not in self.rest_visible:
                    self.scene.write_token(PROXY_LEAVES["rest"][sl["rest"]], "visibility", "inherited", ordinal=o)
                    self.rest_visible.add(sl["rest"])
                self.slides.remove(sl)
                self.slide_free.append(k)
        for k in range(N_SLIDE):                       # idle slide proxies stay parked below the cell
            if k not in active:
                out.append((k, usd_matrix(np.eye(3), (-1.0 - 0.2 * k, 0.0, -5.0))))
        return out

    # -- render-stage mirroring (verified write order: visibility first, transforms last) ------
    def _mirror_poses(self, prime: bool = False, dirty: list[PoolPack] | None = None) -> int:
        o = self.scene.begin()
        for pk in dirty or []:
            for cav in range(bf.N_COLS * bf.N_ROWS):
                want = pk.scheduled.states[cav]
                have = pk.visible_states[cav]
                if want != have:
                    self.scene.write_token(self.leaves(pk.slot, cav, have), "visibility", "invisible", ordinal=o)
                    self.scene.write_token(self.leaves(pk.slot, cav, want), "visibility", "inherited", ordinal=o)
            pk.visible_states = tuple(pk.scheduled.states)
        if self.viz:                                             # air-burst prop follows the valve (scalar write, before xforms)
            want = self.stats.physics_steps <= self.burst_until
            if want != self.burst_visible:
                self.scene.write_token([AIR_BURST_PATH], "visibility", "inherited" if want else "invisible", ordinal=o)
                self.burst_visible = want
        self.pose_b.read(self.pose)
        paths = [pk.path for pk in self.pool]
        mats = [usd_matrix(quat_to_rot(self.pose[pk.idx, 3:7]), self._render_xyz(pk)) for pk in self.pool]
        if self.viz:
            for k, m in self._slide_frames(o):        # visibility writes for landed packs happen inside, before xforms
                paths.append(f"/World/Props/SlidePack_{k}")
                mats.append(m)
        self.scene.write_xforms(paths, np.stack(mats), ordinal=o)
        self.scene.seal(o)
        if prime:
            self._render(PRODUCT_INSPECT, steps=1, keep=False)
        return o

    def _map_ldr(self, frame) -> torch.Tensor:
        """Map this frame's LdrColor and copy it out.  Zero-copy on CUDA, CPU map + upload if the
        interop path is unavailable.  Every attempt unmaps in a finally: a mapping that survived a
        failed import would leak the render var and the retry would map it a second time."""
        key = [k for k in frame.render_vars.keys() if k.endswith("LdrColor")][0]
        for device, path in ((ovrtx.Device.CUDA, "cuda-dlpack"), (ovrtx.Device.CPU, "cpu-fallback")):
            var = None
            try:
                var = frame.render_vars[key].map(device=device)
                img = torch.from_dlpack(var).clone() if path == "cuda-dlpack" else torch.from_numpy(np.from_dlpack(var).copy()).cuda()
                self.stats.capture_path = path
                return img
            except Exception:  # noqa: BLE001
                if path == "cpu-fallback":
                    raise
            finally:
                if var is not None:
                    var.unmap()
                    del var
        raise RuntimeError("unreachable")

    def _render_multi(self, products, steps: int, keep: tuple = ()) -> dict:
        """Every renderer.step goes through StepOutputs; frames are mapped on CUDA and cloned into
        persistent buffers before the outputs are released (no host round trip).

        ``products`` may be several render products, stepped together.  That is not an
        optimisation but a requirement once the dashboard is on: measured on this build, a step
        that renders a DIFFERENT product from the previous step costs ~130 ms (a 3-step triggered
        exposure went 24 ms -> 418 ms when overview frames were interleaved between triggers).
        Stepping the same set every time keeps the triggered exposure at ~35 ms."""
        want = {products} if isinstance(products, str) else set(products)
        out: dict[str, torch.Tensor] = {}
        for i in range(steps):
            with StepOutputs(self.scene.renderer, want, self.scene.ordinal, self.dt) as prods:
                self.stats.renders += 1
                for name, prod in prods.items():
                    for frame in prod.frames:
                        if i == steps - 1 and str(name) in keep:
                            out[str(name)] = self._map_ldr(frame)
                        del frame
                    del prod
        torch.cuda.synchronize()
        assert all(k in out for k in keep), f"render products {set(keep) - set(out)} delivered no frame"
        return out

    def _render(self, product: str, steps: int, keep: bool) -> torch.Tensor | None:
        return self._render_multi(product, steps, keep=(product,) if keep else ()).get(product)

    # -- physics helpers ---------------------------------------------------------------
    def _teleport(self, pk: PoolPack, x: float, y: float, vx: float) -> None:
        self.pose_b.read(self.pose)
        self.pose[pk.idx] = [x, y, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.pose_b.write(self.pose)
        self.vel_b.read(self.vel)
        self.vel[pk.idx] = [vx, 0, 0, 0, 0, 0]
        self.vel_b.write(self.vel)
        self.pose_b.wake_up()

    def _spawn(self, pk: PoolPack, pack_id: int) -> None:
        pk.state, pk.pack_id, pk.spawn_ticks, pk.triggered = "belt", pack_id, self.ticks, False
        pk.scheduled = self.scheduler.sample(pack_id)
        pk.kick_steps_left, pk.max_abs_y, pk.landed = 0, 0.0, False
        pk.exited, pk.settled, pk.still_steps = False, False, 0
        self._teleport(pk, self.tc.x_spawn, 0.0, self.line.v_belt_mps)
        self.outcome[pack_id] = {"scheduled": "PASS" if pk.scheduled.is_nominal else "REJECT", "slot": pk.slot}

    def _park(self, pk: PoolPack) -> None:
        x, y = PARK_SLOTS[pk.slot]
        self._teleport(pk, x, y, 0.0)
        pk.state, pk.pack_id = "parked", None

    def x_nominal(self, pk: PoolPack) -> float:
        return self.tc.x_spawn + (self.ticks - pk.spawn_ticks) * self.line.encoder_pitch_m

    def _drive(self) -> None:
        """Transport: belt packs re-driven to (v, 0, 0); kicked packs keep the belt speed along x
        while their lateral/vertical motion is left to physics; falling packs are free."""
        self.vel_b.read(self.vel)
        for pk in self.pool:
            if pk.state == "belt":
                self.vel[pk.idx] = [self.line.v_belt_mps, 0, 0, 0, 0, 0]
            elif pk.state == "kicked":
                self.vel[pk.idx, 0] = self.line.v_belt_mps
        self.vel_b.write(self.vel)

    def _kick(self) -> None:
        # newly fired packs from W3 (handoff happens at this step)
        while True:
            try:
                pack_id, _ticks = self.actuator.pending.get_nowait()
            except queue.Empty:
                break
            for pk in self.pool:
                if pk.pack_id == pack_id and pk.state == "belt" and not pk.exited:   # past x_exit the nozzle cannot reach it (v1 parked there)
                    pk.state, pk.kick_steps_left = "kicked", self.tc.kick_steps
                    self.stats.kicks += 1
                    self.last_kick_pack = pack_id
                    # seeded angular impulse (roll about x, yaw about z) so ejected packs tumble:
                    # torque about the centre of mass leaves the centre-of-mass path unchanged
                    rng = np.random.default_rng([self.seed, pack_id, 99])
                    pk.kick_torque = (float(rng.uniform(-1.5e-4, 1.5e-4)), 0.0, float(rng.uniform(-1.2e-3, 1.2e-3)))
        active = [pk for pk in self.pool if pk.state == "kicked" and pk.kick_steps_left > 0]
        if not active:
            return
        self.pose_b.read(self.pose)
        self.wrench[:] = 0.0
        self.burst_until = self.stats.physics_steps + 12          # ~50 ms of visible air after the last impulse step
        for pk in active:
            x, y, z = self.pose[pk.idx, 0:3]
            tx, ty, tz = pk.kick_torque
            self.wrench[pk.idx] = [0.0, self.tc.kick_force_n, 0.0, tx, ty, tz, x, y - bf.PACK_W_M / 2, z + bf.PACK_T_M / 2]
            pk.kick_steps_left -= 1
        self.wrench_b.write(self.wrench)

    def _bin_hits(self) -> list[PoolPack]:
        hits = self.physx.overlap(SceneQueryGeometryType.BOX, mode=SceneQueryMode.ALL, half_extent=self.tc.bin_half, position=self.tc.bin_center)
        return [self.handle_to_pack[int(h["rigid_body"])] for h in hits if int(h.get("rigid_body", 0)) in self.handle_to_pack]

    def _append_confirmation(self, pk: PoolPack, pos) -> None:
        rec = {"schema": "blister.confirm/1", "run_id": self.ctl.audit.meta["run_id"], "pack_id": pk.pack_id, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
               "sim_time_s": round(self.sim_time, 4), "encoder_ticks": self.ticks, "bin_overlap": True, "landing_xyz_m": [round(float(v), 4) for v in pos], "source": "physx.overlap(BOX)"}
        self.confirm_events.append(rec)
        self.flash = (pk.pack_id, tuple(float(v) for v in pos), self.sim_time)
        line = json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n"
        with self.ctl.audit._lock:                         # same append-only file and lock as the controller's records
            self.ctl.audit._f.write(line)
            self.ctl.audit._f.flush()
            os.fsync(self.ctl.audit._f.fileno())
        with self.ctl.ledger._lock:                         # physical truth into the in-memory ledger
            rec_l = self.ctl.ledger.packs.get(pk.pack_id)
            if rec_l is not None:
                rec_l.confirmed = True
        self.actuator.confirmed.add(pk.pack_id)

    # -- one physics step ------------------------------------------------------------------
    def step(self) -> None:
        tc, line = self.tc, self.line
        # spawn when the belt has advanced one pitch per pack
        if self.next_pack < self.n_packs and self.belt_dist >= self.next_pack * line.pack_pitch_m:
            free = next((pk for pk in self.pool if pk.state == "parked"), None)
            assert free is not None, "pack pool exhausted - increase --pool"
            self._spawn(free, self.next_pack)
            self.next_pack += 1
        self._kick()
        self._drive()
        self.physx.step(self.dt)
        self.stats.physics_steps += 1
        self.sim_time += self.dt
        self.belt_dist = line.v_belt_mps * self.sim_time
        self.ticks = int(self.belt_dist / line.encoder_pitch_m)
        self.encoder.set(self.ticks)
        self.pose_b.read(self.pose)
        if self.outfeed:
            self.vel_b.read(self.vel)                # fresh velocities for the settle detector
        # bookkeeping: triggers, lateral deviation, outfeed hand-off, bin confirmation
        any_airborne = False
        for pk in self.pool:
            if pk.state == "parked":
                continue
            x, y, z = (float(v) for v in self.pose[pk.idx, 0:3])
            xn = self.x_nominal(pk)
            if pk.state == "belt":
                if not pk.triggered and xn >= tc.x_camera:
                    self._trigger(pk)
                if xn > line.d_nozzle_m:
                    pk.max_abs_y = max(pk.max_abs_y, abs(y))
                if xn > tc.x_exit and not pk.exited:      # bookkeeping point unchanged from v1 (parity)
                    pk.exited = True
                    self.deviation[pk.pack_id] = pk.max_abs_y
                    self.outcome[pk.pack_id].update({"physical": "passed_nozzle", "max_abs_y_m": round(pk.max_abs_y, 5)})
                if x > tc.x_belt_end:                     # centre of mass past the belt collider: free body into the outfeed
                    pk.state, pk.t_outfeed, pk.still_steps, pk.settled = "outfeed", self.sim_time, 0, False
                    self.outfeed.append(pk)
                    self.stats.outfeed_entered += 1
                    self.stats.outfeed_max_in_tote = max(self.stats.outfeed_max_in_tote, len(self.outfeed))
                    self.outcome[pk.pack_id]["outfeed"] = "in_transit"
            elif pk.state == "outfeed":
                if z < tc.z_lost:
                    self.stats.outfeed_lost += 1
                    self.outcome[pk.pack_id].update({"outfeed": "lost", "outfeed_xyz_m": [round(x, 4), round(y, 4), round(z, 4)]})
                    self.outfeed.remove(pk)
                    self._park(pk)
                elif not pk.settled:
                    still = (float(np.linalg.norm(self.vel[pk.idx, 0:3])) < tc.outfeed_v_still
                             and float(np.linalg.norm(self.vel[pk.idx, 3:6])) < tc.outfeed_w_still)
                    pk.still_steps = pk.still_steps + 1 if still else 0
                    dwell = self.sim_time - pk.t_outfeed
                    if pk.still_steps >= tc.outfeed_settle_steps or dwell > tc.outfeed_dwell_max_s:
                        pk.settled = True
                        self.stats.outfeed_settled += 1
                        self.stats.outfeed_settle_s.append(dwell)
                        if pk.still_steps < tc.outfeed_settle_steps:
                            self.stats.outfeed_timeouts += 1
                        inside = OUT_TOTE_X0 < x < OUT_TOTE_X1 and abs(y) < OUT_TOTE_HALF_Y and z < OUT_TOTE_RIM_Z
                        if not inside:
                            self.stats.outfeed_outside += 1
                        self.outcome[pk.pack_id].update({"outfeed": "stacked" if inside else "settled_outside",
                                                         "outfeed_xyz_m": [round(x, 4), round(y, 4), round(z, 4)], "outfeed_settle_s": round(dwell, 3)})
            elif pk.state in ("kicked", "falling"):
                any_airborne = True
                if pk.state == "kicked" and abs(y) > tc.belt_half_width + 0.02:
                    pk.state = "falling"
                if z < tc.z_lost:
                    self.stats.lost += 1
                    self.outcome[pk.pack_id].update({"physical": "lost", "landing_xyz_m": [round(x, 4), round(y, 4), round(z, 4)]})
                    self._park(pk)
        if self.outfeed:
            settled = [pk for pk in self.outfeed if pk.settled]
            while len(settled) > self.outfeed_capacity:   # rolling FIFO: the oldest settled pack (bottom of the stack) recycles
                pk = settled.pop(0)
                self.outfeed.remove(pk)
                self.stats.outfeed_recycled += 1
                self.outcome[pk.pack_id]["outfeed_recycled_s"] = round(self.sim_time, 3)
                self._park(pk)
            while len(self.outfeed) > self.outfeed_max:     # pool guard: the outfeed never holds the pool
                pk = next((p for p in self.outfeed if p.settled), self.outfeed[0])
                self.outfeed.remove(pk)
                if pk.settled:                              # bottom of the stack goes early: an ordinary FIFO turn
                    self.stats.outfeed_recycled += 1
                    self.outcome[pk.pack_id]["outfeed_recycled_s"] = round(self.sim_time, 3)
                else:                                       # every outfeed body still moving: recycle the oldest anyway
                    self.stats.outfeed_forced += 1
                    self.outcome[pk.pack_id]["outfeed"] = "forced_recycle"
                self._park(pk)
        if any_airborne:
            for pk in self._bin_hits():
                if pk.state in ("kicked", "falling") and not pk.landed:
                    pk.landed = True
                    self.stats.confirmed += 1
                    pos = self.pose[pk.idx, 0:3]
                    self.outcome[pk.pack_id].update({"physical": "ejected_confirmed", "landing_xyz_m": [round(float(v), 4) for v in pos]})
                    self._append_confirmation(pk, pos)
                    if self.viz:
                        self._start_slide(pk)
                    self._park(pk)
        if self.mode == "lockstep":
            self._barrier()

    def _barrier(self) -> None:
        """W3 must have observed this tick count before the belt moves again."""
        t0 = time.perf_counter()
        while self.encoder.last_observed < self.ticks:
            if self.ctl.errors:
                raise RuntimeError(f"controller error: {self.ctl.errors}")
            if time.perf_counter() - t0 > 2.0:
                raise RuntimeError("lockstep barrier timeout: W3 did not observe the encoder")
            time.sleep(0.0002)
        self.stats.barrier_waits_ms.append((time.perf_counter() - t0) * 1000)

    def _apply_pack_roi(self, img: torch.Tensor) -> None:
        """Pitch-aware inspection window, applied in place on the CUDA frame.

        Line-design finding from the first closed-loop run: with a 120 mm pitch and a 200 mm field
        of view the neighbouring packs' end cavities are in the frame (a neighbour's edge enters as
        soon as FOV > 2 * (pitch - L_pack / 2) = 150 mm), and the whole-frame verdict then counts
        them (12-13/10 pill_ok, edge-cut pockets read as defects).  A real line gates the exposure
        or the verdict to the triggered pack; the frozen controller has no ROI gate, so the twin
        applies the window here: everything beyond +/-(L_pack/2 + 10 mm) of the frame centre is
        filled with the frame's own belt colour.  Production fix: ROI gate in the verdict or a
        field of view <= 150 mm (tracked as a Module 5 change request)."""
        x0, x1 = bf.IMAGE_W // 2 - ROI_HALF_PX, bf.IMAGE_W // 2 + ROI_HALF_PX
        belt = img[:40, x0:x1, :3].float().mean(dim=(0, 1)).to(img.dtype)     # belt colour from the top strip inside the ROI
        img[:, :x0, :3] = belt
        img[:, x1:, :3] = belt

    def _dirty_packs(self) -> list[PoolPack]:
        return [p for p in self.pool if p.state != "parked" and p.scheduled is not None and tuple(p.scheduled.states) != p.visible_states]

    def _trigger(self, pk: PoolPack) -> None:
        """Camera trigger: mirror poses (visibility first), render, hand the CUDA frame to W1."""
        t0 = time.perf_counter()
        pk.triggered = True
        self._mirror_poses(dirty=self._dirty_packs())
        tr = time.perf_counter()
        img = self._render_multi(self.render_set, steps=self.tc.capture_steps, keep=(PRODUCT_INSPECT,))[PRODUCT_INSPECT]
        self.stats.trig_render_ms.append((time.perf_counter() - tr) * 1000)
        assert img is not None and img.is_cuda
        self._apply_pack_roi(img)
        if pk.pack_id < self.save_frames:
            from PIL import Image

            Image.fromarray(img[..., :3].cpu().numpy()).save(REPORT_DIR / f"trigger_pack{pk.pack_id:02d}.png")
        self.trigger_times.append(self.sim_time)
        self.last_trigger_time = self.sim_time
        if self.viz:                                  # hold this exposure on the dashboard until the next trigger
            tc = time.perf_counter()
            self.shot = {"rgb": img[..., :3].cpu().numpy(), "pack_id": pk.pack_id, "sim_time": self.sim_time}
            self.stats.trig_copy_ms.append((time.perf_counter() - tc) * 1000)
        self.source.push(img, pk.pack_id, self.ticks)
        self.stats.frames_pushed += 1
        self.stats.trigger_wall_ms.append((time.perf_counter() - t0) * 1000)
        if self.mode == "lockstep":                       # deterministic: the verdict exists before the belt moves on
            t1 = time.perf_counter()
            while True:
                rec = self.ctl.ledger.packs.get(pk.pack_id)
                if rec is not None and rec.verdict is not None:
                    break
                if self.ctl.errors or time.perf_counter() - t1 > 5.0:
                    raise RuntimeError(f"no verdict for pack {pk.pack_id}: {self.ctl.errors}")
                time.sleep(0.0002)
            self.stats.verdict_wait_ms.append((time.perf_counter() - t1) * 1000)

    # -- dashboard ---------------------------------------------------------------------------
    def _hud_frame(self, overview_rgb: np.ndarray, wall: float) -> "object":
        hud = self.hud
        led = self.ctl.ledger
        with led._lock:
            rows = [{"pack_id": p, "verdict": led.packs[p].verdict, "confirmed": bool(led.packs[p].confirmed),
                     "latency_ms": led.packs[p].latency_ms} for p in led.order if led.packs[p].verdict is not None]
            for p in led.order:                                   # accumulate class totals once per verdict
                r = led.packs[p]
                if r.verdict is not None and p not in self.class_counted and r.scores:
                    for c, v in (r.scores.get("per_class") or {}).items():
                        self.class_totals[c] = self.class_totals.get(c, 0) + int(v.get("count", 0))
                    self.class_counted.add(p)
            shot_id = self.shot["pack_id"] if self.shot else None
            rec = led.packs.get(shot_id) if shot_id is not None else None
            last = None
            if rec is not None:
                sc = rec.scores or {}
                last = {"pack_id": rec.pack_id, "verdict": rec.verdict, "reason": rec.reason, "latency_ms": rec.latency_ms,
                        "n_ok": sc.get("n_pill_ok_distinct"), "per_class": sc.get("per_class") or {}, "dets": sc.get("detections") or []}
            inv = {"triggers": len(led.packs), "verdicts": sum(1 for r in led.packs.values() if r.verdict is not None),
                   "actions": sum(1 for r in led.packs.values() if r.action is not None)}
        tags = []
        for pk in self.pool:
            if pk.state == "parked" or pk.pack_id is None or (pk.state == "outfeed" and pk.settled):
                continue                                  # stacked packs carry no tag (the tote counter has them)
            r = led.packs.get(pk.pack_id)
            tags.append(hud.PackTag(pk.pack_id, tuple(float(v) for v in self.pose[pk.idx, 0:3]), pk.state,
                                    r.verdict if r else None, bool(r and r.action), bool(r and r.confirmed)))
        s = self.ctl.stats
        flash = None
        if self.flash and self.sim_time - self.flash[2] <= self.vc.flash_s:
            flash = (self.flash[0], self.flash[1])
        if last is not None:
            last["n_cavities"] = bf.N_COLS * bf.N_ROWS
        # throughput from the spacing of the last triggers (sim time), design rate from the line
        trig = self.trigger_times[-9:]
        ppm = 60.0 * (len(trig) - 1) / (trig[-1] - trig[0]) if len(trig) >= 2 and trig[-1] > trig[0] else None
        kicking = [pk for pk in self.pool if pk.state == "kicked" and pk.kick_steps_left > 0]
        kick_active = self.stats.physics_steps <= self.burst_until
        kick_pack = kicking[0].pack_id if kicking else (self.last_kick_pack if kick_active else None)
        return hud.HudFrame(
            sim_time=self.sim_time, wall_time=wall, rtf=self.sim_time / max(wall, 1e-6), mode=self.mode,
            speed=self.play_speed, overview_rgb=overview_rgb,
            inspection_rgb=self.shot["rgb"] if self.shot else None,
            inspection_age_s=(self.sim_time - self.shot["sim_time"]) if self.shot else 0.0,
            last_pack=last, tags=tags, ledger=inv,
            controller={k: s[k] for k in ("frames", "inferences", "overflow", "dropped_frames", "timeouts", "late_actuations")},
            history=rows,
            marks=[((self.tc.x_spawn, 0.0, 0.0), "INFEED"), ((self.tc.x_camera, 0.0, 0.0), "INSPECT  d = 0"),
                   ((self.line.d_nozzle_m, 0.0, 0.0), f"NOZZLE  d = {self.line.d_nozzle_m * 1000:.0f} mm"),
                   ((OUT_X0, 0.0, 0.0), f"OUTFEED  d = {OUT_X0 * 1000:.0f} mm", "near")],
            confirm_flash=flash, footer=self.hud_footer, packs_total=self.n_packs,
            kick_active=kick_active, kick_pack=kick_pack, kick_xyz=(self.line.d_nozzle_m, -0.16, 0.02),
            tote_xyz=(TOTE_C[0] - 0.14, TOTE_C[1] - 0.14, TOTE_RIM_Z), tote_count=self.stats.confirmed,
            encoder_m=self.ticks * self.line.encoder_pitch_m, encoder_ticks=self.ticks,
            throughput_ppm=ppm, design_ppm=60.0 * self.line.v_belt_mps / self.line.pack_pitch_m, belt_mps=self.line.v_belt_mps,
            class_totals=dict(self.class_totals),
            beam_a=BEAM_A, beam_b=BEAM_B,
            beam_broken=any(pk.state == "belt" and abs(float(self.pose[pk.idx, 0])) < bf.PACK_L_M / 2 for pk in self.pool),
            strobe=(self.sim_time - self.last_trigger_time) < 0.012,
            outfeed_xyz=((OUT_TOTE_X0 + OUT_TOTE_X1) / 2, OUT_TOTE_HALF_Y, OUT_TOTE_RIM_Z), outfeed_count=self.stats.outfeed_entered,
            outfeed_in_tote=sum(1 for pk in self.outfeed if pk.settled), outfeed_capacity=self.outfeed_capacity)

    def _capture_display(self, wall: float) -> bool:
        """One dashboard frame.  The simulation thread only mirrors, renders the overview and
        copies the pixels; drawing the HUD and encoding H.264 are pure CPU work and run on the
        dashboard thread, so the closed loop never waits on them.
        Returns False when the operator asked the live window to close."""
        if self._disp_err:
            raise RuntimeError(f"dashboard thread: {self._disp_err[0]}")
        t0 = time.perf_counter()
        self._mirror_poses(dirty=self._dirty_packs())
        t1 = time.perf_counter()
        img = self._render_multi(self.render_set, steps=self.vc.steps, keep=(PRODUCT_OVERVIEW,))[PRODUCT_OVERVIEW]
        hf = self._hud_frame(img[..., :3].cpu().numpy(), wall)
        t2 = time.perf_counter()
        self.stats.disp_mirror_ms.append((t1 - t0) * 1000)
        self.stats.disp_render_ms.append((t2 - t1) * 1000)
        self.stats.display_frames += 1
        try:                                            # bounded: backpressure if the encoder falls behind
            self._disp_q.put(hf, timeout=60)
        except queue.Full:                              # a dead worker must not park the simulation for ever
            raise RuntimeError(f"dashboard thread stalled: {self._disp_err or 'no error reported'}") from None
        if self.window is not None:
            return self._show()
        return True

    def _disp_worker(self) -> None:
        """Compose and encode dashboard frames off the simulation thread."""
        import cv2

        try:
            while True:
                hf = self._disp_q.get()
                if hf is None:
                    return
                t0 = time.perf_counter()
                canvas = self.hud.compose(hf, self.ov_cam, frame_w=bf.IMAGE_W, frame_h=bf.IMAGE_H, imgsz=IMGSZ,
                                          roi_half_px=ROI_HALF_PX, n_cavities=bf.N_COLS * bf.N_ROWS,
                                          budget_ms=self.ctl.policy.inference_budget_s * 1000)
                t1 = time.perf_counter()
                if self.sink is not None:
                    self.sink.write(canvas)
                self._disp_done += 1
                self.stats.disp_compose_ms.append((t1 - t0) * 1000)
                self.stats.disp_write_ms.append((time.perf_counter() - t1) * 1000)
                if self._disp_done in (1, 60):
                    cv2.imwrite(str(REPORT_DIR / f"dashboard_frame{self._disp_done:04d}.png"), canvas)
                if self.window is not None:
                    try:
                        self._disp_out.put_nowait(canvas)
                    except queue.Full:                   # the window may lag the recording; never block
                        pass
        except BaseException as e:  # noqa: BLE001
            self._disp_err.append(f"{type(e).__name__}: {e}")

    def _show(self) -> bool:
        import cv2

        try:
            canvas = self._disp_out.get_nowait()
        except queue.Empty:
            return True
        cv2.imshow(self.window, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")) or self.hud.window_closed(self.window):
            print("live window: stop requested", flush=True)
            return False
        if k == ord(" "):                                  # pause; the realtime pacer is rebased by run()
            t0 = time.perf_counter()
            while True:
                k2 = cv2.waitKey(50) & 0xFF
                if k2 in (27, ord("q")) or self.hud.window_closed(self.window):
                    self.paused_s += time.perf_counter() - t0
                    return False                           # closing the window must end the pause too
                if k2 == ord(" "):
                    break
                if self.ctl.errors or self._disp_err:
                    raise RuntimeError(f"paused with a worker error: {self.ctl.errors or self._disp_err}")
            self.paused_s += time.perf_counter() - t0
        return True

    # -- run ---------------------------------------------------------------------------------
    def run(self) -> dict:
        t_wall0 = time.perf_counter()
        overview_saved = False
        last_overview_check = 0
        done_tail = None
        stopped = False
        while True:
            if self.mode == "realtime":
                target = time.perf_counter() - t_wall0 - self.paused_s
                if self.sim_time >= target:
                    time.sleep(0.0005)
                    continue
                for _ in range(50):                      # bounded catch-up after a blocking render
                    if self.sim_time >= target:
                        break
                    self.step()
                    if not self._tick_display(t_wall0):
                        stopped = True
                        break
            else:
                self.step()
                if not self._tick_display(t_wall0):
                    stopped = True
            if stopped:
                break
            # overview evidence: the first pack in flight after a kick
            if not overview_saved and any(pk.state in ("kicked", "falling") for pk in self.pool) and self.stats.physics_steps - last_overview_check > 12:
                last_overview_check = self.stats.physics_steps
                self._mirror_poses()
                img = self._render_multi(self.render_set if self.viz else PRODUCT_OVERVIEW, steps=3, keep=(PRODUCT_OVERVIEW,))[PRODUCT_OVERVIEW]
                from PIL import Image

                Image.fromarray(img[..., :3].cpu().numpy()).save(REPORT_DIR / "overview_reject_in_flight.png")
                overview_saved = True
            all_spawned = self.next_pack >= self.n_packs
            in_flight = [pk for pk in self.pool if pk.state != "parked" and not (pk.state == "outfeed" and pk.settled)]
            if all_spawned and not in_flight and self.ctl.ledger.pending() == 0:
                if done_tail is None:
                    done_tail = self.sim_time + 0.1
                elif self.sim_time >= done_tail:
                    break
            if self.sim_time > (self.n_packs * self.line.pack_pitch_m + 2.0) / self.line.v_belt_mps + 5.0:
                raise RuntimeError("twin did not converge: packs still in flight")
        wall = time.perf_counter() - t_wall0 - self.paused_s
        if self.viz and not stopped:                      # tail frames so the last ejection is on screen
            for _ in range(self.vc.fps // 2):
                self._capture_display(wall)
        if self.viz:
            self._drain_display()
        return self._report(wall, stopped=stopped)

    def _tick_display(self, t_wall0: float) -> bool:
        if not self.viz or self.stats.physics_steps % self.capture_every:
            return True
        return self._capture_display(time.perf_counter() - t_wall0 - self.paused_s)

    def _report(self, wall: float, stopped: bool = False) -> dict:
        s = self.ctl.summary()
        led = self.ctl.ledger
        rows = []
        for pid in range(self.n_packs):
            rec = led.packs.get(pid)
            o = self.outcome.get(pid, {})
            rows.append({"pack_id": pid, "scheduled": o.get("scheduled"), "verdict": rec.verdict if rec else None, "reason": rec.reason if rec else None, "action": rec.action if rec else None,
                         "physical": o.get("physical"), "confirmed": bool(rec.confirmed) if rec and rec.action == "reject" else None, "max_abs_y_mm": round(o.get("max_abs_y_m", 0.0) * 1000, 2) if "max_abs_y_m" in o else None,
                         "verdict_ms": rec.latency_ms if rec else None, "outfeed": o.get("outfeed"), "outfeed_settle_s": o.get("outfeed_settle_s")})
        rejected = [r for r in rows if r["action"] == "reject"]
        passed = [r for r in rows if r["action"] == "pass"]
        sched_def = [r for r in rows if r["scheduled"] == "REJECT"]
        sched_nom = [r for r in rows if r["scheduled"] == "PASS"]
        metrics = {
            "mode": self.mode, "packs": self.n_packs, "pool": self.n, "wall_s": round(wall, 1), "sim_s": round(self.sim_time, 3), "rtf": round(self.sim_time / wall, 3),
            "physics_steps": self.stats.physics_steps, "renders_guarded": self.stats.renders, "frames_pushed": self.stats.frames_pushed, "capture_path": self.stats.capture_path,
            "kicks": self.stats.kicks, "bin_confirmed": self.stats.confirmed, "lost": self.stats.lost, "confirmation_events": len(self.confirm_events),
            "ledger": s["ledger"], "controller": {k: s[k] for k in ("frames", "inferences", "overflow", "dropped_frames", "timeouts", "late_actuations", "watchdog_events", "errors", "audit_records", "verdict_ms_median", "verdict_ms_p99")},
            "rejected_by_controller": len(rejected), "rejected_ejected_confirmed": sum(1 for r in rejected if r["physical"] == "ejected_confirmed"),
            "passed_by_controller": len(passed), "passed_without_deviation": sum(1 for r in passed if r["physical"] == "passed_nozzle" and (r["max_abs_y_mm"] or 0) < self.tc.nominal_dev_limit * 1000),
            "scheduled_defective": len(sched_def), "scheduled_defective_ejected": sum(1 for r in sched_def if r["physical"] == "ejected_confirmed"),
            "scheduled_nominal": len(sched_nom), "scheduled_nominal_passed_clean": sum(1 for r in sched_nom if r["physical"] == "passed_nozzle" and (r["max_abs_y_mm"] or 0) < self.tc.nominal_dev_limit * 1000),
            "verdict_matches_scheduled": sum(1 for r in rows if r["verdict"] == r["scheduled"]),
            "trigger_wall_ms_median": round(float(np.median(self.stats.trigger_wall_ms)), 1) if self.stats.trigger_wall_ms else None,
            "verdict_wait_ms_median": round(float(np.median(self.stats.verdict_wait_ms)), 1) if self.stats.verdict_wait_ms else None,
            "barrier_wait_ms_median": round(float(np.median(self.stats.barrier_waits_ms)), 2) if self.stats.barrier_waits_ms else None,
            "max_nominal_dev_mm": round(max([r["max_abs_y_mm"] or 0 for r in passed], default=0.0), 2),
            "passed_reached_outfeed": sum(1 for r in passed if r["outfeed"] in ("stacked", "in_transit", "forced_recycle")),
            "passed_stacked": sum(1 for r in passed if r["outfeed"] == "stacked"),
            "outfeed": {"entered": self.stats.outfeed_entered, "settled": self.stats.outfeed_settled, "recycled": self.stats.outfeed_recycled,
                        "forced_recycles": self.stats.outfeed_forced, "lost": self.stats.outfeed_lost, "settled_outside": self.stats.outfeed_outside,
                        "settle_timeouts": self.stats.outfeed_timeouts, "max_in_tote": self.stats.outfeed_max_in_tote,
                        "in_tote_at_end": sum(1 for pk in self.outfeed if pk.settled), "fifo_capacity": self.outfeed_capacity, "max_bodies": self.outfeed_max,
                        "settle_s_median": round(float(np.median(self.stats.outfeed_settle_s)), 3) if self.stats.outfeed_settle_s else None,
                        "settle_s_max": round(float(max(self.stats.outfeed_settle_s)), 3) if self.stats.outfeed_settle_s else None},
        }
        if self.viz:
            med = lambda v: round(float(np.median(v)), 2) if v else None  # noqa: E731
            metrics["dashboard"] = {"display_frames": self.stats.display_frames, "capture_every_steps": self.capture_every,
                                    "playback_speed": round(self.play_speed, 4), "fps": self.vc.fps,
                                    "renderer_steps_per_frame": self.vc.steps, "stopped_by_operator": stopped,
                                    "ms_median": {"mirror": med(self.stats.disp_mirror_ms), "overview_render": med(self.stats.disp_render_ms),
                                                  "hud_compose": med(self.stats.disp_compose_ms), "encode": med(self.stats.disp_write_ms)},
                                    "video": self.video_info}
            metrics["trigger_render_ms_median"] = med(self.stats.trig_render_ms)
            metrics["trigger_snapshot_ms_median"] = med(self.stats.trig_copy_ms)
        return {"metrics": metrics, "rows": rows}

    def _drain_display(self) -> None:
        """Finish every queued dashboard frame, stop the thread, close the encoder.

        Every exit path releases the encoder: a half-written MP4 and an orphaned ffmpeg child are
        worse than the error that got us here, and close() must not silently no-op after this
        raised once."""
        if getattr(self, "_disp_thread", None) is None:
            return
        alive = True
        try:
            try:
                self._disp_q.put(None, timeout=60)      # a worker that already died leaves the queue full
            except queue.Full:
                pass
            self._disp_thread.join(timeout=60)
            alive = self._disp_thread.is_alive()
            if self._disp_err:
                raise RuntimeError(f"dashboard thread: {self._disp_err[0]}")
            if alive:
                raise RuntimeError("dashboard thread did not stop within 60 s")
            assert self._disp_done == self.stats.display_frames, f"dashboard dropped frames: {self._disp_done}/{self.stats.display_frames}"
        finally:
            self._disp_thread = None
            sink, self.sink = self.sink, None
            if sink is not None:
                if alive:                               # the worker may still write: do not pretend the file is finished
                    sink.kill()
                    self.video_info = sink.info()
                else:
                    self.video_info = sink.close()

    def close(self) -> None:
        """Safe to call after a partial setup(): every stage is optional and independently guarded."""
        try:
            self._drain_display()
        except Exception as e:  # noqa: BLE001 - never mask a simulation error with a display error
            print("display teardown:", e, flush=True)
        if self.window is not None:
            try:
                import cv2

                cv2.destroyWindow(self.window)
                cv2.waitKey(1)
            except Exception as e:  # noqa: BLE001
                print("display teardown:", e, flush=True)
            self.window = None
        try:
            if getattr(self, "ctl", None) is not None:
                self.ctl.shutdown()
        finally:
            try:
                if getattr(self, "physx", None) is not None:
                    self.physx.detach_ovstage()
                    self.physx.release()
                if getattr(self, "pstage", None) is not None:
                    self.pstage.destroy()
            finally:
                if getattr(self, "scene", None) is not None:
                    self.scene.close()


def vram_sample() -> dict:
    """Device memory as torch sees it (this process) and as the driver sees it (whole GPU,
    including the renderer and PhysX, which torch cannot see)."""
    out = {"torch_alloc_mb": round(torch.cuda.memory_allocated() / 2**20, 1), "torch_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
           "torch_peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    try:
        import subprocess

        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        out["gpu_used_mb"] = float(q.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        out["gpu_used_mb"] = None
    return out


def demo(n_packs: int, mode: str, seed: int, pool: int, record: Path | None = None, gui: bool = False, vc: VizConfig = VizConfig()) -> int:
    vram = {"before_setup": vram_sample()}
    twin = LiveFactoryTwin(n_packs, seed, mode, pool=pool, record=record, gui=gui, vc=vc)
    t0 = time.perf_counter()
    try:                                   # setup opens the encoder, the window and the dashboard
        twin.setup()                       # thread: a failure after that must still release them
    except BaseException:
        twin.close()
        raise
    print(f"setup {time.perf_counter() - t0:.1f} s | {mode} | pool {twin.n} packs (belt {BELT_OCCUPANCY} + reject transit {REJECT_TRANSIT} + spare 1 reserved -> "
          f"outfeed bodies <= {twin.outfeed_max}, FIFO keeps {twin.outfeed_capacity}) | physics {twin.tc.physics_hz} Hz | line {twin.line.v_belt_mps} m/s, nozzle {twin.line.d_nozzle_m} m, pitch {twin.line.pack_pitch_m} m", flush=True)
    if twin.viz:
        print(f"dashboard {twin.hud.CANVAS_W}x{twin.hud.CANVAS_H} | every {twin.capture_every} physics steps -> {vc.fps} fps at "
              f"{twin.play_speed:.3g}x real time | record {record or '-'} | window {'on' if gui else 'off'}"
              + (f" | encoder {twin.sink.backend}" if twin.sink else ""), flush=True)
    vram["after_setup"] = vram_sample()
    try:
        result = twin.run()
        vram["after_run"] = vram_sample()
    finally:
        twin.close()
    vram["after_close"] = vram_sample()
    m, rows = result["metrics"], result["rows"]
    m["vram"] = vram
    (REPORT_DIR / f"twin_{mode}_{n_packs}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(m, indent=1), flush=True)
    print(f"{'id':>3} {'sched':>6} {'verdict':>7} {'action':>6} {'physical':>18} {'conf':>5} {'dev mm':>7} {'lat ms':>7}  reason")
    for r in rows:
        print(f"{r['pack_id']:3d} {r['scheduled']:>6} {str(r['verdict']):>7} {str(r['action']):>6} {str(r['physical']):>18} {str(r['confirmed']):>5} {str(r['max_abs_y_mm']):>7} {str(r['verdict_ms']):>7}  {r['reason']}")
    problems = []
    n = n_packs
    if not (m["ledger"]["ok"] and m["ledger"]["triggers"] == n):
        problems.append(f"ledger invariant: {m['ledger']}")
    if m["rejected_ejected_confirmed"] != m["rejected_by_controller"]:
        problems.append(f"ejection confirmation: {m['rejected_ejected_confirmed']}/{m['rejected_by_controller']} rejected packs confirmed in the bin")
    if m["scheduled_defective_ejected"] != m["scheduled_defective"]:
        problems.append(f"defective packs ejected: {m['scheduled_defective_ejected']}/{m['scheduled_defective']}")
    if m["passed_without_deviation"] != m["passed_by_controller"]:
        problems.append(f"passed packs with lateral deviation: {m['passed_by_controller'] - m['passed_without_deviation']}")
    if m["scheduled_nominal_passed_clean"] != m["scheduled_nominal"]:
        problems.append(f"nominal packs not passed cleanly: {m['scheduled_nominal_passed_clean']}/{m['scheduled_nominal']} (controller false rejects: {m['scheduled_nominal'] - m['scheduled_nominal_passed_clean']})")
    c = m["controller"]
    if c["frames"] != n or c["overflow"] or c["dropped_frames"] or c["errors"] or c["late_actuations"]:
        problems.append(f"controller counters: {c}")
    if m["frames_pushed"] != n:
        problems.append(f"frames pushed {m['frames_pushed']} != {n}")
    of = m["outfeed"]
    print(f"outfeed: {of['entered']} accepted packs left the belt, {m['passed_stacked']} stacked in the tote, {of['recycled']} recycled by the FIFO "
          f"(keeps {of['fifo_capacity']}, bodies <= {of['max_bodies']}), {of['in_tote_at_end']} in the tote at the end, max {of['max_in_tote']} outfeed bodies at once | "
          f"settle median {of['settle_s_median']} s max {of['settle_s_max']} s, timeouts {of['settle_timeouts']} | forced {of['forced_recycles']}, lost {of['lost']}, outside {of['settled_outside']}", flush=True)
    if m["passed_reached_outfeed"] != m["passed_by_controller"]:
        problems.append(f"accepted packs reaching the outfeed: {m['passed_reached_outfeed']}/{m['passed_by_controller']}")
    if of["lost"] or of["settled_outside"]:
        problems.append(f"outfeed packs lost {of['lost']} / settled outside the tote {of['settled_outside']}")
    if of["forced_recycles"]:
        problems.append(f"outfeed pool guard fired {of['forced_recycles']} time(s) with every outfeed body still moving (raise --pool)")
    # memory: no growth across the run beyond a working-set tolerance (torch: this process; driver: whole GPU)
    a, b = vram["after_setup"], vram["after_run"]
    grow_t = b["torch_alloc_mb"] - a["torch_alloc_mb"]
    grow_g = (b["gpu_used_mb"] - a["gpu_used_mb"]) if (a.get("gpu_used_mb") is not None and b.get("gpu_used_mb") is not None) else 0.0
    print(f"vram: torch alloc {a['torch_alloc_mb']:.0f} -> {b['torch_alloc_mb']:.0f} MB (peak {b['torch_peak_alloc_mb']:.0f}), gpu used {a.get('gpu_used_mb')} -> {b.get('gpu_used_mb')} MB over the run", flush=True)
    if grow_t > 256 or grow_g > 512:
        problems.append(f"device memory grew over the run: torch +{grow_t:.0f} MB, gpu +{grow_g:.0f} MB")
    d = m.get("dashboard")
    if d:
        v = d.get("video")
        print(f"dashboard: {d['display_frames']} frames, 1 per {d['capture_every_steps']} physics steps, playback {d['playback_speed']:.3g}x real time", flush=True)
        if v:
            print(f"VIDEO: {v['path']}  ({v['frames']} frames, {v['duration_s']:.1f} s at {v['fps']} fps, "
                  f"{v['size'][0]}x{v['size'][1]}, {v['bytes'] / 1e6:.1f} MB, {v['backend']})", flush=True)
            if not Path(v["path"]).exists() or v["bytes"] < 10000:
                problems.append(f"video not written: {v}")
    print("DEMO:", "PASS" if not problems else f"FAIL {problems}", flush=True)
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Closed-loop live factory digital twin.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--packs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--pool", type=int, default=16, help=f"dynamic pack pool, {MIN_POOL}..{len(PARK_SLOTS)} (park slots); {BELT_OCCUPANCY} on the belt + "
                                                         f"{REJECT_TRANSIT} reject transit + 1 spare are reserved, the rest may sit in the outfeed (FIFO keeps up to 5 settled)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--lockstep", action="store_true")
    g.add_argument("--realtime", action="store_true")
    ap.add_argument("--record-video", nargs="?", const=str(DEFAULT_VIDEO), default=None,
                    help=f"record the dual-camera dashboard to an MP4 (default {DEFAULT_VIDEO})")
    ap.add_argument("--gui", action="store_true", help="live OpenCV window (space pauses, q/ESC stops)")
    ap.add_argument("--video-fps", type=int, default=VizConfig.fps)
    ap.add_argument("--video-speed", type=float, default=None,
                    help="simulated seconds per playback second (0.25 = 4x slow motion, 1.0 = real time); "
                         "default 0.25 when recording, 1.0 for a live window only")
    ap.add_argument("--video-steps", type=int, default=VizConfig.steps, help="renderer steps per dashboard frame")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    mode = "realtime" if args.realtime else "lockstep"
    speed = args.video_speed if args.video_speed is not None else (VizConfig.speed if args.record_video else 1.0)
    vc = VizConfig(fps=args.video_fps, speed=speed, steps=args.video_steps)
    if args.demo:
        return demo(args.packs, mode, args.seed, args.pool, Path(args.record_video) if args.record_video else None, args.gui, vc)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
