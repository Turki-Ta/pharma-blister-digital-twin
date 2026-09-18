"""Gate 0.5 - simulation stack VERIFY gate for Set A (ovrtx 0.4.1 / ovstage 0.1.1 / ovphysx 0.5.11).

Asserts, on this machine, that the three libraries import at the pinned versions, that a
procedurally authored USD stage opens, that the renderer delivers RGB + semantic frames, that
PhysX steps a rigid pack onto the belt, and that a physics pose mirrored into the shared stage
moves the rendered pack (the live-twin data path).  Exit code is non-zero on any failure.

Run:  uv run python env_verify_sim.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cp1252 consoles (runbook Phase 3)

W, H = 1280, 720
H_CAM = 0.2 * 50.0 / 36.0          # 200 mm FOV along the belt at 50 mm / 36 mm
GSD = 0.2 / W                      # m per px at the belt plane
PACK_SCALE = (0.09, 0.045, 0.01)

USDA = f'''#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 1
    defaultPrim = "World"
)
def Xform "World"
{{
    def PhysicsScene "physicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}
    def DomeLight "DomeLight"
    {{
        float inputs:intensity = 600
        color3f inputs:color = (1, 1, 1)
    }}
    def Camera "Camera"
    {{
        float focalLength = 0.5
        float horizontalAperture = 0.36
        float verticalAperture = 0.2025
        float2 clippingRange = (0.01, 100)
        token projection = "perspective"
        double3 xformOp:translate = (0, 0, {H_CAM:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def Cube "Belt" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        double size = 1
        color3f[] primvars:displayColor = [(0.13, 0.14, 0.15)]
        double3 xformOp:translate = (0, 0, -0.01)
        float3 xformOp:scale = (1.2, 0.3, 0.02)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
    def Cube "Pack" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI", "SemanticsAPI:class"]
    )
    {{
        string semantic:class:params:semanticData = "pack"
        string semantic:class:params:semanticType = "class"
        float physics:mass = 0.012
        double size = 1
        color3f[] primvars:displayColor = [(0.82, 0.83, 0.86)]
        double3 xformOp:translate = (0, 0, 0.05)
        float3 xformOp:scale = ({PACK_SCALE[0]}, {PACK_SCALE[1]}, {PACK_SCALE[2]})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
}}
def Scope "Render"
{{
    def RenderProduct "Camera"
    {{
        rel camera = </World/Camera>
        rel orderedVars = [</Render/Camera/LdrColor>, </Render/Camera/SemanticSegmentation>, </Render/Camera/SemanticIdMap>]
        int2 resolution = ({W}, {H})
        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}
        def RenderVar "SemanticSegmentation"
        {{
            string sourceName = "SemanticSegmentation"
        }}
        def RenderVar "SemanticIdMap"
        {{
            string sourceName = "SemanticIdMap"
        }}
    }}
}}
'''

results: dict[str, tuple[bool, str]] = {}


def record(name: str, ok: bool, detail: str) -> None:
    results[name] = (ok, detail)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<28s} {detail}", flush=True)


def pack_centroid(frame) -> tuple[float, float] | None:
    from simulation.ovx_runtime import masks_by_label

    m = masks_by_label(frame.seg, frame.labels).get("pack")
    if m is None or not m.any():
        return None
    ys, xs = np.where(m)
    return float(xs.mean()), float(ys.mean())


def main() -> int:
    import ovstage

    from simulation.ovx_runtime import Scene, check_versions, quat_to_rot, usd_matrix

    try:
        found = check_versions()
        record("versions (Set A)", True, str(found))
    except Exception as e:  # noqa: BLE001
        record("versions (Set A)", False, str(e))
        return 1

    physx = None
    scene = None
    pstage = None
    try:
        t0 = time.perf_counter()
        scene = Scene("blister.gate")
        record("renderer init", True, f"{time.perf_counter() - t0:.1f} s")

        # Two stages from one USDA: the renderer's stage is populated with the RENDERING domain and
        # PhysX gets its own PHYSICS-domain stage.  (On this build a combined PHYSICS|RENDERING
        # population drops the SemanticsAPI labels; and the renderer never consumes physics poses
        # by itself, so the twin mirrors poses explicitly anyway - this gate exercises that path.)
        t0 = time.perf_counter()
        scene.load_usda(USDA, domains=ovstage.PopulationDomain.RENDERING)
        pstage = ovstage.Stage("blister.gate.physics")
        ovstage.population.open_usd_from_string(pstage, USDA, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
        pstage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        record("open USD stages", True, f"{(time.perf_counter() - t0) * 1000:.0f} ms (render ordinal {scene.ordinal}, physics ordinal 1)")

        # Prime the pack's runtime transform: the first omni:xform write after population is not
        # visible in the very next frame on this build, so write the authored pose once and step.
        scene.write_xforms(["/World/Pack"], usd_matrix(np.eye(3), (0.0, 0.0, 0.05), scale=PACK_SCALE)[None])
        scene.seal()
        scene.render("/Render/Camera", steps=1, want_seg=False)

        # --- render ---------------------------------------------------------------
        times = []
        frame = None
        for i in range(6):
            t = time.perf_counter()
            frame = scene.render("/Render/Camera", steps=1, width=W, height=H)
            times.append(time.perf_counter() - t)
        steady = np.array(times[1:]) * 1000
        mean_rgb = float(frame.rgb.mean())
        c0 = pack_centroid(frame)
        record("render frame", mean_rgb > 20 and frame.rgba.shape == (H, W, 4), f"{frame.rgba.shape} mean RGB {mean_rgb:.1f}, first {times[0] * 1000:.0f} ms, steady median {np.median(steady):.1f} ms ({1000 / np.median(steady):.0f} fps)")
        record("semantic labels", c0 is not None, f"labels {sorted(set(v.strip() for v in frame.labels.values()))}, pack centroid {c0}")

        # --- physics ---------------------------------------------------------------
        from ovphysx import PhysX, TensorType

        physx = PhysX()
        physx.attach_ovstage(pstage, read_ordinal=1)
        pose_b = physx.create_tensor_binding(pattern="/World/Pack", tensor_type=TensorType.RIGID_BODY_POSE)
        pose = np.zeros((1, 7), np.float32)
        pose_b.read(pose)
        z_start = float(pose[0, 2])
        dt = 1.0 / 240.0
        t = time.perf_counter()
        for _ in range(240):
            physx.step(dt)
        ms_step = (time.perf_counter() - t) * 1000 / 240
        pose_b.read(pose)
        z_rest = float(pose[0, 2])
        ok = np.isfinite(pose).all() and z_start > 0.04 and abs(z_rest - PACK_SCALE[2] / 2) < 0.003
        record("physics step (gravity)", bool(ok), f"z {z_start:.4f} -> {z_rest:.4f} m (expected ~{PACK_SCALE[2] / 2:.4f}), {ms_step:.2f} ms/step @240 Hz")

        # rejector kick: 2 N for one step at the side face -> lateral motion
        wr_b = physx.create_tensor_binding(pattern="/World/Pack", tensor_type=TensorType.RIGID_BODY_WRENCH)
        wr = np.array([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, float(pose[0, 0]), float(pose[0, 1]) - PACK_SCALE[1] / 2, float(pose[0, 2])]], np.float32)
        wr_b.write(wr)
        for _ in range(121):
            physx.step(dt)
        pose_b.read(pose)
        dy = float(pose[0, 1])
        record("physics wrench (rejector)", dy > 0.02, f"lateral displacement {dy * 1000:.1f} mm after 0.5 s")

        # --- mirror physics pose into the shared stage and re-render ----------------
        R = quat_to_rot(pose[0, 3:7])
        M = usd_matrix(R, pose[0, 0:3], scale=PACK_SCALE)
        scene.write_xforms(["/World/Pack"], M[None])
        scene.seal()
        frame2 = scene.render("/Render/Camera", steps=2, width=W, height=H)
        c1 = pack_centroid(frame2)
        if c0 and c1:
            # world +Y is image up, so a +dy displacement moves the centroid to a smaller row
            exp_rows = -dy / GSD
            got_rows = c1[1] - c0[1]
            ok = abs(got_rows - exp_rows) < 0.25 * abs(exp_rows) + 8
            record("pose mirror -> render", bool(ok), f"centroid row shift {got_rows:+.1f} px, expected {exp_rows:+.1f} px")
        else:
            record("pose mirror -> render", False, f"centroids {c0} -> {c1}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        record("exception", False, f"{type(e).__name__}: {e}")
    finally:
        try:
            if physx is not None:
                physx.detach_ovstage()
                physx.release()
            if pstage is not None:
                pstage.destroy()
        except Exception as e:  # noqa: BLE001
            print("physx cleanup:", e)
        try:
            if scene is not None:
                scene.close()
        except Exception as e:  # noqa: BLE001
            print("scene cleanup:", e)

    failed = [k for k, (ok, _) in results.items() if not ok]
    print("GATE 0.5      :", "PASS" if not failed else f"FAIL {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
