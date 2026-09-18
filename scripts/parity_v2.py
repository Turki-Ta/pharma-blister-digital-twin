"""Parity of the asset v2 / engine v2 twin runs against the v1 baseline runs.

The defect schedule is a pure function of (seed, pack_id) and does not depend on the asset, so
the v1 and v2 runs at the same seed must produce the same verdicts, ejections and confirmations
pack for pack.  Pixel-level identity is not expected (different asset, different engine): what is
asserted is decision parity, physical parity and clean counters.

    uv run python scripts/parity_v2.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TWIN = ROOT / "reports" / "twin"
KEYS = ("scheduled", "verdict", "action", "physical")
COUNTS = ("rejected_by_controller", "rejected_ejected_confirmed", "passed_by_controller", "passed_without_deviation",
          "scheduled_defective", "scheduled_defective_ejected", "scheduled_nominal", "scheduled_nominal_passed_clean",
          "verdict_matches_scheduled", "kicks", "bin_confirmed", "lost", "confirmation_events", "frames_pushed")
CTL = ("frames", "inferences", "overflow", "dropped_frames", "timeouts", "late_actuations", "watchdog_events")


def leak_lines(log: Path) -> int | None:
    if not log.exists():
        return None
    return sum(1 for ln in log.read_text(encoding="utf-8", errors="replace").splitlines() if "Leaking step result" in ln)


def compare(name: str, log: Path | None) -> list[str]:
    v1 = json.loads((TWIN / "v1" / name).read_text(encoding="utf-8"))
    v2 = json.loads((TWIN / name).read_text(encoding="utf-8"))
    problems = []
    rows = [(a["pack_id"], {k: (a[k], b[k]) for k in KEYS if a[k] != b[k]}) for a, b in zip(v1["rows"], v2["rows"]) if any(a[k] != b[k] for k in KEYS)]
    if len(v1["rows"]) != len(v2["rows"]):
        problems.append(f"{name}: row count {len(v1['rows'])} vs {len(v2['rows'])}")
    if rows:
        problems.append(f"{name}: per-pack decision/physical differences {rows}")
    m1, m2 = v1["metrics"], v2["metrics"]
    for k in COUNTS:
        if m1.get(k) != m2.get(k):
            problems.append(f"{name}: {k} v1={m1.get(k)} v2={m2.get(k)}")
    c1, c2 = m1["controller"], m2["controller"]
    for k in CTL:
        if c2.get(k) and k not in ("frames", "inferences"):
            problems.append(f"{name}: controller {k} = {c2.get(k)} (v1 {c1.get(k)})")
        elif k in ("frames", "inferences") and c1.get(k) != c2.get(k):
            problems.append(f"{name}: controller {k} v1={c1.get(k)} v2={c2.get(k)}")
    if c2.get("errors"):
        problems.append(f"{name}: controller errors {c2['errors']}")
    if log is not None:
        n = leak_lines(log)
        if n:
            problems.append(f"{name}: {n} renderer leak lines in {log.name}")
    vr = m2.get("vram")
    if vr and vr.get("after_setup") and vr.get("after_run"):
        gt = vr["after_run"]["torch_alloc_mb"] - vr["after_setup"]["torch_alloc_mb"]
        gg = (vr["after_run"].get("gpu_used_mb") or 0) - (vr["after_setup"].get("gpu_used_mb") or 0)
        if gt > 256 or gg > 512:
            problems.append(f"{name}: device memory grew torch +{gt:.0f} MB, gpu +{gg:.0f} MB")
    lat = [r["verdict_ms"] for r in v2["rows"] if r["verdict_ms"] is not None]
    print(f"{name}: packs {len(v2['rows'])} | verdict==scheduled {m2['verdict_matches_scheduled']}/{len(v2['rows'])} (v1 {m1['verdict_matches_scheduled']}) | "
          f"rejected {m2['rejected_by_controller']} ejected+confirmed {m2['rejected_ejected_confirmed']} | passed clean {m2['passed_without_deviation']}/{m2['passed_by_controller']} | "
          f"verdict ms median {m2['controller']['verdict_ms_median']} (v1 {m1['controller']['verdict_ms_median']}) | rtf {m2['rtf']} (v1 {m1['rtf']}) | leak lines {leak_lines(log) if log else 'n/a'}")
    return problems


def main() -> int:
    problems = []
    problems += compare("twin_lockstep_20.json", TWIN / "demo_record_20.log")
    problems += compare("twin_lockstep_30.json", TWIN / "demo_lockstep.log")
    if (TWIN / "twin_realtime_12.json").exists():
        v2 = json.loads((TWIN / "twin_realtime_12.json").read_text(encoding="utf-8"))["metrics"]
        print(f"twin_realtime_12.json: verdict==scheduled {v2['verdict_matches_scheduled']}/12 | rtf {v2['rtf']} | timeouts {v2['controller']['timeouts']} (first-inference contention, see report)")
    v2 = json.loads((TWIN / "twin_lockstep_20.json").read_text(encoding="utf-8"))["metrics"]
    vid = (v2.get("dashboard") or {}).get("video")
    if not vid or not Path(vid["path"]).exists() or vid["bytes"] < 10000:
        problems.append(f"video missing or empty: {vid}")
    else:
        print(f"video: {vid['path']} ({vid['frames']} frames, {vid['duration_s']} s, {vid['bytes'] / 1e6:.1f} MB, {vid['backend']})")
    print("PARITY:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("  -", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
