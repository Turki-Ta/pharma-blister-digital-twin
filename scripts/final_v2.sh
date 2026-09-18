#!/usr/bin/env bash
# Post-retrain verification for asset v2 / engine v2 (non-interactive part):
#   1. drift gate re-run with the NEW engine (50 fresh frames, same seed as the pre-retrain gate)
#   2. 20-pack recorded demonstration (the deliverable MP4)
#   3. 30-pack lockstep demonstration (count parity with the v1 baseline)
#   4. parity report against the v1 runs
# Progress goes to reports/final_v2.log; the GUI run is done separately (it opens a window).
set -o pipefail
cd "$(dirname "$0")/.."
LOG=reports/final_v2.log
F='FindAppliedAPIPrimDefinition|Could not find UsdPrimDefinition|OMNI_USD|^#|^$'
{
  echo "=== final start $(date -Iseconds) ==="
  echo "=== [1] drift gate with engine v2 ==="
  uv run python src/inference/drift_check.py --frames 50 2>&1 | grep -vE "$F|^frame " ; echo "drift exit ${PIPESTATUS[0]}"
  cp reports/drift_check.json reports/drift_check_v2_post_retrain.json
  echo "=== [2] 20-pack recorded demonstration ==="
  uv run python src/simulation/live_factory_twin.py --demo --packs 20 --lockstep --record-video 2>&1 | tee reports/twin/demo_record_20.log | grep -vE "$F" | grep -E "^DEMO|^VIDEO|^dashboard|^vram|^setup|Traceback|Error:"
  echo "=== [3] 30-pack lockstep demonstration ==="
  uv run python src/simulation/live_factory_twin.py --demo --packs 30 --lockstep 2>&1 | tee reports/twin/demo_lockstep.log | grep -vE "$F" | grep -E "^DEMO|^vram|^setup|Traceback|Error:"
  echo "=== [4] parity vs v1 ==="
  uv run python scripts/parity_v2.py 2>&1
  echo "=== FINAL DONE $(date -Iseconds) ==="
} > "$LOG" 2>&1
