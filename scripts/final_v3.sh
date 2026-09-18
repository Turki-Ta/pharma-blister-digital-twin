#!/usr/bin/env bash
# Outfeed build verification (cell v6/v7: grounded work light, nominal-product outfeed with a
# rolling FIFO stacking tote, pool 16):
#   0. robustness: 30 packs at seed 1 (9 accepted packs, three in transit on a full stack) and
#      50 packs at the default seed - the two cases the review found the v6 pool guard failing
#   1. 20-pack recorded demonstration (the deliverable MP4)
#   2. 30-pack lockstep stress run (pool guard, FIFO recycling, VRAM)
#   3. parity report against the v1 runs (decisions, physical outcomes, counters, leaks)
# Progress goes to reports/final_v3.log.  Step 0 runs first because the 30-pack JSON is keyed by
# pack count only and step 2 must be the default seed for parity.
set -o pipefail
cd "$(dirname "$0")/.."
LOG=reports/final_v3.log
F='FindAppliedAPIPrimDefinition|Could not find UsdPrimDefinition|OMNI_USD|OmniUsdResolver|^#|^$'
G='^DEMO|^VIDEO|^dashboard|^vram|^setup|^outfeed|Traceback|Error:|pool exhausted'
{
  echo "=== final v3 start $(date -Iseconds) ==="
  echo "=== [0a] robustness: 30 packs, seed 1, pool 16 ==="
  uv run python src/simulation/live_factory_twin.py --demo --packs 30 --seed 1 --pool 16 --lockstep 2>&1 | tee reports/twin/demo_lockstep_30_seed1.log | grep -vE "$F" | grep -E "$G"
  echo "=== [0b] robustness: 50 packs, default seed, pool 16 ==="
  uv run python src/simulation/live_factory_twin.py --demo --packs 50 --pool 16 --lockstep 2>&1 | tee reports/twin/demo_lockstep_50.log | grep -vE "$F" | grep -E "$G"
  echo "=== [1] 20-pack recorded demonstration (pool 16) ==="
  uv run python src/simulation/live_factory_twin.py --demo --packs 20 --pool 16 --lockstep --record-video 2>&1 | tee reports/twin/demo_record_20.log | grep -vE "$F" | grep -E "$G"
  echo "=== [2] 30-pack lockstep stress run (pool 16) ==="
  uv run python src/simulation/live_factory_twin.py --demo --packs 30 --pool 16 --lockstep 2>&1 | tee reports/twin/demo_lockstep.log | grep -vE "$F" | grep -E "$G"
  echo "=== [3] parity vs v1 ==="
  uv run python scripts/parity_v2.py 2>&1
  echo "=== leak lines: seed1 $(grep -c 'Leaking step result' reports/twin/demo_lockstep_30_seed1.log) p50 $(grep -c 'Leaking step result' reports/twin/demo_lockstep_50.log) record $(grep -c 'Leaking step result' reports/twin/demo_record_20.log) lockstep $(grep -c 'Leaking step result' reports/twin/demo_lockstep.log) ==="
  echo "=== FINAL DONE $(date -Iseconds) ==="
} > "$LOG" 2>&1
