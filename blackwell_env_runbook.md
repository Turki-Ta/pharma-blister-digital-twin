# BLACKWELL ENV RUNBOOK — RTX 5090 / 9800X3D / Win11 (AI-to-AI)

**Audience**: an AI coding agent executing in a fresh project folder on THIS machine.
**Contract**: execute phases in order. Every phase ends in a VERIFY gate — run the
check, compare against EXPECTED, and STOP with a report if it mismatches. Do not
improvise substitutions (especially not the torch wheel). Do not ask the user
questions; every decision is pre-made below. Exactly one step (4.1) is a manual
GUI action — print its instructions for the user and continue; do not block on it.

**Hardware ground truth** (verify, don't assume): NVIDIA RTX 5090 32 GB
(Blackwell, compute capability **sm_120 / (12, 0)**), AMD Ryzen 7 9800X3D
(8C/16T, single CCD, 96 MB X3D cache), 64 GB DDR5, Windows 11, NVMe SSD.
Shell is **Windows PowerShell 5.1**: `&&` and `||` DO NOT EXIST — chain with `;`
or separate calls. Default console encoding is cp1252 — Phase 4 fixes this.

**Calibration provenance**: every number marked *(measured)* was benchmarked on
this exact machine in July 2026 during a production training project. Treat them
as calibration references, not folklore.

---

## Phase 0 — Preconditions

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```
**VERIFY**: name contains `RTX 5090`; driver ≥ `570` (this box: 610.62). If the
driver is older than 570, STOP: Blackwell needs a modern driver before anything else.

```powershell
Get-Command uv -ErrorAction SilentlyContinue
```
If `uv` is missing, install it (it is the mandated package manager — verified
fast and correct on this box; do NOT use conda, whose CUDA packaging lags
Blackwell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Disk check: require ≥ 50 GB free on the project drive before installing
(`Get-PSDrive C`). Torch cu130 + CV stack ≈ 10 GB; leave room for data/checkpoints.

## Phase 1 — Project scaffold

From the new project folder root:

```powershell
uv init --python 3.12
```

Replace the generated `pyproject.toml` dependency section with the following
**verified-known-good** Blackwell pinning (the explicit index is what makes
sm_120 work — the default PyPI wheel does NOT support this GPU):

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "torch==2.10.0+cu130",
    "torchvision==0.25.0+cu130",
    "numpy>=2.1",
    "pillow>=10.4",
    "tqdm>=4.66",
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch" }
torchvision = { index = "pytorch" }
```

**Version-drift clause**: if `2.10.0+cu130` is no longer served, use the newest
stable `+cu13x` (or later CUDA line) from the same index and re-run all gates.
NEVER fall back to a CPU wheel, a `cu126`/older wheel ("no kernel image for
sm_120" at runtime), or a conda build.

```powershell
uv sync
```

Add task-appropriate extras only as needed (all verified compatible on this box):
`opencv-python`, `scikit-image`, `scikit-learn`, `pandas`, `pyarrow`,
`transformers`, `diffusers`, `accelerate`, `timm`, `albumentations`, `kornia`,
`einops`, `torchmetrics[image]`, `imagehash`, `pyvips[binary]` (fast
shrink-on-load JPEG decode), `bitsandbytes`, `peft`.

## Phase 2 — GPU stack VERIFY gate (mandatory before any other work)

Create and run `env_verify.py`:

```python
import torch

print("torch        :", torch.__version__)
print("cuda build   :", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device       :", torch.cuda.get_device_name(0))
print("capability   :", torch.cuda.get_device_capability(0))
print("bf16         :", torch.cuda.is_bf16_supported())
print("cudnn        :", torch.backends.cudnn.version())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().mean()
torch.cuda.synchronize()
print("bf16 matmul  : OK", float(y))
```

**EXPECTED**: `cuda available: True`, `capability: (12, 0)`, `bf16: True`,
matmul OK, no warnings about missing kernel images. Any failure here = wrong
wheel; return to Phase 1. Note the cuDNN version: `2.10.0+cu130` bundles cuDNN
**9.12**, which has a known conv/AMP regression lineage (pytorch#167242). If a
conv-heavy workload benchmarks slow, `uv add nvidia-cudnn-cu13` (≥ 9.15) and
A/B the same benchmark before and after.

## Phase 3 — Machine-level environment variables

Set once, user-scoped (new terminals inherit; the current one must re-launch):

```powershell
setx PYTHONUTF8 1
setx HF_HOME "C:\hf-cache"
```

- `PYTHONUTF8=1` is NOT optional. The cp1252 console breaks Python prints of
  Unicode (dashes, Greek letters, filenames) with `UnicodeEncodeError` mid-run —
  this repeatedly interrupted real jobs on this box before the fix.
- `HF_HOME` on a short NVMe path keeps multi-GB model caches off `C:\Users\...`
  long paths and makes them shared across projects.
- Do NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on Windows —
  support is platform-limited; rely on the seatbelt in Phase 4 instead.

## Phase 4 — WDDM spill hazard (the single most important section)

**The hazard** *(measured)*: on Windows WDDM, CUDA over-allocation does **not**
raise OOM. The driver silently spills to system RAM and the job keeps "running"
at a catastrophic slowdown. Measured on this box with a 51M-param GAN fine-tune
at 512²: batch 8 → 22.2 GB, 437 ms/step (healthy); batch 12 → 32.8 GB,
5,843 ms/step (spilling); batch 16 → 43.4 GB, 40,446 ms/step (**92× slower —
a planned 24 h run would take ~3 months, with zero errors reported**).
A SegFormer-B3 at 1024²: batch 2 healthy (18.2 GB), batch 4 spills (35.7 GB),
batch 8 hard-OOMs. "It runs" proves nothing on this platform.

**4.1 Driver policy (manual GUI step — print for the user, then continue).**
NVIDIA Control Panel → Manage 3D Settings → *CUDA - Sysmem Fallback Policy* →
**Prefer No Sysmem Fallback** (globally, or per-program for the project's
`.venv\Scripts\python.exe`). This converts silent spill into an honest OOM
error. There is no environment variable for this; it is a one-time driver
setting.

**4.2 Programmatic seatbelt (mandatory in every training entry point).**
Works regardless of whether 4.1 was done:

```python
import torch
torch.cuda.set_per_process_memory_fraction(0.94)  # ~30.1 GB cap -> OOM, not spill
```

**4.3 Batch-envelope protocol (mandatory before any long run).**
Never choose a batch size by "it didn't crash". Sweep and measure:

```python
import time, torch
for batch in (2, 4, 8, 12, 16):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        run_one_full_training_step(batch)   # fwd + bwd + optimizer, real losses
        torch.cuda.synchronize()
        t0 = time.perf_counter(); run_one_full_training_step(batch)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"batch {batch}: {dt*1000:.0f} ms  peak {peak:.1f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"batch {batch}: OOM"); break
```

**Acceptance rule**: reject any configuration with peak > **30 GB** or a
superlinear step-time jump between adjacent batch sizes (that jump IS the
spill). Prefer the largest batch ≤ 30 GB; recover effective batch with gradient
accumulation, which is nearly free.

**4.4 Long-run OS settings** (commands are exact):
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```
GPU load does NOT count as activity for Windows sleep policy — an idle-timeout
sleep silently freezes multi-day runs *(observed hazard class on this box)*.
Additionally, long-run scripts should hold the system awake defensively:

```python
import ctypes, sys
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
def keep_awake(on: bool) -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
```

Before launching an unattended run, check for a pending Windows Update reboot
(it will restart the machine mid-run; code cannot block it — surface it):
```powershell
Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"
```
`True` → tell the user to reboot before launching.

## Phase 5 — Precision & kernel configuration (put in every training script)

```python
import torch

def configure_backend() -> None:
    # TF32 via the NEW API only; do not mix with legacy allow_tf32 flags.
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:                       # older torch fallback
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.94)
```

Mandates:
- **bf16 autocast, no GradScaler** (`torch.autocast("cuda", dtype=torch.bfloat16)`).
  Blackwell bf16 is first-class; fp16+scaler is legacy on this box.
- **channels_last** for conv nets: model AND input batches.
- **Attention**: use `F.scaled_dot_product_attention`. Do NOT `pip install
  flash-attn` — no Windows sm_120 wheels; SDPA's cuDNN backend is the fast path
  on Blackwell.
- **FFT gotcha**: `torch.fft.rfftn` (and friends) do not run in bf16 on CUDA —
  wrap spectral blocks in an fp32 island (`with torch.autocast("cuda",
  enabled=False):` + `.float()`), then cast back.
- **Loss/metric math in fp32** even under autocast.
- **GAN/R1 note** *(measured)*: gradient-penalty double-backward is stable run
  in fp32 outside the autocast region; keep discriminator steps out of bf16.

`torch.compile` (OPTIONAL — off by default): Inductor-on-CUDA works on Windows
only via community **`triton-windows`** wheels with a STRICT torch↔triton
pairing (torch 2.10 ↔ triton 3.6: `uv add "triton-windows<3.7"`), MSVC Build
Tools installed, and `setx TORCHINDUCTOR_CACHE_DIR "C:\ticache"` (short path —
MAX_PATH failures are real). Always guard with try/except and an eager
fallback; never make compile a correctness dependency.

## Phase 6 — DataLoader & CPU discipline (9800X3D)

The 9800X3D is a single-CCD part — no cross-CCD affinity games needed; SMT on.
Windows uses **spawn** for workers, which drives every rule below:

| Setting | Mandated value | Why *(measured where noted)* |
|---|---|---|
| `num_workers` | **8** (tune 6–10; never 16) | 8 workers ≈ 132 samples/s on a 60 ms/sample CV pipeline *(measured)*; more adds spawn/RAM overhead and starves the main process |
| `persistent_workers` | `True` (mandatory when workers > 0) | spawn re-import costs seconds per epoch otherwise |
| `pin_memory` | `True` (+ `.to(device, non_blocking=True)`) | measurable H2D win |
| `prefetch_factor` | 4 | keeps queue full through decode variance |
| `drop_last` | `True` for training | stable step shapes (also helps compile) |
| Dataset `__init__` | paths/scalars only, picklable | spawn pickles the dataset into every worker |
| Entry point | `if __name__ == "__main__":` guard | REQUIRED on Windows or workers recursively re-launch |
| OpenCV in workers | `cv2.setNumThreads(0)` inside worker code | prevents 8 workers × N threads oversubscription |

**Feed-rate rule**: loader capacity (samples/s) must exceed GPU consumption
(batch / step-time) by ≥ 2×; measure both, don't guess.

**One-time preprocessing principle**: the hot loop must never decode original
assets. Preprocess once (resize/re-encode; `pyvips` shrink-on-load makes this
pass ~10× faster for big JPEGs) and train from the small derived set. On this
box that turned an 88 GB corpus into an 8.3 GB working set and removed decode
from the critical path entirely *(measured)*.

## Phase 7 — `.vscode/settings.json` (create exactly this)

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true,
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.extraPaths": ["src"],
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": { "source.organizeImports": "explicit" }
  },
  "editor.rulers": [100],
  "files.exclude": { "**/__pycache__": true, "**/.ipynb_checkpoints": true },
  "files.watcherExclude": {
    "**/.venv/**": true,
    "**/data/**": true,
    "**/models/**": true,
    "**/output/**": true,
    "**/reports/**": true,
    "**/__pycache__/**": true
  },
  "search.exclude": {
    "**/.venv": true,
    "**/data": true,
    "**/models": true,
    "**/output": true,
    "**/*.parquet": true
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "notebook.output.scrolling": true,
  "terminal.integrated.env.windows": {
    "PYTHONUTF8": "1",
    "PYTHONUNBUFFERED": "1"
  }
}
```

The watcher/search excludes are load-bearing: VS Code file-watching a
multi-GB `data/` directory degrades the whole IDE and burns CPU the training
run needs.

## Phase 8 — Long-run hardening (mandatory patterns for any multi-hour job)

1. **Atomic checkpoints** — a crash mid-`torch.save` leaves a truncated file
   that bricks auto-resume *(fault-injection verified on this box)*:
   ```python
   tmp = path.with_suffix(path.suffix + ".tmp")
   torch.save(payload, tmp)
   os.replace(tmp, path)          # atomic on NTFS
   ```
2. **Resume fallback chain** — try `latest.pt`, then rotated older checkpoints,
   newest first; only hard-fail if none load. Rotate checkpoints (keep ~3):
   a 1 GB payload every 15 min is ~95 GB/day unrotated.
3. **Non-finite guard** — check `torch.isfinite(loss)` (and the value returned
   by `clip_grad_norm_`) BEFORE `optimizer.step()`; skip the step on failure,
   count a streak, abort with checkpoints intact after ~50 consecutive skips.
   bf16 GAN training without this can silently NaN for days.
4. **Keep-awake** (Phase 4.4 snippet) around the run; release in `finally`.
5. **Isolate smoke artifacts** — quick-test runs must checkpoint to a separate
   directory (`ckpt_smoke/`), or auto-resume will pick up garbage weights.

## Appendix — Failure signatures → fixes

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available` | non-cu13x wheel on sm_120 | Phase 1 index pinning |
| Step time jumps 10–90×, no error | WDDM sysmem spill | Phase 4 (policy + seatbelt + envelope) |
| `UnicodeEncodeError: 'charmap' codec...` | cp1252 console | `PYTHONUTF8=1` (Phase 3) |
| `RuntimeError: ... rfftn ... BFloat16` | FFT under bf16 autocast | fp32 island (Phase 5) |
| DataLoader hangs / spawns endlessly | missing `__main__` guard | Phase 6 |
| Workers slow, CPU 100% | OpenCV thread oversubscription | `cv2.setNumThreads(0)` in workers |
| Resume crashes on ckpt load | truncated checkpoint (non-atomic save) | Phase 8.1–8.2 |
| Machine froze training overnight | Windows sleep (GPU ≠ activity) | Phase 4.4 |
| `torch.compile` errors on Windows | missing/mismatched triton-windows or MSVC | Phase 5 pairing, or disable |
| Conv workloads oddly slow | bundled cuDNN 9.12 regression | upgrade `nvidia-cudnn-cu13`, A/B bench |

**END OF RUNBOOK** — report to the user: phases completed, gate outputs, and
any deviation taken under the version-drift clause.
