"""Phase 2 GPU stack VERIFY gate (blackwell_env_runbook.md).

EXPECTED: cuda available True, capability (12, 0), bf16 True, matmul OK,
no 'no kernel image' warnings. Any failure = wrong wheel -> return to Phase 1.
"""
import sys
import warnings

import torch

warnings.simplefilter("always")

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

# --- Gate evaluation (machine-checkable exit code for CI / agent loops) ---
checks = {
    "cuda_available": torch.cuda.is_available(),
    "capability_sm120": torch.cuda.get_device_capability(0) == (12, 0),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "cu130_wheel": "+cu130" in torch.__version__,
    "matmul_finite": torch.isfinite(y).item(),
}
failed = [k for k, v in checks.items() if not v]
print("GATE         :", "PASS" if not failed else f"FAIL {failed}")
sys.exit(0 if not failed else 1)
