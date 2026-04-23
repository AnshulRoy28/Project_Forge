#!/usr/bin/env python3
"""Mock training runner — validates generated code end-to-end."""

import sys
import os
import traceback

os.chdir("/workspace")
sys.path.insert(0, "/workspace")

print("=" * 60)
print("MOCK TRAINING RUN — Code Validation")
print("=" * 60)

errors = []

# Step 1: Import checks
print("\n--- Step 1: Import Checks ---")

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    errors.append(f"PyTorch import: {e}")


try:
    from model import *
    print("✓ model.py imports successfully")
except Exception as e:
    print(f"✗ model.py import failed: {{e}}")
    traceback.print_exc()
    errors.append(f"model.py import: {{e}}")



try:
    from dataset import *
    print("✓ dataset.py imports successfully")
except Exception as e:
    print(f"✗ dataset.py import failed: {{e}}")
    traceback.print_exc()
    errors.append(f"dataset.py import: {{e}}")


# Step 2: Run train.py in mock mode (1 batch, 1 epoch)
print("\n--- Step 2: Mock Training Pass ---")

try:
    # Set environment variable to signal mock mode
    os.environ["NNB_MOCK_RUN"] = "1"
    os.environ["NNB_MAX_BATCHES"] = "2"
    os.environ["NNB_MAX_EPOCHS"] = "1"

    # Execute train.py
    exec(open("/workspace/train.py").read())
    print("\n✓ Training script executed successfully")
except SystemExit as e:
    if e.code == 0:
        print("\n✓ Training script completed (exit 0)")
    else:
        print(f"\n✗ Training script exited with code {e.code}")
        errors.append(f"train.py exit code: {e.code}")
except Exception as e:
    print(f"\n✗ Training script failed: {e}")
    traceback.print_exc()
    errors.append(f"train.py execution: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"✗ MOCK RUN FAILED — {len(errors)} error(s)")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("✓ MOCK RUN PASSED — All checks green!")
    print("=" * 60)
    sys.exit(0)
