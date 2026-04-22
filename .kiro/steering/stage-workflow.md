---
inclusion: auto
description: Detailed workflow for each stage of the neural network builder pipeline.
---

# Stage Workflow

## Stage 0 — Project Init

**Entry Command:** `nnb start`

**Actions:**
1. Generate unique project ID
2. Create `.nnb/` directory structure
3. Initialize `state.json` with `INIT` state
4. Create default steering document
5. Set up logging infrastructure

**Exit Condition:** Directory structure created, state = `SCOPING`

---

## Stage 1 — User Conversation

**Entry State:** `SCOPING`

**Actions:**
1. Open interactive session
2. User describes project in plain language
3. Gemini listens without interrupting
4. Save conversation history after every turn
5. Gemini internally categorizes project type

**Gemini Role:** Interviewer
- Build understanding of domain, goal, constraints
- Identify ambiguities for Stage 2
- Estimate complexity
- Categorize project type (vision, NLP, tabular, etc.)

**Exit Condition:** User indicates they're done describing, state = `SCOPING`

---

## Stage 2 — Scoping Questions

**Entry State:** `SCOPING`

**Actions:**
1. Gemini generates focused questions
2. Ask conversationally, one topic at a time
3. Fold answers into steering document
4. Generate Project Specification
5. Ask user to confirm spec

**Key Questions:**
- Input/output shapes
- Latency vs. accuracy tradeoff
- Pretrained model acceptable?
- Compute budget
- Export requirements (ONNX, TFLite)

**Exit Condition:** User confirms spec, state = `SPEC_CONFIRMED`

**Hard Gate:** User must explicitly confirm before proceeding

---

## Stage 3 — Training Data Requirements

**Entry State:** `SPEC_CONFIRMED`

**Actions:**
1. Gemini generates Data Requirements Report
2. Save to `.nnb/data-requirements.md`
3. Print to terminal
4. Provide checklist via `nnb data status`

**Report Contents:**
- Format (images, CSV, parquet, JSONL, audio)
- Minimum and recommended dataset size
- Required folder/label structure
- Preprocessing handled automatically vs. user responsibility
- Gotchas (class imbalance, resolution, encoding)

**Exit Condition:** Report generated, state = `DATA_REQUIRED`

---

## Stage 4 — Data Validation

**Entry Command:** `nnb data validate --path ./my-data`

**Entry State:** `DATA_REQUIRED`

**Actions:**
1. Take non-destructive sample of data
2. Pass sample to Gemini
3. Gemini checks:
   - Structure matches required format
   - Labels present and consistent
   - No corrupt files
   - Class distribution
   - Size estimate for training time
4. Generate pass/warn/fail report
5. Write data manifest to `.nnb/data-manifest.json`

**Exit Condition:** Data passes validation, state = `DATA_VALIDATED`

**Hard Gate:** Data must pass (not just warn) before proceeding

---

## Stage 5 — Docker Environment Setup

**Entry Command:** `nnb env build`

**Entry State:** `DATA_VALIDATED`

**Actions:**
1. Gemini generates Dockerfile based on spec:
   - Base image (PyTorch CUDA, TensorFlow, JAX)
   - Python version pinned
   - Minimal dependencies
2. Build container
3. Mount data directory (read-only) at `/data`
4. Mount workspace (read-write) at `/workspace`
5. Run health check:
   - GPU/CPU availability
   - CUDA version
   - Framework import
   - Data mount accessible

**Exit Condition:** Container built and healthy, state = `ENV_READY`

**Hard Gate:** Health check must pass

---

## Stage 6 — Code Generation + Mock Run

**Entry Command:** `nnb mock-run`

**Entry State:** `ENV_READY`

**Actions:**

### Code Generation:
1. Gemini generates inside container at `/workspace`:
   - `model.py` — architecture
   - `dataset.py` — data loading
   - `train.py` — training loop
   - `config.yaml` — hyperparameters
   - `requirements.txt` — dependencies
2. Files appear on host via volume mount

### Mock Run:
1. Load tiny synthetic/sampled batch (2-4 samples)
2. Run one forward pass
3. Run one backward pass
4. Run one optimizer step
5. Check:
   - Shapes correct end-to-end
   - Loss is valid (not NaN/Inf)
   - GPU memory footprint

**Gemini Role:** Engineer + Monitor
- Reads tracebacks on failure
- Generates fixes
- Patches files
- Re-runs (up to 3 retries)

**Exit Condition:** Mock run passes, state = `MOCK_PASSED`

**Hard Gate:** Mock run must pass before training

---

## Stage 7 — Training

**Entry Command:** `nnb train`

**Entry State:** `MOCK_PASSED`

**Actions:**
1. Start training as detached container process
2. Stream live view: epoch progress, loss, metrics, ETA
3. Save checkpoints to `/workspace/checkpoints/`
4. Gemini wakes on:
   - Training completion
   - Training crash
   - User runs `nnb check`
5. On completion, Gemini reviews final summary
6. Write Training Report to `.nnb/training-report.md`

**Resilience:**
- Closing terminal doesn't kill training
- `nnb attach <project-id>` to reattach
- Checkpointing enables recovery

**Exit Condition:** Training completes successfully, state = `TRAINING_COMPLETE`

---

## Stage 8 — Inference Setup

**Entry Command:** Automatic after training

**Entry State:** `TRAINING_COMPLETE`

**Actions:**
1. Gemini generates inference artifacts:
   - `inference.py` — standalone inference script
   - `server.py` — optional REST API wrapper
   - Export to ONNX/TFLite if flagged
   - `inference-requirements.txt` — minimal deps
2. Run inference smoke test:
   - Load best checkpoint
   - Run one prediction
   - Verify output shape and type
3. Print final summary

**Exit Condition:** Inference ready, state = `DONE`

---

## State Transitions

```
nnb start          → INIT → SCOPING
(conversation)     → SCOPING (stays)
(confirm spec)     → SPEC_CONFIRMED
(auto)             → DATA_REQUIRED
nnb data validate  → DATA_VALIDATED
nnb env build      → ENV_BUILDING → ENV_READY
nnb mock-run       → MOCK_RUNNING → MOCK_PASSED
nnb train          → TRAINING → TRAINING_COMPLETE
(auto)             → INFERENCE_READY → DONE
```

## Recovery Commands

| Command | Purpose |
|---------|---------|
| `nnb status` | Show current state and next action |
| `nnb resume <id>` | Resume from any state |
| `nnb retry` | Retry failed stage |
| `nnb reset-stage` | Reset current stage (dangerous) |
