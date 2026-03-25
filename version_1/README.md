# 🔬 Parity Agent — Developer Guide

> **An Autonomous Cross-Platform ML Parity Agent**
> The first autonomous agent capable of maintaining behavioral invariance of machine learning models across heterogeneous deployment environments.

---

## Table of Contents

- [What Is This?](#what-is-this)
- [The Problem It Solves](#the-problem-it-solves)
- [How It Works (Big Picture)](#how-it-works-big-picture)
- [Folder Structure](#folder-structure)
- [File-by-File Reference](#file-by-file-reference)
- [The Autonomous Control Loop](#the-autonomous-control-loop)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Key Concepts](#key-concepts)
- [FAQ](#faq)

---

## What Is This?

This is a **Python-based autonomous agent** that acts as a **"debugger for ML deployments"**.

Imagine you trained a YOLOv8 object detection model. It works great on your server (using PyTorch). You export it to TFLite and run it on a phone (using Flutter). But now the phone gives **different results** — fewer detections, wrong bounding boxes, different confidence scores.

**This agent automatically finds out WHY and fixes it.**

It does this through three capabilities:
1. **Self-Observing** — Instruments both pipelines and records everything
2. **Self-Diagnosing** — Finds exactly where the divergence comes from
3. **Self-Correcting** — Runs experiments and fixes the configuration

---

## The Problem It Solves

When you deploy the same ML model to different platforms, the **results differ** even though the **model weights are identical**. This happens because of differences in:

| Factor | Example |
|--------|---------|
| **Normalization** | Server divides by 255, phone divides by 127.5 |
| **Resize method** | Server uses bilinear, phone uses nearest-neighbor |
| **Channel order** | Server expects BGR, phone sends RGB |
| **NMS threshold** | Server uses 0.45, phone uses 0.5 |
| **Sigmoid** | Server applies sigmoid internally, phone may not |
| **Padding** | Server pads with black, phone pads with gray |

These seem like small things, but they cause **real detection failures** in production.

**The Parity Agent finds and fixes all of these automatically.**

---

## How It Works (Big Picture)

The agent runs a loop — think of it like a scientific experiment on repeat:

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE AUTONOMOUS LOOP                          │
│                                                                 │
│   ┌───────┐    ┌──────┐    ┌─────────┐    ┌───────────────┐   │
│   │ TRACE │───▶│ DIFF │───▶│ PROFILE │───▶│  HYPOTHESIZE  │   │
│   └───────┘    └──────┘    └─────────┘    └──────┬────────┘   │
│       ▲                                          │             │
│       │        ┌───────┐    ┌──────────┐         │             │
│       └────────│ ALIGN │◀───│  ABLATE  │◀────────┘             │
│                └───────┘    └──────────┘                       │
│                                                                 │
│   Repeats until parity_loss < threshold (convergence)          │
└─────────────────────────────────────────────────────────────────┘
```

In plain English:
1. **TRACE** — Run the same image through both pipelines, record everything
2. **DIFF** — Compare the recordings, compute how different they are (parity loss)
3. **PROFILE** — Figure out which pipeline stage causes the most difference
4. **HYPOTHESIZE** — Guess what configuration change would fix it
5. **ABLATE** — Test that guess by changing one thing at a time
6. **ALIGN** — Keep the best configuration, go back to step 1

---

## Folder Structure

```
parity_agent/
│
├── README.md                  ← You are here
├── __init__.py                ← Package marker (makes this a Python package)
├── config.yaml                ← All settings in one place
├── requirements-agent.txt     ← Python dependencies
├── run_agent.py               ← MAIN ENTRY POINT — start here to run
│
├── trace/                     ← PHASE 2: "The Eyes" — Making pipelines observable
│   ├── __init__.py
│   ├── schema.py              ← Data models (what a trace looks like)
│   ├── online_tracer.py       ← Records the server/PyTorch pipeline
│   ├── offline_tracer.py      ← Records the phone/TFLite pipeline
│   └── storage.py             ← Saves/loads traces to disk
│
├── diff/                      ← PHASE 3: "The Brain" — Measuring the gap
│   ├── __init__.py
│   ├── metrics.py             ← 5 ways to measure divergence
│   ├── parity_loss.py         ← Combines all 5 into one number
│   └── report.py              ← Pretty-prints the results
│
├── alignment/                 ← PHASE 4: "The Hands" — Tools to fix things
│   ├── __init__.py
│   ├── parameters.py          ← What knobs the agent can turn
│   └── experiment_runner.py   ← Runs controlled experiments
│
├── agents/                    ← PHASE 5: "The Intelligence" — Autonomous reasoning
│   ├── __init__.py
│   ├── profiler.py            ← Finds the biggest problem
│   ├── hypothesis.py          ← Guesses the cause
│   ├── ablation.py            ← Tests the guesses
│   └── alignment.py           ← Keeps the best fix
│
├── utils/                     ← Shared helper functions
│   ├── __init__.py
│   └── image_loader.py        ← Finds test images on disk
│
└── tests/                     ← Unit tests
    └── __init__.py
```

---

## File-by-File Reference

### `run_agent.py` — The Main Entry Point

**What it does:** This is where everything starts. It loads the config, discovers test images, and orchestrates the entire pipeline.

**Key functions:**
- `main()` — Parses command-line arguments, picks which mode to run
- `run_trace()` — Phase 2: generates Golden Traces for all test images
- `run_diff()` — Phase 3: computes diff metrics and generates report
- `run_agent_loop()` — Phase 5: the full autonomous while-loop

**How the control loop works:**
```python
# This is the actual logic (simplified):
while parity_loss > threshold:
    # 1. Run both pipelines on the same images
    online_traces = trace_online(images)
    offline_traces = trace_offline(images, current_config)
    
    # 2. Measure how different the results are
    parity_loss = compute_diff(online_traces, offline_traces)
    
    # 3. Find what's causing the biggest difference
    cause = profiler.analyze(diff_report)
    
    # 4. Guess what config change would fix it
    hypotheses = hypothesis_agent.generate(cause)
    
    # 5. Test each guess in a controlled experiment
    best_config = ablation_agent.test(hypotheses)
    
    # 6. Apply the best fix, loop back
    alignment_agent.apply(best_config)
```

---

### `config.yaml` — Configuration

**What it does:** Central settings file. Everything the agent needs to know is here.

**Sections:**
- **`paths`** — Where to find the model, labels, test images, and output
- **`model`** — Model info (input size, class names) — these are READ-ONLY
- **`online`** — Server pipeline settings (thresholds) — these are the REFERENCE
- **`offline`** — Phone pipeline settings — these are what the AGENT CAN CHANGE
- **`parity_loss.weights`** — How much each metric matters in the total score
- **`agent`** — Agent behavior (max iterations, patience)

**Important:** The `offline` section is what the agent modifies. Everything else stays fixed.

---

### `trace/schema.py` — Data Models

**What it does:** Defines the "shape" of all data the agent works with.

**Key classes:**
- **`Detection`** — One bounding box: class name, confidence, coordinates `[x1, y1, x2, y2]`
- **`PipelineTrace`** — Complete recording of one image through one pipeline, with 4 checkpoints:
  1. `input_tensor` — The preprocessed image (numpy array) fed to the model
  2. `raw_output` — What the model returned (raw numbers, before any decoding)
  3. `decoded_boxes` — Decoded bounding boxes (after sigmoid, before NMS)
  4. `nms_boxes` — Final boxes (after Non-Maximum Suppression removes duplicates)
- **`GoldenTrace`** — A PAIR of traces (online + offline) for the SAME image

**Why 4 checkpoints?** Because the divergence could happen at ANY stage. If we only compared final boxes, we'd know THAT they're different but not WHERE it went wrong. With 4 checkpoints, we can say: "The input tensors are already different → it's a preprocessing problem."

---

### `trace/online_tracer.py` — Server Pipeline Recorder

**What it does:** Runs an image through the Ultralytics/PyTorch pipeline and captures the full trace.

**Key method:** `trace_image(image_path)` → returns a `PipelineTrace`

**How it works:**
1. Opens the image
2. Replicates the YOLO letterbox preprocessing (resize + gray padding)
3. Runs `model.predict()` via Ultralytics
4. Captures the input tensor, raw output, decoded boxes, and NMS boxes
5. Normalizes all bounding boxes to `[0, 1]` range for fair comparison

**This is the REFERENCE pipeline — its results are treated as ground truth.**

---

### `trace/offline_tracer.py` — Phone Pipeline Recorder

**What it does:** Runs an image through the TFLite pipeline **in Python**, replicating exactly what the Flutter/Dart app does.

**This is the most critical file in the whole project.**

**Why run TFLite in Python instead of Flutter?** Because we need both traces in the same format (numpy arrays) to compare them. If we ran Flutter, we'd have to serialize Dart tensors → JSON → Python, which introduces its own errors.

**Key method:** `trace_image(image_path)` → returns a `PipelineTrace`

**What it replicates (from the Dart code):**
1. **Letterbox resize** — Same as `model_service.dart` lines 156-183
2. **Normalization** — `pixel / 255.0` — Same as `model_service.dart` line 192
3. **TFLite inference** — Via `tensorflow.lite.Interpreter`
4. **Output transpose** — Handles `[12, 8400]` vs `[8400, 12]` — Same as `detection_service.dart` lines 36-53
5. **Sigmoid activation** — `1 / (1 + exp(-logit))` — Same as `detection_service.dart` line 105
6. **xywh → xyxy conversion** — Same as `detection_service.dart` lines 154-158
7. **NMS** — Same greedy algorithm as `detection_service.dart` lines 200-219

**Every step references the exact Dart source lines it replicates.** This is critical for scientific reproducibility.

---

### `trace/storage.py` — Persistence

**What it does:** Saves and loads traces to/from disk.

**Storage format:**
- **JSON file** — Human-readable: metadata, detection lists, shapes
- **NPZ file** — Binary: actual numpy tensors (input_tensor, raw_output)

**Why split?** JSON is readable in any text editor. But numpy arrays in JSON would be massive and imprecise. NPZ is compact and preserves exact floating-point values.

**Directory layout:**
```
traces/
├── online/           ← Server traces
│   ├── mango_01_online.json
│   └── mango_01_online.npz
├── offline/          ← Phone traces
│   ├── mango_01_offline.json
│   └── mango_01_offline.npz
└── paired/           ← Golden Trace index
    └── mango_01.json
```

---

### `diff/metrics.py` — 5 Divergence Metrics

**What it does:** Measures HOW MUCH the online and offline pipelines disagree, using 5 different methods.

| # | Function | What it measures | Catches |
|---|----------|-----------------|---------|
| 1 | `tensor_l2()` | L2 distance of input tensors | Normalization, resize, or channel bugs |
| 2 | `logits_diff()` | Mean absolute diff of raw model outputs | Quantization drift, runtime differences |
| 3 | `iou_mismatch()` | 1 − mean(IoU of matched boxes) | Bounding box coordinate errors |
| 4 | `count_diff()` | Absolute detection count difference | NMS threshold causing extra/missing boxes |
| 5 | `confidence_kl()` | KL divergence of confidence distributions | Sigmoid or calibration problems |

**Why 5 metrics?** Each metric is a "lens" that sees a different type of problem. No single metric catches everything. Together, they give a complete picture.

**`compute_all_metrics()`** — Runs all 5 on a trace pair, returns a dict.

---

### `diff/parity_loss.py` — The Objective Function

**What it does:** Combines all 5 metrics into a single number — the **Parity Loss**.

**Formula:**
```
ParityLoss = (w₁ × tensor_l2 + w₂ × logits_diff + w₃ × iou_mismatch 
              + w₄ × count_diff + w₅ × confidence_kl) / total_weight
```

**The weights are configurable** in `config.yaml`. Default:
- `tensor_l2`: 1.0, `logits_diff`: 1.0, `iou_mismatch`: 1.0
- `count_diff`: 0.5, `confidence_kl`: 0.5

**Why weighted?** Because some metrics matter more. IoU mismatch (boxes in wrong place) is worse than a slight confidence difference.

**Key methods:**
- `compute()` — Parity loss for ONE image pair
- `compute_batch()` — Aggregate loss across ALL test images (mean, std, min, max)

**This number is what the agent tries to minimize to zero.**

---

### `diff/report.py` — Report Generator

**What it does:** Turns raw numbers into human-readable reports.

**Two formats:**
- **Console report** — Formatted table printed to terminal
- **Markdown report** — Saved as `.md` file (good for papers/presentations)

**Example output:**
```
======================================================================
  PARITY AGENT — DIFF REPORT
======================================================================
  Images analyzed:  15
  Mean Parity Loss: 0.234567
  
  PER-IMAGE METRICS
  Image                TensorL2     Logits       IoU-M      Count    KL         TOTAL
  mango_01             0.001234     0.005678     0.1200     2        0.034521   0.234567
  apple_03             0.000012     0.000034     0.0000     0        0.000001   0.000012
```

---

### `alignment/parameters.py` — The Agent's Toolbox

**What it does:** Defines every "knob" the agent is allowed to turn.

**The 8 tunable parameters:**

| Parameter | Type | Options | What it controls |
|-----------|------|---------|------------------|
| `normalization` | categorical | `divide_255`, `neg1_pos1`, `none` | How pixel values are scaled |
| `resize_method` | categorical | `bilinear`, `nearest`, `area`, `lanczos` | How images are resized |
| `channel_order` | categorical | `rgb`, `bgr` | Color channel ordering |
| `confidence_threshold` | continuous | 0.1 to 0.9 (step 0.05) | Min detection confidence |
| `iou_threshold` | continuous | 0.2 to 0.8 (step 0.05) | NMS overlap threshold |
| `apply_sigmoid` | categorical | `true`, `false` | Whether to sigmoid raw logits |
| `letterbox_padding` | categorical | `true`, `false` | Letterbox vs stretch resize |
| `padding_color` | categorical | `[114,114,114]`, `[0,0,0]`, `[128,128,128]` | Letterbox fill color |

**Key function:** `generate_ablation_configs()` — Creates configs where ONE parameter is changed, others stay fixed. This is the controlled experiment principle.

**Important:** The agent NEVER touches model weights, labels, or training data. Only these 8 parameters.

---

### `alignment/experiment_runner.py` — Experiment Engine

**What it does:** Runs controlled experiments — change one thing, measure the effect.

**Key methods:**
- `run_single_experiment()` — Run offline tracer with a config, compute parity loss
- `run_ablation()` — Sweep all values of ONE parameter, find which is best
- `save_log()` — Save experiment history as JSON for reproducibility

**How an ablation works:**
```
Parameter: confidence_threshold
Baseline:  0.5  → Parity Loss: 0.2345
Test 0.25: → Parity Loss: 0.1890  ✓ Better!
Test 0.30: → Parity Loss: 0.2001
Test 0.35: → Parity Loss: 0.2134
Best: 0.25 (improvement: 0.0455)
```

---

### `agents/profiler.py` — "What's the biggest problem?"

**What it does:** Analyzes parity metrics to rank which pipeline stage causes the most divergence.

**How it works:**
1. Averages each metric across all test images
2. Maps metrics to pipeline stages:
   - `tensor_l2` → **preprocessing** stage
   - `logits_diff` → **model runtime** stage
   - `iou_mismatch` → **localization** stage
   - `count_diff` → **NMS behavior** stage
   - `confidence_kl` → **calibration** stage
3. Ranks stages by divergence magnitude
4. Suggests which parameters to investigate

**Example output:**
```python
{
    "dominant_stage": "preprocessing",  # ← THIS is where the problem is
    "ranking": [
        {"stage": "preprocessing", "value": 0.15, "pct_of_total": 64%},
        {"stage": "nms_behavior", "value": 0.05, "pct_of_total": 21%},
        ...
    ],
    "suggested_params": ["normalization", "resize_method", "channel_order"]
}
```

---

### `agents/hypothesis.py` — "What could be causing this?"

**What it does:** Given the profiler's findings, generates ranked hypotheses about the cause.

**It uses 7 rules, each encoding domain knowledge:**

| Rule | Trigger | Hypothesis |
|------|---------|-----------|
| 1 | High tensor_l2 + preprocessing dominant | `normalization_mismatch` |
| 2 | Medium tensor_l2 + preprocessing dominant | `resize_method_mismatch` |
| 3 | Preprocessing dominant | `channel_order_swap` (RGB/BGR) |
| 4 | High count_diff + NMS dominant | `nms_threshold_mismatch` |
| 5 | High confidence_kl + calibration dominant | `confidence_threshold_mismatch` |
| 6 | Calibration dominant | `sigmoid_missing_or_double` |
| 7 | Very high tensor_l2 | `padding_color_mismatch` |

**Priority levels:** `critical` > `high` > `medium` > `low`

**These rules encode the knowledge you'd gain from manual debugging (Phase 3).**

---

### `agents/ablation.py` — "Let me test each guess"

**What it does:** Takes the hypotheses and runs controlled experiments to verify them.

**Key method:** `test_hypotheses()`
1. For each hypothesis, gets the suggested parameters
2. For each parameter, runs a full ablation sweep
3. Records which config change gives the best improvement
4. Returns results ranked by improvement

**Example:**
```
Hypothesis: normalization_mismatch
  Testing: normalization
    divide_255 → loss: 0.23  (baseline)
    neg1_pos1  → loss: 0.45  (worse)
    none       → loss: 0.89  (much worse)
  Testing: channel_order
    rgb → loss: 0.23  (baseline)
    bgr → loss: 0.08  ← BEST! (improvement: 0.15)
```

---

### `agents/alignment.py` — "Keep the best fix"

**What it does:** Takes the ablation results and applies the winning config change.

**Safety check:** Only applies a change if improvement > `min_improvement` (default: 0.001). This prevents random noise from changing the config.

**Key features:**
- `apply_best()` — Applies the top-performing config change
- `save_config()` — Writes the updated config back to `config.yaml`
- `save_history()` — Saves a log of ALL changes made (for reproducibility)
- `get_summary()` — Human-readable list: "Changed X from A to B (Δ=0.05)"

---

### `utils/image_loader.py` — Image Discovery

**What it does:** Scans a directory for image files.

**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

**Returns:** Sorted list of absolute file paths.

---

## The Autonomous Control Loop

Here's the full loop in detail:

```
Iteration 1:
  [1/5] Trace offline pipeline with current config
  [2/5] Compute parity loss: 0.345678
  [3/5] Profile: dominant stage = "preprocessing" (64% of divergence)
  [4/5] Hypotheses:
        • channel_order_swap (high)
        • normalization_mismatch (high)
  [5/5] Ablation results:
        Best: channel_order = "bgr" → loss: 0.189012

Iteration 2:
  [1/5] Trace offline pipeline with updated config
  [2/5] Compute parity loss: 0.189012
  [3/5] Profile: dominant stage = "nms_behavior" (45% of divergence)
  [4/5] Hypotheses:
        • confidence_threshold_mismatch (high)
  [5/5] Ablation results:
        Best: confidence_threshold = 0.25 → loss: 0.034567

Iteration 3:
  [2/5] Compute parity loss: 0.034567
  ✓ CONVERGED! (below threshold 0.05)
```

**Total: 3 iterations, loss reduced from 0.35 → 0.03 — fully autonomous.**

---

## Configuration

### `config.yaml` Overview

```yaml
# WHERE things are
paths:
  model_tflite: "assets/model.tflite"      # The frozen YOLO model
  labels: "assets/labels.txt"               # Class names
  test_images: "test_images/"               # Your test images go here
  traces_dir: "traces/"                     # Output: recorded traces
  results_dir: "results/"                   # Output: reports & experiments

# WHAT the model expects (read-only, never changed)
model:
  input_size: 640
  num_classes: 8
  class_names: [apple, watermelon, mango, strawberry, banana, orange, pineapple, grape]

# REFERENCE pipeline (server — treated as ground truth)
online:
  confidence_threshold: 0.25
  iou_threshold: 0.45

# TUNABLE pipeline (phone — what the agent modifies)
offline:
  normalization: "divide_255"
  resize_method: "bilinear"
  channel_order: "rgb"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  apply_sigmoid: true

# HOW to score (weights for each metric)
parity_loss:
  weights: { tensor_l2: 1.0, logits_diff: 1.0, iou_mismatch: 1.0, count_diff: 0.5, confidence_kl: 0.5 }
  threshold: 0.05               # Stop when loss < this

# AGENT behavior
agent:
  max_iterations: 50
  patience: 5                   # Stop if no improvement for 5 rounds
```

---

## How to Run

### Prerequisites

```bash
# Install dependencies (from the project root)
pip install -r parity_agent/requirements-agent.txt
```

### Step 1: Add Test Images

Create `test_images/` at the project root and add 10–20 fruit images:
```bash
mkdir test_images
# Copy or download fruit images into this folder
```

### Step 2: Run

```bash
# Option A: Run everything end-to-end
python parity_agent/run_agent.py --mode full --images test_images/

# Option B: Run one phase at a time
python parity_agent/run_agent.py --mode trace --images test_images/   # Generate traces
python parity_agent/run_agent.py --mode diff                          # Compute diffs
python parity_agent/run_agent.py --mode agent --images test_images/   # Run agent loop
```

### Output

```
traces/                  ← Golden Traces (JSON + NPZ)
results/
├── diffs/               ← Diff reports (Markdown)
├── experiments/         ← Experiment logs (JSON)
└── alignment_history.json  ← What the agent changed
```

---

## Key Concepts

### Golden Trace
A complete recording of what happens inside a pipeline for one image. Like an X-ray of the inference process. Without it, you only see "it gave wrong boxes" — with it, you see exactly WHERE things went wrong.

### Parity Loss
A single number (0 = perfect match, higher = worse) measuring how different the online and offline pipelines behave. The agent's job is to minimize this number.

### Ablation
A controlled experiment where you change ONE thing and measure the effect. If changing normalization from `divide_255` to `neg1_pos1` makes things worse, you know `divide_255` was correct.

### Frozen Model
The model weights (what the neural network learned) are NEVER changed. The agent only adjusts the "plumbing" around the model — how images are preprocessed, how outputs are decoded, what thresholds are used.

---

## FAQ

**Q: Does this agent use GPT/LLM/ChatGPT?**
No. The agent uses deterministic rules and controlled experiments. No language models, no prompting. It's systems intelligence, not generative AI.

**Q: Why not just copy the server's preprocessing to the phone?**
That's exactly what the agent helps you do — but automatically. It finds which settings need to match and which values to use. For a single model, you could do it manually. But for 10 models across 5 platforms, you need automation.

**Q: Can this work with models other than YOLO?**
Yes. The system is model-agnostic. The trace schema works for any model that takes an image tensor and outputs detection boxes. Just update `config.yaml` with the right class names and input size.

**Q: Why are there separate online and offline tracers?**
The online tracer uses Ultralytics (the official YOLO library). The offline tracer uses raw TFLite (replicating what Flutter does). They intentionally use different code paths — because that's the real-world situation we're testing.

**Q: What if the parity loss never reaches zero?**
It won't — some numerical differences are inherent to quantization (FP32 vs INT8) or runtime differences (PyTorch vs TFLite). The threshold in `config.yaml` defines "good enough." A loss of 0.01 typically means detections are functionally equivalent.
