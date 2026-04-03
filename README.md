# AutoParity — Autonomous ML Deployment Alignment

> Autonomous ML deployment alignment agent that detects and fixes cross-runtime inference drift (PyTorch → TFLite).

---

## Version 1 — Core Parity Agent

The foundational autonomous agent capable of self-observation, self-diagnosis, and self-correction of cross-platform ML inference drift.

**📂 [`version_1/`](./version_1/)**

### Features
- **Golden Trace System** — 4-checkpoint instrumentation of both online (PyTorch) and offline (TFLite) pipelines
- **5 Divergence Metrics** — tensor L2, logits diff, IoU mismatch, count diff, confidence KL
- **Parity Loss** — Weighted composite objective function the agent minimises
- **Hypothesis Engine** — 7 domain-knowledge rules to diagnose drift causes
- **Ablation Framework** — Controlled experiments testing one parameter at a time
- **8 Tunable Parameters** — normalization, resize method, channel order, thresholds, sigmoid, padding

### How to Run
```bash
pip install -r version_1/requirements-agent.txt
python version_1/run_agent.py --mode full --images test_images/
```

### Documentation
See [`version_1/README.md`](./version_1/README.md) for full file-by-file documentation.

---

## Version 2 — Dynamic Analysis, IoU Metrics & Visual Comparison

Version 2 builds on the core agent with **automated analysis, publication-quality figures, IoU-based detection metrics, and visual before/after comparison**. Everything is fully portable — works with any user's project and test images, with zero hardcoded values.

**📂 [`version_2/`](./version_2/)**

### What's New in v0.2

| Feature | Description |
|---------|-------------|
| **Snapshot System** | Auto-captures baseline (pre-fix) and final (post-fix) detection data as JSON |
| **IoU-Based Metrics** | Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95 using IoU matching |
| **5 Publication Figures** | Dynamic figures generated from real data at 300 DPI |
| **Before/After Images** | Side-by-side detection box visualisation (broken vs fixed) |
| **Streamlit Dashboard** | 5-tab interactive dashboard with metrics tables and trace viewer |
| **LangGraph Orchestration** | State machine control loop replacing the while-loop |
| **Auto-Apply to Dart** | Agent can directly patch Flutter/Dart source code |
| **BMVC Proposal** | Full academic proposal with formal metric definitions |

### New Scripts in v0.2

| Script | What It Does | How to Run |
|--------|--------------|------------|
| `utils/snapshots.py` | Saves baseline/final detection state + IoU metrics | Auto-called by the agent |
| `utils/parity_metrics.py` | Computes precision, recall, F1, mAP using IoU matching | Auto-called by snapshots |
| `bmvc_figures.py` | Generates 5 publication-quality figures from snapshots | `python version_2/bmvc_figures.py` |
| `save_annotated_images.py` | Renders bounding boxes on images (before/after) | `python version_2/save_annotated_images.py` |
| `dashboard/app.py` | Interactive Streamlit dashboard with 5 tabs | `streamlit run version_2/dashboard/app.py` |
| `agents/graph.py` | LangGraph state machine orchestration | `python version_2/run_agent.py --mode graph` |
| `agents/auto_apply.py` | Auto-patches Dart source code with fixes | Auto-called by the agent |

### How to Run (Version 2)

```bash
# 1. Install dependencies
pip install -r version_2/requirements-agent.txt

# 2. Place your model and test images
#    - assets/model.tflite (your TFLite model)
#    - assets/labels.txt (one class per line)
#    - test_images/ (10-20 test images)

# 3. Update config
#    Edit version_2/config.yaml with your class names and paths

# 4. Run the agent (captures baseline + fixes + captures final — all automatic)
python version_2/run_agent.py --mode graph --images test_images/

# 5. Generate analysis outputs
python version_2/bmvc_figures.py              # 5 publication figures
python version_2/save_annotated_images.py     # Before/after annotated images
streamlit run version_2/dashboard/app.py      # Interactive dashboard
```

### Portability

The entire system is **100% portable**:
- All scripts read from `config.yaml` for model paths, class names, and test image folder
- The snapshot system captures YOUR project's broken state automatically at iteration 0
- Figures, tables, and annotated images are generated from snapshots — no hardcoded values
- If you drop in different test images and re-run, everything updates automatically

### Documentation
See [`version_2/README.md`](./version_2/README.md) for full file-by-file documentation, architecture diagram, and detailed explanations.

---

## Architecture Overview

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

## License

See [LICENSE](./LICENSE) for details.

## Citation

See [CITATION.cff](./CITATION.cff) for citation information.
