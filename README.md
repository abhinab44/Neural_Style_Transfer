# Real-Time Neural Style Transfer

A CPU-only, multithreaded neural style transfer application that applies classical painting styles to a live webcam feed in real time — built with TensorFlow Hub (Magenta) and OpenCV. Benchmarked and documented with a full technical report.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange?logo=tensorflow) ![OpenCV](https://img.shields.io/badge/OpenCV-latest-green?logo=opencv) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table of Contents
- [Real-Time Neural Style Transfer](#real-time-neural-style-transfer)
  - [Table of Contents](#table-of-contents)
  - [Performance](#performance)
  - [Architecture](#architecture)
    - [Why this threading model matters](#why-this-threading-model-matters)
  - [Styles](#styles)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Arguments](#arguments)
    - [Controls](#controls)
  - [Performance Output](#performance-output)
  - [Requirements](#requirements)
  - [Environment](#environment)
  - [Bug Fix: `ZeroDivisionError` on `--frame-skip 0`](#bug-fix-zerodivisionerror-on---frame-skip-0)
  - [Known Limitations](#known-limitations)
  - [Key Takeaways](#key-takeaways)
  - [Full Technical Report](#full-technical-report)
  - [License](#license)

---

## Performance

Benchmarked on **Intel 12th Gen i7-1255U — CPU only, no GPU.**

| Metric | 384px input | 256px input |
|---|---|---|
| Avg inference time | 3.05 s | 1.83 s |
| True stylization FPS | 0.33 FPS | **0.55 FPS** |
| Avg end-to-end latency | 8.98 s | **5.40 s** |
| Display FPS (UI) | ~30 FPS | ~30 FPS |

> Display and inference run on separate threads — the UI stays smooth at ~30 FPS regardless of inference speed.

---

## Architecture

```
┌─────────────────────────────┐     ┌──────────────────────────────────┐
│        MAIN THREAD          │     │          WORKER THREAD           │
│                             │     │                                  │
│  Webcam → Frame Skip Logic  │────▶│  Preprocess → TF Hub Inference   │
│  Display Loop (~30 FPS)     │◀────│  Post-process → Result           │
│                             │     │                                  │
│     [Frame Queue max=2]     │     │     [Result Queue max=2]         │
└─────────────────────────────┘     └──────────────────────────────────┘
```

Bounded queues (maxsize=2) act as back-pressure — preventing unbounded memory growth when inference is slower than capture. Latency is approximately 3× inference time due to queue saturation, which is expected deterministic behavior.

### Why this threading model matters

Neural inference (~1.83s/frame) and UI rendering (~33ms/frame) differ by **55×**. A naïve single-threaded design blocks the display loop on every inference call, freezing the UI for seconds at a time.

This app solves that with a **producer-consumer pattern**:

| Concern | Thread | Rate |
|---|---|---|
| Camera capture + display | Main | ~30 FPS |
| TF Hub inference | Daemon worker | ~0.55 FPS |
| Communication | Bounded `Queue(maxsize=2)` | — |

Key design decisions:
- **Daemon thread** — the worker exits automatically when the main thread ends, no manual cleanup needed
- **`maxsize=2` queues** — if the worker falls behind, new frames are dropped rather than queued forever; RAM usage stays flat over hours of runtime (verified over 20-minute stress tests)
- **Cached last frame** — the display loop always has something to show; it renders the most recent stylized result while waiting for the next one, keeping the UI at a smooth 30 FPS even when inference is slow
- **Per-frame timestamps** — each frame carries a `time.time()` stamp from the moment it enters the input queue, so true end-to-end latency (not just inference time) can be measured accurately

---

## Styles

| # | Style | Artist |
|---|---|---|
| 1 | Starry Night | Van Gogh |
| 2 | Les Demoiselles d'Avignon | Picasso |
| 3 | Impression, Sunrise | Monet |
| 4 | The Great Wave | Hokusai |
| 5 | Composition VII | Kandinsky |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/abhinab44/Neural_Style_Transfer.git
cd Neural_Style_Transfer

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Basic run (recommended settings)
python webtrans1.py

# With frame skipping (reduces queue pressure, not throughput)
python webtrans1.py --frame-skip 2

# Select camera if you have multiple
python webtrans1.py --camera-id 1
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--frame-skip` | `2` | Submit every N-th frame to the inference queue. `0` = every frame |
| `--camera-id` | `0` | OpenCV camera device index |

### Controls

| Key | Action |
|---|---|
| `1` – `5` | Switch between the 5 styles |
| `Space` | Toggle between original and stylized view |
| `F` | Toggle FPS display |
| `H` | Toggle help overlay |
| `S` | Save screenshot to `screenshots/` folder |
| `Q` | Quit |

---

## Performance Output

On exit, the app prints a full benchmark summary:

```
===== Performance Summary =====
Total Stylized Frames  : 142
Average Inference Time : 1.8342 sec
Min Inference Time     : 1.6820 sec
Max Inference Time     : 3.9359 sec
True Stylization FPS   : 0.55
Avg End-to-End Latency : 5.4010 sec
================================
```

> The first inference is always slower (~3–5s) due to TensorFlow JIT graph compilation. This is expected and excluded from reported averages.

---

## Requirements

- Python 3.11
- Webcam (built-in or external)
- ~2 GB disk space for model download (cached after first run)
- No GPU required — runs on CPU only

---

## Environment

Tested on:

```
OS       : Windows 11
CPU      : Intel 12th Gen Core i7-1255U
RAM      : 16 GB
GPU      : None
Python   : 3.11
TF       : 2.19.0
```

---

## Bug Fix: `ZeroDivisionError` on `--frame-skip 0`

An early version crashed with `ZeroDivisionError` when `--frame-skip 0` was passed. The original frame-skipping logic used the modulo operator unconditionally:

```python
# Before — crashes when frame_skip=0
should_process = (self.frame_counter % self.frame_skip == 0)
```

`frame_skip=0` means *"process every frame"* — a valid and useful setting — but `% 0` raises `ZeroDivisionError` in Python regardless of context.

The fix adds an explicit guard before the modulo is ever evaluated:

```python
# After — handles zero correctly
if self.frame_skip == 0:
    should_process = True          # process every frame
else:
    should_process = (self.frame_counter % self.frame_skip == 0)
```

This matters beyond just avoiding a crash — `frame_skip=0` is the correct setting for **benchmarking**, since it maximizes inference throughput and removes any artificial frame-drop logic from the latency measurements. All benchmark data in this repo was collected with `--frame-skip 0`.

---

## Known Limitations

- CPU-only inference is slow (~0.55 FPS) — a GPU would achieve 10–30× speedup
- End-to-end latency (~5.4s) is not suitable for interactive AR use cases
- Only the TF Hub 256 model variant is used (no official 512 checkpoint exists)

---

## Key Takeaways

Building this project surfaced several non-obvious lessons that don't come from coursework:

- **Display FPS and inference FPS are entirely separate problems.** The naive assumption is that a slow model means a slow UI. The real solution is architectural — decouple the two concerns into separate threads rather than trying to optimize inference to match display speed.

- **Bounded queues are a memory safety tool, not just a performance one.** Setting `maxsize=2` means the app drops frames under load rather than accumulating them indefinitely. RAM usage stayed flat across 20-minute sessions — something that only becomes obvious when you actually measure it.

- **Benchmarking requires deliberate methodology.** The first inference frame includes TensorFlow's JIT graph compilation overhead (~3–5s), which would skew every reported average if included. Excluding it isn't obvious until you see the numbers and ask why frame 0 is always an outlier.

- **Latency ≈ 3× inference time is deterministic, not a bug.** With the queue filling instantly at 30 FPS input vs. 0.55 FPS consumption, each frame waits roughly two inference cycles before being processed. Understanding why helped distinguish expected architecture behavior from something worth fixing.

---

## Full Technical Report

A complete performance report with architecture diagrams, benchmark charts, and latency analysis is available in [`NeuralStyleTransfer_Report.pdf`](./NeuralStyleTransfer_Report.pdf).

---

## License

MIT
