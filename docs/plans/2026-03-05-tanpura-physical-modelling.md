# Tanpura Physical Modelling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure tanpura synthesis into three independent physical modelling methods (modal, FDTD, waveguide) sharing common infrastructure, with a multi-objective loss function combining spectral and perceptual features.

**Architecture:** Shared params/bridge/loss/IO layer consumed by three independent synthesizers, each with identical interface. Optimization harness is method-agnostic. Current tanpura.py code is refactored into modal/ as the first working method; FDTD and waveguide are new implementations.

**Tech Stack:** Python 3.9+, numpy, numba, scipy, librosa, optuna, cmaes, uv

---

### Task 1: Create shared/params.py

**Files:**
- Create: `shared/__init__.py`
- Create: `shared/params.py`

Extract `TanpuraParams`, `_derived()`, `compute_modal_frequencies()`, `compute_mode_shapes()` from `tanpura.py`. These are used by all methods.

---

### Task 2: Create shared/bridge.py

**Files:**
- Create: `shared/bridge.py`

Extract `create_bridge_profile()` from `tanpura.py`. The bridge geometry (parabolic curve + jiva Gaussian bump) is shared across all methods.

---

### Task 3: Create shared/loss.py

**Files:**
- Create: `shared/loss.py`

Enhanced multi-objective loss combining:
- Spectral: Log Spectral Distance, spectral centroid trajectory, spectral flux
- Perceptual: MFCC distance, temporal envelope correlation
- Weighted combination

---

### Task 4: Create shared/audio_io.py

**Files:**
- Create: `shared/audio_io.py`

Common functions: load reference audio, write WAV output, normalize audio.

---

### Task 5: Create shared/optimize.py

**Files:**
- Create: `shared/optimize.py`

Method-agnostic Optuna harness. Takes any synthesize() callable and a loss function, runs optimization. Extracts the optimizer from tanpura.py.

---

### Task 6: Create modal/synthesizer.py

**Files:**
- Create: `modal/__init__.py`
- Create: `modal/synthesizer.py`
- Create: `modal/kernels.py`

Port existing synthesis code from `tanpura.py` with enhancements:
- Energy-conserving Stormer-Verlet integration (replace explicit Euler)
- Tension modulation (string stretching during bridge contact)
- 2-polarisation vibration (vertical + horizontal)

---

### Task 7: Create modal/run.py

**Files:**
- Create: `modal/run.py`

CLI entry point for modal method: `uv run modal/run.py render` / `uv run modal/run.py optimize`

---

### Task 8: Create fdtd/synthesizer.py

**Files:**
- Create: `fdtd/__init__.py`
- Create: `fdtd/synthesizer.py`
- Create: `fdtd/kernels.py`

Finite difference time domain solver:
- Spatial grid discretisation of stiff string PDE
- Energy-conserving implicit scheme for bridge contact
- CFL-stable time stepping

---

### Task 9: Create fdtd/run.py

**Files:**
- Create: `fdtd/run.py`

CLI entry point for FDTD method.

---

### Task 10: Create waveguide/synthesizer.py

**Files:**
- Create: `waveguide/__init__.py`
- Create: `waveguide/synthesizer.py`
- Create: `waveguide/kernels.py`

Digital waveguide with contact:
- Bidirectional delay lines for free string
- Allpass dispersion filters for stiffness
- Penalty contact at bridge boundary

---

### Task 11: Create waveguide/run.py

**Files:**
- Create: `waveguide/run.py`

CLI entry point for waveguide method.

---

### Task 12: Update project files

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `.gitignore`
- Delete: `tanpura.py`

Update deps, README with new structure, gitignore references/ directory.

---
