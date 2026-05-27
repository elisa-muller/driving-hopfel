# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A Python research project studying phantom traffic jams and Hopf bifurcation in ring-road traffic models. All scripts live in `protoype/` (note the typo in the folder name — it's intentional as committed).

Dependencies: `numpy`, `matplotlib` (no `requirements.txt`; install via `pip install numpy matplotlib`).

## Running the scripts

```bash
# Analytical Hopf bifurcation scan — saves 3 PNG files, no display required
python protoype/traffic_hopf_analysis.py

# True branch diagram — saves PNG + CSV, imports from traffic_hopf_analysis.py
python protoype/traffic_hopf_true_branch.py

# Animated simulation (IDM-style model, requires a display/GUI)
python protoype/traffic_jam.py

# Enhanced animated simulation with live Hopf diagnostics (requires a display/GUI)
python protoype/traffic_jam_enhanced.py
```

## Architecture

### Module dependency
`traffic_hopf_true_branch.py` imports core functions from `traffic_hopf_analysis.py`:
```
traffic_hopf_true_branch.py
    └── traffic_hopf_analysis.py  (TrafficParams, simulate, jacobian, etc.)
```

The two animation scripts (`traffic_jam.py`, `traffic_jam_enhanced.py`) are fully standalone.

### Two distinct traffic models

**Optimal-Velocity (OV) model** — used in `traffic_hopf_analysis.py`, `traffic_hopf_true_branch.py`, and `traffic_jam_enhanced.py`:
- State: gap `s_i` and speed `v_i` per car
- `ds_i/dt = v_{i+1} − v_i`, `dv_i/dt = α(V(s_i) − v_i)`
- `V(s) = 0.5·v_max·(tanh((s − s_c)/w) + 1)` — smooth monotone OV function
- AV feedback: `u_i = −k_s(s_i − s_eq) − k_v(v_i − v_eq) − k_r(v_i − v_{i+1})`

**IDM-like model** — used only in `traffic_jam.py`:
- Uses `s_star`, `a_max`, `b` (deceleration) parameters
- AV controller targets a large safe gap with strong damping gains

### Hopf bifurcation mechanics
The equilibrium loses stability when `α` crosses a critical value `α_c(k)` for Fourier mode `k`:
- Analytic formula: `α_c(k) = V′(s_eq)·(1 + cos(θ_k))`
- `traffic_hopf_analysis.py` sweeps `α`, computes the leading non-neutral eigenvalue of the block-circulant Jacobian, and finds sign crossings
- `traffic_hopf_true_branch.py` runs nonlinear simulations at each `α` and extracts min/max speed envelopes to plot the classical pitchfork-like branch diagram

### Key data structures
- `TrafficParams` (frozen dataclass in `traffic_hopf_analysis.py`) — single source of truth for all model parameters
- State vector: `[s_0, …, s_{N-1}, v_0, …, v_{N-1}]` (gaps first, speeds second)
- `simulate()` uses RK4; `traffic_jam*.py` use forward Euler for real-time animation speed

### Output files (written to current working directory)
- `hopf_bifurcation_diagram.png` — growth-rate scan
- `hopf_trajectory_below.png`, `hopf_trajectory_above.png` — time traces on each side of the Hopf point
- `hopf_true_branch_diagram.png`, `hopf_true_branch_data.csv` — branch diagram and raw data
