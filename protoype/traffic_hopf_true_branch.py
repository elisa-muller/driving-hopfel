"""True Hopf branch diagram for the traffic ring-road model.

This script builds the classical branch diagram shape for the same traffic
model used in traffic_hopf_analysis.py.

What is plotted
---------------
For each value of alpha, we simulate the nonlinear system, remove transients,
and compute for one observed car:

    v_min(alpha) = min_t v_i(t)  over the tail window
    v_max(alpha) = max_t v_i(t)  over the tail window

If the system converges to equilibrium, then v_min ~= v_max and the branch is
single-valued. When a limit cycle exists (Hopf side), v_min and v_max separate,
giving the classical "one branch splitting into two" picture.

Equations solved (same model as traffic_hopf_analysis.py)
---------------------------------------------------------
Baseline dynamics:

    ds_i/dt = v_{i+1} - v_i
    dv_i/dt = alpha * (V(s_i) - v_i)

with

    V(s) = 0.5*v_max * (tanh((s - s_c)/w) + 1)

Optional AV feedback for selected cars i:

    u_i = -k_s (s_i - s_eq) - k_v (v_i - v_eq) - k_r (v_i - v_{i+1})
    dv_i/dt = alpha * (V(s_i) - v_i) + u_i

Outputs
-------
1. hopf_true_branch_diagram.png
   True branch diagram: equilibrium branch plus periodic-orbit envelopes.
2. hopf_true_branch_data.csv
   Raw data table for alpha, growth rate, mean speed, min/max envelopes,
   and measured amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import matplotlib.pyplot as plt

from traffic_hopf_analysis import (
    TrafficParams,
    equilibrium_speed,
    leading_nonneutral_real_part,
    locate_crossing,
    select_controlled_indices,
    simulate,
)


OUTPUT_PNG = "hopf_true_branch_diagram.png"
OUTPUT_CSV = "hopf_true_branch_data.csv"


@dataclass(frozen=True)
class BranchConfig:
    """Configuration for branch-diagram extraction."""

    use_control: bool = False
    alpha_min: float = 0.2
    alpha_max: float = 6.0
    alpha_steps: int = 36

    # Long simulation and transient removal are needed for branch extraction.
    sim_time: float = 90.0
    dt: float = 0.05
    transient_fraction: float = 0.60

    # Small perturbation to excite the critical mode.
    perturbation_amplitude: float = 0.2

    # Measured branch observable: speed of this car index.
    observed_car: int = 0

    # Amplitude below this threshold is considered effectively equilibrium.
    amplitude_tol: float = 0.15


def build_params(config: BranchConfig) -> TrafficParams:
    """Create the simulation parameter set used for branch extraction."""

    base = TrafficParams()
    return replace(
        base,
        use_control=config.use_control,
        dt=config.dt,
        sim_time=config.sim_time,
        perturbation_amplitude=config.perturbation_amplitude,
    )


def extract_branches(alpha_values: np.ndarray,
                     params: TrafficParams,
                     config: BranchConfig,
                     controlled_indices: np.ndarray | None) -> dict[str, np.ndarray]:
    """Run nonlinear simulations and extract equilibrium/periodic envelopes."""

    n = alpha_values.size
    growth = np.zeros(n, dtype=float)
    v_mean = np.zeros(n, dtype=float)
    v_min = np.zeros(n, dtype=float)
    v_max = np.zeros(n, dtype=float)
    amp = np.zeros(n, dtype=float)

    ctrl = controlled_indices if (config.use_control and controlled_indices is not None) else None
    obs = int(np.clip(config.observed_car, 0, params.n_cars - 1))

    for i, alpha in enumerate(alpha_values):
        if i % 6 == 0 or i == n - 1:
            print(f"Running branch point {i + 1}/{n} at alpha={alpha:.4f}")
        growth[i] = leading_nonneutral_real_part(alpha, params, controlled_indices=ctrl)
        sim = simulate(alpha, params, controlled_indices=ctrl)

        v_signal = sim["v"][:, obs]
        tail_start = int(config.transient_fraction * v_signal.size)
        tail = v_signal[tail_start:]

        v_mean[i] = float(np.mean(tail))
        v_min[i] = float(np.min(tail))
        v_max[i] = float(np.max(tail))
        amp[i] = 0.5 * (v_max[i] - v_min[i])

        # Collapse tiny numerical oscillations onto the equilibrium branch.
        if amp[i] < config.amplitude_tol:
            v_min[i] = v_mean[i]
            v_max[i] = v_mean[i]

    return {
        "growth": growth,
        "v_mean": v_mean,
        "v_min": v_min,
        "v_max": v_max,
        "amp": amp,
    }


def save_csv(alpha_values: np.ndarray, data: dict[str, np.ndarray]) -> None:
    """Save branch data for later plotting/reporting."""

    matrix = np.column_stack([
        alpha_values,
        data["growth"],
        data["v_mean"],
        data["v_min"],
        data["v_max"],
        data["amp"],
    ])
    header = "alpha,growth_realpart,v_mean,v_min,v_max,amplitude"
    np.savetxt(OUTPUT_CSV, matrix, delimiter=",", header=header, comments="")


def plot_true_branch(alpha_values: np.ndarray,
                     data: dict[str, np.ndarray],
                     alpha_c: float | None,
                     params: TrafficParams,
                     config: BranchConfig) -> None:
    """Plot the true Hopf branch diagram (single branch -> split envelopes)."""

    growth = data["growth"]
    v_mean = data["v_mean"]
    v_min = data["v_min"]
    v_max = data["v_max"]
    amp = data["amp"]

    stable_eq = growth < 0.0
    unstable_eq = growth > 0.0
    periodic = amp >= config.amplitude_tol

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alpha_values[stable_eq], v_mean[stable_eq], color="black", lw=2,
            label="Equilibrium branch (stable)")
    ax.plot(alpha_values[unstable_eq], v_mean[unstable_eq], color="black", lw=2, ls="--",
            label="Equilibrium branch (unstable)")

    ax.plot(alpha_values[periodic], v_max[periodic], color="tab:blue", lw=1.8,
            label="Periodic branch: upper envelope")
    ax.plot(alpha_values[periodic], v_min[periodic], color="tab:red", lw=1.8,
            label="Periodic branch: lower envelope")

    if alpha_c is not None:
        ax.axvline(alpha_c, color="tab:green", lw=2, ls=":",
                   label=f"Hopf threshold (linear) alpha ≈ {alpha_c:.4f}")

    ax.axhline(equilibrium_speed(params), color="gray", lw=1, ls=":")
    ax.set_xlabel("alpha")
    ax.set_ylabel(f"Observed speed branch (car {config.observed_car})")
    ax.set_title("True Hopf Branch Diagram (Traffic Ring Road)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")


def main() -> None:
    config = BranchConfig()
    params = build_params(config)

    controlled_indices = select_controlled_indices(params.n_cars, params.controlled_stride)
    if not config.use_control:
        controlled_indices = np.array([], dtype=int)

    alpha_values = np.linspace(config.alpha_min, config.alpha_max, config.alpha_steps)

    data = extract_branches(alpha_values, params, config, controlled_indices)

    alpha_c, idx = locate_crossing(alpha_values, data["growth"])

    save_csv(alpha_values, data)
    plot_true_branch(alpha_values, data, alpha_c, params, config)

    print("\n=== True Branch Diagram Generated ===")
    print(f"Saved diagram: {OUTPUT_PNG}")
    print(f"Saved data:    {OUTPUT_CSV}")
    print(f"Mode:          {'controlled' if config.use_control else 'baseline'}")
    if alpha_c is None:
        print("Linear scan did not find a Hopf crossing in this alpha range.")
    else:
        print(f"Linear Hopf estimate alpha ≈ {alpha_c:.6f} (between samples {idx} and {idx + 1})")

    periodic_points = int(np.count_nonzero(data["amp"] >= config.amplitude_tol))
    print(f"Periodic-branch points detected: {periodic_points}/{alpha_values.size}")

    plt.close("all")


if __name__ == "__main__":
    main()