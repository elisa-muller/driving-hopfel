"""Hopf bifurcation analysis for traffic flow on a ring road.

This script is intentionally written as a self-contained analysis tool.
It does three things:

1. Defines a traffic model that can undergo a Hopf bifurcation.
2. Derives the equilibrium, linearization, and critical Hopf condition.
3. Numerically verifies the stability change by sweeping a parameter and
   simulating trajectories below and above the critical point.

The baseline model is a classic optimal-velocity ring-road system.

State variables
---------------
We use gap and speed variables for each car:

    s_i = x_{i+1} - x_i - l
    v_i = speed of car i

with periodic indexing on a ring road. Here l is the physical vehicle length.

Equations of motion
-------------------
For the human-driver baseline, the equations are:

    ds_i/dt = v_{i+1} - v_i
    dv_i/dt = alpha * (V(s_i) - v_i)

where V(s) is an optimal-velocity function and alpha is the driver reaction
gain. This is the core model that can lose stability through a Hopf
bifurcation when alpha or the traffic density changes.

Optional AV feedback
--------------------
For selected cars, we can add a linear feedback controller:

    u_i = -k_s (s_i - s_eq) - k_v (v_i - v_eq) - k_r (v_i - v_{i+1})

so that the controlled dynamics become:

    dv_i/dt = alpha * (V(s_i) - v_i) + u_i

If no cars are selected for control, then u_i = 0 and the model reduces to
the baseline human-driver system. The Hopf analysis in this script is most
transparent in the baseline case, but the controlled case is included so you
can see how feedback shifts the critical point.

Equilibrium
-----------
For a homogeneous ring-road equilibrium,

    s_i = s_eq = (L - N*l)/N
    v_i = v_eq = V(s_eq)

so every car has the same spacing and speed.

Linearization of the baseline model
-----------------------------------
Let delta s_i and delta v_i be small perturbations around equilibrium.
Then:

    d/dt (delta s_i) = delta v_{i+1} - delta v_i
    d/dt (delta v_i) = alpha * V'(s_eq) * delta s_i - alpha * delta v_i

For Fourier mode k with theta_k = 2*pi*k/N, the characteristic equation is:

    lambda^2 + alpha * lambda + alpha * V'(s_eq) * (1 - exp(i*theta_k)) = 0

The Hopf crossing for the baseline model occurs when a complex-conjugate pair
of eigenvalues crosses the imaginary axis. For each mode k != 0, the critical
gain is:

    alpha_c(k) = V'(s_eq) * (1 + cos(theta_k))
    omega_c(k) = V'(s_eq) * sin(theta_k)

The first mode to destabilize is usually k = 1. That is the place where Hopf
should appear in this script.

Important note
--------------
This is a mathematical traffic-flow model, not the forced animation in the
original file. The current script is autonomous apart from the parameter scan,
so it is appropriate for actual bifurcation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


OUTPUT_BIFURCATION_PNG = "hopf_bifurcation_diagram.png"
OUTPUT_TRAJECTORY_BELOW_PNG = "hopf_trajectory_below.png"
OUTPUT_TRAJECTORY_ABOVE_PNG = "hopf_trajectory_above.png"


@dataclass(frozen=True)
class TrafficParams:
    """Model parameters.

    The numbers are chosen so that the baseline system has a visible Hopf
    crossing in a reasonable parameter range.
    """

    n_cars: int = 20
    road_length: float = 500.0
    vehicle_length: float = 5.0

    # Optimal-velocity curve parameters.
    v_max: float = 30.0
    v_shape_center: float = 20.0
    v_shape_width: float = 10.0

    # Driver reaction gain to sweep for Hopf detection.
    alpha_min: float = 0.2
    alpha_max: float = 6.0
    alpha_steps: int = 500

    # Optional controller gains.
    use_control: bool = True
    control_gain_spacing: float = 1.2
    control_gain_speed: float = 0.8
    control_gain_relative_speed: float = 0.4

    # Which cars are controlled if use_control is True.
    controlled_stride: int = 5

    # Simulation settings.
    dt: float = 0.02
    sim_time: float = 120.0
    perturbation_amplitude: float = 0.15


def select_controlled_indices(n_cars: int, stride: int) -> np.ndarray:
    """Pick a simple evenly spaced set of controlled cars."""

    if stride <= 0:
        return np.array([], dtype=int)
    return np.arange(0, n_cars, stride, dtype=int)


def optimal_velocity(gap: np.ndarray, params: TrafficParams) -> np.ndarray:
    """Smooth optimal-velocity function V(s).

    We use a shifted and rescaled tanh profile:

        V(s) = v_max/2 * [tanh((s - s_c)/w) + 1]

    This is smooth, monotone, and easy to differentiate.
    """

    z = (gap - params.v_shape_center) / params.v_shape_width
    return 0.5 * params.v_max * (np.tanh(z) + 1.0)


def optimal_velocity_prime(gap: float, params: TrafficParams) -> float:
    """Derivative V'(s) at a scalar gap value."""

    z = (gap - params.v_shape_center) / params.v_shape_width
    sech2 = 1.0 - np.tanh(z) ** 2
    return 0.5 * params.v_max * sech2 / params.v_shape_width


def equilibrium_gap(params: TrafficParams) -> float:
    """Homogeneous spacing at the ring-road equilibrium."""

    usable_road = params.road_length - params.n_cars * params.vehicle_length
    return usable_road / params.n_cars


def equilibrium_speed(params: TrafficParams) -> float:
    """Homogeneous equilibrium speed."""

    return float(optimal_velocity(np.array([equilibrium_gap(params)]), params)[0])


def equilibrium_state(params: TrafficParams) -> np.ndarray:
    """Return the stacked equilibrium state [s_0..s_{N-1}, v_0..v_{N-1}]."""

    s_eq = equilibrium_gap(params)
    v_eq = equilibrium_speed(params)
    return np.concatenate([
        np.full(params.n_cars, s_eq, dtype=float),
        np.full(params.n_cars, v_eq, dtype=float),
    ])


def rhs(state: np.ndarray, alpha: float, params: TrafficParams,
        controlled_indices: Iterable[int] | None = None) -> np.ndarray:
    """Nonlinear right-hand side for the ring-road system.

    State ordering is:
        [s_0, ..., s_{N-1}, v_0, ..., v_{N-1}]
    """

    n = params.n_cars
    s = state[:n]
    v = state[n:]
    v_lead = np.roll(v, -1)

    ds = v_lead - v
    dv = alpha * (optimal_velocity(s, params) - v)

    if params.use_control and controlled_indices is not None:
        s_eq = equilibrium_gap(params)
        v_eq = equilibrium_speed(params)
        controlled_indices = np.asarray(list(controlled_indices), dtype=int)
        if controlled_indices.size > 0:
            gap_term = -params.control_gain_spacing * (s[controlled_indices] - s_eq)
            speed_term = -params.control_gain_speed * (v[controlled_indices] - v_eq)
            rel_speed_term = -params.control_gain_relative_speed * (
                v[controlled_indices] - v_lead[controlled_indices]
            )
            dv[controlled_indices] += gap_term + speed_term + rel_speed_term

    return np.concatenate([ds, dv])


def jacobian(alpha: float, params: TrafficParams,
             controlled_indices: Iterable[int] | None = None) -> np.ndarray:
    """Analytical Jacobian of the nonlinear model at the homogeneous equilibrium.

    For the baseline system, the Jacobian has a simple block-circulant form.
    If control is enabled, the controlled cars receive extra linear terms.
    """

    n = params.n_cars
    j = np.zeros((2 * n, 2 * n), dtype=float)

    s_eq = equilibrium_gap(params)
    vp = optimal_velocity_prime(s_eq, params)

    for i in range(n):
        lead = (i + 1) % n

        # d/dt(delta s_i) = delta v_{i+1} - delta v_i
        j[i, n + i] = -1.0
        j[i, n + lead] = 1.0

        # d/dt(delta v_i) = alpha*V'(s_eq)*delta s_i - alpha*delta v_i
        j[n + i, i] = alpha * vp
        j[n + i, n + i] = -alpha

    if params.use_control and controlled_indices is not None:
        controlled_indices = np.asarray(list(controlled_indices), dtype=int)
        if controlled_indices.size > 0:
            for i in controlled_indices:
                lead = (i + 1) % n
                # Control law:
                #   u_i = -k_s (s_i - s_eq) - k_v (v_i - v_eq) - k_r (v_i - v_{i+1})
                j[n + i, i] += -params.control_gain_spacing
                j[n + i, n + i] += -params.control_gain_speed - params.control_gain_relative_speed
                j[n + i, n + lead] += params.control_gain_relative_speed

    return j


def mode_roots(alpha: float, params: TrafficParams) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form mode roots for the baseline model.

    This formula is valid when control is off. It is useful because it shows
    exactly where the Hopf bifurcation should appear.
    """

    s_eq = equilibrium_gap(params)
    vp = optimal_velocity_prime(s_eq, params)
    roots = []
    thetas = []
    for k in range(params.n_cars):
        theta = 2.0 * np.pi * k / params.n_cars
        q = np.exp(1j * theta)
        coeff = [1.0, alpha, alpha * vp * (1.0 - q)]
        r = np.roots(coeff)
        roots.append(r)
        thetas.append(theta)
    return np.asarray(thetas), np.asarray(roots)


def analytic_hopf_predictions(params: TrafficParams) -> list[dict[str, float]]:
    """Return analytic Hopf predictions for each nonzero Fourier mode.

    This is the clean baseline-only formula:

        alpha_c(k) = V'(s_eq) * (1 + cos(theta_k))
        omega_c(k) = V'(s_eq) * sin(theta_k)

    The mode k = 1 is typically the first one to destabilize.
    """

    s_eq = equilibrium_gap(params)
    vp = optimal_velocity_prime(s_eq, params)
    predictions = []
    for k in range(1, params.n_cars):
        theta = 2.0 * np.pi * k / params.n_cars
        alpha_c = vp * (1.0 + np.cos(theta))
        omega_c = vp * np.sin(theta)
        predictions.append({
            "mode": float(k),
            "theta": float(theta),
            "alpha_c": float(alpha_c),
            "omega_c": float(omega_c),
        })
    return predictions


def leading_nonneutral_real_part(alpha: float, params: TrafficParams,
                                 controlled_indices: Iterable[int] | None = None,
                                 neutral_tol: float = 1e-9) -> float:
    """Largest real part excluding the neutral translation/conservation modes."""

    eigs = np.linalg.eigvals(jacobian(alpha, params, controlled_indices))
    filtered = eigs[np.abs(eigs) > neutral_tol]
    return float(np.max(filtered.real))


def locate_crossing(alpha_values: np.ndarray, growth_rates: np.ndarray) -> tuple[float | None, int | None]:
    """Find the first sign change from negative to positive in a growth-rate curve."""

    for i in range(len(alpha_values) - 1):
        y0 = growth_rates[i]
        y1 = growth_rates[i + 1]
        if y0 == 0.0:
            return float(alpha_values[i]), i
        if y1 == 0.0:
            return float(alpha_values[i + 1]), i
        if y0 * y1 < 0.0:
            x0 = alpha_values[i]
            x1 = alpha_values[i + 1]
            t = -y0 / (y1 - y0)
            return float(x0 + t * (x1 - x0)), i
    return None, None


def rk4_step(state: np.ndarray, alpha: float, params: TrafficParams,
             controlled_indices: Iterable[int] | None, dt: float) -> np.ndarray:
    """One Runge-Kutta 4 step for the nonlinear simulation."""

    k1 = rhs(state, alpha, params, controlled_indices)
    k2 = rhs(state + 0.5 * dt * k1, alpha, params, controlled_indices)
    k3 = rhs(state + 0.5 * dt * k2, alpha, params, controlled_indices)
    k4 = rhs(state + dt * k3, alpha, params, controlled_indices)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(alpha: float, params: TrafficParams,
             controlled_indices: Iterable[int] | None = None) -> dict[str, np.ndarray]:
    """Simulate the nonlinear model from a small perturbation around equilibrium."""

    n_steps = int(np.ceil(params.sim_time / params.dt))
    times = np.linspace(0.0, params.sim_time, n_steps + 1)

    state = equilibrium_state(params)

    rng = np.random.default_rng(7)
    perturb = params.perturbation_amplitude * rng.standard_normal(state.shape)
    state = state + perturb

    s_hist = np.zeros((n_steps + 1, params.n_cars), dtype=float)
    v_hist = np.zeros((n_steps + 1, params.n_cars), dtype=float)
    s_hist[0] = state[:params.n_cars]
    v_hist[0] = state[params.n_cars:]

    for i in range(n_steps):
        state = rk4_step(state, alpha, params, controlled_indices, params.dt)
        s_hist[i + 1] = state[:params.n_cars]
        v_hist[i + 1] = state[params.n_cars:]

    return {"t": times, "s": s_hist, "v": v_hist}


def print_model_summary(params: TrafficParams, controlled_indices: np.ndarray) -> None:
    """Print the equations and key numbers in plain text."""

    s_eq = equilibrium_gap(params)
    v_eq = equilibrium_speed(params)
    vp = optimal_velocity_prime(s_eq, params)

    print("\n=== Traffic Hopf Analysis ===")
    print("Baseline equations:")
    print("  ds_i/dt = v_{i+1} - v_i")
    print("  dv_i/dt = alpha * (V(s_i) - v_i)")
    if params.use_control and controlled_indices.size > 0:
        print("Controlled cars:", controlled_indices.tolist())
        print("Controller:")
        print("  u_i = -k_s (s_i - s_eq) - k_v (v_i - v_eq) - k_r (v_i - v_{i+1})")
    else:
        print("Controller: off")

    print("\nEquilibrium:")
    print(f"  s_eq = {s_eq:.6f} m")
    print(f"  v_eq = {v_eq:.6f} m/s")
    print(f"  V'(s_eq) = {vp:.6f}")


def print_hopf_table(params: TrafficParams) -> None:
    """Print the analytic Hopf prediction for every nonzero Fourier mode."""

    print("\nAnalytic Hopf predictions for the baseline model:")
    print("  mode k | theta_k (rad) | alpha_c(k) | omega_c(k)")
    print("  ------ | ------------- | ---------- | ----------")
    for pred in analytic_hopf_predictions(params):
        print(
            f"  {int(pred['mode']):>6d} |"
            f" {pred['theta']:.6f} |"
            f" {pred['alpha_c']:.6f} |"
            f" {pred['omega_c']:.6f}"
        )


def plot_bifurcation_scan(alpha_values: np.ndarray,
                          baseline_growth: np.ndarray,
                          controlled_growth: np.ndarray,
                          alpha_c_baseline: float | None,
                          alpha_c_controlled: float | None) -> None:
    """Plot the growth-rate scan used to detect Hopf crossings."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alpha_values, baseline_growth, label="Baseline: max Re(lambda)", lw=2)
    ax.plot(alpha_values, controlled_growth, label="Controlled: max Re(lambda)", lw=2, ls="--")
    ax.axhline(0.0, color="black", lw=1)

    if alpha_c_baseline is not None:
        ax.axvline(alpha_c_baseline, color="tab:blue", ls=":", lw=2,
                   label=f"Baseline Hopf approx. alpha = {alpha_c_baseline:.3f}")
    if alpha_c_controlled is not None:
        ax.axvline(alpha_c_controlled, color="tab:orange", ls=":", lw=2,
                   label=f"Controlled Hopf approx. alpha = {alpha_c_controlled:.3f}")

    ax.set_title("Hopf scan: leading non-neutral eigenvalue real part")
    ax.set_xlabel("alpha")
    ax.set_ylabel("max Re(lambda) excluding neutral mode")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_BIFURCATION_PNG, dpi=200, bbox_inches="tight")


def plot_trajectories(results: dict[str, np.ndarray], title: str) -> None:
    """Plot one representative gap and speed trajectory."""

    t = results["t"]
    s = results["s"]
    v = results["v"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, s[:, 0], lw=2)
    axes[0].set_ylabel("Gap s_0")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, v[:, 0], lw=2, color="tab:red")
    axes[1].set_ylabel("Speed v_0")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()

    if title.startswith("Below Hopf point"):
        fig.savefig(OUTPUT_TRAJECTORY_BELOW_PNG, dpi=200, bbox_inches="tight")
    elif title.startswith("Above Hopf point"):
        fig.savefig(OUTPUT_TRAJECTORY_ABOVE_PNG, dpi=200, bbox_inches="tight")


def main() -> None:
    params = TrafficParams()
    controlled_indices = select_controlled_indices(params.n_cars, params.controlled_stride)

    print_model_summary(params, controlled_indices)
    print_hopf_table(params)

    alpha_values = np.linspace(params.alpha_min, params.alpha_max, params.alpha_steps)

    baseline_growth = np.array([
        leading_nonneutral_real_part(alpha, params, controlled_indices=None)
        for alpha in alpha_values
    ])

    if params.use_control and controlled_indices.size > 0:
        controlled_growth = np.array([
            leading_nonneutral_real_part(alpha, params, controlled_indices=controlled_indices)
            for alpha in alpha_values
        ])
    else:
        controlled_growth = baseline_growth.copy()

    alpha_c_baseline, baseline_idx = locate_crossing(alpha_values, baseline_growth)
    alpha_c_controlled, controlled_idx = locate_crossing(alpha_values, controlled_growth)

    print("\nNumerical scan:")
    if alpha_c_baseline is None:
        print("  Baseline scan did not find a sign change in the chosen alpha range.")
    else:
        print(f"  Baseline Hopf crossing approximately at alpha = {alpha_c_baseline:.6f}")
        print(f"  Crossing detected between samples {baseline_idx} and {baseline_idx + 1}")

    if params.use_control and controlled_indices.size > 0:
        if alpha_c_controlled is None:
            print("  Controlled scan did not find a sign change in the chosen alpha range.")
        else:
            print(f"  Controlled Hopf crossing approximately at alpha = {alpha_c_controlled:.6f}")
            print(f"  Crossing detected between samples {controlled_idx} and {controlled_idx + 1}")

    # Simulate on both sides of the baseline Hopf point if the crossing exists.
    if alpha_c_baseline is not None:
        alpha_below = max(params.alpha_min, 0.9 * alpha_c_baseline)
        alpha_above = min(params.alpha_max, 1.1 * alpha_c_baseline)
    else:
        alpha_below = 0.8 * params.alpha_max
        alpha_above = 0.95 * params.alpha_max

    below_results = simulate(alpha_below, params, controlled_indices=None)
    above_results = simulate(alpha_above, params, controlled_indices=None)

    print("\nSimulation values:")
    print(f"  Below-critical alpha = {alpha_below:.6f}")
    print(f"  Above-critical alpha = {alpha_above:.6f}")
    below_growth = leading_nonneutral_real_part(alpha_below, params, controlled_indices=None)
    above_growth = leading_nonneutral_real_part(alpha_above, params, controlled_indices=None)
    print(f"  Leading growth rate at alpha_below = {below_growth:.6f}")
    print(f"  Leading growth rate at alpha_above = {above_growth:.6f}")
    print("  One side of the Hopf point should have a negative leading real part")
    print("  and decay back to the equilibrium; the other side should have a")
    print("  positive leading real part and produce growing oscillations.")

    plot_bifurcation_scan(
        alpha_values,
        baseline_growth,
        controlled_growth,
        alpha_c_baseline,
        alpha_c_controlled,
    )
    plot_trajectories(below_results, f"Below Hopf point: alpha = {alpha_below:.3f}")
    plot_trajectories(above_results, f"Above Hopf point: alpha = {alpha_above:.3f}")

    plt.show()


if __name__ == "__main__":
    main()