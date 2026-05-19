"""Enhanced phantom-traffic simulation with Hopf-aware diagnostics.

This script is a new simulation file that merges:
1. Ring-road traffic animation (baseline vs controlled), and
2. Dynamics and considerations from the Hopf analysis scripts.

Model (for each vehicle i on a ring)
------------------------------------
State:
    x_i(t): position on ring
    v_i(t): speed

Gap and leader speed:
    s_i = (x_{i+1} - x_i) mod L - l
    v_{i+1} = leader speed

Human-driver baseline acceleration:
    a_i = alpha * (V(s_i) - v_i)

Optimal-velocity function:
    V(s) = 0.5 * v_max * (tanh((s - s_c)/w) + 1)

AV feedback (for selected AV cars):
    u_i = -k_s (s_i - s_eq) - k_v (v_i - v_eq) - k_r (v_i - v_{i+1})
    a_i <- a_i + u_i

Integrators:
    xdot_i = v_i
    vdot_i = a_i

Two systems are simulated side-by-side:
    - Baseline (human only)
    - Controlled (AV feedback enabled after activation time)

Live diagnostics shown in the figure:
    - Dominant non-zero Fourier mode amplitude of speed wave
    - Running speed envelope of one observed car (branch-like indicator)
    - Leading Jacobian real part at equilibrium (linear stability indicator)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors


@dataclass(frozen=True)
class Config:
    # Geometry and discretization
    road_length: float = 500.0
    n_cars: int = 20
    car_length: float = 5.0
    dt: float = 0.05

    # Optimal-velocity dynamics
    alpha: float = 2.8
    v_max: float = 30.0
    v_shape_center: float = 20.0
    v_shape_width: float = 10.0

    # AV controller
    av_count: int = 4
    av_activation_time: float = 30.0
    k_s: float = 1.2
    k_v: float = 0.8
    k_r: float = 0.4

    # Disturbance events
    auto_brake_start: float = 10.0
    auto_brake_end: float = 12.0
    auto_brake_decel: float = -8.0
    manual_brake_duration: float = 1.2
    manual_brake_decel: float = -9.0

    # Numerical/visual settings
    sim_steps_per_frame: int = 4
    animation_interval_ms: int = 12
    total_frames: int = 1800

    # Initial random perturbations around homogeneous equilibrium
    init_position_noise: float = 0.20
    init_speed_noise: float = 0.40

    # Safety clips for acceleration and speed
    accel_min: float = -10.0
    accel_max: float = 4.0
    speed_min: float = 0.0

    # Diagnostics
    observed_car: int = 0
    envelope_window: float = 20.0  # seconds
    diag_every_n_frames: int = 10


CFG = Config()


def select_av_indices(total_cars: int, av_count: int) -> np.ndarray:
    av_count = int(np.clip(av_count, 0, total_cars))
    if av_count == 0:
        return np.array([], dtype=int)
    return ((np.arange(av_count) * total_cars) / av_count).astype(int)


def optimal_velocity(gap: np.ndarray) -> np.ndarray:
    z = (gap - CFG.v_shape_center) / CFG.v_shape_width
    return 0.5 * CFG.v_max * (np.tanh(z) + 1.0)


def optimal_velocity_prime(gap_scalar: float) -> float:
    z = (gap_scalar - CFG.v_shape_center) / CFG.v_shape_width
    sech2 = 1.0 - np.tanh(z) ** 2
    return 0.5 * CFG.v_max * sech2 / CFG.v_shape_width


def equilibrium_gap() -> float:
    usable = CFG.road_length - CFG.n_cars * CFG.car_length
    return usable / CFG.n_cars


def equilibrium_speed() -> float:
    return float(optimal_velocity(np.array([equilibrium_gap()]))[0])


def gaps_and_leaders(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = (np.roll(x, -1) - x) % CFG.road_length
    gaps = dx - CFG.car_length
    v_lead = np.roll(v, -1)
    return gaps, v_lead


def accelerations(x: np.ndarray, v: np.ndarray, av_on: bool, av_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gaps, v_lead = gaps_and_leaders(x, v)
    acc = CFG.alpha * (optimal_velocity(gaps) - v)

    if av_on and av_indices.size > 0:
        s_eq = equilibrium_gap()
        v_eq = equilibrium_speed()
        idx = av_indices

        u = (
            -CFG.k_s * (gaps[idx] - s_eq)
            -CFG.k_v * (v[idx] - v_eq)
            -CFG.k_r * (v[idx] - v_lead[idx])
        )
        acc[idx] += u

    acc = np.clip(acc, CFG.accel_min, CFG.accel_max)
    return acc, gaps


def integrate_euler(x: np.ndarray, v: np.ndarray, acc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Fast explicit integrator used for real-time animation speed."""

    v_new = np.clip(v + acc * dt, CFG.speed_min, CFG.v_max)
    x_new = (x + v_new * dt) % CFG.road_length
    return x_new, v_new


def jacobian_equilibrium(av_on: bool, av_indices: np.ndarray) -> np.ndarray:
    n = CFG.n_cars
    j = np.zeros((2 * n, 2 * n), dtype=float)
    vp = optimal_velocity_prime(equilibrium_gap())

    for i in range(n):
        lead = (i + 1) % n

        # d/dt delta s_i = delta v_{i+1} - delta v_i
        j[i, n + i] = -1.0
        j[i, n + lead] = 1.0

        # d/dt delta v_i = alpha*V'(s_eq) delta s_i - alpha delta v_i
        j[n + i, i] = CFG.alpha * vp
        j[n + i, n + i] = -CFG.alpha

    if av_on and av_indices.size > 0:
        for i in av_indices:
            lead = (i + 1) % n
            j[n + i, i] += -CFG.k_s
            j[n + i, n + i] += -CFG.k_v - CFG.k_r
            j[n + i, n + lead] += CFG.k_r

    return j


def leading_real_part(av_on: bool, av_indices: np.ndarray) -> float:
    eigs = np.linalg.eigvals(jacobian_equilibrium(av_on, av_indices))
    # Drop neutral translation-like near-zero modes.
    filt = eigs[np.abs(eigs) > 1e-9]
    return float(np.max(filt.real))


def dominant_mode_amplitude(v: np.ndarray) -> float:
    vk = np.fft.fft(v - np.mean(v))
    amps = np.abs(vk) / v.size
    if v.size <= 2:
        return 0.0
    return float(np.max(amps[1:v.size // 2 + 1]))


def initial_state(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_x = np.linspace(0.0, CFG.road_length, CFG.n_cars, endpoint=False)
    x = base_x + rng.normal(0.0, CFG.init_position_noise, CFG.n_cars)
    x = np.sort(x % CFG.road_length)

    v_eq = equilibrium_speed()
    v = np.full(CFG.n_cars, v_eq) + rng.normal(0.0, CFG.init_speed_noise, CFG.n_cars)
    v = np.clip(v, CFG.speed_min, CFG.v_max)
    return x, v


AV_INDICES = select_av_indices(CFG.n_cars, CFG.av_count)

# Two parallel worlds: controlled vs baseline
x_control, v_control = initial_state(seed=8)
x_base, v_base = x_control.copy(), v_control.copy()

global_time = 0.0
av_active = False

manual_brake_car: int | None = None
manual_brake_remaining = 0.0
frame_counter = 0
cached_lead_real_control = 0.0
cached_lead_real_base = 0.0

time_hist: list[float] = []
wave_control_hist: list[float] = []
wave_base_hist: list[float] = []
obs_control_hist: list[float] = []
obs_base_hist: list[float] = []


fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.1])

ax_control = fig.add_subplot(gs[0, 0], polar=True)
ax_base = fig.add_subplot(gs[0, 1], polar=True)
ax_speed = fig.add_subplot(gs[1, 0])
ax_wave = fig.add_subplot(gs[1, 1])

for ax in (ax_control, ax_base):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)

ax_control.set_title(f"Controlled System (AV count = {CFG.av_count})", color="green")
ax_base.set_title("Baseline System (Human Only)", color="crimson")

norm = mcolors.Normalize(vmin=0.0, vmax=CFG.v_max)
cmap = plt.colormaps["RdYlGn"]

theta0 = 2.0 * np.pi * (x_control / CFG.road_length)
scatter_control = ax_control.scatter(theta0, np.ones(CFG.n_cars), s=55, zorder=3)
scatter_base = ax_base.scatter(theta0, np.ones(CFG.n_cars), s=55, zorder=3)

line_obs_control, = ax_speed.plot([], [], lw=2, color="tab:green", label="Observed car (controlled)")
line_obs_base, = ax_speed.plot([], [], lw=2, ls="--", color="tab:red", label="Observed car (baseline)")
ax_speed.set_ylabel("Observed Speed (m/s)")
ax_speed.set_xlabel("Time (s)")
ax_speed.grid(True, alpha=0.3)
ax_speed.legend(loc="upper right", fontsize=8)

line_wave_control, = ax_wave.plot([], [], lw=2, color="tab:blue", label="Wave amplitude (controlled)")
line_wave_base, = ax_wave.plot([], [], lw=2, ls="--", color="tab:orange", label="Wave amplitude (baseline)")
ax_wave.set_ylabel("Dominant Mode Amplitude")
ax_wave.set_xlabel("Time (s)")
ax_wave.grid(True, alpha=0.3)
ax_wave.legend(loc="upper right", fontsize=8)

title_text = fig.text(0.5, 0.955, "Enhanced Phantom Jam: Hopf-Aware Simulation", ha="center", fontsize=14, fontweight="bold")
time_text = fig.text(0.5, 0.92, "", ha="center", fontsize=11)
status_text = fig.text(0.5, 0.895, "", ha="center", fontsize=11, color="dimgray")
diag_text = fig.text(0.5, 0.87, "", ha="center", fontsize=10, color="dimgray")
control_hint_text = fig.text(0.5, 0.845, "Press SPACE to trigger manual hard brake on random car", ha="center", fontsize=10, color="gray")


def on_key_press(event) -> None:
    global manual_brake_car, manual_brake_remaining
    if event.key in (" ", "space"):
        manual_brake_car = int(np.random.randint(0, CFG.n_cars))
        manual_brake_remaining = CFG.manual_brake_duration
        print(f"Manual brake triggered on car {manual_brake_car} for {manual_brake_remaining:.2f}s")


fig.canvas.mpl_connect("key_press_event", on_key_press)


def apply_forced_brakes(acc_control: np.ndarray, acc_base: np.ndarray) -> None:
    global manual_brake_remaining, manual_brake_car

    # Scheduled perturbation.
    if CFG.auto_brake_start < global_time < CFG.auto_brake_end:
        idx = CFG.n_cars // 2
        acc_control[idx] = CFG.auto_brake_decel
        acc_base[idx] = CFG.auto_brake_decel

    # Manual perturbation.
    if manual_brake_remaining > 0.0 and manual_brake_car is not None:
        idx = manual_brake_car
        acc_control[idx] = CFG.manual_brake_decel
        acc_base[idx] = CFG.manual_brake_decel
        manual_brake_remaining -= CFG.dt
        if manual_brake_remaining <= 0.0:
            manual_brake_remaining = 0.0
            manual_brake_car = None


def update(frame: int):
    global x_control, v_control, x_base, v_base, global_time, av_active
    global frame_counter, cached_lead_real_control, cached_lead_real_base

    # Advance multiple physics steps per rendered frame for faster simulated time.
    for _ in range(CFG.sim_steps_per_frame):
        global_time += CFG.dt
        if (not av_active) and global_time >= CFG.av_activation_time:
            av_active = True

        acc_control, _ = accelerations(x_control, v_control, av_active, AV_INDICES)
        acc_base, _ = accelerations(x_base, v_base, False, AV_INDICES)
        apply_forced_brakes(acc_control, acc_base)

        x_control, v_control = integrate_euler(x_control, v_control, acc_control, CFG.dt)
        x_base, v_base = integrate_euler(x_base, v_base, acc_base, CFG.dt)

    frame_counter += 1

    # Update ring positions.
    scatter_control.set_offsets(np.column_stack([2.0 * np.pi * (x_control / CFG.road_length), np.ones(CFG.n_cars)]))
    scatter_base.set_offsets(np.column_stack([2.0 * np.pi * (x_base / CFG.road_length), np.ones(CFG.n_cars)]))

    c_control = cmap(norm(v_control))
    c_base = cmap(norm(v_base))
    if AV_INDICES.size > 0:
        c_control[AV_INDICES] = mcolors.to_rgba("magenta")
        c_base[AV_INDICES] = mcolors.to_rgba("cyan")
    scatter_control.set_color(c_control)
    scatter_base.set_color(c_base)

    # Diagnostics
    obs = int(np.clip(CFG.observed_car, 0, CFG.n_cars - 1))
    wave_control = dominant_mode_amplitude(v_control)
    wave_base = dominant_mode_amplitude(v_base)

    time_hist.append(global_time)
    wave_control_hist.append(wave_control)
    wave_base_hist.append(wave_base)
    obs_control_hist.append(float(v_control[obs]))
    obs_base_hist.append(float(v_base[obs]))

    line_obs_control.set_data(time_hist, obs_control_hist)
    line_obs_base.set_data(time_hist, obs_base_hist)
    line_wave_control.set_data(time_hist, wave_control_hist)
    line_wave_base.set_data(time_hist, wave_base_hist)

    # Dynamic x-limits
    for ax in (ax_speed, ax_wave):
        right = max(60.0, global_time + 5.0)
        left = max(0.0, right - 120.0)
        ax.set_xlim(left, right)

    # Adaptive y-limits
    if obs_control_hist and obs_base_hist:
        vmax_plot = max(np.max(obs_control_hist[-500:]), np.max(obs_base_hist[-500:]), 1.0)
        ax_speed.set_ylim(0.0, min(CFG.v_max + 1.0, vmax_plot + 2.0))
    if wave_control_hist and wave_base_hist:
        wmax = max(np.max(wave_control_hist[-500:]), np.max(wave_base_hist[-500:]), 1e-3)
        ax_wave.set_ylim(0.0, wmax * 1.25)

    # Running envelope over trailing window.
    n_win = max(5, int(CFG.envelope_window / CFG.dt))
    tail_control = np.array(obs_control_hist[-n_win:])
    tail_base = np.array(obs_base_hist[-n_win:])
    amp_control = 0.5 * float(np.max(tail_control) - np.min(tail_control))
    amp_base = 0.5 * float(np.max(tail_base) - np.min(tail_base))

    # Cache expensive eigenvalue diagnostics instead of recomputing every frame.
    if frame_counter % CFG.diag_every_n_frames == 0:
        cached_lead_real_control = leading_real_part(av_active, AV_INDICES)
        cached_lead_real_base = leading_real_part(False, AV_INDICES)

    lead_real_control = cached_lead_real_control
    lead_real_base = cached_lead_real_base

    time_text.set_text(f"t = {global_time:.1f}s | alpha = {CFG.alpha:.2f} | observed car = {obs}")
    status_text.set_text(
        f"AV {'ON' if av_active else 'OFF'} | Leading Re(lambda): controlled={lead_real_control:+.4f}, baseline={lead_real_base:+.4f}"
    )
    status_text.set_color("green" if av_active else "dimgray")
    diag_text.set_text(
        f"Wave amp: controlled={wave_control:.4f}, baseline={wave_base:.4f} | "
        f"Running branch envelope (speed): controlled={amp_control:.4f}, baseline={amp_base:.4f}"
    )

    artists = [
        scatter_control,
        scatter_base,
        line_obs_control,
        line_obs_base,
        line_wave_control,
        line_wave_base,
        title_text,
        time_text,
        status_text,
        diag_text,
        control_hint_text,
    ]
    return tuple(artists)


ani = animation.FuncAnimation(
    fig,
    update,
    frames=CFG.total_frames,
    interval=CFG.animation_interval_ms,
    blit=False,
)

# Keep a second strong reference so aggressive cleanup paths do not collect it.
_ANI_REF = ani

plt.tight_layout()
plt.subplots_adjust(top=0.80, bottom=0.08, wspace=0.22, hspace=0.35)
plt.show()
