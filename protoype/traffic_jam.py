import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==========================================
# 1. SIMULATION PARAMETERS
# ==========================================
L = 600.0             # Track length (expanded to give room to accelerate)
N = 55               # Number of cars
car_length = 5.0      # Length of each car (meters)
dt = 0.05             # Time step (seconds)

# Aggressive Human Drivers (Guarantees infinite shockwaves)
v0 = 20.0             # Desired speed (m/s)
T = 0.7               # Safe time headway (driving closer)
a_max = 1.2           # Maximum acceleration (m/s^2)
b = 0.5               # Weaker comfortable deceleration (forces overreaction)
s0 = 2.0              # Minimum gap (m)

# ==========================================
# SUPERCHARGED AV PARAMETERS
# ==========================================
# Number of autonomous vehicles in the controlled system (0..N)
AV_COUNT = 5
v_target = 6.0        # Drive slow to create a buffer
k1 = 0.5              # Strong speed tracking
k2 = 1.5              # Extreme damping (reacts instantly to lead car)
k3 = 0.1              # Gap control
s_safe = 25.0         # Demand a massive gap to absorb waves

# Event timing (seconds)
av_activation_time = 30.0
braking_start_time = 10.0
braking_end_time = 12.0

# Manual braking trigger (keyboard)
manual_brake_duration = 1.5
manual_brake_deceleration = -10.0


def select_av_indices(total_cars, av_count):
    av_count = int(np.clip(av_count, 0, total_cars))
    if av_count == 0:
        return np.array([], dtype=int)
    # Spread AVs around the ring to maximize damping coverage.
    return ((np.arange(av_count) * total_cars) / av_count).astype(int)


AV_INDICES = select_av_indices(N, AV_COUNT)

# ==========================================
# 2. INITIALIZATION
# ==========================================
np.random.seed(42) 
x_init = np.sort(np.linspace(0, L, N, endpoint=False))
# Start cars fast so they slam into the perturbation
v_init = np.ones(N) * 15.0 + np.random.uniform(-1, 1, N)

# State 1: With AV
x1 = x_init.copy()
v1 = v_init.copy()
AV_ACTIVE = False     

# State 2: Without AV (Baseline)
x2 = x_init.copy()
v2 = v_init.copy()

global_time = 0.0
manual_brake_car = None
manual_brake_time_remaining = 0.0

# Data tracking for the live graph
time_history = []
v1_marked_histories = {idx: [] for idx in AV_INDICES}
v2_marked_histories = {idx: [] for idx in AV_INDICES}

# ==========================================
# 3. PHYSICS ENGINE
# ==========================================
def calculate_accelerations(x, v, av_active, av_indices=None):
    dx = (np.roll(x, -1) - x) % L
    s = dx - car_length
    v_lead = np.roll(v, -1)
    delta_v = v - v_lead 
    
    s_safe_den = np.maximum(s, 0.5) 
    s_star = s0 + np.maximum(0, v * T + (v * delta_v) / (2 * np.sqrt(a_max * b)))
    accel = a_max * (1 - (v / v0)**4 - (s_star / s_safe_den)**2)
    
    if av_active and av_indices is not None and len(av_indices) > 0:
        gap_error = s[av_indices] - s_safe
        a_av = k1 * (v_target - v[av_indices]) - k2 * delta_v[av_indices] + k3 * gap_error
        a_av = np.clip(a_av, -3.0, 1.5)

        emergency_mask = s[av_indices] < s0 * 2
        accel[av_indices] = np.where(emergency_mask, accel[av_indices], a_av)
            
    return accel

# ==========================================
# 4. VISUALIZATION SETUP
# ==========================================
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

# Top Rings
ax1 = fig.add_subplot(gs[0, 0], polar=True)
ax2 = fig.add_subplot(gs[0, 1], polar=True)
# Bottom Graph
ax3 = fig.add_subplot(gs[1, :])

fig.suptitle("Phantom Traffic Jam: AV Damper vs Human Baseline", fontsize=16)

for ax in [ax1, ax2]:
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.grid(False)

ax1.set_title(f"System WITH {AV_COUNT} AV Dampers", color='green', pad=10)
ax2.set_title("System WITHOUT AV", color='red', pad=10)

# Heatmap colormap
norm = mcolors.Normalize(vmin=0, vmax=v0)
cmap = cm.get_cmap('RdYlGn')

theta_init = 2 * np.pi * (x_init / L)
r_init = np.ones(N)

scat1 = ax1.scatter(theta_init, r_init, s=50, zorder=3)
scat2 = ax2.scatter(theta_init, r_init, s=50, zorder=3)

# Setup Live Graph (speeds for marked cars)
line_pairs = []
line_palette = plt.cm.tab10(np.linspace(0, 1, max(1, AV_INDICES.size)))
for i, idx in enumerate(AV_INDICES):
    color = line_palette[i]
    line_av, = ax3.plot([], [], color=color, lw=2, label=f'AV car {idx}')
    line_base, = ax3.plot([], [], color=color, lw=2, ls='--', label=f'Baseline car {idx}')
    line_pairs.append((idx, line_av, line_base))

ax3.set_xlim(0, 100)
ax3.set_ylim(0, v0 + 2)
ax3.set_ylabel("Speed of Marked Cars (m/s)")
ax3.set_xlabel("Time (s)")
ax3.grid(True)
if AV_INDICES.size > 0:
    av_handles = [line_av for _, line_av, _ in line_pairs]
    base_handles = [line_base for _, _, line_base in line_pairs]
    legend_handles = av_handles + base_handles
    legend_labels = [h.get_label() for h in legend_handles]
    ax3.legend(legend_handles, legend_labels, loc='upper right', ncol=2, fontsize=8)
else:
    ax3.text(0.5, 0.5, 'No marked cars: set AV_COUNT > 0', transform=ax3.transAxes,
             ha='center', va='center', fontsize=11, color='gray')

time_text = fig.text(0.5, 0.45, '', ha='center', fontsize=12, fontweight='bold')
status_text = fig.text(0.5, 0.42, f'AV: OFF (Waiting for jam) | AV cars: {AV_COUNT}', ha='center', fontsize=12, color='gray')
control_text = fig.text(0.5, 0.39, 'Press SPACE to trigger random-car hard braking', ha='center', fontsize=10, color='dimgray')


def on_key_press(event):
    global manual_brake_car, manual_brake_time_remaining
    if event.key in (' ', 'space'):
        manual_brake_car = int(np.random.randint(0, N))
        manual_brake_time_remaining = manual_brake_duration
        print(f"Manual brake triggered: car {manual_brake_car} for {manual_brake_duration:.1f}s")


fig.canvas.mpl_connect('key_press_event', on_key_press)

def update(frame):
    global x1, v1, x2, v2, AV_ACTIVE, global_time, manual_brake_car, manual_brake_time_remaining
    
    steps_per_frame = 5
    for _ in range(steps_per_frame):
        global_time += dt
        
        if global_time > av_activation_time and not AV_ACTIVE:
            AV_ACTIVE = True
            status_text.set_text(f'AV: ON (Damping Wave) | AV cars: {AV_COUNT}')
            status_text.set_color('green')
            
        accel1 = calculate_accelerations(x1, v1, AV_ACTIVE, AV_INDICES)
        accel2 = calculate_accelerations(x2, v2, False)
        
        # Severe braking to cause the jam
        if braking_start_time < global_time < braking_end_time:
            accel1[N // 2] = -10.0
            accel2[N // 2] = -10.0

        # Optional user-triggered perturbation (SPACE key)
        if manual_brake_time_remaining > 0 and manual_brake_car is not None:
            accel1[manual_brake_car] = manual_brake_deceleration
            accel2[manual_brake_car] = manual_brake_deceleration
            manual_brake_time_remaining -= dt
            if manual_brake_time_remaining <= 0:
                manual_brake_time_remaining = 0.0
                manual_brake_car = None

        v1 = np.clip(v1 + accel1 * dt, 0, v0)
        x1 = (x1 + v1 * dt) % L
        
        v2 = np.clip(v2 + accel2 * dt, 0, v0)
        x2 = (x2 + v2 * dt) % L
        
    # Update Rings
    scat1.set_offsets(np.column_stack([2 * np.pi * (x1 / L), np.ones(N)]))
    scat2.set_offsets(np.column_stack([2 * np.pi * (x2 / L), np.ones(N)]))
    
    # Update Colors
    c1, c2 = cmap(norm(v1)), cmap(norm(v2))
    if AV_INDICES.size > 0:
        c1[AV_INDICES] = mcolors.to_rgba('magenta')  # AVs in controlled system
        c2[AV_INDICES] = mcolors.to_rgba('cyan')     # Same car IDs in baseline
    scat1.set_color(c1)
    scat2.set_color(c2)
    
    # Update graph with marked-car speeds
    time_history.append(global_time)
    for idx, line_av, line_base in line_pairs:
        v1_marked_histories[idx].append(v1[idx])
        v2_marked_histories[idx].append(v2[idx])
        line_av.set_data(time_history, v1_marked_histories[idx])
        line_base.set_data(time_history, v2_marked_histories[idx])
    
    # Scroll graph if time exceeds x-axis limit
    if global_time > ax3.get_xlim()[1]:
        ax3.set_xlim(global_time - 100, global_time + 20)
    
    time_text.set_text(f'Time: {global_time:.1f} s')

    line_artists = []
    for _, line_av, line_base in line_pairs:
        line_artists.extend([line_av, line_base])
    return (scat1, scat2, time_text, status_text, *line_artists)

ani = animation.FuncAnimation(fig, update, frames=2000, blit=False, interval=10)
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.10, hspace=0.3)
plt.show()
