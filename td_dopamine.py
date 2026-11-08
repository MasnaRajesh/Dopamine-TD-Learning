"""
Project 22: Dopamine and temporal-difference (TD) learning
-----------------------------------------------------------
Reproduces the classic reward-prediction-error results (Schultz, Dayan &
Montague, 1997) with tabular TD(0) on a bell -> wait -> food task.

Task (one "trial"):
    time steps t = 0 .. T ; bell (cue) appears at t = CUE ; food (reward = 1)
    arrives at t = REWARD.  State = time step since the trial began
    (a "complete serial compound" representation: one table entry per step).

TD(0):
    delta_t = r_t + gamma * V[t] - V[t-1]        (TD error = "surprise")
    V[t-1] <- V[t-1] + alpha * delta_t            (only for t-1 >= CUE)
States before the cue are never updated (V = 0): the bell arrives at an
unpredictable moment, so the agent cannot learn to anticipate it.

Run:  python td_dopamine.py        (makes results in ./results and results.txt)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- settings ----------------
T = 12            # last time step of a trial
CUE = 2           # bell appears at this step
REWARD = 8        # food arrives at this step
ALPHA = 0.1       # learning rate
GAMMA = 1.0       # discount factor (episodic task, so 1 is fine)
N_TRIALS = 300
SEED = 0

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "results")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREY, RED, GREEN = "#2563eb", "#ea580c", "#6b7280", "#dc2626", "#16a34a"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


# ---------------- core TD(0) ----------------
def run_trial(V, alpha, p_reward=1.0, reward_time=REWARD, learn=True, rng=None):
    """Run one trial. Returns delta[t] for t=0..T (delta[0] is always 0)."""
    delta = np.zeros(T + 1)
    got_reward = True if p_reward >= 1.0 else (rng.random() < p_reward)
    for t in range(1, T + 1):
        r = 1.0 if (t == reward_time and got_reward) else 0.0
        delta[t] = r + GAMMA * V[t] - V[t - 1]
        if learn and t - 1 >= CUE:
            V[t - 1] += alpha * delta[t]
    V[T] = 0.0          # terminal state keeps value 0
    return delta


def train(n_trials=N_TRIALS, alpha=ALPHA, p_reward=1.0, seed=SEED, learn=True):
    rng = np.random.default_rng(seed)
    V = np.zeros(T + 1)
    deltas = np.zeros((n_trials, T + 1))
    for k in range(n_trials):
        deltas[k] = run_trial(V, alpha, p_reward, learn=learn, rng=rng)
    return V, deltas


def probe(V, reward_time=REWARD, deliver=True):
    """One test trial with NO learning (so the agent is not changed)."""
    rng = np.random.default_rng(1)
    return run_trial(V.copy(), 0.0, 1.0 if deliver else 0.0,
                     reward_time=reward_time, learn=False, rng=rng)


lines = []          # numbers that go in the report / slides
def log(s):
    print(s)
    lines.append(s)


# =========================================================
# Experiment 1 (core): the surprise moves from food to bell
# =========================================================
V, D = train()
_, D_base = train(learn=False)       # baseline: an agent that never learns
trials = np.arange(1, N_TRIALS + 1)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(trials, D[:, CUE], color=BLUE, lw=2.4, label="TD agent: surprise at BELL")
ax.plot(trials, D[:, REWARD], color=ORANGE, lw=2.4, label="TD agent: surprise at FOOD")
ax.plot(trials, D_base[:, REWARD], color=GREY, lw=2, ls="--",
        label="Baseline (no learning): surprise at FOOD")
ax.set_xlabel("Trial number")
ax.set_ylabel("TD error (surprise), δ")
ax.set_title("Learning curve: surprise moves from the food to the bell")
ax.legend(frameon=False, fontsize=10, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_learning_curve.png"))
plt.close(fig)

log("== Experiment 1: learning curve (alpha=%.2f, %d trials) ==" % (ALPHA, N_TRIALS))
for k in (1, 10, 25, 50, 100, 300):
    log("trial %3d: delta@bell=%.3f  delta@food=%.3f" % (k, D[k - 1, CUE], D[k - 1, REWARD]))
log("baseline (no learning) delta@food on every trial = %.3f" % D_base[:, REWARD].mean())
log("learned values V[cue..reward-1] = " + ", ".join("%.3f" % v for v in V[CUE:REWARD]))

# =========================================================
# Experiment 2 (extra): heatmap of delta over time x trials
# =========================================================
fig, ax = plt.subplots(figsize=(7.5, 4.6))
lim = 1.0
im = ax.imshow(D[:150].T, aspect="auto", origin="lower", cmap="RdBu_r",
               vmin=-lim, vmax=lim, extent=[0.5, 150.5, -0.5, T + 0.5])
ax.axhline(CUE, color="k", lw=0.8, ls=":")
ax.axhline(REWARD, color="k", lw=0.8, ls=":")
ax.text(151.5, CUE, "bell", va="center", fontsize=10)
ax.text(151.5, REWARD, "food", va="center", fontsize=10)
ax.set_xlabel("Trial number")
ax.set_ylabel("Time step within trial")
ax.set_title("TD error over time and trials (red = positive surprise)")
cb = fig.colorbar(im, ax=ax, pad=0.1)
cb.set_label("δ")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_heatmap.png"))
plt.close(fig)

# =========================================================
# Experiment 3: the three patterns as time traces (Schultz-style)
#   before learning / after learning / food omitted
# =========================================================
d_before = D[0]
d_after = probe(V)
d_omit = probe(V, deliver=False)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
panels = [(d_before, "1. Before learning\n(food is a surprise)"),
          (d_after, "2. After learning\n(bell is the surprise)"),
          (d_omit, "3. Food omitted\n(negative dip)")]
for ax, (d, title) in zip(axes, panels):
    cols = [RED if x < -1e-9 else BLUE for x in d]
    ax.bar(np.arange(T + 1), d, color=cols, width=0.7)
    ax.axvline(CUE, color="k", lw=0.8, ls=":")
    ax.axvline(REWARD, color="k", lw=0.8, ls=":")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time step")
    ax.set_ylim(-1.25, 1.25)
    ax.text(CUE, 1.12, "bell", ha="center", fontsize=9)
    ax.text(REWARD, 1.12, "food", ha="center", fontsize=9)
axes[0].set_ylabel("TD error δ")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_three_patterns.png"))
plt.close(fig)

log("")
log("== Experiment 3: three patterns ==")
log("before learning : delta@bell=%.3f delta@food=%.3f" % (d_before[CUE], d_before[REWARD]))
log("after learning  : delta@bell=%.3f delta@food=%.3f" % (d_after[CUE], d_after[REWARD]))
log("food omitted    : delta@bell=%.3f delta@food=%.3f" % (d_omit[CUE], d_omit[REWARD]))

# =========================================================
# Experiment 4 (extra): effect of the learning rate alpha
# =========================================================
fig, ax = plt.subplots(figsize=(7.5, 4.2))
log("")
log("== Experiment 4: learning-rate sweep (trials until |delta@food| < 0.05) ==")
for a, c in [(0.03, GREY), (0.1, BLUE), (0.3, ORANGE), (0.6, GREEN)]:
    _, Da = train(alpha=a)
    ax.plot(trials, Da[:, REWARD], color=c, lw=2.2, label="α = %.2f" % a)
    below = np.where(np.abs(Da[:, REWARD]) < 0.05)[0]
    n_conv = (below[0] + 1) if len(below) else None
    log("alpha=%.2f: %s trials" % (a, n_conv if n_conv else ">%d" % N_TRIALS))
ax.set_xlabel("Trial number")
ax.set_ylabel("δ at food")
ax.set_title("Bigger learning rate → surprise at the food vanishes sooner")
ax.legend(frameon=False)
ax.set_xlim(0, 150)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_alpha_sweep.png"))
plt.close(fig)

# =========================================================
# Experiment 5 (extra): reward probability (Fiorillo et al., 2003)
# =========================================================
probs = [0.0, 0.25, 0.5, 0.75, 1.0]
log("")
log("== Experiment 5: reward probability (mean over last 1000 of 3000 trials) ==")
cue_resp, rew_resp, omit_resp = [], [], []
for p in probs:
    n = 3000
    rng = np.random.default_rng(SEED)
    Vp = np.zeros(T + 1)
    cue_l, rew_l, omit_l = [], [], []
    for k in range(n):
        delivered = (p >= 1.0) or (rng.random() < p)
        delta = np.zeros(T + 1)
        for t in range(1, T + 1):
            r = 1.0 if (t == REWARD and delivered) else 0.0
            delta[t] = r + GAMMA * Vp[t] - Vp[t - 1]
            if t - 1 >= CUE:
                Vp[t - 1] += 0.05 * delta[t]
        Vp[T] = 0.0
        if k >= n - 1000:
            cue_l.append(delta[CUE])
            (rew_l if delivered else omit_l).append(delta[REWARD])
    cue_resp.append(np.mean(cue_l))
    rew_resp.append(np.mean(rew_l) if rew_l else np.nan)
    omit_resp.append(np.mean(omit_l) if omit_l else np.nan)
    log("p=%.2f: delta@bell=%.2f  delta@food(if given)=%s  delta@food(if omitted)=%s" % (
        p, cue_resp[-1],
        "%.2f" % rew_resp[-1] if not np.isnan(rew_resp[-1]) else "n/a",
        "%.2f" % omit_resp[-1] if not np.isnan(omit_resp[-1]) else "n/a"))

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(probs, cue_resp, "o-", color=BLUE, lw=2.4, ms=7, label="δ at bell")
ax.plot(probs, rew_resp, "s-", color=ORANGE, lw=2.4, ms=7, label="δ at food (when given)")
ax.plot(probs, omit_resp, "^-", color=RED, lw=2.4, ms=7, label="δ at food time (when omitted)")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("Probability that the bell is followed by food")
ax.set_ylabel("Average TD error δ")
ax.set_title("Bell response grows with the probability of food")
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_reward_probability.png"))
plt.close(fig)

# =========================================================
# Experiment 6 (extra): timing - food comes earlier / later than learned
# =========================================================
d_early = probe(V, reward_time=REWARD - 2)
d_late = probe(V, reward_time=REWARD + 2)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
panels = [(d_after, "On time (as learned)"),
          (d_early, "Food arrives 2 steps EARLY"),
          (d_late, "Food arrives 2 steps LATE")]
for ax, (d, title) in zip(axes, panels):
    cols = [RED if x < -1e-9 else BLUE for x in d]
    ax.bar(np.arange(T + 1), d, color=cols, width=0.7)
    ax.axvline(REWARD, color="k", lw=0.8, ls=":")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time step")
    ax.set_ylim(-1.25, 1.75)
    ax.text(REWARD, 1.2, "learned\nfood time", ha="center", fontsize=8)
axes[0].set_ylabel("TD error δ")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_timing.png"))
plt.close(fig)

log("")
log("== Experiment 6: timing ==")
log("early (t=%d): delta@early=%.3f  delta@learned time(t=%d)=%.3f" % (
    REWARD - 2, d_early[REWARD - 2], REWARD, d_early[REWARD]))
log("late  (t=%d): delta@learned time(t=%d)=%.3f  delta@late(t=%d)=%.3f" % (
    REWARD + 2, REWARD, d_late[REWARD], REWARD + 2, d_late[REWARD + 2]))

with open(os.path.join(OUT, "results.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nSaved figures to", FIG)


# =========================================================
# Slide-sized versions of the figures (bigger text, sized for the deck)
# =========================================================
import json
from PIL import Image

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})
dims = {}

def save_slide(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    with Image.open(path) as im:
        dims[name] = im.size

# --- learning curve (slide 3, left)
fig, ax = plt.subplots(figsize=(4.7, 2.75))
ax.plot(trials, D[:, CUE], color=BLUE, lw=2.4, label="Bell (TD agent)")
ax.plot(trials, D[:, REWARD], color=ORANGE, lw=2.4, label="Food (TD agent)")
ax.plot(trials, D_base[:, REWARD], color=GREY, lw=2, ls="--", label="Food (no-learning baseline)")
ax.set_xlabel("Trial number")
ax.set_ylabel("TD error δ")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.45))
save_slide(fig, "slide_learning_curve.png")

# --- three patterns, stacked (slide 3, right)
fig, axes = plt.subplots(3, 1, figsize=(4.3, 3.9), sharex=True, sharey=True)
for ax, (d, title) in zip(axes, [(d_before, "1. Before learning: food is the surprise"),
                                 (d_after, "2. After learning: bell is the surprise"),
                                 (d_omit, "3. Food omitted: negative dip")]):
    ax.bar(np.arange(T + 1), d, color=[RED if x < -1e-9 else BLUE for x in d], width=0.7)
    ax.axvline(CUE, color="k", lw=0.8, ls=":")
    ax.axvline(REWARD, color="k", lw=0.8, ls=":")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylim(-1.3, 1.3)
    ax.set_yticks([-1, 0, 1])
axes[0].text(CUE + 0.15, 1.02, "bell", fontsize=10)
axes[0].text(REWARD + 0.15, 1.02, "food", fontsize=10)
axes[2].set_xlabel("Time step within the trial")
axes[1].set_ylabel("TD error δ")
fig.tight_layout()
save_slide(fig, "slide_three_patterns.png")

# --- heatmap (backup 1)
fig, ax = plt.subplots(figsize=(4.4, 3.0))
im = ax.imshow(D[:150].T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
               extent=[0.5, 150.5, -0.5, T + 0.5])
ax.axhline(CUE, color="k", lw=0.8, ls=":")
ax.axhline(REWARD, color="k", lw=0.8, ls=":")
ax.text(3, CUE + 0.35, "bell", fontsize=10, color="k")
ax.text(3, REWARD + 0.35, "food", fontsize=10, color="k")
ax.set_xlabel("Trial number")
ax.set_ylabel("Time step")
cb = fig.colorbar(im, ax=ax, pad=0.03)
cb.set_label("TD error δ")
fig.tight_layout()
save_slide(fig, "slide_heatmap.png")

# --- reward probability (backup 1)
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.plot(probs, cue_resp, "o-", color=BLUE, lw=2.2, ms=6, label="δ at bell")
ax.plot(probs, rew_resp, "s-", color=ORANGE, lw=2.2, ms=6, label="δ at food (given)")
ax.plot(probs, omit_resp, "^-", color=RED, lw=2.2, ms=6, label="δ at food time (omitted)")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("Probability that the bell is followed by food")
ax.set_ylabel("Average TD error δ")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=10)
save_slide(fig, "slide_reward_probability.png")

# --- learning-rate sweep (backup 2)
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for a, c in [(0.03, GREY), (0.1, BLUE), (0.3, ORANGE), (0.6, GREEN)]:
    _, Da = train(alpha=a)
    ax.plot(trials, Da[:, REWARD], color=c, lw=2.2, label="α = %.2f" % a)
ax.set_xlim(0, 150)
ax.set_xlabel("Trial number")
ax.set_ylabel("TD error δ at food")
ax.legend(frameon=False)
save_slide(fig, "slide_alpha_sweep.png")

# --- timing (backup 2)
fig, axes = plt.subplots(3, 1, figsize=(4.4, 3.4), sharex=True, sharey=True)
for ax, (d, title) in zip(axes, [(d_after, "On time (as learned)"),
                                 (d_early, "Food 2 steps early"),
                                 (d_late, "Food 2 steps late")]):
    ax.bar(np.arange(T + 1), d, color=[RED if x < -1e-9 else BLUE for x in d], width=0.7)
    ax.axvline(REWARD, color="k", lw=0.8, ls=":")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylim(-1.3, 1.3)
    ax.set_yticks([-1, 0, 1])
axes[0].text(REWARD + 0.15, 0.8, "learned food time", fontsize=10)
axes[2].set_xlabel("Time step within the trial")
axes[1].set_ylabel("TD error δ")
fig.tight_layout()
save_slide(fig, "slide_timing.png")

with open(os.path.join(FIG, "slide_dims.json"), "w") as f:
    json.dump(dims, f)
print("Saved slide-sized results:", ", ".join(dims)) 
