# Project 22: Dopamine and TD Learning

This project shows how a simple reinforcement learning rule, **TD(0)**, behaves like
dopamine neurons in the brain (Schultz, Dayan & Montague, 1997).

An agent hears a **bell**, waits a few steps, then gets **food**. At first the food is a
surprise. After learning, the agent expects the food, so the surprise moves to the bell.
If the food is then taken away, the surprise goes negative. Dopamine neurons show the
same three patterns.

## How it works

- Each trial has time steps `t = 0 .. 12`. The bell rings at `t = 2`, food (reward = 1) comes at `t = 8`.
- The state is just the time step, and the agent keeps one value `V[t]` per step (a lookup table).
- At every step it computes the **TD error** (the "surprise", like a dopamine signal) and updates `V`:

  ```
  delta_t  = r_t + gamma * V[t] - V[t-1]
  V[t-1]  += alpha * delta_t        (only for steps from the bell onwards)
  ```

- Steps before the bell are never updated, because the bell comes at a time the agent cannot predict.

Settings (top of `td_dopamine.py`): `alpha = 0.1`, `gamma = 1`, `300` trials, `seed = 0`.

## Run it

```
pip install -r requirements.txt
python td_dopamine.py
```

Needs Python 3 with numpy, matplotlib and Pillow. It runs in a few seconds on a CPU.
Running it writes the graphs to a new `results/` folder and the numbers to `results.txt`.
Results are the same on every run (fixed random seed).

## Experiments

| # | What it tests | Key result |
|---|---|---|
| 1 | Learning curve | Surprise at food falls 1 → 0, surprise at bell rises 0 → 1. A no-learning baseline stays at 1. |
| 2 | Heatmap of TD error | Shows the surprise moving from the food step to the bell step over trials. |
| 3 | Three patterns | Before learning: food = 1. After: bell = 1, food = 0. Food omitted: dip of −1. |
| 4 | Learning rate | Bigger `alpha` learns faster (0.03 → 100 trials, 0.6 → 5 trials). |
| 5 | Reward probability | Bell response grows with the chance of food (like Fiorillo et al., 2003). |
| 6 | Timing | Food 2 steps early → positive spike then a dip at the expected time; late → dip then spike. |

## Files

| File | What it is |
|---|---|
| `td_dopamine.py` | All the code: TD agent, experiments and plots |
| `results.txt` | Numbers from all experiments |
| `results/` | Saved graphs. `fig*.png` = normal size, `slide_*.png` = same graphs with bigger text for slides |
| `Project22_Dopamine_TD_Learning.pptx` | Presentation slides |
| `Project22_Problem_Formulation.pdf` | Problem formulation write-up |
| `requirements.txt` | Python packages needed |
