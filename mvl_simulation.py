#!/usr/bin/env python3
"""
mvl_simulation.py
Multi-scale network benchmark for the Median Value Law.
"""

import numpy as np
import matplotlib.pyplot as plt

N = 60
L_min, L_max = 1e-35, 4.4e26
levels = np.logspace(np.log10(L_min), np.log10(L_max), N)
log_levels = np.log10(levels)
human_idx = np.argmin(np.abs(levels - 1.71))

# Propagation coefficients
p_up = 0.78
p_down = 0.62

# Weighting schemes
w_uniform = np.ones(N)

mu = np.log10(1.71)
sigma = 2.5
w_human = np.exp(-0.5 * ((log_levels - mu) / sigma) ** 2)

def total_influence(k, weights):
    impact = np.zeros(N)
    impact[k] = 1.0
    for i in range(k + 1, N):
        impact[i] = impact[i - 1] * p_up
    for i in range(k - 1, -1, -1):
        impact[i] = impact[i + 1] * p_down
    return np.sum(impact * weights)

scores_uniform = np.array([total_influence(k, w_uniform) for k in range(N)])
scores_human = np.array([total_influence(k, w_human) for k in range(N)])

opt_uniform = np.argmax(scores_uniform)
opt_human = np.argmax(scores_human)

print(f"Uniform optimum:   index={opt_uniform}, scale={levels[opt_uniform]:.2e} m")
print(f"Human optimum:     index={opt_human}, scale={levels[opt_human]:.2e} m")
print(f"Human scale:       index={human_idx}, scale={levels[human_idx]:.2e} m")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(log_levels, scores_uniform, "o-", markersize=3, label="Uniform weighting")
ax.plot(log_levels, scores_human, "s-", markersize=3, label="Organism-centred weighting")
ax.axvline(np.log10(1.71), color="red", linestyle="--", label="Human scale")
ax.axvline(log_levels[opt_uniform], color="C0", linestyle=":", label="Uniform optimum")
ax.axvline(log_levels[opt_human], color="C1", linestyle=":", label="Organism optimum")
ax.set_xlabel(r"$\log_{10}(L\,/\,\mathrm{m})$")
ax.set_ylabel("Total influence")
ax.set_title("MVL benchmark: perturbation influence vs. scale")
ax.legend()
plt.tight_layout()
plt.savefig("mvl_simulation.png", dpi=300)
print("Saved mvl_simulation.png")
