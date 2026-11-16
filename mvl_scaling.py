#!/usr/bin/env python3
"""
mvl_scaling.py
Compute MVL scaling relations between human height, Planck length,
and the observable Universe.
"""

import numpy as np
import pandas as pd

# Publicly available reference scales (SI metres)
PLANCK_LENGTH = 1.616255e-35          # CODATA 2018
UNIVERSE_RADIUS = 4.4e26              # Planck 2015, comoving observable radius
HUMAN_HEIGHT = 1.71                   # NCD-RisC global average adult height

phi = (1.0 + np.sqrt(5.0)) / 2.0

# Author's proposed reciprocal scaling
a = UNIVERSE_RADIUS / HUMAN_HEIGHT
h = HUMAN_HEIGHT / a  # equivalent to H**2 / R_U

# Golden-section variants
logR = np.log10(UNIVERSE_RADIUS)
logH = np.log10(HUMAN_HEIGHT)

tau_top = 1.0 / phi          # measured from largest scale
logLmin_top = (logH - tau_top * logR) / (1.0 - tau_top)

tau_bot = 1.0 / phi**2       # measured from smallest scale
logLmin_bot = (logH - tau_bot * logR) / (1.0 - tau_bot)

# Geometric mean between Planck length and Universe
L_GM = np.sqrt(PLANCK_LENGTH * UNIVERSE_RADIUS)

results = pd.Series({
    "Scaling factor a": a,
    "Predicted minimal scale h (m)": h,
    "h / Planck length": h / PLANCK_LENGTH,
    "Geometric mean Planck-Universe (m)": L_GM,
    "Implied L_min (tau=1/phi) (m)": 10**logLmin_top,
    "Implied L_min (tau=1/phi^2) (m)": 10**logLmin_bot,
})

print(results)
