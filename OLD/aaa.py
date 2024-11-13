# Photon-Photon Luminosity Spectrum Calculation --- Hamzeh Khanpour 2024

import numpy as np
import math
import scipy.integrate as integrate

# Constants in GeV
ALPHA = 1 / 137  # Fine-structure constant
emass = 5.1099895e-4  # Electron mass in GeV
pmass = 0.938272081  # Proton mass in GeV

# Integration limits and parameters
Q2e_max = 100.0  # Maximum photon virtuality for electron in GeV^2
Q2p_max = 100.0  # Maximum photon virtuality for proton in GeV^2
y_e_max = 1.0  # Maximum value of energy fraction carried by photon from electron
y_p_max = 1.0  # Maximum value of energy fraction carried by photon from proton

# Photon Flux from Electron

def photon_flux_electron(y_e, Q2_e):
    if y_e <= 0 or y_e >= 1:
        return 0.0
    Q2_min_e = emass**2 * y_e**2 / (1 - y_e)
    if Q2_e < Q2_min_e or Q2_e > Q2e_max:
        return 0.0
    factor = ALPHA / (math.pi * y_e * Q2_e)
    flux = (1 - y_e) * (1 - Q2_min_e / Q2_e) + 0.5 * y_e**2
    return factor * flux

# Photon Flux from Proton

def photon_flux_proton(y_p, Q2_p):
    if y_p <= 0 or y_p >= 1:
        return 0.0
    Q2_min_p = pmass**2 * y_p**2 / (1 - y_p)
    if Q2_p < Q2_min_p or Q2_p > Q2p_max:
        return 0.0
    factor = ALPHA / (math.pi * y_p * Q2_p)
    flux = (1 - y_p) * (1 - Q2_min_p / Q2_p) + 0.5 * y_p**2
    return factor * flux

# Photon-Photon Luminosity Spectrum Calculation

def photon_photon_luminosity(W, s):
    def integrand(y_e, y_p, Q2_e, Q2_p):
        delta_arg = W**2 - (y_e * y_p * s - Q2_e - Q2_p)
        if abs(delta_arg) < 1e-3:  # Approximate handling of delta function
            return photon_flux_electron(y_e, Q2_e) * photon_flux_proton(y_p, Q2_p)
        return 0.0

    y_e_min = lambda Q2_e, Q2_p: (W**2 + Q2_e + Q2_p) / s
    y_p_min = lambda Q2_e, Q2_p: (W**2 + Q2_e + Q2_p) / s

    result, _ = integrate.nquad(
        integrand,
        [[0, y_e_max], [0, y_p_max], [0, Q2e_max], [0, Q2p_max]],
        opts={'epsrel': 1e-3}
    )
    return result

# Example Calculation
s_cms = 4 * 50.0 * 7000.0  # Center-of-mass energy squared for E_e = 50 GeV, E_p = 7000 GeV
W_values = np.linspace(10.0, 100.0, 10)  # Range of W values from 10 GeV to 100 GeV

for W in W_values:
    luminosity = photon_photon_luminosity(W, s_cms)
    print(f"Photon-Photon Luminosity at W = {W:.2f} GeV: {luminosity:.6e} GeV^-1")

    
