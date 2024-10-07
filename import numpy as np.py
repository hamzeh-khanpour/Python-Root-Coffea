import numpy as np
from scipy.integrate import quad

# Constants
alpha = 1/137  # Fine-structure constant
hbarc2 = 0.389  # Conversion factor for cross sections
mtau = 1.777  # Mass of tau lepton in GeV

# Photon Flux for Electron
def photon_flux_electron(ye, Q2_min, Q2_max):
    def integrand(Q2, ye):
        return (1 - ye) * (1 - Q2_min / Q2) / Q2 + (ye**2 / 2) / Q2
    flux, _ = quad(integrand, Q2_min, Q2_max, args=(ye,))
    return alpha / np.pi / ye * flux

# Proton Elastic Photon Flux
def photon_flux_proton(yp, Q2_min, Q2_max):
    Mp = 0.938  # Mass of proton in GeV
    def dipole_form_factor(Q2):
        GE = (1 + Q2 / 0.71)**-2
        GM = GE / 7.78
        return (4 * Mp**2 * GE**2 + Q2 * GM**2) / (4 * Mp**2 + Q2)
    
    def integrand(Q2, yp):
        return dipole_form_factor(Q2) / Q2
    
    flux, _ = quad(integrand, Q2_min, Q2_max, args=(yp,))
    return alpha / np.pi / yp * flux

# Photon-Photon Cross Section for tau production
def cs_tautau_w(wvalue):
    alpha2 = (1.0 / 137.0) ** 2
    beta = np.sqrt(1.0 - (4.0 * mtau**2) / wvalue**2)
    
    cs = (4.0 * np.pi * alpha2 * hbarc2 / wvalue**2) * (beta * (3 - beta**4) / (2.0 * beta) * np.log((1 + beta) / (1 - beta)) - 2.0 + beta**2) * 1e9
    return cs

# Integrating the fluxes to calculate electron-proton cross section
def ep_cross_section(Ee, Ep, W_min, W_max, Q2_min_e, Q2_max_e, Q2_min_p, Q2_max_p):
    S = 4 * Ee * Ep
    def integrand(W):
        ye_min = W**2 / S
        ye_max = 1.0
        yp_min = W**2 / (ye_min * S)
        yp_max = 1.0
        flux_e = photon_flux_electron(ye_min, Q2_min_e, Q2_max_e)
        flux_p = photon_flux_proton(yp_min, Q2_min_p, Q2_max_p)
        sigma_gg = cs_tautau_w(W)
        return flux_e * flux_p * sigma_gg / W

    total_cross_section, _ = quad(integrand, W_min, W_max)
    return total_cross_section

# Example usage
Ee = 50  # Electron energy in GeV
Ep = 7000  # Proton energy in GeV
W_min = 10  # Minimum W value in GeV
W_max = 1000  # Maximum W value in GeV
Q2_min_e = 1e-4  # Electron photon flux minimum virtuality in GeV^2
Q2_max_e = 1.0  # Electron photon flux maximum virtuality in GeV^2
Q2_min_p = 1e-4  # Proton photon flux minimum virtuality in GeV^2
Q2_max_p = 1.0  # Proton photon flux maximum virtuality in GeV^2

total_cross_section = ep_cross_section(Ee, Ep, W_min, W_max, Q2_min_e, Q2_max_e, Q2_min_p, Q2_max_p)
print(f"Total electron-proton cross section: {total_cross_section:.6e} pb")
