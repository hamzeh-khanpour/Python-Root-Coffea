# Elastic Photon-Photon Luminosity Spectrum at LHeC --- Hamzeh Khanpour October 2024

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4  # Electron mass
pmass = 0.938272081   # Proton mass

q2emax = 10.0      # Maximum photon virtuality for electron in GeV^2 (matching your settings)
q2pmax = 10.0      # Maximum photon virtuality for proton in GeV^2 (matching your settings)

# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)

def G_M(Q2):
    return 7.78 * G_E(Q2)

# Minimum Photon Virtuality
def qmin2(mass, y):
    return mass * mass * y * y / (1 - y)

# Exact Photon Flux from Electron (Bethe-Heitler/Weizsäcker-Williams Formalism)
def flux_y_electron(ye, qmax2):
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2(emass, ye)

    # Integration over Q2 from qmin2 to qmax2
    def integrand(lnQ2):
        Q2 = np.exp(lnQ2)
        # Using exact photon flux expression for an electron (Bethe-Heitler formalism)
        y1 = 0.5 * (1.0 + (1.0 - ye) ** 2) / ye
        flux = y1 / (Q2 + emass**2)  # Retain Q^2 dependence accurately
        return flux * Q2  # Multiply by Q2 to account for change of variable

    try:
        result, _ = integrate.quad(integrand, math.log(qmin2v), math.log(qmax2), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for electron flux did not converge for ye={ye}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for electron flux: {e}")
        result = 0.0

    return ALPHA2PI * result

# Elastic Photon Flux from Proton (Exact QED Form)
def flux_y_proton(yp, qmax2):
    if yp <= 0 or yp >= 1:
        return 0.0
    qmin2v = qmin2(pmass, yp)

    # Integration over ln(Q2) from qmin2 to qmax2
    def integrand(lnQ2):
        Q2 = np.exp(lnQ2)
        gE2 = G_E(Q2)
        gM2 = G_M(Q2)
        formE = (4 * pmass ** 2 * gE2 + Q2 * gM2) / (4 * pmass ** 2 + Q2)
        formM = gM2
        flux_tmp = (1 - yp) * (1 - qmin2v / Q2) * formE + 0.5 * yp ** 2 * formM
        return flux_tmp * ALPHA2PI / (yp * Q2) * Q2  # Multiply by Q2 to account for change of variable

    try:
        result, _ = integrate.quad(integrand, math.log(qmin2v), math.log(qmax2), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for proton flux did not converge for yp={yp}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for proton flux: {e}")
        result = 0.0
    return result

# Elastic Photon-Photon Luminosity Spectrum Calculation at Given W
def flux_el_yy_atW(W, eEbeam, pEbeam, qmax2e, qmax2p):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared
    ymin = W * W / s_cms

    # Integration over ye from ymin to 1
    def integrand(ye):
        yp = W * W / (s_cms * ye)
        if yp <= 0.0 or yp >= 1.0:
            return 0.0
        return flux_y_proton(yp, qmax2p) * yp * flux_y_electron(ye, qmax2e)

    try:
        result, _ = integrate.quad(integrand, ymin, 1.0, epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for elastic luminosity did not converge for W={W}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for elastic luminosity: {e}")
        result = 0.0
    return result * 2.0 / W

# Tau-Tau Production Cross-Section Calculation at Given W
def cs_tautau_w_condition_Hamzeh(W):
    alpha = 1 / 137.0
    hbarc2 = 0.389  # Conversion factor to pb
    mtau = 1.77686  # Tau mass in GeV

    if W < 2 * mtau:
        return 0.0
    beta = math.sqrt(1.0 - 4.0 * mtau**2 / W**2)
    cross_section = (4 * math.pi * alpha**2 * hbarc2) / W**2 * beta * (
        (3 - beta**4) / (2 * beta) * math.log((1 + beta) / (1 - beta)) - 2 + beta**2
    ) * 1e9
    return cross_section

# Integrated Tau-Tau Production Cross-Section from W_0 to sqrt(s_cms)
def integrated_tau_tau_cross_section(W0, eEbeam, pEbeam, qmax2e, qmax2p):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared

    try:
        result, _ = integrate.quad(
            lambda W: cs_tautau_w_condition_Hamzeh(W) * flux_el_yy_atW(W, eEbeam, pEbeam, qmax2e, qmax2p),
            W0, np.sqrt(s_cms), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for tau-tau production cross-section did not converge for W_0={W0}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for tau-tau production cross-section: {e}")
        result = 0.0
    return result

# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W_values = np.logspace(1.0, 3.0, 101)  # Range of W values from 10 GeV to 1000 GeV

# Calculate the Elastic Photon-Photon Luminosity Spectrum at W = 10 GeV
W_value = 10.0  # GeV
luminosity_at_W10 = flux_el_yy_atW(W_value, eEbeam, pEbeam, q2emax, q2pmax)
print(f"Elastic Photon-Photon Luminosity Spectrum at W = {W_value} GeV: {luminosity_at_W10:.6e} GeV^-1")

# Calculate the Elastic Photon-Photon Luminosity Spectrum
luminosity_values = [flux_el_yy_atW(W, eEbeam, pEbeam, q2emax, q2pmax) for W in W_values]

# Calculate Integrated Tau-Tau Production Cross-Section at W_0 = 10 GeV
W0_value = 10.0  # GeV
integrated_cross_section_value = integrated_tau_tau_cross_section(W0_value, eEbeam, pEbeam, q2emax, q2pmax)
print(f"Integrated Tau-Tau Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")

# Plot the Elastic Photon-Photon Luminosity Spectrum
plt.figure(figsize=(10, 8))

# Set plotting range
plt.xlim(10.0, 1000.0)
plt.ylim(1.e-7, 1.e-1)

plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Elastic Photon-Photon Luminosity Spectrum')
plt.xlabel(r"Center-of-Mass Energy $W$ [GeV]", fontsize=18)
plt.ylabel(r"Photon-Photon Luminosity $S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
plt.title("Elastic Photon-Photon Luminosity Spectrum at LHeC", fontsize=20)
plt.grid(True, which="both", linestyle="--")
plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^1 \, \mathrm{GeV}^2$', fontsize=14)

# Save the plot as a PDF
plt.savefig("elastic_photon_luminosity_spectrum_exact_flux.pdf")
plt.show()
