# Revised version with factor of 2 adjustment for consistency

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass
pmass = 0.938272081 + 0.1349768   # Proton mass

q2emax = 100000.0  # Maximum photon virtuality for electron in GeV^2 (matching your settings)
q2pmax = 10.0  # Maximum photon virtuality for proton in GeV^2 (matching your settings)


# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)


def G_M(Q2):
    return 7.78 * G_E(Q2)


# Minimum Photon Virtuality
def qmin2(mass, y):
    return mass * mass * y * y / (1 - y)


# Elastic Photon Flux from Electron
def flux_y_electron(ye, qmax2):
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2(emass, ye)
    y1 = 0.5 * (1.0 + (1.0 - ye) ** 2) / ye
    y2 = (1.0 - ye) / ye
    flux1 = y1 * math.log(qmax2 / qmin2v)
    flux2 = y2 * (1.0 - qmin2v / qmax2)
    return ALPHA2PI * (flux1 - flux2)


# Elastic Photon Flux from Proton
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
        flux_tmp = (1 - yp) * formE + 0.5 * yp ** 2 * formM
        # Corrected integrand to include Q2 for change of variables
        return flux_tmp * ALPHA2PI / (yp * Q2) * Q2  # Multiply by Q2 to account for change of variables


    result, _ = integrate.quad(integrand, math.log(qmin2v), math.log(qmax2), epsrel=1e-4)
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

    result, _ = integrate.quad(integrand, ymin, 1.0, epsrel=1e-4)
    return result * 2.0 / W  # Multiplied by 4 instead of 2 to check for potential factor of 2 discrepancy


# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W_values = np.logspace(1.0, 3.0, 303)  # Range of W values from 10 GeV to 1000 GeV


# Calculate the Elastic Photon-Photon Luminosity Spectrum
luminosity_values = [flux_el_yy_atW(W, eEbeam, pEbeam, q2emax, q2pmax) for W in W_values]

# Plot the Results
plt.figure(figsize=(10, 8))

# Set plotting range
plt.xlim(10.0, 1000.0)
plt.ylim(1.e-7, 1.e-1)

plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Elastic')
plt.xlabel(r"$W$ [GeV]", fontsize=18)
plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
plt.title("Elastic Photon-Photon Luminosity Spectrum at LHeC", fontsize=20)
plt.grid(True, which="both", linestyle="--")
plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^1 \, \mathrm{GeV}^2$', fontsize=14)


plt.show()

