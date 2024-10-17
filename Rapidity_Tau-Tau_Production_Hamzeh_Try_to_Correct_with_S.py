
# Differential tau-tau production cross-Section as a function of rapidity Y 

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass
pmass = 0.938272081    # Proton mass
mtau = 1.77686  # Tau mass in GeV
q2emax = 100000.0  # Maximum photon virtuality for electron in GeV^2
q2pmax = 100000.0  # Maximum photon virtuality for proton in GeV^2


# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)

def G_M(Q2):
    return 7.78 * G_E(Q2)


# Minimum Photon Virtuality
def qmin2(mass, y):
    return mass * mass * y * y / (1 - y)


# Suppression Factor for Large Photon Virtuality (Exponential Form)
def suppression_factor(Q2, W, c=0.2):
    return np.exp(-Q2 / (c * W**2))


# Elastic Photon Flux from Electron
def flux_y_electron(ye, qmax2, W):
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2(emass, ye)

    # Integration over ln(Q2) from ln(qmin2) to ln(qmax2)
    def integrand(lnQ2):
        Q2 = np.exp(lnQ2)
        y1 = 0.5 * (1.0 + (1.0 - ye) ** 2) / ye
        y2 = (1.0 - ye) / ye
        flux1 = y1 / Q2
        flux2 = y2 / qmax2
        suppression = suppression_factor(Q2, W)
        return (flux1 - flux2) * suppression * Q2

    try:
        result, _ = integrate.quad(integrand, math.log(qmin2v), math.log(qmax2), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for electron flux did not converge for ye={ye}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for electron flux: {e}")
        result = 0.0

    return ALPHA2PI * result


# Elastic Photon Flux from Proton
def flux_y_proton(yp, qmax2, W):
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
        suppression = suppression_factor(Q2, W)
        return flux_tmp * ALPHA2PI / (yp * Q2) * Q2 * suppression

    try:
        result, _ = integrate.quad(integrand, math.log(qmin2v), math.log(qmax2), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for proton flux did not converge for yp={yp}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for proton flux: {e}")
        result = 0.0
    return result


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


# Differential Tau-Tau Production Cross-Section with Respect to Rapidity
def differential_cross_section_rapidity(Y, W0, eEbeam, pEbeam, qmax2e, qmax2p):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared
    sqrt_s = np.sqrt(s_cms)
    
    def integrand(W):
        yp = W * np.exp(Y) / (2 * pEbeam)
        ye = W * np.exp(-Y) / (2 * eEbeam)
        
        if yp > 1.0 or ye > 1.0:
            return 0.0
        
        flux_e = flux_y_electron(ye, qmax2e, W)
        flux_p = flux_y_proton(yp, qmax2p, W)
        sigma_gg = cs_tautau_w_condition_Hamzeh(W)
        
        return sigma_gg * W * flux_e * flux_p

    try:
        result, _ = integrate.quad(integrand, W0, sqrt_s, epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for differential cross-section did not converge for Y={Y}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for differential cross-section: {e}")
        result = 0.0
    
    return 2.0 / s_cms * result


# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W0_value = 10.0  # GeV, starting value for integration


# Calculate the Differential Cross-Section for a Range of Y Values
Y_values = np.linspace(-10.0, 10.0, 100)  # Range of rapidity values
differential_cross_section_values = [differential_cross_section_rapidity(Y, W0_value, eEbeam, pEbeam, q2emax, q2pmax) for Y in Y_values]


# Calculate the Total Cross-Section (Area Under the Curve)
total_cross_section = np.trapz(differential_cross_section_values, Y_values)


# Plot the Differential Tau-Tau Production Cross-Section as a Function of Rapidity Y
plt.figure(figsize=(10, 8))
plt.plot(Y_values, differential_cross_section_values, linestyle='solid', linewidth=2, label='Elastic')

# Set y-axis limit from 0 to 12
plt.ylim(0, 14)

# Add additional information to the plot
plt.text(-9, 0.8 * max(differential_cross_section_values), f'q2emax = {q2emax:.1f} GeV^2', fontsize=14, color='blue')
plt.text(-9, 0.7 * max(differential_cross_section_values), f'q2pmax = {q2pmax:.1f} GeV^2', fontsize=14, color='blue')
plt.text(-9, 0.6 * max(differential_cross_section_values), f'Total Cross-Section = {total_cross_section:.2e} pb', fontsize=14, color='blue')



plt.xlabel(r"Rapidity $Y$", fontsize=18)
plt.ylabel(r"$\frac{d\sigma_{\tau^+\tau^-}}{dY}$ [pb]", fontsize=18)
plt.title("Rapidity distribution at LHeC (Corrected)", fontsize=20)
plt.grid(True, which="both", linestyle="--")
plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)


# Save the plot
plt.savefig("differential_cross_section_rapidity_corrected.pdf")
plt.savefig("differential_cross_section_rapidity_corrected.jpg")

plt.show()


