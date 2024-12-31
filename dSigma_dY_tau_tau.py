
#----------------------------------------------------------------
#----------------------------------------------------------------

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4  # Electron mass in GeV
pmass = 0.938272081   # Proton mass in GeV

q2emax = 100.0         # Maximum photon virtuality for electron in GeV^2
q2pmax = 100.0         # Maximum photon virtuality for proton in GeV^2

W0 = 10.0             # Minimum W value in GeV

eEbeam = 50.0         # Electron beam energy in GeV
pEbeam = 7000.0       # Proton beam energy in GeV

# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)

def G_M(Q2):
    return 7.78 * G_E(Q2)

# Minimum Photon Virtuality
def qmin2(mass, y):
    return mass * mass * y * y / (1 - y) if y < 1 else float('inf')


#----------------------------------------------------------------

# Photon Flux from Electron
def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2(emass, ye)
    if qmin2v <= 0 or Q2e < qmin2v or Q2e > q2emax:
        return 0.0
    flux = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2)
    return flux * Q2e  # Multiply by Q2e to account for dQ^2 = Q^2 d(lnQ^2)

#----------------------------------------------------------------

# Photon Flux from Proton
def flux_y_proton(yp):
    if yp <= 0 or yp >= 1:
        return 0.0
    qmin2p = qmin2(pmass, yp)
    if qmin2p <= 0:
        return 0.0

    def lnQ2p_integrand(lnQ2p):
        Q2p = np.exp(lnQ2p)
        if Q2p < qmin2p or Q2p > q2pmax:
            return 0.0
        gE2 = G_E(Q2p)
        gM2 = G_M(Q2p)
        formE = (4 * pmass**2 * gE2 + Q2p * gM2) / (4 * pmass**2 + Q2p)
        formM = gM2
        flux = ALPHA2PI / (yp * Q2p) * ((1 - yp) * (1 - qmin2p / Q2p) * formE + 0.5 * yp**2 * formM)
        return flux * Q2p  # Multiply by Q2p to account for dQ^2 = Q^2 d(lnQ^2)

    result_lnQ2p, _ = integrate.quad(lnQ2p_integrand, math.log(qmin2p), math.log(q2pmax), epsrel=1e-4)
    return result_lnQ2p

#----------------------------------------------------------------

# Tau-Tau Production Cross-Section Calculation at Given W
def cs_tautau_w_condition(W):
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


#----------------------------------------------------------------

# Calculate dSigma/dY for a given Y
def dSigma_dY(Y):
    def integrand(W):
        yp = W * math.exp(Y) / (2 * pEbeam)
        ye = W * math.exp(-Y) / (2 * eEbeam)

        if yp <= 0 or yp >= 1 or ye <= 0 or ye >= 1:
            return 0.0
        
        lnQ2e_min = math.log(qmin2(emass, ye))
        lnQ2e_max = math.log(q2emax)

        def electron_flux_integrand(lnQ2e):
            return flux_y_electron(ye, lnQ2e)
        
        flux_ye, _ = integrate.quad(electron_flux_integrand, lnQ2e_min, lnQ2e_max, epsrel=1e-4)

        return flux_ye * flux_y_proton(yp) * cs_tautau_w_condition(W) * W 


    s_cms = 4.0 * eEbeam * pEbeam
    W_min = W0
    W_max = math.sqrt(s_cms)
    integral_result, _ = integrate.quad(integrand, W_min, W_max, epsrel=1e-4)

    return (2.0 / s_cms) * integral_result


#----------------------------------------------------------------

# Compute dSigma/dY for a range of Y values
Y_values = np.linspace(-10, 10, 303)
dSigma_values = [dSigma_dY(Y) for Y in Y_values]


#----------------------------------------------------------------

# Save results to a file
output_data = np.column_stack((Y_values, dSigma_values))

header = (
    "Rapidity (Y)   dSigma/dY [pb]\n"
    f"eEbeam: {eEbeam} GeV, pEbeam: {pEbeam} GeV\n"
    f"q2emax: {q2emax}, q2pmax: {q2pmax}"
)
np.savetxt("dSigma_dY_tau_tau.txt", output_data, header=header, fmt="%0.8e")


#----------------------------------------------------------------

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(Y_values, dSigma_values, label=r"$d\sigma/dY$")
plt.xlabel(r"$Y$")
plt.ylabel(r"$d\sigma/dY$ [pb]")
plt.title(r"Differential Cross-Section $d\sigma/dY$ for $\tau^+\tau^-$ Production")
plt.legend()
plt.grid()
plt.savefig("dSigma_dY_tau_tau.pdf")
plt.show()


#----------------------------------------------------------------
#----------------------------------------------------------------

