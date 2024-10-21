# Elastic Photon-Photon Luminosity Spectrum at LHeC --- Updated Version with Parallel Computing


import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from multiprocessing import Pool


# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass
pmass = 0.938272081    # Proton mass 

q2emax = 100.0  # Maximum photon virtuality for electron in GeV^2 (matching your settings)
q2pmax = 100.0  # Maximum photon virtuality for proton in GeV^2 (matching your settings)


# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)

def G_M(Q2):
    return 7.78 * G_E(Q2)

# Minimum Photon Virtuality
def qmin2(mass, y):
    return (mass * y)**2 / (1 - y)



# Photon Flux from Electron
def flux_y_electron(ye, Q2e):
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2(emass, ye)
    if Q2e < qmin2v or Q2e > q2emax:
        return 0.0

    flux = ALPHA2PI / (ye * Q2e) * (
        (1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2
    )
    return flux



# Photon Flux from Proton
def flux_y_proton(yp, Q2p):
    if yp <= 0 or yp >= 1:
        return 0.0
    qmin2v = qmin2(pmass, yp)
    if Q2p < qmin2v or Q2p > q2pmax:
        return 0.0

    gE2 = G_E(Q2p)
    gM2 = G_M(Q2p)
    formE = (4 * pmass**2 * gE2 + Q2p * gM2) / (4 * pmass**2 + Q2p)
    formM = gM2

    flux = ALPHA2PI / (yp * Q2p) * (
        (1 - yp) * (1 - qmin2v / Q2p) * formE + 0.5 * yp**2 * formM
    )
    return flux



# Elastic Photon-Photon Luminosity Spectrum Calculation at Given W (Exact formula for W^2)
def flux_el_yy_atW(W, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared

    # Integration over ye from ye_min to ye_max (which is 1)
    ye_min = W**2 / s_cms

    def integrand(ye):
        yp_min = W**2 / (ye * s_cms)
        yp_max = 1.0


        def yp_integrand(yp):
            Q2e_min = 0.0
            Q2e_max = ye * yp * s_cms - W**2


            def Q2e_integrand(Q2e):
                Q2p = ye * yp * s_cms - W**2 - Q2e
                if Q2p < 0.0 or Q2p > q2pmax:
                    return 0.0


                # Calculate photon fluxes
                flux_e = flux_y_electron(ye, Q2e)
                flux_p = flux_y_proton(yp, Q2p)

                return flux_e * flux_p

            result_Q2e, _ = integrate.quad(Q2e_integrand, Q2e_min, Q2e_max, epsrel=1e-4)
            return result_Q2e


        result_yp, _ = integrate.quad(yp_integrand, yp_min, yp_max, epsrel=1e-4)
        return result_yp


    result_ye, _ = integrate.quad(integrand, ye_min, 1.0, epsrel=1e-4)
    return result_ye





# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV

W_values = np.logspace(1.0, 3.0, 101)  # Range of W values from 10 GeV to 1000 GeV


# Wrapper function for parallel processing
def wrapper_flux_el_yy_atW(W):
    return flux_el_yy_atW(W, eEbeam, pEbeam)


# Parallel Calculation of the Elastic Photon-Photon Luminosity Spectrum
if __name__ == "__main__":
    with Pool() as pool:
        luminosity_values = pool.map(wrapper_flux_el_yy_atW, W_values)


    # Calculate the Elastic Photon-Photon Luminosity Spectrum at W = 10 GeV
    W_value = 10.0  # GeV
    luminosity_at_W10 = flux_el_yy_atW(W_value, eEbeam, pEbeam)
    print(f"Elastic Photon-Photon Luminosity Spectrum at W = {W_value} GeV: {luminosity_at_W10:.6e} GeV^-1")

    # Plot the Results
    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)

    plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Elastic')


    # Marking W_0 = 10 GeV on the plot
    plt.scatter(W_value, luminosity_at_W10, color='red', zorder=5)
    plt.text(W_value, luminosity_at_W10 * 1.5, f'$W_0 = 10$ GeV\n$S_{{\gamma\gamma}} = {luminosity_at_W10:.2e}$', color='red', fontsize=10, ha='center')


    # Plot settings
    plt.xlabel(r"$W$ [GeV]", fontsize=18)
    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    plt.title("Elastic Photon-Photon Luminosity Spectrum at LHeC", fontsize=20)


    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_{e,\text{max}} = 100 \, \mathrm{GeV}^2, \, Q^2_{p,\text{max}} = 100 \, \mathrm{GeV}^2$', fontsize=14)

    # Save the plot as a PDF and JPG
    plt.savefig("Elastic-Photon-Photon-Luminosity-Spectrum-Hamzeh_with_W_Parallel_Final_Expression.pdf")
    plt.savefig("Elastic-Photon-Photon-Luminosity-Spectrum-Hamzeh_with_W_Parallel_Final_Expression.jpg")

    plt.show()

    
