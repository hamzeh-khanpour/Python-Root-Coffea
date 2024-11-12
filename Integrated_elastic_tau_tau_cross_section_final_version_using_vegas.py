# This code calculates the elastic Photon Luminosity Spectrum Syy
# Final Version (with corrected W^2 = -Q2e + ye * yp * s ) to solve the high photon virtuality issues
# This code also calculates the integrated elastic tau tau cross section (ep -> e p tau^+ tau^-)
# Hamzeh Khanpour, Laurent Forthomme, and Krzysztof Piotrzkowski -- November 2024

################################################################################

import numpy as np
import math
import vegas  # Import vegas for Monte Carlo integration
import scipy.integrate as integrate  # Add this line to fix the error
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass in GeV
pmass = 0.938272081    # Proton mass in GeV

q2emax = 10.0  # Maximum photon virtuality for electron in GeV^2
q2pmax = 10.0  # Maximum photon virtuality for proton in GeV^2


# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)


def G_M(Q2):
    return 7.78 * G_E(Q2)


# Minimum Photon Virtuality
def qmin2(mass, y):
    return mass * mass * y * y / (1 - y) if y < 1 else float('inf')


# Function to compute y_p using new W: W = sqrt(-Q_e^2 + y_e y_p s)
def compute_yp(W, Q2e, ye, Ee, Ep):
    s = 4 * Ee * Ep
    numerator = W**2 + Q2e
    denominator = ye * s
    return numerator / denominator if denominator != 0 else 0


# Function to compute the Jacobian (partial derivative of W with respect to y_p)
def compute_jacobian(ye, yp, Q2e, Ee, Ep):
    s = 4 * Ee * Ep
    W = np.sqrt(-Q2e + ye * yp * s)
    return abs(ye * s / (2 * W)) if W != 0 else 0


# Photon Flux from Electron (using lnQ2 as the integration variable)
def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    if 0 < ye < 1 and qmin2(emass, ye) <= Q2e <= q2emax:
        return ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2(emass, ye) / Q2e) + 0.5 * ye**2) * Q2e
    return 0.0



# Photon Flux from Proton (using lnQ2 as the integration variable)
def flux_y_proton(yp):
    if 0 < yp < 1:
        qmin2p = qmin2(pmass, yp)
        if qmin2p <= 0:
            return 0.0

        def lnQ2p_integrand(lnQ2p):
            Q2p = np.exp(lnQ2p)
            if qmin2p <= Q2p <= q2pmax:
                gE2 = G_E(Q2p)
                gM2 = G_M(Q2p)
                formE = (4 * pmass**2 * gE2 + Q2p * gM2) / (4 * pmass**2 + Q2p)
                formM = gM2
                return ALPHA2PI / (yp * Q2p) * ((1 - yp) * (1 - qmin2p / Q2p) * formE + 0.5 * yp**2 * formM) * Q2p
            return 0.0

        lnQ2p_min = math.log(qmin2p)
        lnQ2p_max = math.log(q2pmax)
        result_lnQ2p, _ = integrate.quad(lnQ2p_integrand, lnQ2p_min, lnQ2p_max, epsrel=1e-4)
        return result_lnQ2p
    return 0.0


# -------------------------------------------------------------------------------------


# Photon Luminosity Spectrum Calculation (using vegas for integration)
def flux_el_yy_atW(W, eEbeam, pEbeam, n_samples=10000):
    s_cms = 4.0 * eEbeam * pEbeam
    ye_min, ye_max = W**2.0 / s_cms, 1.0


    # Define integrand for vegas
    def integrand(x):
        ye = x[0] * (ye_max - ye_min) + ye_min
        lnQ2e = x[1] * (math.log(q2emax) - math.log(qmin2(emass, ye))) + math.log(qmin2(emass, ye))
        Q2e = np.exp(lnQ2e)
        yp_value = compute_yp(W, Q2e, ye, eEbeam, pEbeam)
        jacobian = compute_jacobian(ye, yp_value, Q2e, eEbeam, pEbeam)

        if 0 < yp_value < 1 and jacobian != 0:
            flux_e = flux_y_electron(ye, lnQ2e)
            proton_flux = flux_y_proton(yp_value)
            return flux_e * proton_flux / jacobian
        return 0.0

    integ = vegas.Integrator([[0, 1], [0, 1]])
    result = integ(integrand, nitn=10, neval=n_samples)
    return result.mean



# Tau-Tau Production Cross-Section Calculation at Given W
def cs_tautau_w_condition_Hamzeh(W):
    alpha = 1 / 137.0
    hbarc2 = 0.389
    mtau = 1.77686
    if W < 2 * mtau:
        return 0.0
    beta = math.sqrt(1.0 - 4.0 * mtau**2 / W**2)
    return (4 * math.pi * alpha**2 * hbarc2) / W**2 * beta * (
        (3 - beta**4) / (2 * beta) * math.log((1 + beta) / (1 - beta)) - 2 + beta**2
    ) * 1e9



# Integrated Tau-Tau Production Cross-Section from W_0 to sqrt(s_cms) using vegas
def integrated_tau_tau_cross_section(W0, eEbeam, pEbeam, n_samples=10000):
    s_cms = 4.0 * eEbeam * pEbeam
    upper_limit = np.sqrt(s_cms)

    def integrand(W):
        return cs_tautau_w_condition_Hamzeh(W[0]) * flux_el_yy_atW(W[0], eEbeam, pEbeam)

    integ = vegas.Integrator([[W0, upper_limit]])
    result = integ(integrand, nitn=10, neval=n_samples)
    return result.mean


################################################################################


if __name__ == "__main__":
    num_cores = 10  # Set this to the number of cores you want to use

        # Parameters
    eEbeam = 50.0  # Electron beam energy in GeV
    pEbeam = 7000.0  # Proton beam energy in GeV
    W_values = np.logspace(1.0, 3.0, 101)



    # Calculate the Elastic Photon Luminosity Spectrum in Parallel
    with Pool(processes=num_cores) as pool:
        luminosity_values = pool.starmap(flux_el_yy_atW, [(W, eEbeam, pEbeam) for W in W_values])



    # Save results to a text file with q2emax and q2pmax included in the filename
    filename_txt = f"Elastic_Photon_Luminosity_Spectrum_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.txt"
    with open(filename_txt, "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values, luminosity_values):
            file.write(f"{W:.6e}    {S_yy:.6e}\n")



    # Calculate Elastic Photon-Photon Luminosity Spectrum at W0_value
    W0_value = 10.0  # GeV
    luminosity_at_W10 = flux_el_yy_atW(W0_value, eEbeam, pEbeam)
    print(f"Elastic Photon-Photon Luminosity Spectrum at W = {W0_value} GeV: {luminosity_at_W10:.6e} GeV^-1")



    # Calculate Integrated Tau-Tau Production Cross-Section at W_0 = 10 GeV
    integrated_cross_section_value = integrated_tau_tau_cross_section(W0_value, eEbeam, pEbeam)
    print(f"Integrated Tau-Tau Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")


    # Plot the Elastic Photon-Photon Luminosity Spectrum
    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)

    plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Elastic')
    
    plt.text(15, 5.e-6, f'q2emax = {q2emax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 2.e-6, f'q2pmax = {q2pmax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 1.e-6, f'Luminosity at W={W0_value} GeV = {luminosity_at_W10:.2e} GeV^-1', fontsize=14, color='blue')

    plt.xlabel(r"$W$ [GeV]", fontsize=18)

    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    plt.title("Elastic Syy at LHeC with Correct W", fontsize=20)
    plt.grid(True, which="both", linestyle="--")


    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)

    filename_pdf = f"Elastic_Photon_Luminosity_Spectrum_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.pdf"
    filename_jpg = f"Elastic_Photon_Luminosity_Spectrum_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.jpg"


    plt.savefig(filename_pdf)
    plt.savefig(filename_jpg)
    plt.show()


################################################################################


    # Plot the Tau-Tau Production Cross-Section as a Function of W_0
    W0_range = np.arange(10.0, 1001.0, 1.0)
    with Pool(processes=num_cores) as pool:
        cross_section_values = pool.starmap(integrated_tau_tau_cross_section, [(W0, eEbeam, pEbeam) for W0 in W0_range])


    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-3, 1.e2)
    plt.loglog(W0_range, cross_section_values, linestyle='solid', linewidth=2, label='Elastic')

    plt.text(15, 2.e-2, f'q2emax = {q2emax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 1.e-2, f'q2pmax = {q2pmax:.1e} GeV^2', fontsize=14, color='blue')

    plt.text(15, 5.e-3, f'Integrated Tau-Tau Cross-Section at W_0={W0_value} GeV = {integrated_cross_section_value:.2e} pb', fontsize=14, color='blue')
    plt.xlabel(r"$W_0$ [GeV]", fontsize=18)
    plt.ylabel(r"$\sigma_{\tau^+\tau^-}$ (W > $W_0$) [pb]", fontsize=18)
    plt.title("Integrated Tau-Tau Production Cross-Section at LHeC (Corrected)", fontsize=20)

    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)


    filename_pdf = f"Integrated_elastic_tau_tau_cross_section_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.pdf"
    filename_jpg = f"Integrated_elastic_tau_tau_cross_section_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.jpg"

    plt.savefig(filename_pdf)
    plt.savefig(filename_jpg)

    plt.show()

################################################################################
