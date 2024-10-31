
# Integrated_tau_tau_cross_section_Jacobian_Krzysztof

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from multiprocessing import Pool


# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass in GeV
pmass = 0.938272081    # Proton mass in GeV

q2emax = 1000.0  # Maximum photon virtuality for electron in GeV^2
q2pmax = 1000.0  # Maximum photon virtuality for proton in GeV^2

# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)


def G_M(Q2):
    return 7.78 * G_E(Q2)


# Minimum Photon Virtuality
def qmin2(mass, y):
    if y >= 1:
        return float('inf')  # This would indicate a non-physical scenario, so return infinity
    return mass * mass * y * y / (1 - y)


# Function to compute y_p using Equation (C.5)
def compute_yp(W, Q2e, ye, Ee, Ep):
    numerator = W**2 + Q2e
    denominator = 2 * ye * Ee * Ep + 2 * Ep * np.sqrt((ye * Ee)**2 + Q2e) * (1 - Q2e / (2 * Ee**2 * (1 - ye)))
    if denominator == 0:
        return 0
    yp = numerator / denominator
    return yp


# Function to compute the Jacobian (partial derivative of f with respect to y_p)
def compute_jacobian(ye, yp, Q2e, Ee, Ep):
    # Calculate the inner term g(y_e, y_p, Q2_e)
    g = (-Q2e + 2 * ye * yp * Ee * Ep 
         + 2 * yp * Ep * np.sqrt((ye * Ee) ** 2 + Q2e) * (1 - Q2e / (2 * Ee ** 2 * (1 - ye))))
    
    # Partial derivative of g with respect to y_p
    partial_g = (2 * ye * Ee * Ep 
                 + 2 * Ep * np.sqrt((ye * Ee) ** 2 + Q2e) * (1 - Q2e / (2 * Ee ** 2 * (1 - ye))))
    
    # Calculate the Jacobian
    jacobian = abs(partial_g / (2 * np.sqrt(g)))
    
    return jacobian



# Photon Flux from Electron (using lnQ2 as the integration variable)
def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2(emass, ye)
    if qmin2v <= 0 or Q2e < qmin2v or Q2e > q2emax:
        return 0.0

    flux = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2)
    return flux * Q2e  # Multiply by Q2e to account for dQ^2 = Q^2 d(lnQ^2)



# Photon Flux from Proton (using lnQ2 as the integration variable)
def flux_y_proton(yp):
    if yp <= 0 or yp >= 1:
        return 0.0
    qmin2p = qmin2(pmass, yp)
    if qmin2p <= 0:
        return 0.0

    lnQ2p_min = math.log(qmin2p)
    lnQ2p_max = math.log(q2pmax)

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

    result_lnQ2p, _ = integrate.quad(lnQ2p_integrand, lnQ2p_min, lnQ2p_max, epsrel=1e-4)
    return result_lnQ2p



# Photon-Photon Luminosity Spectrum Calculation (Final Form using the Jacobian)
def flux_el_yy_atW(W, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared

    # Integration over ye from ye_min to ye_max (which is 1)
    ye_min = W**2.0 / s_cms
    ye_max = 1.0

    def integrand(ye):
        # Update lnQ2e_min and lnQ2e_max using physical limits
        qmin2e = qmin2(emass, ye)
        if qmin2e <= 0:
            return 0.0

        lnQ2e_min = math.log(qmin2e)
        lnQ2e_max = math.log(q2emax)

        def lnQ2e_integrand(lnQ2e):
            Q2e = np.exp(lnQ2e)
            # Calculate y_p using Equation (C.5)
            yp_value = compute_yp(W, Q2e, ye, eEbeam, pEbeam)

            if yp_value <= 0 or yp_value >= 1:
                return 0.0

            # Calculate the Jacobian
            jacobian = compute_jacobian(ye, yp_value, Q2e, eEbeam, pEbeam)
            if jacobian == 0:
                return 0.0

            # Calculate the photon flux from the proton at y_p = y_p*
            proton_flux = flux_y_proton(yp_value)

            # Calculate the photon flux from the electron
            flux_e = flux_y_electron(ye, lnQ2e)

            return flux_e * proton_flux / jacobian

        result_lnQ2e, _ = integrate.quad(lnQ2e_integrand, lnQ2e_min, lnQ2e_max, epsrel=1e-4)
        return result_lnQ2e

    result_ye, _ = integrate.quad(integrand, ye_min, ye_max, epsrel=1e-4)
    return result_ye



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
def integrated_tau_tau_cross_section(W0, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared
    
    try:
        result, _ = integrate.quad(
            lambda W: cs_tautau_w_condition_Hamzeh(W) * flux_el_yy_atW(W, eEbeam, pEbeam),
            W0, np.sqrt(s_cms), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for tau-tau production cross-section did not converge for W_0={W0}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for tau-tau production cross-section: {e}")
        result = 0.0
    return result


################################################################################


if __name__ == "__main__":

    num_cores = 20  # Set this to the number of cores you want to use

    # Parameters
    eEbeam = 50.0  # Electron beam energy in GeV
    pEbeam = 7000.0  # Proton beam energy in GeV
    W_values = np.logspace(1.0, 3.0, 101)  # Range of W values from 10 GeV to 1000 GeV

    # Calculate the Elastic Photon-Photon Luminosity Spectrum in Parallel
    with Pool() as pool:
        luminosity_values = pool.starmap(flux_el_yy_atW, [(W, eEbeam, pEbeam) for W in W_values])

    # Save results to a text file
    with open("With_Suppresion_Factor.txt", "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values, luminosity_values):
            file.write(f"{W:.6e}    {S_yy:.6e}\n")


    # Calculate Integrated Tau-Tau Production Cross-Section at W_0 = 10 GeV
    W0_value = 10.0  # GeV
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


    plt.savefig("Jacobian_Krzysztof.pdf")
    plt.savefig("Jacobian_Krzysztof.jpg")
    plt.show()


################################################################################


    # Plot the Tau-Tau Production Cross-Section as a Function of W_0
    W0_range = np.arange(10.0, 1001.0, 1.0)  # Range of W_0 values from 10 GeV to 1000 GeV

    with Pool() as pool:
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
    plt.title("Integrated Tau-Tau Production Cross-Section at LHeC  (Corrected)", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)


    plt.savefig("integrated_tau_tau_cross_section_Jacobian_Krzysztof.pdf")
    plt.savefig("integrated_tau_tau_cross_section_Jacobian_Krzysztof.jpg")


    plt.show()

################################################################################
