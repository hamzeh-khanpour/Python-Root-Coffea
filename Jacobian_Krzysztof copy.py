# Photon-Photon-Luminosity-Spectrum-Hamzeh_with_W_Parallel_Final_Expression_dlnq2_Jacobian

import numpy as np
import math
import scipy.integrate as integrate
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
    if y >= 1:
        return float('inf')  # This would indicate a non-physical scenario, so return infinity
    return mass * mass * y * y / (1 - y)




# Function to compute y_p using new W: W = sqrt(-Q_e^2 + y_e y_p s)
def compute_yp(W, Q2e, ye, Ee, Ep):
    s = 4 * Ee * Ep  # Center-of-mass energy squared
    numerator = W**2 + Q2e
    denominator = ye * s
    if denominator == 0:
        return 0
    yp = numerator / denominator
    return yp



# Function to compute the Jacobian (partial derivative of W with respect to y_p)
def compute_jacobian(ye, yp, Q2e, Ee, Ep):
    s = 4 * Ee * Ep  # Center-of-mass energy squared
    W = np.sqrt(-Q2e + ye * yp * s)  # Updated W using new W formula
    if W == 0:
        return 0
    jacobian = abs(ye * s / (2 * W))
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



# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W_values = np.logspace(1.0, 3.0, 101)  # Range of W values from 10 GeV to 1000 GeV



# Wrapper function for parallel processing
def wrapper_flux_el_yy_atW(W):
    return flux_el_yy_atW(W, eEbeam, pEbeam)



# Parallel Calculation of the Photon-Photon Luminosity Spectrum
if __name__ == "__main__":

    num_cores = 10  # Set this to the number of cores you want to use
    
    with Pool() as pool:
        luminosity_values = pool.map(wrapper_flux_el_yy_atW, W_values)



    # Save results to a text file
    with open("Jacobian_Krzysztof.txt", "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values, luminosity_values):
            file.write(f"{W:.6e}    {S_yy:.6e}\n")



    W_value = 10.0  # GeV
    luminosity_at_W10 = flux_el_yy_atW(W_value, eEbeam, pEbeam)
    print(f"Photon-Photon Luminosity Spectrum at W = {W_value} GeV: {luminosity_at_W10:.6e} GeV^-1")


    # Plot the Results
    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)


    plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Elastic')


    # Add additional information to the plot
    plt.text(15, 5.e-6, f'q2emax = {q2emax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 2.e-6, f'q2pmax = {q2pmax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 1.e-6, f'Luminosity at W={W_value} GeV = {luminosity_at_W10:.2e} GeV^-1', fontsize=14, color='blue')



    # Plot settings
    plt.xlabel(r"$W$ [GeV]", fontsize=18)
    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    plt.title("Elastic Syy at LHeC with Correct W", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)


    # Save the plot as a PDF and JPG
    plt.savefig("Jacobian_Krzysztof.pdf")
    plt.savefig("Jacobian_Krzysztof.jpg")
    plt.show()


