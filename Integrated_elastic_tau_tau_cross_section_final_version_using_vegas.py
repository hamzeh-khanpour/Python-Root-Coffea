#  This code calculates the elastic Photon Luminosity Spectrum Syy : using Vegas for the integration
#  Final Version (with corrected W^2 = -Q2e + ye * yp * s) to solve high photon virtuality issues
#  This code also calculates the integrated elastic tau tau cross section (ep -> e p tau^+ tau^-)
#  Hamzeh Khanpour, Laurent Forthomme, and Krzysztof Piotrzkowski -- November 2024

################################################################################


import numpy as np
import math
import vegas  # For Monte Carlo integration with vegas
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy.integrate as integrate


# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi

emass = 5.1099895e-4  # Electron mass in GeV
pmass = 0.938272081   # Proton mass in GeV

q2emax = 10.0         # Maximum photon virtuality for electron in GeV^2
q2pmax = 10.0         # Maximum photon virtuality for proton in GeV^2

# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)


def G_M(Q2):
    return 7.78 * G_E(Q2)


# Minimum Photon Virtuality
def qmin2(mass, y):
    return mass * mass * y * y / (1 - y) if y < 1 else float('inf')


# Function to compute y_p using W = sqrt(-Q_e^2 + y_e y_p s)
def compute_yp(W, Q2e, ye, Ee, Ep):
    s = 4 * Ee * Ep  # Center-of-mass energy squared
    denominator = ye * s
    return (W**2 + Q2e) / denominator if denominator != 0 else 0


# Function to compute the Jacobian (partial derivative of W with respect to y_p)
def compute_jacobian(ye, yp, Q2e, Ee, Ep):
    s = 4 * Ee * Ep  # Center-of-mass energy squared
    W = np.sqrt(-Q2e + ye * yp * s)
    return abs(ye * s / (2 * W)) if W != 0 else 0


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
    if yp <= 0 or yp >= 1:                     # Hamzeh tagged elastic   ==>  if (yp <= 0.01 or yp >= 0.20):
        print('invalid yp value: ', yp)
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


################################################################################


# Photon Luminosity Spectrum Calculation (using vegas for integration)
def flux_el_yy_atW(W, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam
    ye_min, ye_max = W**2.0 / s_cms, 1.0

    # Define the integrand function
    def vegas_integrand(x):
        ye, lnQ2e = ye_min + x[0] * (ye_max - ye_min), math.log(qmin2(emass, ye_min)) + x[1] * (math.log(q2emax) - math.log(qmin2(emass, ye_min)))
        Q2e = np.exp(lnQ2e)
        
        # Compute necessary values
        yp_value = compute_yp(W, Q2e, ye, eEbeam, pEbeam)
        if yp_value <= 0.0 or yp_value >= 1.0:              
#            print('invalid yp_value value: ', yp_value)
            return 0.0
        

        # Calculate the Jacobian
        jacobian = compute_jacobian(ye, yp_value, Q2e, eEbeam, pEbeam)
        if jacobian == 0:
            return 0.0

        # Photon flux calculations
        proton_flux = flux_y_proton(yp_value)
        flux_e = flux_y_electron(ye, lnQ2e)
        
        # Multiply by the volume elements
        return flux_e * proton_flux / jacobian * (ye_max - ye_min) * (math.log(q2emax) - math.log(qmin2(emass, ye_min)))

    # Set up the vegas Integrator with bounds for ye and lnQ2e as [0, 1] for vegas
    integrator = vegas.Integrator([[0, 1], [0, 1]])

    # Training phase
    integrator(vegas_integrand, nitn=5, neval=1000) 

    # Final evaluation
    result = integrator(vegas_integrand, nitn=10, neval=10000)

    # Optional: Print summary for debugging
    #print(result.summary())
    #print('Result Elastic Syy =', result, 'Q =', result.Q)

    return result.mean if result.Q > 0.1 else None  # Return only if Q indicates stable result


################################################################################



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




# Integrated Tau-Tau Production Cross-Section using vegas
def integrated_tau_tau_cross_section(W0, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam
    upper_limit = np.sqrt(s_cms)

    # Define the vegas integrand for the cross-section
    def vegas_integrand(x):
        # Scale x from [0, 1] to [W0, upper_limit]
        W = W0 + x[0] * (upper_limit - W0)
        
        # Calculate the cross-section and luminosity spectrum
        tau_tau_cross_section = cs_tautau_w_condition_Hamzeh(W)
        luminosity_spectrum = flux_el_yy_atW(W, eEbeam, pEbeam)
        
        if luminosity_spectrum is None:
            luminosity_spectrum = 0.0  # Handle NoneType if vegas fails
        
        # Return the product and scale by volume element (upper_limit - W0)
        return tau_tau_cross_section * luminosity_spectrum * (upper_limit - W0)

    # Set up the vegas Integrator for W range [W0, upper_limit] mapped to [0, 1]
    integrator = vegas.Integrator([[0, 1]])

    # Training phase
    integrator(vegas_integrand, nitn=5, neval=1000) 

    # Final evaluation
    result = integrator(vegas_integrand, nitn=10, neval=10000)

    # Optional: Print summary for debugging
    #print(result.summary())
    #print('Result Elastic tau tau xs =', result, 'Q =', result.Q)

    # Return mean of the result if Q is stable, otherwise 0.0
    return result.mean if result.Q > 0.1 else 0.0



################################################################################

# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W_values = np.logspace(1.0, 3.0, 303)  # Range of W values from 10 GeV to 1000 GeV
#W_values = np.logspace(1.0, 2.874, 404)  # Range of W values from 10 GeV to 1000 GeV  _LHeC750GeV

num_cores = 10  # Set this to the number of cores you want to use

# Wrapper function for parallel processing
def wrapper_flux_el_yy_atW(W):
    return flux_el_yy_atW(W, eEbeam, pEbeam)

# Parallel Calculation of the Photon-Photon Luminosity Spectrum
if __name__ == "__main__":

    # Calculate the Elastic Photon Luminosity Spectrum in Parallel
    with Pool(processes=num_cores) as pool:
        luminosity_values = pool.starmap(flux_el_yy_atW, [(W, eEbeam, pEbeam) for W in W_values])

    # Save results to a text file with q2emax and q2pmax included in the filename
    filename_txt = f"Elastic_Photon_Luminosity_Spectrum_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.txt"
    with open(filename_txt, "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values, luminosity_values):

            # Only write non-zero and non-None S_yy values
            if S_yy is not None and S_yy != 0.0:
                file.write(f"{W:.6e}    {S_yy:.6e}\n")


    # Calculate Elastic Photon-Photon Luminosity Spectrum at W0_value
    W0_value = 10.0  # GeV
    luminosity_at_W10 = flux_el_yy_atW(W0_value, eEbeam, pEbeam)
    print(f"Elastic Photon-Photon Luminosity Spectrum at W = {W0_value} GeV: {luminosity_at_W10:.6e} GeV^-1")

    # Calculate Integrated Tau-Tau Production Cross-Section at W_0 = 10 GeV
    integrated_cross_section_value = integrated_tau_tau_cross_section(W0_value, eEbeam, pEbeam)
    print(f"Integrated Elastic Tau-Tau Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")


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


    # Save plot
    plt.savefig(f"Elastic_Photon_Luminosity_Spectrum_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.pdf")
    plt.savefig(f"Elastic_Photon_Luminosity_Spectrum_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.jpg")
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


# Define filenames with q2emax and q2pmax values included
    filename_pdf = f"Integrated_elastic_tau_tau_cross_section_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.pdf"
    filename_jpg = f"Integrated_elastic_tau_tau_cross_section_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.jpg"

# Save the plot with the customized filenames
    plt.savefig(filename_pdf)
    plt.savefig(filename_jpg)

    plt.show()


################################################################################
