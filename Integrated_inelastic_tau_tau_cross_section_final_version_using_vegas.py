#  This code calculates the "inelastic" Photon Luminosity Spectrum Syy : using Vegas for the integration
#  Final Version (with corrected W^2 = .....) to solve high photon virtuality issues
#  This code also calculates the integrated "inelastic" tau tau cross section (ep -> e p tau^+ tau^-)
#  Hamzeh Khanpour, Laurent Forthomme, and Krzysztof Piotrzkowski -- November 2024

################################################################################


import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from multiprocessing import Pool
import vegas


#import pycepgen  # For the inelastic structure function calculations
#import ALLM

# 202 ALLM97 (continuum, FT/HERA photoprod. tot.x-s 1356 points fit)
# sf_ALLM97 = pycepgen.StructureFunctionsFactory.build(202)

# 11 Suri-Yennie
# sf_Suri_Yennie = pycepgen.StructureFunctionsFactory.build(11)

# 301 LUXlike (hybrid)
# sf_luxlike = pycepgen.StructureFunctionsFactory.build(301)

# 303 Kulagin-Barinov (hybrid)
# sf_Kulagin_Barinov = pycepgen.StructureFunctionsFactory.build(303)


# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass in GeV
pmass = 0.938272081    # Proton mass in GeV
pi0mass = 0.1349768    # Pion mass in GeV

q2emax = 100000.0  # Maximum photon virtuality for electron in GeV^2
q2pmax = 100000.0  # Maximum photon virtuality for proton in GeV^2
MN_max = 100.0  # Maximum MN in GeV


#=========================================================================


Mass2_0 = 0.31985
Mass2_P = 49.457
Mass2_R = 0.15052
Q2_0    = 0.52544
Lambda2 = 0.06527

Ccp = (0.28067, 0.22291,  2.1979)
Cap = (-0.0808, -0.44812, 1.1709)
Cbp = (0.36292, 1.8917,   1.8439)

Ccr = (0.80107, 0.97307, 3.4942)
Car = (0.58400, 0.37888, 2.6063)
Cbr = (0.01147, 3.7582,  0.49338)


# --------------------------------------------------------------


def tvalue(Q2):
    return math.log \
           ((math.log((Q2 + Q2_0) / Lambda2) / math.log(Q2_0 / Lambda2)))


def xP(xbj, Q2):
    if xbj == 0:
        print("xbj zero")
        return -1.
    xPinv = 1. + Q2 / (Q2 + Mass2_P) * (1. / xbj - 1.)
    return 1. / xPinv


def xR(xbj, Q2):
    if xbj == 0:
        print("xbj zero")
        return -1.
    xPinv = 1. + Q2 / (Q2 + Mass2_R) * (1. / xbj - 1.)
    return 1. / xPinv


def type1(tval, tuple1):
    return tuple1[0] + tuple1[1] * (tval ** tuple1[2])


def type2(tval, tuple1):
    return tuple1[0] +\
           (tuple1[0] - tuple1[1]) * (1. / (1. + tval ** tuple1[2]) - 1.)


def aP(tval):
    return type2(tval, Cap)


def bP(tval):
    return type1(tval, Cbp)


def cP(tval):
    return type2(tval, Ccp)


def aR(tval):
    return type1(tval, Car)


def bR(tval):
    return type1(tval, Cbr)


def cR(tval):
    return type1(tval, Ccr)


def allm_f2P(xbj, Q2):
    tval = tvalue(Q2)
    return cP(tval) * (xP(xbj, Q2) ** aP(tval)) * ((1. - xbj) ** bP(tval))


def allm_f2R(xbj, Q2):
    tval = tvalue(Q2)
    return cR(tval) * (xR(xbj, Q2) ** aR(tval)) * ((1. - xbj) ** bR(tval))


def allm_f2(xbj, Q2):
    return Q2 / (Q2 + Mass2_0) * (allm_f2P(xbj, Q2) + allm_f2R(xbj, Q2))


#=========================================================================



# Minimum Photon Virtuality
def qmin2_electron(mass, y):
    if y >= 1:
        return float('inf')  # This would indicate a non-physical scenario, so return infinity
    return mass * mass * y * y / (1 - y)



# Minimum Photon Virtuality for Proton Case
def qmin2_proton(MN, y):
    if y >= 1:
        return float('inf')  # Non-physical scenario, return infinity
    return ((MN**2) / (1 - y) - pmass**2) * y



# Function to compute y_p using given equation (F.18)
def compute_yp(W, Q2e, Q2p, ye, Ee, Ep, MN):
    numerator = W**2 + Q2e + Q2p - (Q2e * (Q2p + MN**2 - pmass**2)) / (4 * Ee * Ep)
    denominator = ye * 4 * Ee * Ep
    yp_value = numerator / denominator
    return yp_value



# Function to compute the Jacobian with respect to y_p
def compute_jacobian(ye, Ee, Ep, W):
    jacobian = abs(2 * ye * Ee * Ep / W)
    return jacobian



# Photon Flux from Electron (using lnQ2 as the integration variable) (Equation F.13)
def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2_electron(emass, ye)
    if qmin2v <= 0 or Q2e < qmin2v or Q2e > q2emax:
        return 0.0

    flux = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2)
    return flux * Q2e  # Multiply by Q2e to account for dQ^2 = Q^2 d(lnQ^2)



# Photon Flux from Proton for Inelastic Case (Equation F.14) using lnQ2p as the integration variable
def flux_y_proton(yp, lnQ2p, MN):
    Q2p = np.exp(lnQ2p)
    xbj = Q2p / (MN**2 - pmass**2 + Q2p)

    if yp <= 0 or yp >= 1:
        return 0.0
    qmin2p = qmin2_proton(MN, yp)  # Using new function for proton Qmin2
    if qmin2p <= 0 or Q2p < qmin2p or Q2p > q2pmax:
        return 0.0

    # Structure function calculations for inelastic case using cepgen
    FE = allm_f2(xbj, Q2p) * (2 * MN / (MN**2 - pmass**2 + Q2p))
    FM = allm_f2(xbj, Q2p) * (2 * MN * (MN**2 - pmass**2 + Q2p)) / (Q2p * Q2p)

    flux = ALPHA2PI / (yp * Q2p) * ((1 - yp) * (1 - qmin2p / Q2p) * FE + 0.5 * yp**2 * FM)
    return flux * Q2p  # Multiply by Q2p to account for dQ^2 = Q^2 d(lnQ^2)



# The Inelastic Photon-Photon Luminosity Spectrum Calculation (Final Form using the Jacobian) (F.19)
# The Inelastic Photon-Photon Luminosity Spectrum Calculation using vegas
def flux_el_yy_atW(W, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared
    ye_min = W**2.0 / s_cms
    ye_max = 1.0

    # Define the vegas integrand for the inelastic photon-photon luminosity spectrum
    def vegas_integrand(x):
        # Map vegas inputs [0,1] to physical ranges for each variable
        ye = ye_min + x[0] * (ye_max - ye_min)
        lnQ2e_min = math.log(qmin2_electron(emass, ye))
        lnQ2e_max = math.log(q2emax)
        lnQ2e = lnQ2e_min + x[1] * (lnQ2e_max - lnQ2e_min)
        Q2e = np.exp(lnQ2e)

        MN_min, MN_max = pmass + pi0mass, 100.0           #  MN_max = 100.0
        MN = MN_min + x[2] * (MN_max - MN_min)

        # Calculate the Jacobian based on W and current variables
        jacobian = compute_jacobian(ye, eEbeam, pEbeam, W)
        if jacobian == 0:
            return 0.0

        # Compute y_p value based on current integration variables
        yp_value = compute_yp(W, Q2e, 0.0, ye=ye, Ee=eEbeam, Ep=pEbeam, MN=MN)
        if yp_value <= 0 or yp_value >= 1:
            return 0.0

        # Map [0,1] interval for Q2p
        qmin2p = qmin2_proton(MN, yp_value)
        lnQ2p_min = math.log(qmin2p)
        lnQ2p_max = math.log(q2pmax)
        lnQ2p = lnQ2p_min + x[3] * (lnQ2p_max - lnQ2p_min)
        Q2p = np.exp(lnQ2p)

        # Compute y_p value based on current integration variables
        yp_value = compute_yp(W, Q2e, Q2p, ye=ye, Ee=eEbeam, Ep=pEbeam, MN=MN)
        if yp_value <= 0 or yp_value >= 1:
            return 0.0

        # Calculate the fluxes from electron and proton
        proton_flux = flux_y_proton(yp_value, lnQ2p, MN)
        electron_flux = flux_y_electron(ye, lnQ2e)

        # Multiply by scaling factors for each variable's range
        return (
            electron_flux * proton_flux / jacobian *
            (ye_max - ye_min) *
            (lnQ2e_max - lnQ2e_min) *
            (MN_max - MN_min) *
            (lnQ2p_max - lnQ2p_min)
        )

    # Set up vegas integrator for 4-dimensional integration over ye, lnQ2e, MN, lnQ2p
    integrator = vegas.Integrator([[0, 1], [0, 1], [0, 1], [0, 1]])

    # Training phase
    integrator(vegas_integrand, nitn=5, neval=1000)

    # Final evaluation
    result = integrator(vegas_integrand, nitn=10, neval=10000)

    # Optional: Print summary for debugging
    #print(result.summary())
    #print('Result Inelastic Syy =', result, 'Q =', result.Q)

    # Return mean of result if Q value indicates stability, otherwise None
    return result.mean if result.Q > 0.1 else None



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
    #print('Result Inelastic tau tau xs =', result, 'Q =', result.Q)

    # Return mean of the result if Q is stable, otherwise 0.0
    return result.mean if result.Q > 0.1 else 0.0



################################################################################


# Parameters
eEbeam = 20.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
#W_values = np.logspace(1.0, 3.0, 303)  # Range of W values from 10 GeV to 1000 GeV
W_values = np.logspace(1.0, 2.874, 303)  # Range of W values from 10 GeV to 1000 GeV  LHeC@750GeV


num_cores = 10  # Set this to the number of cores you want to use

# Wrapper function for parallel processing
def wrapper_flux_el_yy_atW(W):
    return flux_el_yy_atW(W, eEbeam, pEbeam)


# Parallel Calculation of the Photon-Photon Luminosity Spectrum
if __name__ == "__main__":


    with Pool(processes=num_cores) as pool:
        luminosity_values = pool.map(wrapper_flux_el_yy_atW, W_values)


# Save results with None handling and formatted filename
    filename = f"Inelastic_Photon_Luminosity_Spectrum_MNmax_{int(MN_max)}_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas_LHeC750GeV.txt"

    with open(filename, "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values, luminosity_values):
        # Only write non-zero and non-None S_yy values
            if S_yy is not None and S_yy != 0.0:
                file.write(f"{W:.6e}    {S_yy:.6e}\n")


    W_value = 10.0  # GeV
    luminosity_at_W10 = flux_el_yy_atW(W_value, eEbeam, pEbeam)
    print(f"Photon-Photon Luminosity Spectrum at W = {W_value} GeV: {luminosity_at_W10:.6e} GeV^-1")


    # Calculate Integrated Tau-Tau Production Cross-Section at W_0 = 10 GeV
    integrated_cross_section_value = integrated_tau_tau_cross_section(W_value, eEbeam, pEbeam)
    print(f"Integrated Inelastic Tau-Tau Production Cross-Section at W_0 = {W_value} GeV: {integrated_cross_section_value:.6e} pb")

    # Plot the Results
    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)


    plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Inelastic')


    # Add additional information to the plot
    plt.text(15, 5.e-6, f'q2emax = {q2emax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 2.e-6, f'q2pmax = {q2pmax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 1.e-6, f'Luminosity at W={W_value} GeV = {luminosity_at_W10:.2e} GeV^-1', fontsize=14, color='blue')


    # Plot settings
    plt.xlabel(r"$W$ [GeV]", fontsize=18)
    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    plt.title("Inelastic $S_{\gamma\gamma}$ at LHeC with Correct W", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)


# Save the plot as a PDF and JPG
# Construct filenames for the plots including MN_max, q2emax, and q2pmax
    plot_filename_pdf = f"Inelastic_Photon_Luminosity_Spectrum_MNmax_{int(MN_max)}_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.pdf"
    plot_filename_jpg = f"Inelastic_Photon_Luminosity_Spectrum_MNmax_{int(MN_max)}_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.jpg"

# Save the plot with dynamic filenames
    plt.savefig(plot_filename_pdf)
    plt.savefig(plot_filename_jpg)

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
    filename_pdf = f"Integrated_inelastic_tau_tau_cross_section_MNmax_{int(MN_max)}_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.pdf"
    filename_jpg = f"Integrated_inelastic_tau_tau_cross_section_MNmax_{int(MN_max)}_q2emax_{int(q2emax)}_q2pmax_{int(q2pmax)}_using_vegas.jpg"

# Save the plot with the customized filenames
    plt.savefig(filename_pdf)
    plt.savefig(filename_jpg)

    plt.show()


################################################################################

