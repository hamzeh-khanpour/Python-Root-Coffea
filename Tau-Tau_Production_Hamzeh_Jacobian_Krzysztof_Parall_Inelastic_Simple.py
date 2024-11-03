
# Integrated_tau_tau_cross_section_Jacobian_Krzysztof_Parallel_Inelastic_Simple   Hamzeh Khanpour 2024

################################################################################


import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from multiprocessing import Pool
#import ALLM
#import pycepgen  # For the inelastic structure function calculations

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
q2pmax = 10.0  # Maximum photon virtuality for proton in GeV^2
MN_max = 10.0  # Maximum MN in GeV


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



# Function to compute W given ye, yp, Ee, and Ep
def compute_W(ye, yp, Ee, Ep):
    W = np.sqrt(ye * yp * 4 * Ee * Ep)
    return W

# Function to compute y_p using the updated equation (F.18) based on W^2 = y_e y_p 4 E_e E_p
def compute_yp(W, ye, Ee, Ep):
    numerator = W**2
    denominator = ye * 4 * Ee * Ep
    yp_value = numerator / denominator
    return yp_value

# Function to compute the Jacobian with respect to y_p using updated W
def compute_jacobian(W, ye, Ee, Ep):
    jacobian = abs(2 * ye * Ee * Ep / W)
    return jacobian

# Photon Flux from Electron (using lnQ2 as the integration variable)
def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2_electron(emass, ye)
    if qmin2v <= 0 or Q2e < qmin2v or Q2e > q2emax:
        return 0.0

    flux = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2)
    return flux * Q2e  # Multiply by Q2e to account for dQ^2 = Q^2 d(lnQ^2)

# Photon Flux from Proton for Inelastic Case (Equation F.13) using lnQ2p as the integration variable
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


# Photon-Photon Luminosity Spectrum Calculation (Final Form using the Jacobian)
def flux_el_yy_atW(W, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared

    # Integration over ye from ye_min to ye_max (which is 1)
    ye_min = W**2.0 / s_cms
    ye_max = 1.0

    def integrand(ye):
        # Update lnQ2e_min and lnQ2e_max using physical limits
        qmin2e = qmin2_electron(emass, ye)
        if qmin2e <= 0:
            return 0.0

        lnQ2e_min = math.log(qmin2e)
        lnQ2e_max = math.log(q2emax)

        def lnQ2e_integrand(lnQ2e):
            Q2e = np.exp(lnQ2e)

            # Integration over MN
            MN_min = pmass + pi0mass
            MN_max = 10.0  # Given as a fixed upper bound for MN

            def integrand_MN(MN):
                # Calculate y_p using Equation (F.18) dynamically based on current Q2p
                yp_value = compute_yp(W, ye, eEbeam, pEbeam)

                # Check if the computed yp is valid
                if yp_value <= 0 or yp_value >= 1:
                    return 0.0

                # Integration over Q2p using lnQ2p as the integration variable
                qmin2p = qmin2_proton(MN, yp_value)  # Using the new function for proton Qmin2
                lnQ2p_min = math.log(qmin2p)
                lnQ2p_max = math.log(q2pmax)

                def lnQ2p_integrand(lnQ2p):
                    Q2p = np.exp(lnQ2p)

                    # Calculate the Jacobian with respect to y_p using updated method
                    jacobian = compute_jacobian(W, ye, eEbeam, pEbeam)
                    if jacobian == 0:
                        return 0.0

                    # Calculate the photon flux from the proton using the inelastic structure function (F.19)
                    proton_flux = flux_y_proton(yp_value, lnQ2p, MN)
                    return proton_flux / jacobian

                # Integrate over lnQ2p
                result_lnQ2p, _ = integrate.quad(lnQ2p_integrand, lnQ2p_min, lnQ2p_max, epsrel=1e-4)
                return result_lnQ2p

            # Integrate over MN
            result_MN, _ = integrate.quad(integrand_MN, MN_min, MN_max, epsrel=1e-4)
            return result_MN * flux_y_electron(ye, lnQ2e)

        # Integrate over lnQ2e
        result_lnQ2e, _ = integrate.quad(lnQ2e_integrand, lnQ2e_min, lnQ2e_max, epsrel=1e-4)
        return result_lnQ2e

    # Integrate over ye
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

    num_cores = 100  # Set this to the number of cores you want to use


    # Parameters
    eEbeam = 50.0  # Electron beam energy in GeV
    pEbeam = 7000.0  # Proton beam energy in GeV
    W_values = np.logspace(1.0, 3.0, 101)  # Range of W values from 10 GeV to 1000 GeV


    # Calculate the Elastic Photon-Photon Luminosity Spectrum in Parallel
    with Pool(processes=num_cores) as pool:
        luminosity_values = pool.starmap(flux_el_yy_atW, [(W, eEbeam, pEbeam) for W in W_values])


    # Save results to a text file
    with open("Jacobian_Krzysztof_Parallel_Inelastic_Simple.txt", "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values, luminosity_values):
            file.write(f"{W:.6e}    {S_yy:.6e}\n")



    # Calculate Elastic Photon-Photon Luminosity Spectrum at W0_value
    W0_value = 10.0  # GeV
    luminosity_at_W10 = flux_el_yy_atW(W0_value, eEbeam, pEbeam)
    
    

    # Calculate Integrated Tau-Tau Production Cross-Section at W_0 = 10 GeV
    integrated_cross_section_value = integrated_tau_tau_cross_section(W0_value, eEbeam, pEbeam)
    print(f"Integrated inelastic Tau-Tau Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")


    # Plot the Elastic Photon-Photon Luminosity Spectrum
    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)
    plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Elastic')
    plt.text(15, 5.e-6, f'q2emax = {q2emax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 2.e-6, f'q2pmax = {q2pmax:.1e} GeV^2', fontsize=14, color='blue')


    # Corrected line with luminosity_at_W10 now defined
    plt.text(15, 1.e-6, f'Luminosity at W={W0_value} GeV = {luminosity_at_W10:.2e} GeV^-1', fontsize=14, color='blue')


    plt.xlabel(r"$W$ [GeV]", fontsize=18)
    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    plt.title("Inelastic Syy at LHeC with Correct W", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)

    plt.savefig("Jacobian_Krzysztof_Parallel_Inelastic_Simple.pdf")
    plt.savefig("Jacobian_Krzysztof_Parallel_Inelastic_Simple.jpg")
    
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
    
    
    plt.text(15, 5.e-3, f'Integrated Inelastic Tau-Tau Cross-Section at W_0={W0_value} GeV = {integrated_cross_section_value:.2e} pb', fontsize=14, color='blue')

    plt.xlabel(r"$W_0$ [GeV]", fontsize=18)
    plt.ylabel(r"$\sigma_{\tau^+\tau^-}$ (W > $W_0$) [pb]", fontsize=18)
    
    plt.title("Integrated Inelastic Tau-Tau Production Cross-Section at LHeC (Corrected)", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)


    plt.savefig("integrated_tau_tau_cross_section_Jacobian_Krzysztof_Parallel_Inelastic_Simple.pdf")
    plt.savefig("integrated_tau_tau_cross_section_Jacobian_Krzysztof_Parallel_Inelastic_Simple.jpg")
    
    plt.show()
    
    
################################################################################



