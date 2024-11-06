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
pi0mass = 0.1349768    # Pion mass in GeV

q2emax = 100000.0  # Maximum photon virtuality for electron in GeV^2
q2pmax = 10.0  # Maximum photon virtuality for proton in GeV^2
MN_max = 10.0  # Maximum MN in GeV

# Parameters for the ALLM structure function model
Mass2_0 = 0.31985
Mass2_P = 49.457
Mass2_R = 0.15052
Q2_0 = 0.52544
Lambda2 = 0.06527

Ccp = (0.28067, 0.22291, 2.1979)
Cap = (-0.0808, -0.44812, 1.1709)
Cbp = (0.36292, 1.8917, 1.8439)
Ccr = (0.80107, 0.97307, 3.4942)
Car = (0.58400, 0.37888, 2.6063)
Cbr = (0.01147, 3.7582, 0.49338)

def tvalue(Q2):
    """Compute t-value for the ALLM model."""
    return math.log(math.log((Q2 + Q2_0) / Lambda2) / math.log(Q2_0 / Lambda2))

def xP(xbj, Q2):
    """Calculate xP for the ALLM model."""
    if xbj == 0:
        return -1.0
    xPinv = 1.0 + Q2 / (Q2 + Mass2_P) * (1.0 / xbj - 1.0)
    return 1.0 / xPinv

def xR(xbj, Q2):
    """Calculate xR for the ALLM model."""
    if xbj == 0:
        return -1.0
    xPinv = 1.0 + Q2 / (Q2 + Mass2_R) * (1.0 / xbj - 1.0)
    return 1.0 / xPinv

def type1(tval, tuple1):
    return tuple1[0] + tuple1[1] * (tval ** tuple1[2])

def type2(tval, tuple1):
    return tuple1[0] + (tuple1[0] - tuple1[1]) * (1.0 / (1.0 + tval ** tuple1[2]) - 1.0)

def allm_f2(xbj, Q2):
    """Calculate the ALLM structure function F2."""
    tval = tvalue(Q2)
    f2P = type2(tval, Ccp) * (xP(xbj, Q2) ** type2(tval, Cap)) * ((1.0 - xbj) ** type1(tval, Cbp))
    f2R = type1(tval, Ccr) * (xR(xbj, Q2) ** type1(tval, Car)) * ((1.0 - xbj) ** type1(tval, Cbr))
    return Q2 / (Q2 + Mass2_0) * (f2P + f2R)

def qmin2_electron(mass, y):
    if y >= 1:
        return float('inf')  # Non-physical scenario
    return mass * mass * y * y / (1 - y)

def qmin2_proton(MN, y):
    if y >= 1:
        return float('inf')  # Non-physical scenario
    return ((MN**2) / (1 - y) - pmass**2) * y

def compute_yp(W, Q2e, Q2p, ye, Ee, Ep):
    numerator = W**2 + Q2e + Q2p
    denominator = ye * 4 * Ee * Ep
    return numerator / denominator

def compute_jacobian(ye, Ee, Ep, W):
    return abs(2 * ye * Ee * Ep / W)

# Precompute MN-integrated flux_y_proton
def precompute_flux_y_proton(lnQ2p, yp):
    def proton_flux_integrand(MN):
        Q2p = np.exp(lnQ2p)
        xbj = Q2p / (MN**2 - pmass**2 + Q2p)
        qmin2p = qmin2_proton(MN, yp)

        if yp <= 0 or yp >= 1 or qmin2p <= 0 or Q2p < qmin2p or Q2p > q2pmax:
            return 0.0

        # Calculate the structure function contributions
        FE = allm_f2(xbj, Q2p) * (2 * MN / (MN**2 - pmass**2 + Q2p))
        FM = allm_f2(xbj, Q2p) * (2 * MN * (MN**2 - pmass**2 + Q2p)) / (Q2p * Q2p)

        # Photon flux contribution from proton
        flux = ALPHA2PI / (yp * Q2p) * ((1 - yp) * (1 - qmin2p / Q2p) * FE + 0.5 * yp**2 * FM)
        return flux * Q2p  # Multiply by Q2p to account for dQ^2 = Q^2 d(lnQ^2)

    # Integrate over MN
    MN_min = pmass + pi0mass
    MN_max = 10.0
    result_MN, _ = integrate.quad(proton_flux_integrand, MN_min, MN_max, epsrel=1e-3)
    return result_MN




def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    qmin2v = qmin2_electron(emass, ye)

    if ye <= 0 or ye >= 1 or qmin2v <= 0 or Q2e < qmin2v or Q2e > q2emax:
        return 0.0

    # Calculate electron flux
    flux = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2)
    return flux * Q2e


def flux_el_yy_atW(W, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared
    ye_min = W**2.0 / s_cms
    ye_max = 1.0

    def integrand(ye):
        qmin2e = qmin2_electron(emass, ye)
        if qmin2e <= 0:
            return 0.0

        lnQ2e_min = math.log(qmin2e)
        lnQ2e_max = math.log(q2emax)

        def lnQ2e_integrand(lnQ2e):
            Q2e = np.exp(lnQ2e)
            jacobian = compute_jacobian(ye, eEbeam, pEbeam, W)
            if jacobian == 0:
                return 0.0

            # Integration over Q2p using lnQ2p as the integration variable
            def lnQ2p_integrand(lnQ2p):
                Q2p = np.exp(lnQ2p)
                yp_value = compute_yp(W, Q2e, Q2p, ye, eEbeam, pEbeam)

                if yp_value <= 0 or yp_value >= 1:
                    return 0.0

                # Use precomputed proton flux with MN integration
                proton_flux = precompute_flux_y_proton(lnQ2p, yp_value)
                return proton_flux / jacobian

            # Integrate over lnQ2p
            result_lnQ2p, _ = integrate.quad(lnQ2p_integrand, math.log(q2pmax), math.log(q2pmax), epsrel=1e-4)
            return result_lnQ2p * flux_y_electron(ye, lnQ2e)

        # Integrate over lnQ2e
        result_lnQ2e, _ = integrate.quad(lnQ2e_integrand, lnQ2e_min, lnQ2e_max, epsrel=1e-4)
        return result_lnQ2e

    # Integrate over ye
    result_ye, _ = integrate.quad(integrand, ye_min, ye_max, epsrel=1e-4)
    return result_ye

# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W_values = np.logspace(1.0, 3.0, 101)

def wrapper_flux_el_yy_atW(W):
    return flux_el_yy_atW(W, eEbeam, pEbeam)

# Parallel Calculation of Photon-Photon Luminosity Spectrum
if __name__ == "__main__":
    num_cores = 16  # Adjust based on available resources

    # Use multiprocessing to parallelize computation over W values
    with Pool(num_cores) as pool:
        luminosity_values = pool.map(wrapper_flux_el_yy_atW, W_values)

    # Save results to a text file
    with open("Jacobian_Krzysztof_Inelastic_Updated_noMN.txt", "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values, luminosity_values):
            file.write(f"{W:.6e}    {S_yy:.6e}\n")

    # Example calculation at a specific W value for verification
    W_value = 10.0  # GeV
    luminosity_at_W10 = flux_el_yy_atW(W_value, eEbeam, pEbeam)
    print(f"Photon-Photon Luminosity Spectrum at W = {W_value} GeV: {luminosity_at_W10:.6e} GeV^-1")

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
    plt.savefig("Jacobian_Krzysztof_Inelastic_Updated_noMN.pdf")
    plt.savefig("Jacobian_Krzysztof_Inelastic_Updated_noMN.jpg")
    plt.show()
