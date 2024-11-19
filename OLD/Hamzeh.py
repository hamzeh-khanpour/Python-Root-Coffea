# Photon-Photon-Luminosity-Spectrum-Hamzeh_with_W_Parallel_Final_Expression_dlnq2_Jacobian_Inelastic_Case
# Hamzeh, Laurent and Krzysztof ---- 4 November 2024

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import lru_cache
import numba

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass in GeV
pmass = 0.938272081    # Proton mass in GeV
pi0mass = 0.1349768    # Pion mass in GeV

q2emax = 1e5  # Maximum photon virtuality for electron in GeV^2
q2pmax = 10.0  # Maximum photon virtuality for proton in GeV^2
MN_min = pmass + pi0mass  # Minimum MN in GeV
MN_max = 10.0             # Maximum MN in GeV

# Parameters for the ALLM function
Mass2_0, Mass2_P, Mass2_R, Q2_0, Lambda2 = 0.31985, 49.457, 0.15052, 0.52544, 0.06527
Ccp, Cap, Cbp = (0.28067, 0.22291, 2.1979), (-0.0808, -0.44812, 1.1709), (0.36292, 1.8917, 1.8439)
Ccr, Car, Cbr = (0.80107, 0.97307, 3.4942), (0.58400, 0.37888, 2.6063), (0.01147, 3.7582, 0.49338)

@numba.njit
def compute_yp(W, Q2e, Q2p, ye, Ee, Ep, MN):
    numerator = W**2 + Q2e + Q2p - (Q2e * (Q2p + MN**2 - pmass**2)) / (4 * Ee * Ep)
    denominator = ye * 4 * Ee * Ep
    return numerator / denominator if denominator != 0 else 0

@numba.njit
def compute_jacobian(ye, Ee, Ep, W):
    return abs(2 * ye * Ee * Ep / W) if W != 0 else 0

@lru_cache(maxsize=None)
def tvalue(Q2):
    return math.log(math.log((Q2 + Q2_0) / Lambda2) / math.log(Q2_0 / Lambda2))

@lru_cache(maxsize=None)
def allm_f2(xbj, Q2):
    tval = tvalue(Q2)
    def type1(tval, tuple1):
        return tuple1[0] + tuple1[1] * (tval ** tuple1[2])

    def type2(tval, tuple1):
        return tuple1[0] + (tuple1[0] - tuple1[1]) * (1. / (1. + tval ** tuple1[2]) - 1.)

    cP_val = type2(tval, Ccp)
    aP_val, bP_val = type2(tval, Cap), type1(tval, Cbp)
    xPinv = 1 + Q2 / (Q2 + Mass2_P) * (1. / xbj - 1.) if xbj != 0 else 0
    f2P = cP_val * ((1 / xPinv) ** aP_val) * ((1 - xbj) ** bP_val) if xbj != 0 else 0

    cR_val = type1(tval, Ccr)
    aR_val, bR_val = type1(tval, Car), type1(tval, Cbr)
    xRinv = 1 + Q2 / (Q2 + Mass2_R) * (1. / xbj - 1.) if xbj != 0 else 0
    f2R = cR_val * ((1 / xRinv) ** aR_val) * ((1 - xbj) ** bR_val) if xbj != 0 else 0

    return Q2 / (Q2 + Mass2_0) * (f2P + f2R)

@numba.njit
def qmin2_electron(mass, y):
    return mass * mass * y * y / (1 - y) if y < 1 else float('inf')

@numba.njit
def qmin2_proton(MN, y):
    return ((MN**2) / (1 - y) - pmass**2) * y if y < 1 else float('inf')

@numba.njit
def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2_electron(emass, ye)
    if qmin2v <= 0 or Q2e < qmin2v or Q2e > q2emax:
        return 0.0
    flux = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2)
    return flux * Q2e

@lru_cache(maxsize=None)
def flux_y_proton(yp, lnQ2p, MN):
    Q2p = np.exp(lnQ2p)
    xbj = Q2p / (MN**2 - pmass**2 + Q2p) if (MN**2 - pmass**2 + Q2p) != 0 else 0
    if yp <= 0 or yp >= 1:
        return 0.0
    qmin2p = qmin2_proton(MN, yp)
    if qmin2p <= 0 or Q2p < qmin2p or Q2p > q2pmax:
        return 0.0
    FE = allm_f2(xbj, Q2p) * (2 * MN / (MN**2 - pmass**2 + Q2p))
    FM = allm_f2(xbj, Q2p) * (2 * MN * (MN**2 - pmass**2 + Q2p)) / (Q2p * Q2p)
    flux = ALPHA2PI / (yp * Q2p) * ((1 - yp) * (1 - qmin2p / Q2p) * FE + 0.5 * yp**2 * FM)
    return flux * Q2p

def flux_el_yy_atW(W, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam
    ye_min = W**2.0 / s_cms
    ye_max = 1.0

    def integrand(ye):
        qmin2e = qmin2_electron(emass, ye)
        lnQ2e_min, lnQ2e_max = math.log(qmin2e), math.log(q2emax)

        def lnQ2e_integrand(lnQ2e):
            Q2e = np.exp(lnQ2e)

            # Integration over MN
            MN_min = pmass + pi0mass
            MN_max = 10.0  # Given as a fixed upper bound for MN

            def integrand_MN(MN):
                jacobian = compute_jacobian(ye, eEbeam, pEbeam, W)
                if jacobian == 0:
                    return 0.0
                lnQ2p_min, lnQ2p_max = math.log(qmin2_proton(MN, 0.01)), math.log(q2pmax)

                def lnQ2p_integrand(lnQ2p):
                    Q2p = np.exp(lnQ2p)
                    yp_value = compute_yp(W, Q2e, Q2p, ye, eEbeam, pEbeam, MN)
                    if yp_value <= 0 or yp_value >= 1:
                        return 0.0
                    proton_flux = flux_y_proton(yp_value, lnQ2p, MN)
                    return proton_flux / jacobian

                # Using quad_vec here for inner integration over lnQ2p
                return integrate.quad_vec(lnQ2p_integrand, lnQ2p_min, lnQ2p_max, epsrel=1e-3)[0]

            # Using quad_vec for MN integration
            return integrate.quad_vec(integrand_MN, MN_min, MN_max, epsrel=1e-3)[0] * flux_y_electron(ye, lnQ2e)

        # Using quad_vec for lnQ2e integration
        return integrate.quad_vec(lnQ2e_integrand, lnQ2e_min, lnQ2e_max, epsrel=1e-3)[0]

    return integrate.quad_vec(integrand, ye_min, ye_max, epsrel=1e-3)[0]

# Parameters
eEbeam, pEbeam = 50.0, 7000.0
W_values = np.logspace(1.0, 3.0, 101)

def wrapper_flux_el_yy_atW(W):
    return flux_el_yy_atW(W, eEbeam, pEbeam)

if __name__ == "__main__":
    with Pool() as pool:
        luminosity_values = pool.map(wrapper_flux_el_yy_atW, W_values)

    with open("Jacobian_Krzysztof_Inelastic_Updated.txt", "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values, luminosity_values):
            file.write(f"{W:.6e}    {S_yy:.6e}\n")

    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)
    plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Inelastic')
    plt.xlabel(r"$W$ [GeV]", fontsize=18)
    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    plt.title("Inelastic $S_{\gamma\gamma}$ at LHeC with Correct W", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(fontsize=14)
    plt.savefig("Jacobian_Krzysztof_Inelastic_Updated.pdf")
    plt.savefig("Jacobian_Krzysztof_Inelastic_Updated.jpg")
    plt.show()
