# Photon-Photon-Luminosity-Spectrum-Hamzeh_with_W_Parallel_Final_Expression_dlnq2_Jacobian_Inelastic_Case
# Hamzeh, Laurent and Krzysztof ---- 4 November 2024

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import jit

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi
emass = 5.1099895e-4   # Electron mass in GeV
pmass = 0.938272081    # Proton mass in GeV
pi0mass = 0.1349768    # Pion mass in GeV
q2emax = 100000.0      # Maximum photon virtuality for electron in GeV^2
q2pmax = 10.0          # Maximum photon virtuality for proton in GeV^2
MN_max = 10.0          # Maximum MN in GeV

# Parameters for structure functions
Mass2_0 = 0.31985
Mass2_P = 49.457
Mass2_R = 0.15052
Q2_0 = 0.52544
Lambda2 = 0.06527

# Coefficients for ALLM97
Ccp = (0.28067, 0.22291, 2.1979)
Cap = (-0.0808, -0.44812, 1.1709)
Cbp = (0.36292, 1.8917, 1.8439)
Ccr = (0.80107, 0.97307, 3.4942)
Car = (0.58400, 0.37888, 2.6063)
Cbr = (0.01147, 3.7582, 0.49338)

@jit(nopython=True)
def tvalue(Q2):
    return math.log((math.log((Q2 + Q2_0) / Lambda2) / math.log(Q2_0 / Lambda2)))

@jit(nopython=True)
def type1(tval, tuple1):
    return tuple1[0] + tuple1[1] * (tval ** tuple1[2])

@jit(nopython=True)
def type2(tval, tuple1):
    return tuple1[0] + (tuple1[0] - tuple1[1]) * (1. / (1. + tval ** tuple1[2]) - 1.)

@jit(nopython=True)
def aP(tval):
    return type2(tval, Cap)

@jit(nopython=True)
def bP(tval):
    return type1(tval, Cbp)

@jit(nopython=True)
def cP(tval):
    return type2(tval, Ccp)

@jit(nopython=True)
def aR(tval):
    return type1(tval, Car)

@jit(nopython=True)
def bR(tval):
    return type1(tval, Cbr)

@jit(nopython=True)
def cR(tval):
    return type1(tval, Ccr)

@jit(nopython=True)
def allm_f2P(xbj, Q2):
    tval = tvalue(Q2)
    return cP(tval) * (xP(xbj, Q2) ** aP(tval)) * ((1. - xbj) ** bP(tval))

@jit(nopython=True)
def allm_f2R(xbj, Q2):
    tval = tvalue(Q2)
    return cR(tval) * (xR(xbj, Q2) ** aR(tval)) * ((1. - xbj) ** bR(tval))

@jit(nopython=True)
def allm_f2(xbj, Q2):
    return Q2 / (Q2 + Mass2_0) * (allm_f2P(xbj, Q2) + allm_f2R(xbj, Q2))

@jit(nopython=True)
def qmin2_electron(mass, y):
    if y >= 1:
        return float('inf')
    return mass * mass * y * y / (1 - y)

@jit(nopython=True)
def qmin2_proton(MN, y):
    if y >= 1:
        return float('inf')
    return ((MN**2) / (1 - y) - pmass**2) * y

@jit(nopython=True)
def compute_yp(W, Q2e, Q2p, ye, Ee, Ep, MN):
    numerator = W**2 + Q2e + Q2p - (Q2e * (Q2p + MN**2 - pmass**2)) / (4 * Ee * Ep)
    denominator = ye * 4 * Ee * Ep
    return numerator / denominator

@jit(nopython=True)
def compute_jacobian(ye, Ee, Ep, W):
    return abs(2 * ye * Ee * Ep / W)

@jit(nopython=True)
def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2_electron(emass, ye)
    if qmin2v <= 0 or Q2e < qmin2v or Q2e > q2emax:
        return 0.0
    flux = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2)
    return flux * Q2e

@jit(nopython=True)
def flux_y_proton(yp, lnQ2p, MN):
    Q2p = np.exp(lnQ2p)
    xbj = Q2p / (MN**2 - pmass**2 + Q2p)
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
        if qmin2e <= 0:
            return 0.0
        lnQ2e_min, lnQ2e_max = math.log(qmin2e), math.log(q2emax)

        def lnQ2e_integrand(lnQ2e):
            Q2e = np.exp(lnQ2e)
            MN_max = 10.0          # Maximum MN in GeV
            MN_min, MN_max = pmass + pi0mass, MN_max

            def integrand_MN(MN):
                jacobian = compute_jacobian(ye, eEbeam, pEbeam, W)
                if jacobian == 0:
                    return 0.0
                qmin2p = qmin2_proton(MN, 0.01)
                lnQ2p_min, lnQ2p_max = np.log(qmin2p), np.log(q2pmax)

                def lnQ2p_integrand(lnQ2p):
                    Q2p = np.exp(lnQ2p)
                    yp_value = compute_yp(W, Q2e, Q2p, ye, eEbeam, pEbeam, MN)
                    if yp_value <= 0 or yp_value >= 1:
                        return 0.0
                    proton_flux = flux_y_proton(yp_value, lnQ2p, MN)
                    return proton_flux / jacobian

                return integrate.quad(lnQ2p_integrand, lnQ2p_min, lnQ2p_max, epsrel=1e-3)[0]

            return integrate.quad(integrand_MN, MN_min, MN_max, epsrel=1e-3)[0] * flux_y_electron(ye, lnQ2e)

        return integrate.quad(lnQ2e_integrand, lnQ2e_min, lnQ2e_max, epsrel=1e-3)[0]

    return integrate.quad(integrand, ye_min, ye_max, epsrel=1e-3)[0]

# Parameters
eEbeam, pEbeam = 50.0, 7000.0
W_values = np.logspace(1.0, 3.0, 101)

def wrapper_flux_el_yy_atW(W):
    return flux_el_yy_atW(W, eEbeam, pEbeam)

if __name__ == "__main__":
    with Pool() as pool:
        luminosity_values = pool.map(wrapper_flux_el_yy_atW, W_values)
    np.savetxt("Jacobian_Krzysztof_Inelastic_Updated.txt", np.column_stack((W_values, luminosity_values)), header="W [GeV]    S_yy [GeV^-1]")

    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)
    plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Inelastic')
    plt.xlabel(r"$W$ [GeV]", fontsize=18)
    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    plt.title("Inelastic $S_{\gamma\gamma}$ at LHeC with Correct W", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)
    plt.savefig("Jacobian_Krzysztof_Inelastic_Updated.pdf")
    plt.savefig("Jacobian_Krzysztof_Inelastic_Updated.jpg")
    plt.show()
