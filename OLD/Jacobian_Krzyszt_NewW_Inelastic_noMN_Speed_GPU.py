# Photon-Photon-Luminosity-Spectrum-Hamzeh_with_W_Parallel_Final_Expression_dlnq2_Jacobian

import cupy as cp  # Use CuPy instead of NumPy for GPU acceleration

print(cp.cuda.runtime.getDeviceCount())


import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import lru_cache

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

# Caching for frequently used functions to speed up calculations
@lru_cache(maxsize=None)
def tvalue(Q2):
    """Cached calculation of t value for structure functions."""
    return math.log((math.log((Q2 + Q2_0) / Lambda2) / math.log(Q2_0 / Lambda2)))

@lru_cache(maxsize=None)
def xP(xbj, Q2):
    """Calculate xP, with caching for performance."""
    if xbj == 0:
        return -1.0  # Avoid division by zero
    xPinv = 1.0 + Q2 / (Q2 + Mass2_P) * (1.0 / xbj - 1.0)
    return 1.0 / xPinv

@lru_cache(maxsize=None)
def xR(xbj, Q2):
    """Calculate xR, with caching for performance."""
    if xbj == 0:
        return -1.0
    xPinv = 1.0 + Q2 / (Q2 + Mass2_R) * (1.0 / xbj - 1.0)
    return 1.0 / xPinv

def type1(tval, tuple1):
    return tuple1[0] + tuple1[1] * (tval ** tuple1[2])

def type2(tval, tuple1):
    return tuple1[0] + (tuple1[0] - tuple1[1]) * (1.0 / (1.0 + tval ** tuple1[2]) - 1.0)

@lru_cache(maxsize=None)
def allm_f2(xbj, Q2):
    """Cached ALLM structure function f2."""
    tval = tvalue(Q2)
    return Q2 / (Q2 + Mass2_0) * (
        type2(tval, Ccp) * (xP(xbj, Q2) ** type2(tval, Cap)) * ((1.0 - xbj) ** type1(tval, Cbp)) +
        type1(tval, Ccr) * (xR(xbj, Q2) ** type1(tval, Car)) * ((1.0 - xbj) ** type1(tval, Cbr))
    )

def qmin2_electron(mass, y):
    if y >= 1:
        return float('inf')  # Return infinity for non-physical scenario
    return mass * mass * y * y / (1 - y)

def qmin2_proton(MN, y):
    if y >= 1:
        return float('inf')  # Return infinity for non-physical scenario
    return ((MN**2) / (1 - y) - pmass**2) * y

def compute_yp(W, Q2e, Q2p, ye, Ee, Ep):
    numerator = W**2 + Q2e + Q2p
    denominator = ye * 4 * Ee * Ep
    return numerator / denominator

def compute_jacobian(ye, Ee, Ep, W):
    return abs(2 * ye * Ee * Ep / W)

# Electron flux calculation with caching
@lru_cache(maxsize=None)
def flux_y_electron(ye, lnQ2e):
    """Cached photon flux from electron, uses lnQ2 as the integration variable."""
    Q2e = cp.exp(lnQ2e)  # Use CuPy for GPU acceleration
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2_electron(emass, ye)
    if qmin2v <= 0 or Q2e < qmin2v or Q2e > q2emax:
        return 0.0
    flux = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2)
    return flux * Q2e

# Proton flux calculation optimized for inelastic case with caching
@lru_cache(maxsize=None)
def flux_y_proton(yp, lnQ2p, MN):
    """Cached photon flux from proton for inelastic case, using lnQ2p as the integration variable."""
    Q2p = cp.exp(lnQ2p)  # Use CuPy for GPU acceleration
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
    """Photon-Photon Luminosity Spectrum Calculation with optimized nested integrations."""
    s_cms = 4.0 * eEbeam * pEbeam
    ye_min = W**2.0 / s_cms
    ye_max = 1.0

    def integrand(ye):
        qmin2e = qmin2_electron(emass, ye)
        if qmin2e <= 0:
            return 0.0
        lnQ2e_min = math.log(qmin2e)
        lnQ2e_max = math.log(q2emax)

        def lnQ2e_integrand(lnQ2e):
            Q2e = cp.exp(lnQ2e)  # Use CuPy for GPU acceleration
            MN_min = pmass + pi0mass
            MN_max = 10.0

            def integrand_MN(MN):
                jacobian = compute_jacobian(ye, eEbeam, pEbeam, W)
                if jacobian == 0:
                    return 0.0
                qmin2p = qmin2_proton(MN, 0.01)
                lnQ2p_min = math.log(qmin2p)
                lnQ2p_max = math.log(q2pmax)

                def lnQ2p_integrand(lnQ2p):
                    Q2p = cp.exp(lnQ2p)
                    yp_value = compute_yp(W, Q2e, Q2p, ye, eEbeam, pEbeam)
                    if yp_value <= 0 or yp_value >= 1:
                        return 0.0
                    proton_flux = flux_y_proton(yp_value, lnQ2p, MN)
                    return proton_flux / jacobian

                return integrate.quad_vec(lnQ2p_integrand, lnQ2p_min, lnQ2p_max, epsrel=1e-3)[0]

            return integrate.quad_vec(integrand_MN, MN_min, MN_max, epsrel=1e-3)[0] * flux_y_electron(ye, lnQ2e)

        return integrate.quad_vec(lnQ2e_integrand, lnQ2e_min, lnQ2e_max, epsrel=1e-3)[0]

    return integrate.quad_vec(integrand, ye_min, ye_max, epsrel=1e-4)[0]

# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W_values = cp.logspace(1.0, 3.0, 101)  # Range of W values from 10 GeV to 1000 GeV using CuPy

def wrapper_flux_el_yy_atW(W):
    """Wrapper function to calculate flux at given W value, for parallel processing."""
    return flux_el_yy_atW(W, eEbeam, pEbeam)

# Parallel Calculation of Photon-Photon Luminosity Spectrum
if __name__ == "__main__":
    num_cores = 10  # Adjust based on available resources
    with Pool(num_cores) as pool:
        luminosity_values = pool.map(wrapper_flux_el_yy_atW, W_values.get())  # Transfer W_values to CPU

    # Save results to a text file
    with open("Jacobian_Krzysztof_Inelastic_Updated_noMN.txt", "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values.get(), luminosity_values):  # Transfer results back to CPU
            file.write(f"{W:.6e}    {S_yy:.6e}\n")

    W_value = 10.0  # GeV
    luminosity_at_W10 = flux_el_yy_atW(W_value, eEbeam, pEbeam)
    print(f"Photon-Photon Luminosity Spectrum at W = {W_value} GeV: {luminosity_at_W10:.6e} GeV^-1")

    # Plotting (data must be transferred back to CPU)
    W_values_cpu = W_values.get()
    luminosity_values_cpu = cp.asnumpy(luminosity_values)

    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)
    plt.loglog(W_values_cpu, luminosity_values_cpu, linestyle='solid', linewidth=2, label='Inelastic')

    plt.text(15, 5.e-6, f'q2emax = {q2emax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 2.e-6, f'q2pmax = {q2pmax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 1.e-6, f'Luminosity at W={W_value} GeV = {luminosity_at_W10:.2e} GeV^-1', fontsize=14, color='blue')

    plt.xlabel(r"$W$ [GeV]", fontsize=18)
    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    plt.title("Inelastic $S_{\gamma\gamma}$ at LHeC with Correct W", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)

    plt.savefig("Jacobian_Krzysztof_Inelastic_Updated_noMN.pdf")
    plt.savefig("Jacobian_Krzysztof_Inelastic_Updated_noMN.jpg")
    plt.show()
