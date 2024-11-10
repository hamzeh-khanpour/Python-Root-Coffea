# Photon-Photon-Luminosity-Spectrum-Hamzeh_with_W_Parallel_MonteCarlo

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / np.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4  # Electron mass in GeV
pmass = 0.938272081  # Proton mass in GeV
pi0mass = 0.1349768  # Pion mass in GeV

q2emax = 100000.0  # Maximum photon virtuality for electron in GeV^2
q2pmax = 10.0      # Maximum photon virtuality for proton in GeV^2
MN_max = 10.0      # Maximum MN in GeV


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
    return np.log((np.log((Q2 + Q2_0) / Lambda2) / np.log(Q2_0 / Lambda2)))


def xP(xbj, Q2):
    xPinv = 1.0 + Q2 / (Q2 + Mass2_P) * (1.0 / xbj - 1.0)
    return 1.0 / xPinv


def xR(xbj, Q2):
    xPinv = 1.0 + Q2 / (Q2 + Mass2_R) * (1.0 / xbj - 1.0)
    return 1.0 / xPinv


def type1(tval, tuple1):
    return tuple1[0] + tuple1[1] * (tval ** tuple1[2])


def type2(tval, tuple1):
    return tuple1[0] + (tuple1[0] - tuple1[1]) * (1.0 / (1.0 + tval ** tuple1[2]) - 1.0)


def allm_f2(xbj, Q2):
    tval = tvalue(Q2)
    return Q2 / (Q2 + Mass2_0) * (
        type2(tval, Ccp) * (xP(xbj, Q2) ** type2(tval, Cap)) * ((1.0 - xbj) ** type1(tval, Cbp)) +
        type1(tval, Ccr) * (xR(xbj, Q2) ** type1(tval, Car)) * ((1.0 - xbj) ** type1(tval, Cbr))
    )


def qmin2_electron(mass, y):
    return mass**2 * y**2 / (1 - y)


def qmin2_proton(MN, y):
    return ((MN**2) / (1 - y) - pmass**2) * y


def compute_yp(W, Q2e, Q2p, ye, Ee, Ep):
    return (W**2 + Q2e + Q2p) / (ye * 4 * Ee * Ep)


def compute_jacobian(ye, Ee, Ep, W):
    return abs(2 * ye * Ee * Ep / W)



def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    qmin2v = qmin2_electron(emass, ye)
    if qmin2v <= 0 or Q2e < qmin2v or Q2e > q2emax:
        return 0.0
    return ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2) * Q2e



def flux_y_proton(yp, lnQ2p, MN):
    Q2p = np.exp(lnQ2p)
    xbj = Q2p / (MN**2 - pmass**2 + Q2p)
    qmin2p = qmin2_proton(MN, yp)
    if qmin2p <= 0 or Q2p < qmin2p or Q2p > q2pmax:
        return 0.0
    FE = allm_f2(xbj, Q2p) * (2 * MN / (MN**2 - pmass**2 + Q2p))
    FM = allm_f2(xbj, Q2p) * (2 * MN * (MN**2 - pmass**2 + Q2p)) / (Q2p * Q2p)
    return ALPHA2PI / (yp * Q2p) * ((1 - yp) * (1 - qmin2p / Q2p) * FE + 0.5 * yp**2 * FM) * Q2p



# Monte Carlo integration to calculate luminosity
def monte_carlo_flux_el_yy_atW(W, eEbeam, pEbeam, num_samples=10000):
    ye_min = W**2 / (4 * eEbeam * pEbeam)
    ye_values = np.random.uniform(ye_min, 1.0, num_samples)

    lnQ2e_values = np.random.uniform(np.log(qmin2_electron(emass, ye_values)), np.log(q2emax), num_samples)

    MN_values = np.random.uniform(pmass + pi0mass, MN_max, num_samples)

    lnQ2p_values = np.random.uniform(np.log(qmin2_proton(MN_values, 0.01)), np.log(q2pmax), num_samples)
    
    flux_sum = 0.0
    for ye, lnQ2e, MN, lnQ2p in zip(ye_values, lnQ2e_values, MN_values, lnQ2p_values):
        Q2e = np.exp(lnQ2e)
        Q2p = np.exp(lnQ2p)

        jacobian = compute_jacobian(ye, eEbeam, pEbeam, W)
        yp_value = compute_yp(W, Q2e, Q2p, ye, eEbeam, pEbeam)
        
        if yp_value > 0 and yp_value < 1:
            flux = flux_y_proton(yp_value, lnQ2p, MN) / jacobian * flux_y_electron(ye, lnQ2e)
            flux_sum += flux
    
    volume = (1.0 - ye_min) * (np.log(q2emax) - np.log(qmin2_electron(emass, ye_min))) * (MN_max - (pmass + pi0mass)) * (np.log(q2pmax) - np.log(qmin2_proton(MN_max, 0.01)))
    return flux_sum / num_samples * volume



# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W_values = np.logspace(1.0, 3.0, 101)


# Wrapper function for parallel processing
def wrapper_monte_carlo_flux(W):
    return monte_carlo_flux_el_yy_atW(W, eEbeam, pEbeam)


# Parallel Monte Carlo Calculation of the Photon-Photon Luminosity Spectrum
if __name__ == "__main__":
    num_cores = 10  # Set this based on available resources

    with Pool(num_cores) as pool:
        luminosity_values = pool.map(wrapper_monte_carlo_flux, W_values)

    with open("Jacobian_Krzysztof_Inelastic_MonteCarlo.txt", "w") as file:
        file.write("# W [GeV]    S_yy [GeV^-1]\n")
        for W, S_yy in zip(W_values, luminosity_values):
            file.write(f"{W:.6e}    {S_yy:.6e}\n")


    # Plot the Results
    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)
    plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Inelastic')


    plt.xlabel(r"$W$ [GeV]", fontsize=18)
    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    plt.title("Inelastic $S_{\gamma\gamma}$ at LHeC with Monte Carlo Integration", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10 \, \mathrm{GeV}^2$', fontsize=14)


    plt.savefig("Jacobian_Krzysztof_Inelastic_MonteCarlo.pdf")
    plt.savefig("Jacobian_Krzysztof_Inelastic_MonteCarlo.jpg")
    plt.show()

