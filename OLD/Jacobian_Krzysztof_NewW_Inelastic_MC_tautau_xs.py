
# Photon-Photon-Luminosity-Spectrum-Hamzeh_with_W_Parallel_MonteCarlo

#===========================================================================

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
from multiprocessing import Pool

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass in GeV
pmass = 0.938272081    # Proton mass in GeV
pi0mass = 0.1349768    # Pion mass in GeV

mtau = 1.77686         # Tau mass in GeV

q2emax = 10.0      # Maximum photon virtuality for electron in GeV^2
q2pmax = 10.0          # Maximum photon virtuality for proton in GeV^2
MN_max = 10.0          # Maximum MN in GeV

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
    return math.log((math.log((Q2 + Q2_0) / Lambda2) / math.log(Q2_0 / Lambda2)))


def xP(xbj, Q2):
    if xbj == 0:
        return -1.
    xPinv = 1. + Q2 / (Q2 + Mass2_P) * (1. / xbj - 1.)
    return 1. / xPinv


def xR(xbj, Q2):
    if xbj == 0:
        return -1.
    xPinv = 1. + Q2 / (Q2 + Mass2_R) * (1. / xbj - 1.)
    return 1. / xPinv


def type1(tval, tuple1):
    return tuple1[0] + tuple1[1] * (tval ** tuple1[2])


def type2(tval, tuple1):
    return tuple1[0] + (tuple1[0] - tuple1[1]) * (1. / (1. + tval ** tuple1[2]) - 1.)


def allm_f2(xbj, Q2):
    tval = tvalue(Q2)
    allm_p = type2(tval, Ccp) * (xP(xbj, Q2) ** type2(tval, Cap)) * ((1. - xbj) ** type1(tval, Cbp))
    allm_r = type1(tval, Ccr) * (xR(xbj, Q2) ** type1(tval, Car)) * ((1. - xbj) ** type1(tval, Cbr))
    return Q2 / (Q2 + Mass2_0) * (allm_p + allm_r)


# Minimum Photon Virtualities
def qmin2_electron(mass, y):
    return mass * mass * y * y / (1 - y) if y < 1 else float('inf')


def qmin2_proton(MN, y):
    return ((MN**2) / (1 - y) - pmass**2) * y if y < 1 else float('inf')


# Calculation of y_p and Jacobian
def compute_yp(W, Q2e, Q2p, ye, Ee, Ep, MN):
    numerator = W**2 + Q2e + Q2p - (Q2e * (Q2p + MN**2 - pmass**2)) / (4 * Ee * Ep)
    return numerator / (ye * 4 * Ee * Ep)


def compute_jacobian(ye, Ee, Ep, W):
    return abs(2 * ye * Ee * Ep / W)


# Photon Flux Functions
def flux_y_electron(ye, lnQ2e):
    Q2e = np.exp(lnQ2e)
    qmin2v = qmin2_electron(emass, ye)
    if 0 < ye < 1 and qmin2v <= Q2e <= q2emax:
        return ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2) * Q2e
    return 0.0


def flux_y_proton(yp, lnQ2p, MN):
    Q2p = np.exp(lnQ2p)
    xbj = Q2p / (MN**2 - pmass**2 + Q2p)
    qmin2p = qmin2_proton(MN, yp)
    if 0 < yp < 1 and qmin2p <= Q2p <= q2pmax:
        FE = allm_f2(xbj, Q2p) * (2 * MN / (MN**2 - pmass**2 + Q2p))
        FM = allm_f2(xbj, Q2p) * (2 * MN * (MN**2 - pmass**2 + Q2p)) / (Q2p * Q2p)
        return ALPHA2PI / (yp * Q2p) * ((1 - yp) * (1 - qmin2p / Q2p) * FE + 0.5 * yp**2 * FM) * Q2p
    return 0.0



# Monte Carlo Integration
def monte_carlo_integration(func, bounds, n_samples=100000):
    samples = [np.random.uniform(low, high, n_samples) for low, high in bounds]
    total_sum = 0
    for ye, lnQ2e, MN, lnQ2p in zip(*samples):
        total_sum += func(ye, lnQ2e, MN, lnQ2p)
    volume = np.prod([high - low for low, high in bounds])
    return (total_sum / n_samples) * volume



# Inelastic Photon-Photon Luminosity Spectrum Calculation
def flux_el_yy_atW(W, eEbeam, pEbeam, n_samples=100000):
    s_cms = 4.0 * eEbeam * pEbeam
    ye_min, ye_max = W**2 / s_cms, 1.0

    def integrand(ye, lnQ2e, MN, lnQ2p):

        Q2e, Q2p = np.exp(lnQ2e), np.exp(lnQ2p)

        jacobian = compute_jacobian(ye, eEbeam, pEbeam, W)

        yp_value = compute_yp(W, Q2e, Q2p, ye, eEbeam, pEbeam, MN)

        if 0 < yp_value < 1:

            return flux_y_electron(ye, lnQ2e) * flux_y_proton(yp_value, lnQ2p, MN) / jacobian
        
        return 0.0

    bounds = [
        (ye_min, ye_max),
        (math.log(qmin2_electron(emass, ye_min)), math.log(q2emax)),
        (pmass + pi0mass, MN_max),
        (math.log(qmin2_proton(pmass + pi0mass, ye_min)), math.log(q2pmax))
    ]

    return monte_carlo_integration(integrand, bounds, n_samples)




# Tau-Tau Production Cross-Section Calculation
def cs_tautau_w_condition(W):
    alpha = 1 / 137.0
    hbarc2 = 0.389  # Conversion factor to pb
    if W < 2 * mtau:
        return 0.0
    beta = math.sqrt(1.0 - 4.0 * mtau**2 / W**2)
    cross_section = (4 * math.pi * alpha**2 * hbarc2) / W**2 * beta * (
        (3 - beta**4) / (2 * beta) * math.log((1 + beta) / (1 - beta)) - 2 + beta**2
    ) * 1e9  # in pb
    return cross_section



# Integrated Tau-Tau Production Cross-Section from W0 to sqrt(s_cms)
def integrated_tau_tau_cross_section(W0, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared
    upper_limit = np.sqrt(s_cms)  # Maximum possible W
    result, _ = integrate.quad(
        lambda W: cs_tautau_w_condition(W) * flux_el_yy_atW(W, eEbeam, pEbeam),
        W0, upper_limit, epsrel=1e-4
    )
    return result


# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W_values = np.logspace(1.0, 3.0, 101)


#===========================================================================




# Wrapper function for parallel processing
def wrapper_flux_el_yy_atW(W):
    return flux_el_yy_atW(W, eEbeam, pEbeam)


# Parallel Calculation of the Photon-Photon Luminosity Spectrum
if __name__ == "__main__":

    num_cores = 100  # Adjust based on available resources

    with Pool(num_cores) as pool:
        luminosity_values = pool.map(wrapper_flux_el_yy_atW, W_values)


    W_value = 10.0  # GeV
    luminosity_at_W10 = flux_el_yy_atW(W_value, eEbeam, pEbeam)
    print(f"Photon-Photon Luminosity Spectrum at W = {W_value} GeV: {luminosity_at_W10:.6e} GeV^-1")

# Format the file name to include MN, q2emax, and q2pmax
file_name = f"Jacobian_Krzysztof_Inelastic_MonteCarlo_MN{MN_max}_q2emax{int(q2emax)}_q2pmax{int(q2pmax)}.txt"

with open(file_name, "w") as file:
    file.write("# W [GeV]    S_yy [GeV^-1]\n")
    for W, S_yy in zip(W_values, luminosity_values):
        file.write(f"{W:.6e}    {S_yy:.6e}\n")



#===========================================================================



    # Plot the Photon-Photon Luminosity Spectrum
    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-7, 1.e-1)

    plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Inelastic')


    # Add additional information to the plot
    plt.text(15, 5.e-6, f'q2emax = {q2emax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 2.e-6, f'q2pmax = {q2pmax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 1.e-6, f'Luminosity at W={W_value} GeV = {luminosity_at_W10:.2e} GeV^-1', fontsize=14, color='blue')


    plt.xlabel(r"$W$ [GeV]", fontsize=18)
    plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
    
    plt.title("Inelastic $S_{\gamma\gamma}$ at LHeC with Monte Carlo Integration", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(fontsize=14)

# Format the base file name to include MN_max, q2emax, and q2pmax
file_base = f"Jacobian_Krzysztof_Inelastic_MonteCarlo_MN{MN_max}_q2emax{int(q2emax)}_q2pmax{int(q2pmax)}"

# Save the plots with the formatted file name
plt.savefig(f"{file_base}.pdf")
plt.savefig(f"{file_base}.jpg")
plt.show()


#===========================================================================


# Integrated Tau-Tau Production Cross-Section at W0
W0_value = 10.0  # GeV
integrated_cross_section_value = integrated_tau_tau_cross_section(W0_value, eEbeam, pEbeam)
print(f"Integrated Tau-Tau Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")

# Tau-Tau Cross-Section as a Function of W0
W0_range = np.arange(10.0, 1001.0, 1.0)

# Wrapper function for parallel processing
def wrapper_integrated_tau_tau_cross_section(W0):
    return integrated_tau_tau_cross_section(W0, eEbeam, pEbeam)

# Parallel computation for cross-section values over W0_range
if __name__ == "__main__":

    num_cores = 100  # Adjust based on available resources
    
    with Pool(num_cores) as pool:
        cross_section_values = pool.map(wrapper_integrated_tau_tau_cross_section, W0_range)

    # Plot the Tau-Tau Production Cross-Section as a Function of W0
    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-3, 1.e2)
    plt.loglog(W0_range, cross_section_values, linestyle='solid', linewidth=2, label='Tau-Tau Production Cross-Section')

    plt.xlabel(r"$W_0$ [GeV]", fontsize=18)
    plt.ylabel(r"$\sigma_{\tau^+\tau^-}$ (W > $W_0$) [pb]", fontsize=18)
    plt.title("Integrated Tau-Tau Production Cross-Section at LHeC", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(fontsize=14)

# Format the base file name to include MN_max, q2emax, and q2pmax
file_base_cross_section = f"integrated_tau_tau_cross_section_MC_MN{MN_max}_q2emax{int(q2emax)}_q2pmax{int(q2pmax)}"

# Save the cross-section plot with the formatted file name
plt.savefig(f"{file_base_cross_section}.pdf")
plt.savefig(f"{file_base_cross_section}.jpg")

plt.show()

#===========================================================================
