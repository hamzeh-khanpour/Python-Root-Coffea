import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi

emass = 5.1099895e-4   # Electron mass in GeV
pmass = 0.938272081    # Proton mass in GeV
pi0mass = 0.1349768    # Pion mass in GeV

q2emax = 100.0      # Maximum photon virtuality for electron in GeV^2
q2pmax = 100.0          # Maximum photon virtuality for proton in GeV^2
MN_max = 10.0          # Maximum MN in GeV

# Load photon-photon luminosity data from the text file
data = np.loadtxt('Jacobian_Krzysztof_Inelastic_MonteCarlo_MN10.0_q2emax100_q2pmax100.txt', comments='#')
W_data = data[:, 0]
S_yy_data = data[:, 1]

# Create an interpolation function for S_yy
S_yy_interp = interp1d(W_data, S_yy_data, kind='linear', bounds_error=False, fill_value=0.0)

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
            lambda W: cs_tautau_w_condition_Hamzeh(W) * S_yy_interp(W),
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
    W0_value = 10.0  # GeV

    # Calculate Integrated Tau-Tau Production Cross-Section at W_0 = 10 GeV
    integrated_cross_section_value = integrated_tau_tau_cross_section(W0_value, eEbeam, pEbeam)
    print(f"Integrated inelastic Tau-Tau Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")

    # Plot the Tau-Tau Production Cross-Section as a Function of W_0
    W0_range = np.arange(10.0, 1001.0, 1.0)
    cross_section_values = [integrated_tau_tau_cross_section(W0, eEbeam, pEbeam) for W0 in W0_range]

    plt.figure(figsize=(10, 8))
    plt.xlim(10.0, 1000.0)
    plt.ylim(1.e-3, 1.e2)

    plt.loglog(W0_range, cross_section_values, linestyle='solid', linewidth=2, label='Inelastic')
    plt.text(15, 2.e-2, f'q2emax = {q2emax:.1e} GeV^2', fontsize=14, color='blue')
    plt.text(15, 1.e-2, f'q2pmax = {q2pmax:.1e} GeV^2', fontsize=14, color='blue')

    plt.text(15, 5.e-3, f'Integrated Inelastic Tau-Tau Cross-Section at W_0={W0_value} GeV = {integrated_cross_section_value:.2e} pb', fontsize=14, color='blue')

    plt.xlabel(r"$W_0$ [GeV]", fontsize=18)
    plt.ylabel(r"$\sigma_{\tau^+\tau^-}$ (W > $W_0$) [pb]", fontsize=18)

    plt.title("Integrated Inelastic Tau-Tau Production Cross-Section at LHeC (Corrected)", fontsize=20)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)

    plt.savefig("integrated_tau_tau_cross_section_Jacobian_Krzysztof_Parallel_Inelastic_Simple_Numerical.pdf")
    plt.savefig("integrated_tau_tau_cross_section_Jacobian_Krzysztof_Parallel_Inelastic_Simple_Numerical.jpg")

    plt.show()
    
################################################################################


