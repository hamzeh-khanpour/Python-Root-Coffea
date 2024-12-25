# The `integrated_tau_tau_cross_section` function computes the integrated tau-tau production cross-section 
# for a given threshold energy `W0` and electron/proton beam energies `eEbeam` and `pEbeam`.
#   - It defines an integrand combining tau-tau cross-section and the interpolated S_yy.
#   - If S_yy is zero at a given W, it skips that W value to avoid unnecessary computations.
#   - The result is integrated over W from W0 up to the maximum value set by `sqrt(s_cms)`.
#   - Integration warnings are caught, ensuring stable results.
#   - This function returns the integrated cross-section result in picobarns (pb).
# Hamzeh Khanpour, Laurent Forthomme, and Krzysztof Piotrzkowski -- November 2024


import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import mplhep as hep

# Use CMS style for plots
hep.style.use("CMS")

# Load photon luminosity data from the text file
data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_3_q2emax_10_q2pmax_10_using_vegas.txt', comments='#')
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

    def integrand(W):
        tau_tau_cross_section = cs_tautau_w_condition_Hamzeh(W)
        S_yy_value = S_yy_interp(W)

        if S_yy_value == 0.0:
            return 0.0
        return tau_tau_cross_section * S_yy_value

    try:
        result, _ = integrate.quad(integrand, W0, np.sqrt(s_cms), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration did not converge for W_0={W0}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration: {e}")
        result = 0.0
    return result

################################################################################

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

# Set up the figure with CMS style
fig, ax = plt.subplots(figsize=(8.0, 8.0))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

# Set plot limits
ax.set_xlim(10.0, 1000.0)
ax.set_ylim(1e-3, 1e2)

# Plot the cross-section values
plt.loglog(W0_range, cross_section_values, linestyle='solid', linewidth=2, label='Inelastic')

# Annotate the integrated cross-section value
plt.text(15, 5.e-3, f'Integrated Cross-Section\nW_0={W0_value} GeV\n{integrated_cross_section_value:.2e} pb',
         fontsize=14, color='blue')

# Add labels, title, and legend
plt.xlabel(r"$W_0$ [GeV]", fontsize=18)
plt.ylabel(r"$\sigma_{{\rm ep}\to {\rm e}(\gamma\gamma\to\tau^+\tau^-){\rm p}^{(\ast)}}$ (W > W$_0$) [pb]", fontsize=18)
plt.title("Inelastic Tau-Tau Production Cross-Section", fontsize=20)
plt.legend(title=r"$eE_{beam}$=50 GeV, $pE_{beam}$=7000 GeV", fontsize=14)

# Add gridlines
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Save the plot
plt.savefig("tau-tau_cross_section.pdf")
plt.savefig("tau-tau_cross_section.jpg")

# Display the plot
plt.show()


    
################################################################################


