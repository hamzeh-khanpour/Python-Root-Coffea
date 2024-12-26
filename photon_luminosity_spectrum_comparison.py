import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Apply CMS style
hep.style.use("CMS")

# Load data from files
inelastic_data = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_3_q2emax_10_q2pmax_10_using_vegas.txt", skiprows=1)
elastic_data = np.loadtxt("Elastic_Photon_Luminosity_Spectrum_q2emax_10_q2pmax_10_using_vegas.txt", skiprows=1)

# Extract W and S_yy values
w_inelastic = inelastic_data[:, 0]
s_yy_inelastic = inelastic_data[:, 1]

w_elastic = elastic_data[:, 0]
s_yy_elastic = elastic_data[:, 1]

# Integration using trapezoidal rule
def integrate_luminosity(w, s_yy):
    integrated = []
    for i in range(len(w)):
        integrated.append(np.trapz(s_yy[i:], w[i:]))
    return np.array(integrated)

# Perform integration
sigma_inelastic = integrate_luminosity(w_inelastic, s_yy_inelastic)
sigma_elastic = integrate_luminosity(w_elastic, s_yy_elastic)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

# Set axis limits
ax.set_xlim(10, 1000)
ax.set_ylim(1e-3, 1e3)

# Plot curves
ax.loglog(w_elastic, sigma_elastic, label="Elastic", linewidth=2.5, linestyle="solid")
ax.loglog(w_inelastic, sigma_inelastic, label=r"$M_N < 3$ GeV ($Q^2_p < 10$ GeV$^2$)", linewidth=2.5, linestyle="dotted")

# Add title and legend
ax.legend(title=r"$Q^2_e < 10$ GeV$^2$", loc="upper right")

# Add labels
ax.set_xlabel(r"$W_0$ [GeV]")
ax.set_ylabel(r"$\sigma_{\mathrm{ep} \to \mathrm{e}(\gamma\gamma \to \tau^+\tau^-)\mathrm{p}^{(*)}}$ ($W > W_0$) [pb]")

# Save and show plot
plt.savefig("photon_luminosity_spectrum_comparison.pdf")
plt.savefig("photon_luminosity_spectrum_comparison.jpg")
plt.show()
