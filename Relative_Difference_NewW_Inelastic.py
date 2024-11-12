
# Comparison of Photon-Photon Luminosity Spectrum -- Inelastic

import numpy as np
import matplotlib.pyplot as plt


# Load data from both files
W_simple, Syy_simple = np.loadtxt("Jacobian_Krzysztof_Inelastic_Updated_Simple.txt", skiprows=1, unpack=True)
W_krzysztof, Syy_krzysztof = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_10_q2emax_100000_q2pmax_100000_using_vegas.txt", skiprows=1, unpack=True)

# Plot the comparison of S_yy as a function of W
plt.figure(figsize=(10, 8))


# Set plotting range
plt.xlim(10.0, 1000.0)
plt.ylim(1.e-7, 1.e-1)

plt.loglog(W_simple, Syy_simple, 'r-', label="W^2 = ye yp s", linewidth=2)
plt.loglog(W_krzysztof, Syy_krzysztof, 'b--', label="W^2 = full inelastic case", linewidth=2)



# Find the index corresponding to W = 10 GeV
W_value = 10.0
idx_simple = (np.abs(W_simple - W_value)).argmin()
idx_krzysztof = (np.abs(W_krzysztof - W_value)).argmin()

# Get the corresponding S_yy values
Syy_simple_value = Syy_simple[idx_simple]
Syy_krzysztof_value = Syy_krzysztof[idx_krzysztof]


# Add text annotations for S_yy values at W = 10 GeV
plt.text(W_value, Syy_simple_value, f"Syy with (ye yp s) at 10 GeV =  {Syy_simple_value:.2e}", color='red', fontsize=12, ha='left', va='bottom')
plt.text(W_value, Syy_krzysztof_value, f"Syy with full inelastic case at 10 GeV =  {Syy_krzysztof_value:.2e}", color='blue', fontsize=12, ha='left', va='top')


# Add additional information to the plot
plt.xlabel(r"$W$ [GeV]", fontsize=18)
plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
plt.title("Comparison of (ye yp s) and full inelastic case", fontsize=20)
plt.grid(True, which="both", linestyle="--")
plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)

# Save the plot as a PDF and JPG
plt.savefig("Comparison_Luminosity_Spectrum_NewW_Inelastic.pdf")
plt.savefig("Comparison_Luminosity_Spectrum_NewW_Inelastic.jpg")





# Plot the relative difference between the two results
relative_difference =  np.abs((Syy_simple - Syy_krzysztof) / Syy_krzysztof)  # 1 - Syy_simple / Syy_krzysztof   # np.abs((Syy_simple - Syy_krzysztof) / Syy_krzysztof) * 100


plt.figure(figsize=(10, 8))


# Set plotting range
plt.xlim(10.0, 1000.0)
plt.ylim(-1.e0, 1.e0)

plt.semilogx(W_simple, relative_difference, 'g-', linewidth=2, label="Relative Difference")
plt.xlabel(r"$W$ [GeV]", fontsize=18)
plt.ylabel("Relative Difference |(Old-Corrected)/Corrected|", fontsize=18)
plt.title("Relative Difference between (ye yp s) and full inelastic case", fontsize=20)
plt.grid(True, which="both", linestyle="--")
plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)


# Save the plot as a PDF and JPG
plt.savefig("Relative_Difference_NewW_Inelastic.pdf")
plt.savefig("Relative_Difference_NewW_Inelastic.jpg")

# Show plots
plt.show()


