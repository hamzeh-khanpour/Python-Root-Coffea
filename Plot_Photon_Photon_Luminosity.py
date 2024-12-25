
# Description: Plotting Photon-Photon Luminosity Spectrum
# This Python script plots the photon-photon luminosity spectrum 𝑆𝛾𝛾 as a function of the invariant mass 
# 𝑊 based on pre-calculated data stored in text files. The text files are generated from analysis codes that compute the 
# 𝑆𝛾𝛾 values for different physical scenarios (elastic and inelastic cases).
# Author: Hamzeh Khanpour, Laurent Forthomme, and Krzysztof Piotrzkowski -- November 2024


import numpy as np
import matplotlib.pyplot as plt

# List of input files
files = [
    "Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_10_using_vegas.txt",
    "Inelastic_Photon_Luminosity_Spectrum_MNmax_10_q2emax_100000_q2pmax_10_using_vegas.txt",
    "Inelastic_Photon_Luminosity_Spectrum_MNmax_50_q2emax_100000_q2pmax_1000_using_vegas.txt",
    "Inelastic_Photon_Luminosity_Spectrum_MNmax_300_q2emax_100000_q2pmax_100000_using_vegas.txt",
]

# Corresponding labels for the plot
labels = [
    r"Elastic",
    r"$M_N < 10 \, \mathrm{GeV}$ ($Q^2_p < 10 \, \mathrm{GeV}^2$)",
    r"$M_N < 50 \, \mathrm{GeV}$ ($Q^2_p < 10^3 \, \mathrm{GeV}^2$)",
    r"$M_N < 300 \, \mathrm{GeV}$ ($Q^2_p < 10^5 \, \mathrm{GeV}^2$)",
]

# Line styles for each dataset
line_styles = ['-', ':', '-.', '--']

# Plot settings
plt.figure(figsize=(10, 8))

# Loop over files to read and plot data
for file, label, style in zip(files, labels, line_styles):
    # Load data from the file (skip comments)
    data = np.loadtxt(file, comments="#")
    W = data[:, 0]  # W [GeV]
    S_yy = data[:, 1]  # S_yy [GeV^-1]
    
    # Plot the data
    plt.loglog(W, S_yy, style, label=label, linewidth=2)

# Set plot range
plt.xlim(10.0, 1000.0)
plt.ylim(1.e-7, 1.e-1)

# Plot labels, title, and legend
plt.xlabel(r"$W$ [GeV]", fontsize=18)
plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
plt.title(r"Corrected Photon-Photon Luminosity Spectrum $S_{\gamma\gamma}$", fontsize=20)
plt.grid(True, which="both", linestyle="--")
plt.legend(title=r"$Q^2_e < 10^5 \, \mathrm{GeV}^2$", fontsize=14)

# Save the plot as PDF and JPG
plt.savefig("Photon_Photon_Luminosity_Spectrum_Comparison.pdf")
plt.savefig("Photon_Photon_Luminosity_Spectrum_Comparison.jpg")

# Show the plot
plt.show()
