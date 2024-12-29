
# Description: Plotting Photon-Photon Luminosity Spectrum
# This Python script plots the photon-photon luminosity spectrum 𝑆𝛾𝛾 as a function of the invariant mass 
# 𝑊 based on pre-calculated data stored in text files. The text files are generated from analysis codes that compute the 
# 𝑆𝛾𝛾 values for different physical scenarios (elastic and inelastic cases).
# Author: Hamzeh Khanpour, Laurent Forthomme, and Krzysztof Piotrzkowski -- January 2025


# ================================================================================

import mplhep as hep
import numpy as np
import matplotlib.pyplot as plt
import sys

hep.style.use("CMS")
#plt.style.use(hep.style.ROOT)

'''plt.rcParams["axes.linewidth"] = 1.8
plt.rcParams["xtick.major.width"] = 1.8
plt.rcParams["xtick.minor.width"] = 1.8
plt.rcParams["ytick.major.width"] = 1.8
plt.rcParams["ytick.minor.width"] = 1.8

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

plt.rcParams["legend.fontsize"] = 15

plt.rcParams['legend.title_fontsize'] = 'x-large' '''


# ================================================================================


# Load data from input files
inelastic_data_I = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_10_q2emax_100000_q2pmax_10_using_vegas_LHeC750GeV.txt", skiprows=1)
inelastic_data_II = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_50_q2emax_100000_q2pmax_1000_using_vegas_LHeC750GeV.txt", skiprows=1)
inelastic_data_III = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_300_q2emax_100000_q2pmax_100000_using_vegas_LHeC750GeV.txt", skiprows=1)

elastic_data = np.loadtxt("Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_10_using_vegas_LHeC750GeV.txt", skiprows=1)


# Extract W values and luminosity spectra
wv_inelastic_I = inelastic_data_I[:, 0]
s_yy_inelastic_I = inelastic_data_I[:, 1]

wv_inelastic_II = inelastic_data_II[:, 0]
s_yy_inelastic_II = inelastic_data_II[:, 1]

wv_inelastic_III = inelastic_data_III[:, 0]
s_yy_inelastic_III = inelastic_data_III[:, 1]

wv_elastic = elastic_data[:, 0]
s_yy_elastic = elastic_data[:, 1]

# Debugging input data
print("Inelastic W values (first 10):", wv_inelastic_I[:10])
print("Inelastic S_yy values (first 10):", s_yy_inelastic_I[:10])

print("Inelastic W values (first 10):", wv_inelastic_II[:10])
print("Inelastic S_yy values (first 10):", s_yy_inelastic_II[:10])

print("Inelastic W values (first 10):", wv_inelastic_III[:10])
print("Inelastic S_yy values (first 10):", s_yy_inelastic_III[:10])

print("Elastic W values (first 10):", wv_elastic[:10])
print("Elastic S_yy values (first 10):", s_yy_elastic[:10])



# Plotting
fig, ax = plt.subplots(figsize=(8.0, 9.0))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

ax.set_xlim(10.0, 1000.0)
ax.set_ylim(1.e-7, 1.e-1)


# Plot the luminosity spectra
ax.loglog(wv_elastic, s_yy_elastic, label="Elastic", linestyle="solid", linewidth=3)
ax.loglog(wv_inelastic_I, s_yy_inelastic_I, label=r"$M_N < 10$ GeV ($Q^2_p < 10$ GeV$^2$)", linestyle="dotted", linewidth=3)
ax.loglog(wv_inelastic_II, s_yy_inelastic_II, label=r"$M_N < 50$ GeV ($Q^2_p < 10^3$ GeV$^2$)", linestyle="dashed", linewidth=3)
ax.loglog(wv_inelastic_III, s_yy_inelastic_III, label=r"$M_N < 300$ GeV ($Q^2_p < 10^5$ GeV$^2$)", linestyle="dashdot", linewidth=3)

# Add labels and legend
ax.set_xlabel(r"$W$ [GeV]")
ax.set_ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]")
ax.legend(title=r"$Q^2_e < 10^5$ GeV$^2$", loc="upper right")



# Add additional information
info_text = "LHeC@750 GeV"
plt.text(0.35, 0.11, info_text, transform=ax.transAxes, ha='center', va='center', fontsize=25, color='blue', fontweight='bold')

info_text_2 = r"$E_e$=20 GeV; $E_p$=7000 GeV"
plt.text(0.35, 0.05, info_text_2, transform=ax.transAxes, ha='center', va='center', fontsize=25, color='blue', fontweight='bold')


# Save the plot
plt.savefig("Exact_Photon_Luminosity_Spectrum_LHeC750GeV_JHEP.pdf")
plt.savefig("Exact_Photon_Luminosity_Spectrum_LHeC750GeV_JHEP.jpg")

# Show the plot
plt.show()


##################################################################

