
# Description: Plotting Integrated Photon-Photon Luminosity Spectrum
# This Python script plots the Integrated photon-photon luminosity spectrum as a function of the minimal invariant mass 
# 𝑊_0 based on pre-calculated data stored in text files. The text files are generated from analysis codes that compute the 
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
inelastic_data_I = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_10_q2emax_100000_q2pmax_10_using_vegas.txt", skiprows=1)
inelastic_data_II = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_100_q2emax_100000_q2pmax_100000_using_vegas_LHeC750GeV_303.txt", skiprows=1)
inelastic_data_III = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_300_q2emax_100000_q2pmax_100000_using_vegas.txt", skiprows=1)


elastic_data_I = np.loadtxt("Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_10_using_vegas.txt", skiprows=1)
elastic_data_II = np.loadtxt("Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_10_using_vegas_tagged_elastic_Newyp.txt", skiprows=1)
elastic_data_III = np.loadtxt("Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_10_using_vegas_tagged_elastic_LHeC750GeV_Newyp.txt", skiprows=1)


# Extract W values and luminosity spectra
wv_inelastic_I = inelastic_data_I[:, 0]
s_yy_inelastic_I = inelastic_data_I[:, 1]

wv_inelastic_II = inelastic_data_II[:, 0]
s_yy_inelastic_II = inelastic_data_II[:, 1]

wv_inelastic_III = inelastic_data_III[:, 0]
s_yy_inelastic_III = inelastic_data_III[:, 1]

wv_elastic_I = elastic_data_I[:, 0]
s_yy_elastic_I = elastic_data_I[:, 1]


wv_elastic_II = elastic_data_II[:, 0]
s_yy_elastic_II = elastic_data_II[:, 1]


wv_elastic_III = elastic_data_III[:, 0]
s_yy_elastic_III = elastic_data_III[:, 1]


# Debugging input data
print("Inelastic W values (first 10):", wv_inelastic_I[:10])
print("Inelastic S_yy values (first 10):", s_yy_inelastic_I[:10])

print("Inelastic W values (first 10):", wv_inelastic_II[:10])
print("Inelastic S_yy values (first 10):", s_yy_inelastic_II[:10])

print("Inelastic W values (first 10):", wv_inelastic_III[:10])
print("Inelastic S_yy values (first 10):", s_yy_inelastic_III[:10])

print("Elastic W values (first 10):", wv_elastic_I[:10])
print("Elastic S_yy values (first 10):", s_yy_elastic_I[:10])

print("Elastic W values (first 10):", wv_elastic_III[:10])
print("Elastic S_yy values (first 10):", s_yy_elastic_III[:10])

print("Elastic W values (first 10):", wv_elastic_III[:10])
print("Elastic S_yy values (first 10):", s_yy_elastic_III[:10])



# ================================================================================

# Integration using trapezoidal rule
def trap_integ(wv, fluxv):
    wmin = np.zeros(len(wv) - 1)
    integ = np.zeros(len(wv) - 1)
    for i in range(len(wv) - 2, -1, -1):
        wvwid = wv[i + 1] - wv[i]
        traparea = wvwid * 0.5 * (fluxv[i] + fluxv[i + 1])
        wmin[i] = wv[i]
        if i == len(wv) - 2:
            integ[i] = traparea
        else:
            integ[i] = integ[i + 1] + traparea

    return wmin, integ



##################################################################


# Perform integration for both grids
wv_inel_trap_I, int_inel_I = trap_integ(wv_inelastic_I, s_yy_inelastic_I)
wv_inel_trap_II, int_inel_II = trap_integ(wv_inelastic_II, s_yy_inelastic_II)
wv_inel_trap_III, int_inel_III = trap_integ(wv_inelastic_III, s_yy_inelastic_III)

wv_el_trap_I, int_el_I = trap_integ(wv_elastic_I, s_yy_elastic_I)
wv_el_trap_II, int_el_II = trap_integ(wv_elastic_II, s_yy_elastic_II)
wv_el_trap_III, int_el_III = trap_integ(wv_elastic_III, s_yy_elastic_III)


# Debugging integration results
print("Integrated inelastic cross-section (partial):", int_inel_I[:200])
print("Integrated elastic cross-section (partial):", int_el_I[:200])


# Ensure all arrays are of the same length
min_length_I = min(len(wv_el_trap_I), len(int_el_I), len(wv_inel_trap_I), len(int_inel_I))
min_length_II = min(len(wv_el_trap_I), len(int_el_I), len(wv_inel_trap_II), len(int_inel_II))
min_length_III = min(len(wv_el_trap_I), len(int_el_I), len(wv_inel_trap_III), len(int_inel_III))

wv_el_trap_I = wv_el_trap_I[:min_length_I]
int_el_I = int_el_I[:min_length_I]

wv_el_trap_II = wv_el_trap_II[:min_length_I]
int_el_II = int_el_II[:min_length_I]

wv_el_trap_III = wv_el_trap_III[:min_length_I]
int_el_III = int_el_III[:min_length_I]


wv_inel_trap_I = wv_inel_trap_I[:min_length_I]
int_inel_I = int_inel_I[:min_length_I]

wv_inel_trap_II = wv_inel_trap_II[:min_length_II]
int_inel_II = int_inel_II[:min_length_II]

wv_inel_trap_III = wv_inel_trap_III[:min_length_III]
int_inel_III = int_inel_III[:min_length_III]




# Plotting
fig, ax = plt.subplots(figsize=(10.0, 11.0))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

ax.set_xlim(10.0, 1000.0)
ax.set_ylim(1.e-5, 2.0)


# Plot elastic and inelastic cross-sections
ax.loglog(wv_el_trap_I, int_el_I, label="elastic", linestyle="solid", linewidth=3, color="blue")
ax.loglog(wv_el_trap_II, int_el_II, label="elastic - p detected", linestyle="dashdot", linewidth=3, color="orange")
#ax.loglog(wv_el_trap_III, int_el_III, label="elastic - p detected (LHeC@0.75TeV)", linestyle="dotted", linewidth=3, color="black")

ax.loglog(wv_inel_trap_I, int_inel_I, label=r"$M_N < 10$ GeV ($Q^2_p < 10$ GeV$^2$)", linestyle=(0, (5, 2, 1, 2, 1, 2)), linewidth=3, color="red")
ax.loglog(wv_inel_trap_II, int_inel_II, label=r"$M_N < 100$ GeV ($Q^2_p < 10^5$ GeV$^2$) ($\sqrt{s}=0.75$ TeV)", linestyle="dotted", linewidth=3, color="magenta")
ax.loglog(wv_inel_trap_III, int_inel_III, label=r"$M_N < 100$ GeV ($Q^2_p < 10^5$ GeV$^2$)", linestyle="dashed", linewidth=3, color="green")



# Add labels and legend
ax.set_xlabel(r"$W_0$ [GeV]")
ax.set_ylabel(r"Integrated S$_{\gamma \gamma}$ (W > W$_0$)")
ax.legend(
    title=r"$Q^2_e < 10^5$ GeV$^2$", 
    loc="upper right"#, 
#    fontsize="small",  # Adjusts the font size of the legend labels
#    title_fontsize="small"  # Adjusts the font size of the legend title
)



# Save and show the plot
plt.savefig("Exact_Integrated_Photon_Luminosity_Spectrum_JHEP_Krzysztof.pdf")
#plt.savefig("Exact_Integrated_Photon_Luminosity_Spectrum_JHEP_Krzysztof.jpg")

plt.show()



##################################################################

