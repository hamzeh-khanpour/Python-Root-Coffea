
# The exact elastic/inelastic `integrated ttbar production cross-section` for the `ep -> e(gamma gamma -> t \bar{t})p(*)` process 
# Final Version -- January 2025 -- Hamzeh Khanpour

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
inelastic_data_II = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_50_q2emax_100000_q2pmax_1000_using_vegas.txt", skiprows=1)
inelastic_data_III = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_300_q2emax_100000_q2pmax_100000_using_vegas.txt", skiprows=1)

elastic_data = np.loadtxt("Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_10_using_vegas.txt", skiprows=1)


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



# Function to calculate the ttbar production cross-section
##################################################################

def cs_ttbar_w_condition_Hamzeh(wvalue):  # Eq.62 of Physics Reports 364 (2002) 359-450
    mtop = 172.50
    Qf = 2.0/3.0
    Nc = 3.0
    hbarc2 =  0.389
    alpha2 = (1.0/137.0)*(1.0/137.0)

    # Element-wise calculation of beta using np.where
    beta = np.sqrt(np.where(1.0 - 4.0 * mtop * mtop / wvalue**2.0 >= 0.0, 1.0 - 4.0 * mtop * mtop / wvalue**2.0, np.nan))

    # Element-wise calculation of cs using np.where
    cs = np.where(wvalue > mtop, ( 4.0 * np.pi * alpha2 * Qf**4.0 * Nc * hbarc2 ) / wvalue**2.0 * (beta) * \
             ( (3.0 - (beta**4.0))/(2.0 * beta) * np.log((1.0 + beta)/(1.0 - beta)) - 2.0 + beta**2.0 ), 0.0) * 1e9

    return cs


##################################################################


# Debugging cross-section calculation
print("Cross-section for inelastic W values (first 10):", cs_ttbar_w_condition_Hamzeh(wv_inelastic_I)[:10])
print("Cross-section for inelastic W values (first 10):", cs_ttbar_w_condition_Hamzeh(wv_inelastic_II)[:10])
print("Cross-section for inelastic W values (first 10):", cs_ttbar_w_condition_Hamzeh(wv_inelastic_III)[:10])

print("Cross-section for elastic W values (first 10):", cs_ttbar_w_condition_Hamzeh(wv_elastic)[:10])


##################################################################

# Integration using trapezoidal rule
def trap_integ(wv, fluxv):
    wmin = np.zeros(len(wv) - 1)
    integ = np.zeros(len(wv) - 1)

    for i in range(len(wv) - 2, -1, -1):
        wvwid = wv[i + 1] - wv[i]
        cs_0 = cs_ttbar_w_condition_Hamzeh(wv[i])
        cs_1 = cs_ttbar_w_condition_Hamzeh(wv[i + 1])
        traparea = wvwid * 0.5 * (fluxv[i] * cs_0 + fluxv[i + 1] * cs_1)
        wmin[i] = wv[i]
        if i == len(wv) - 2:
            integ[i] = traparea
        else:
            integ[i] = integ[i + 1] + traparea

    nanobarn = 1.e+40

    return wmin, integ  # * nanobarn


##################################################################


# Perform integration for both grids
wv_inel_trap_I, int_inel_I = trap_integ(wv_inelastic_I, s_yy_inelastic_I)
wv_inel_trap_II, int_inel_II = trap_integ(wv_inelastic_II, s_yy_inelastic_II)
wv_inel_trap_III, int_inel_III = trap_integ(wv_inelastic_III, s_yy_inelastic_III)

wv_el_trap, int_el = trap_integ(wv_elastic, s_yy_elastic)


# Debugging integration results
print("Integrated inelastic cross-section (partial):", int_inel_I[:200])
print("Integrated elastic cross-section (partial):", int_el[:200])


# Ensure all arrays are of the same length
min_length_I = min(len(wv_el_trap), len(int_el), len(wv_inel_trap_I), len(int_inel_I))
min_length_II = min(len(wv_el_trap), len(int_el), len(wv_inel_trap_II), len(int_inel_II))
min_length_III = min(len(wv_el_trap), len(int_el), len(wv_inel_trap_III), len(int_inel_III))

wv_el_trap = wv_el_trap[:min_length_I]
int_el = int_el[:min_length_I]

wv_inel_trap_I = wv_inel_trap_I[:min_length_I]
int_inel_I = int_inel_I[:min_length_I]

wv_inel_trap_II = wv_inel_trap_II[:min_length_II]
int_inel_II = int_inel_II[:min_length_II]

wv_inel_trap_III = wv_inel_trap_III[:min_length_III]
int_inel_III = int_inel_III[:min_length_III]

# Plotting
fig, ax = plt.subplots(figsize=(8.0, 9.0))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

ax.set_xlim(350.0, 1000.0)
ax.set_ylim(1.e-7, 1.e-2)




# Plot elastic and inelastic cross-sections
ax.loglog(wv_el_trap, int_el, label="Elastic", linestyle="solid", linewidth=3)
ax.loglog(wv_inel_trap_I, int_inel_I, label=r"$M_N < 10$ GeV ($Q^2_p < 10$ GeV$^2$)", linestyle="dotted", linewidth=3)
ax.loglog(wv_inel_trap_II, int_inel_II, label=r"$M_N < 50$ GeV ($Q^2_p < 10^3$ GeV$^2$)", linestyle="dashed", linewidth=3)
ax.loglog(wv_inel_trap_III, int_inel_III, label=r"$M_N < 300$ GeV ($Q^2_p < 10^5$ GeV$^2$)", linestyle="dashdot", linewidth=3)

# Add labels and legend
ax.set_xlabel(r"$W_0$ [GeV]")
ax.set_ylabel(r"$\sigma_{{\rm ep}\to {\rm e}(\gamma\gamma\to t \bar{t}){\rm p}^{(\ast)}}$ (W > W$_0$) [pb]")
ax.legend(title=r"$Q^2_e < 10^5$ GeV$^2$", loc="upper right")



# Save output values
output_data = np.column_stack((wv_el_trap, int_el, int_inel_I, int_inel_II, int_inel_III))
header = "W_Value [GeV] Elastic [pb] Inelastic_I [pb] Inelastic_II [pb] Inelastic_III [pb]"
np.savetxt("exact_ttbar_cross_section.txt", output_data, header=header, fmt="%0.8e", delimiter="\t")




# Save and show the plot
plt.savefig("exact_ttbar_cross_section_JHEP.pdf")
plt.savefig("exact_ttbar_cross_section_JHEP.jpg")

plt.show()


##################################################################

