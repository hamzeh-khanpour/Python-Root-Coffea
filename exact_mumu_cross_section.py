
# The exact elastic/inelastic `integrated mu-mu production cross-section` for the `ep -> e(gamma gamma -> mu+ mu-)p(*)` process 
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
inelastic_data = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_3_q2emax_50_q2pmax_50_using_vegas.txt", skiprows=1)
elastic_data = np.loadtxt("Elastic_Photon_Luminosity_Spectrum_q2emax_50_q2pmax_50_using_vegas.txt", skiprows=1)

# Extract W values and luminosity spectra
wv_inelastic = inelastic_data[:, 0]
s_yy_inelastic = inelastic_data[:, 1]

wv_elastic = elastic_data[:, 0]
s_yy_elastic = elastic_data[:, 1]

# Debugging input data
print("Inelastic W values (first 10):", wv_inelastic[:10])
print("Inelastic S_yy values (first 10):", s_yy_inelastic[:10])
print("Elastic W values (first 10):", wv_elastic[:10])
print("Elastic S_yy values (first 10):", s_yy_elastic[:10])



# Function to calculate the tau-tau production cross-section
##################################################################

def cs_mumu_w_condition_Hamzeh(wvalue):  # Eq.62 of Physics Reports 364 (2002) 359-450
    re = 2.8179403262e-15 * 137.0 / 128.0
    me = 0.510998950e-3
    mmuon = 0.105658
    hbarc2 =  0.389
    alpha2 = (1.0/137.0)*(1.0/137.0)

    # Element-wise calculation of beta using np.where
    beta = np.sqrt(np.where(1.0 - 4.0 * mmuon * mmuon / wvalue**2.0 >= 0.0, 1.0 - 4.0 * mmuon * mmuon / wvalue**2.0, np.nan))

    # Element-wise calculation of cs using np.where
    cs = np.where(wvalue > mmuon, ( 4.0 * np.pi * alpha2 * hbarc2 ) / wvalue**2.0 * (beta) * \
             ( (3.0 - (beta**4.0))/(2.0 * beta) * np.log((1.0 + beta)/(1.0 - beta)) - 2.0 + beta**2.0 ), 0.0) * 1e9

    return cs


##################################################################


def cs_mumu_w_condition_Krzysztof(wvalue):
    re = 2.8179403262e-15 * 137.0 / 128.0
    me = 0.510998950e-3
    mmuon = 0.105658
    hbarc2 = 0.389
    alpha2 = (1.0/137.0)*(1.0/137.0)

    # Element-wise calculation of beta using np.where
    beta = np.sqrt(np.where(1.0 - 4.0 * mmuon * mmuon / wvalue**2.0 >= 0, 1.0 - 4.0 * mmuon * mmuon / wvalue**2.0, np.nan))

    # Element-wise calculation of cs using np.where
    cs =  4.0 * np.pi * hbarc2 * alpha2 / wvalue**2.0 * \
         (2.0 * (1.0 + 4.0 * mmuon**2.0 / wvalue**2.0 - 8.0 * mmuon**4.0 / wvalue**4.0) * np.log(2.0 * wvalue / (mmuon * (1.0 + beta))) -
          beta * (1.0 + 4.0 * mmuon**2.0 / wvalue**2.0)) * 1e9


    return cs



##################################################################


# Debugging cross-section calculation
print("Cross-section for inelastic W values (first 10):", cs_mumu_w_condition_Hamzeh(wv_inelastic)[:10])
print("Cross-section for elastic W values (first 10):", cs_mumu_w_condition_Hamzeh(wv_elastic)[:10])


# Integration using trapezoidal rule
def trap_integ(wv, fluxv):
    wmin = np.zeros(len(wv) - 1)
    integ = np.zeros(len(wv) - 1)

    for i in range(len(wv) - 2, -1, -1):
        wvwid = wv[i + 1] - wv[i]
        cs_0 = cs_mumu_w_condition_Hamzeh(wv[i])
        cs_1 = cs_mumu_w_condition_Hamzeh(wv[i + 1])
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
wv_inel_trap, int_inel = trap_integ(wv_inelastic, s_yy_inelastic)
wv_el_trap, int_el = trap_integ(wv_elastic, s_yy_elastic)


# Debugging integration results
print("Integrated inelastic cross-section (partial):", int_inel[:10])
print("Integrated elastic cross-section (partial):", int_el[:10])


# Ensure all arrays are of the same length
min_length = min(len(wv_el_trap), len(int_el), len(wv_inel_trap), len(int_inel))

wv_el_trap = wv_el_trap[:min_length]
int_el = int_el[:min_length]

wv_inel_trap = wv_inel_trap[:min_length]
int_inel = int_inel[:min_length]


# Plotting
fig, ax = plt.subplots(figsize=(8.0, 9.0))

plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)
ax.set_xlim(10.0, 1000.0)
ax.set_ylim(1e-3, 1e3)


# Plot elastic and inelastic cross-sections
ax.loglog(wv_el_trap, int_el, label="Elastic", linestyle="solid", linewidth=3)
ax.loglog(wv_inel_trap, int_inel, label=r"$M_N < 3$ GeV ($Q^2_p < 50$ GeV$^2$)", linestyle="dotted", linewidth=3)


# Add labels and legend
ax.set_xlabel(r"$W_0$ [GeV]")
ax.set_ylabel(r"$\sigma_{\mathrm{ep} \to \mathrm{e}(\gamma\gamma \to \mu^+ \mu^-)\mathrm{p}^{(*)}}$ ($W > W_0$) [pb]")
ax.legend(title=r"$Q^2_e < 50$ GeV$^2$", loc="upper right")


# Save output values
output_data = np.column_stack((wv_el_trap, int_el, int_inel))
header = "W_Value [GeV] Elastic [pb] Inelastic [pb]"
np.savetxt("exact_mumu_cross_section_3_50_50.txt", output_data, header=header, fmt="%0.8e", delimiter="\t")



# Save and show the plot
plt.savefig("exact_mumu_cross_section_3_50_50_JHEP.pdf")
plt.savefig("exact_mumu_cross_section_3_50_50_JHEP.jpg")

plt.show()


##################################################################

