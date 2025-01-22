
# The exact elastic/inelastic `integrated WW production cross-section` for the `ep -> e(gamma gamma -> W^+W^-)p(*)` process
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
inelastic_data_I = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_100_q2emax_100000_q2pmax_100000_using_vegas.txt", skiprows=1)
inelastic_data_II = np.loadtxt("Inelastic_Photon_Luminosity_Spectrum_MNmax_100_q2emax_100000_q2pmax_100000_using_vegas_LHeC750GeV_303.txt", skiprows=1)


elastic_data_I = np.loadtxt("Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_100000_using_vegas_tagged_elastic_Newyp.txt", skiprows=1)
elastic_data_II = np.loadtxt("Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_100000_using_vegas_tagged_elastic_LHeC750GeV_Newyp.txt", skiprows=1)



# Extract W values and luminosity spectra
wv_inelastic_I = inelastic_data_I[:, 0]
s_yy_inelastic_I = inelastic_data_I[:, 1]

wv_inelastic_II = inelastic_data_II[:, 0]
s_yy_inelastic_II = inelastic_data_II[:, 1]

wv_elastic_II = elastic_data_II[:, 0]
s_yy_elastic_II = elastic_data_II[:, 1]

wv_elastic_I = elastic_data_I[:, 0]
s_yy_elastic_I = elastic_data_I[:, 1]


# Debugging input data
print("Inelastic W values (first 10):", wv_inelastic_I[:10])
print("Inelastic S_yy values (first 10):", s_yy_inelastic_I[:10])

print("Inelastic W values (first 10):", wv_inelastic_II[:10])
print("Inelastic S_yy values (first 10):", s_yy_inelastic_II[:10])



print("Elastic W values (first 10):", wv_elastic_II[:10])
print("Eelastic S_yy values (first 10):", s_yy_elastic_II[:10])

print("Elastic W values (first 10):", wv_elastic_I[:10])
print("Elastic S_yy values (first 10):", s_yy_elastic_I[:10])



# Function to calculate the WW production cross-section
##################################################################
def cs_ww_w(wvalue):

    re = 2.8179403262e-15 * 137.0 / 128.0
    me = 0.510998950e-3
    mw = 80.379
    hbarc2 =  0.389
    alpha2 = (1.0/128.0)*(1.0/128.0)

    #if wvalue > 2.0 * mw:
        #cs = (19.0/2.0) * np.pi * re * re * me * me / mw / mw \
             #* np.sqrt(wvalue * wvalue - 4.0 * mw * mw) / wvalue
    #elif wvalue > 300.0:
        #cs = 8.0 * np.pi * re * re * me * me / mw / mw
    #else:
        #cs = 0.0

    if wvalue > 2.0 * mw:
        cs = (19.0/2.0) * np.pi * hbarc2 * alpha2  / mw / mw \
             * np.sqrt(wvalue * wvalue - 4.0 * mw * mw) / wvalue         * 1e9
    elif wvalue > 300.0:
        cs = 8.0 * np.pi * hbarc2 * alpha2  / mw / mw                    * 1e9
    else:
        cs = 0.0

    return cs

##################################################################


def cs_ww_w_PR364(wvalue):

    re = 2.8179403262e-15 * 137.0 / 128.0
    me = 0.510998950e-3
    mw = 80.379
    hbarc2 =  0.389
    alpha2 = (1.0/128.0)*(1.0/128.0)

    beta = np.sqrt(np.where(1.0 - 4.0 * mw * mw / wvalue**2.0 >= 0, 1.0 - 4.0 * mw * mw / wvalue**2.0, np.nan))

    cs = np.pi * hbarc2 * alpha2 / wvalue**2.0 * beta * \
         ( -3.0 * (1.0 - beta**4.0)/beta * np.log((1.0 + beta)/(1.0 - beta)) + \
           2.0* (22.0 - 9.0 * beta**2.0 + 3.0*beta**4.0)/(1.0 - beta**2.0) )*1e9

    return cs

##################################################################


# Debugging cross-section calculation
#print("Cross-section for inelastic W values (first 10):", cs_ww_w(wv_inelastic_I)[:10])
#print("Cross-section for inelastic W values (first 10):", cs_ww_w(wv_inelastic_II)[:10])
#print("Cross-section for inelastic W values (first 10):", cs_ww_w(wv_inelastic_III)[:10])

#print("Cross-section for elastic W values (first 10):", cs_ww_w(wv_elastic)[:10])


##################################################################

# Integration using trapezoidal rule
def trap_integ(wv, fluxv):
    wmin = np.zeros(len(wv) - 1)
    integ = np.zeros(len(wv) - 1)

    for i in range(len(wv) - 2, -1, -1):
        wvwid = wv[i + 1] - wv[i]
        cs_0 = cs_ww_w(wv[i])
        cs_1 = cs_ww_w(wv[i + 1])
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


wv_el_trap_I, int_el_I = trap_integ(wv_elastic_I, s_yy_elastic_I)
wv_el_trap_II, int_el_II = trap_integ(wv_elastic_II, s_yy_elastic_II)


# Debugging integration results
print("Integrated inelastic cross-section (partial):", int_inel_I[:200])
print("Integrated elastic cross-section (partial):", int_el_I[:200])




# Find the minimum length among all arrays
min_length = min(len(wv_el_trap_I), len(int_el_I), len(wv_inel_trap_I), len(int_inel_I), len(wv_inel_trap_II), len(int_inel_II), len(wv_el_trap_II), len(int_el_II))


# Truncate all arrays to the minimum length
wv_el_trap_I = wv_el_trap_I[:min_length]
int_el_I = int_el_I[:min_length]


wv_inel_trap_I = wv_inel_trap_I[:min_length]
int_inel_I = int_inel_I[:min_length]


wv_inel_trap_II = wv_inel_trap_II[:min_length]
int_inel_II = int_inel_II[:min_length]


wv_el_trap_II = wv_el_trap_II[:min_length]
int_el_II = int_el_II[:min_length]

# Now stack the arrays
output_data = np.column_stack((wv_el_trap_I, int_el_I, int_inel_I, int_inel_II, int_el_II))



# Plotting
fig, ax = plt.subplots(figsize=(10.0, 11.0))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

ax.set_xlim(161.0, 1000.0)
ax.set_ylim(1.0e-4, 1.0e0)



# Plot elastic and inelastic cross-sections
ax.loglog(wv_el_trap_I, int_el_I, label="elastic - p detected", linestyle="solid", linewidth=3, color="blue")
ax.loglog(wv_el_trap_II, int_el_II, label=r"elastic - p detected ($\sqrt{s}=0.75$ TeV)", linestyle=(0, (5, 2, 1, 2, 1, 2)), linewidth=3, color="red")


ax.loglog(wv_inel_trap_I, int_inel_I, label=r"$M_N < 100$ GeV ($Q^2_p < 10^5$ GeV$^2$)", linestyle="dashed", linewidth=3, color="green")
ax.loglog(wv_inel_trap_II, int_inel_II, label=r"$M_N < 100$ GeV ($Q^2_p < 10^5$ GeV$^2$) ($\sqrt{s}=0.75$ TeV)", linestyle="dotted", linewidth=3, color="magenta")



# Add labels and legend
ax.set_xlabel(r"$W_0$ [GeV]")
ax.set_ylabel(r"$\sigma_{{\rm ep}\to {\rm e}(\gamma\gamma \to W^+W^-){\rm p}^{(\ast)}}$ (W > W$_0$) [pb]")
ax.legend(title=r"$Q^2_e < 10^5$ GeV$^2$", loc="upper right")



# Save output values
output_data = np.column_stack((wv_el_trap_I, int_el_I, int_inel_I, int_inel_II, int_el_II))
header = "W_Value [GeV] Elastic [pb] Inelastic_I [pb] Inelastic_II [pb] Inelastic_III [pb]"
np.savetxt("exact_ww_cross_section_JHEP_Krzysztof.txt", output_data, header=header, fmt="%0.8e", delimiter="\t")




# Save and show the plot
plt.savefig("exact_ww_cross_section_JHEP_Krzysztof.pdf")
#plt.savefig("exact_ww_cross_section_JHEP_Krzysztof.jpg")

plt.show()


##################################################################

