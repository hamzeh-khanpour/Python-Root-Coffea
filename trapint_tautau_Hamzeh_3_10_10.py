# Final Version -- January 2025 -- Hamzeh Khanpour

# ================================================================================

# Imports
import mplhep as hep
import numpy as np
import matplotlib.pyplot as plt
import sys

# Use CMS style for plots
hep.style.use("CMS")

# ================================================================================

# Function Definitions

def cs_tautau_w_condition_Hamzeh(wvalue):
    """Calculate tau-tau cross-section as a function of W."""
    re = 2.8179403262e-15 * 137.0 / 128.0
    me = 0.510998950e-3
    mtau = 1.77686
    hbarc2 = 0.389
    alpha2 = (1.0 / 137.0) ** 2

    # Calculate beta
    beta = np.sqrt(np.maximum(0.0, 1.0 - 4.0 * mtau**2 / wvalue**2))

    # Compute cross-section
    cs = np.where(
        wvalue > mtau,
        (4.0 * np.pi * alpha2 * hbarc2 / wvalue**2) * beta * (
            (3.0 - beta**4) / (2.0 * beta) * np.log((1.0 + beta) / (1.0 - beta)) - 2.0 + beta**2),
        0.0
    ) * 1e9  # Convert to pb
    return cs


def trap_integ(wv, fluxv):
    """Perform trapezoidal integration with tau-tau cross-section weighting."""
    # Ensure the arrays have compatible lengths
    if len(wv) != len(fluxv):
        raise ValueError("Input arrays `wv` and `fluxv` must have the same length.")
    
    wmin = np.zeros(len(wv) - 1)
    integ = np.zeros(len(wv) - 1)

    for i in range(len(wv) - 1):
        wvwid = wv[i + 1] - wv[i]
        cs_0 = cs_tautau_w_condition_Hamzeh(wv[i])
        cs_1 = cs_tautau_w_condition_Hamzeh(wv[i + 1])
        traparea = wvwid * 0.5 * (fluxv[i] * cs_0 + fluxv[i + 1] * cs_1)
        wmin[i] = wv[i]
        if i == 0:
            integ[i] = traparea
        else:
            integ[i] = integ[i - 1] + traparea

    return wmin, integ

# ================================================================================

# Main Script

# Add the directory containing the data module to the Python path
sys.path.append('./values')

# Import photon luminosity data
from wgrid_3_10_10_exact_inelastic import *

# Validate array lengths before accessing
print("Validating data lengths...")
print(f"Length of wvalueselas: {len(wvalueselas)}")
print(f"Length of wvaluesinel: {len(wvaluesinel)}")
print(f"Length of elas: {len(elas)}")
print(f"Length of inel: {len(inel)}")

# Check if index 3 is within range
index = 3
if len(wvalueselas) > index and len(wvaluesinel) > index and len(elas) > index and len(inel) > index:
    wvelas = np.array(wvalueselas[index])
    wvinel = np.array(wvaluesinel[index])

    ie = np.array(inel[index])
    el = np.array(elas[index])

    # Perform integration for inelastic and elastic components
    min_len_inel = min(len(wvinel), len(ie))  # Truncate to the smallest size
    wvinel, ie = wvinel[:min_len_inel], ie[:min_len_inel]

    min_len_el = min(len(wvelas), len(el))  # Truncate to the smallest size
    wvelas, el = wvelas[:min_len_el], el[:min_len_el]

    wv1, int_inel = trap_integ(wvinel, ie)
    wv2, int_el = trap_integ(wvelas, el)

    # Ensure output arrays have the same length for stacking
    min_len_output = min(len(wv2), len(int_el), len(int_inel))
    wv2, int_el, int_inel = wv2[:min_len_output], int_el[:min_len_output], int_inel[:min_len_output]

    # ================================================================================

    # Plotting
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

    # Set axis limits
    ax.set_xlim(10.0, 1000.0)
    ax.set_ylim(1.e-3, 10.e2)

    # Labels and title
    inel_label = (f"$M_N<$ {inel[0]:g} GeV") + (f" ($Q^2_p<$ {inel[2]:g} GeV$^2$)")
    title_label = f"$Q^2_e<$ {10:g} GeV$^2$"

    plt.loglog(wv2, int_el, linestyle='solid', linewidth=4, label='Elastic')
    plt.loglog(wv1, int_inel, linestyle='dotted', linewidth=4, label=inel_label)

    # Legend
    plt.legend(title=title_label)

    # Axis labels
    plt.xlabel("W$_0$ [GeV]", fontsize=24)
    plt.ylabel(r"$\sigma_{{\rm ep}\to {\rm e}(\gamma\gamma\to\tau^+\tau^-){\rm p}^{(\ast)}}$ (W > W$_0$) [pb]", fontsize=24)

    # Save the plot
    plt.savefig("cs_tautau_30_10_10.pdf")
    plt.savefig("cs_tautau_30_10_10.jpg")

    # Display the plot
    plt.show()

    # ================================================================================

    # Save Results

    # Save integration results to a text file
    output_data = np.column_stack((wv2, int_el, int_inel))
    header = 'W_Value Elastic Inelastic'
    np.savetxt('output_values_tau.txt', output_data, header=header, fmt='%0.8e', delimiter='\t')

else:
    print(f"Error: Index {index} is out of range for the given data arrays.")
