# Elastic Photon-Photon Luminosity Spectrum at LHeC --- Updated Version with CUDA Support and Improvements

import numpy as np
import math
import matplotlib.pyplot as plt
from numba import cuda, float64


# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass
pmass = 0.938272081    # Proton mass


q2emax = 100.0  # Maximum photon virtuality for electron in GeV^2
q2pmax = 100.0  # Maximum photon virtuality for proton in GeV^2


# CUDA Kernel for Photon-Photon Luminosity Spectrum Calculation
@cuda.jit
def flux_el_yy_atW_cuda(W_values, eEbeam, pEbeam, luminosity_values):
    idx = cuda.grid(1)
    if idx < W_values.size:
        W = W_values[idx]
        s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared
        ye_min = W**2 / s_cms

        result_ye = 0.0
        ye_steps = 200  # Increased for more parallel work
        Q2e_steps = 100
        yp_steps = 100
        Q2p_steps = 100

        for i in range(ye_steps):
            ye = ye_min + i * (1.0 - ye_min) / (ye_steps - 1)
            Q2e_min = (emass * ye)**2 / (1 - ye)
            Q2e_max = q2emax
            result_Q2e = 0.0

            for j in range(Q2e_steps):
                Q2e = Q2e_min + j * (Q2e_max - Q2e_min) / (Q2e_steps - 1)
                yp_min = (W**2 + Q2e) / (s_cms * ye)
                yp_max = 1.0
                result_yp = 0.0

                for k in range(yp_steps):
                    yp = yp_min + k * (yp_max - yp_min) / (yp_steps - 1)
                    Q2p_min = (pmass * yp)**2 / (1 - yp)
                    Q2p_max = q2pmax
                    result_Q2p = 0.0

                    for l in range(Q2p_steps):
                        Q2p = Q2p_min + l * (Q2p_max - Q2p_min) / (Q2p_steps - 1)
                        W2_exact = ye * yp * s_cms - Q2e - Q2p
                        if W2_exact <= 0.0:
                            continue

                        # Calculate photon fluxes
                        qmin2v_e = (emass * ye)**2 / (1 - ye)
                        flux_e = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v_e / Q2e) + 0.5 * ye**2)

                        qmin2v_p = (pmass * yp)**2 / (1 - yp)
                        gE2 = (1 + Q2p / 0.71) ** (-4)
                        gM2 = 7.78 * gE2
                        formE = (4 * pmass**2 * gE2 + Q2p * gM2) / (4 * pmass**2 + Q2p)
                        formM = gM2
                        flux_p = ALPHA2PI / (yp * Q2p) * ((1 - yp) * (1 - qmin2v_p / Q2p) * formE + 0.5 * yp**2 * formM)

                        flux_product = flux_e * flux_p
                        if not (math.isnan(flux_product) or math.isinf(flux_product)):
                            result_Q2p += flux_product

                    result_yp += result_Q2p

                result_Q2e += result_yp

            result_ye += result_Q2e

        luminosity_values[idx] = result_ye

# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV

W_values = np.logspace(1.0, 3.0, 101)  # Range of W values from 10 GeV to 1000 GeV
luminosity_values = np.zeros_like(W_values)

# Allocate memory on the GPU
d_W_values = cuda.to_device(W_values)
d_luminosity_values = cuda.to_device(luminosity_values)

# CUDA Grid and Block Dimensions
threads_per_block = 16  # Reduced for more blocks
blocks_per_grid = (W_values.size + (threads_per_block - 1)) // threads_per_block

# Launch CUDA kernel to calculate the Elastic Photon-Photon Luminosity Spectrum
flux_el_yy_atW_cuda[blocks_per_grid, threads_per_block](d_W_values, eEbeam, pEbeam, d_luminosity_values)

# Copy results back to the host
luminosity_values = d_luminosity_values.copy_to_host()

# Handle NaN or Inf values in the results
luminosity_values = np.nan_to_num(luminosity_values, nan=0.0, posinf=0.0, neginf=0.0)

# Plot the Results
plt.figure(figsize=(10, 8))
plt.xlim(10.0, 1000.0)
plt.ylim(1.e-7, 1.e-1)

plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Elastic')

# Marking W_0 = 10 GeV on the plot
W_value = 10.0  # GeV
luminosity_at_W10_idx = np.where(W_values == W_value)[0]
if len(luminosity_at_W10_idx) > 0:
    luminosity_at_W10 = luminosity_values[luminosity_at_W10_idx[0]]
    plt.scatter(W_value, luminosity_at_W10, color='red', zorder=5)
    plt.text(W_value, luminosity_at_W10 * 1.5, f'$W_0 = 10$ GeV\n$S_{{\gamma\gamma}} = {luminosity_at_W10:.2e}$', color='red', fontsize=10, ha='center')

# Plot settings
plt.xlabel(r"$W$ [GeV]", fontsize=18)
plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
plt.title("Elastic Photon-Photon Luminosity Spectrum at LHeC", fontsize=20)

plt.grid(True, which="both", linestyle="--")
plt.legend(title=r'$Q^2_{e,\text{max}} = 100 \, \mathrm{GeV}^2, \, Q^2_{p,\text{max}} = 100 \, \mathrm{GeV}^2$', fontsize=14)

# Save the plot as a PDF and JPG
plt.savefig("elastic_photon_luminosity_spectrum_with_W10.pdf")
plt.savefig("elastic_photon_luminosity_spectrum_with_W10.jpg")

plt.show()
