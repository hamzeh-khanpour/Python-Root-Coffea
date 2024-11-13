
# qmin2p as a function of yp

import numpy as np
import matplotlib.pyplot as plt

# Constants
MN = 10.0  # Nucleon mass in GeV
Mp = 0.938272081  # Proton mass in GeV

# Define function for Q_p,min^2
def qmin2_proton(MN, yp):
    return ((MN**2) / (1 - yp) - Mp**2) * yp

# Generate values for y_p from 0 to 1 (but not including 1 to avoid division by zero)
y_p_values = np.logspace(-4, -0.01, 500)  # Log spaced values from 0.01 to 0.99

# Calculate Q_p,min^2 for each value of y_p
qmin2p_values = [qmin2_proton(MN, yp) for yp in y_p_values]

# Plot the results using log-log scale
plt.figure(figsize=(10, 6))
plt.loglog(y_p_values, qmin2p_values, label=r'$Q^2_{p,\mathrm{min}} \, = \left( \frac{M_N^2}{1 - y_p} - M_p^2 \right) y_p$', color='b')
plt.xlabel(r'$y_p$', fontsize=14)
plt.ylabel(r'$Q^2_{p,\mathrm{min}} \, \mathrm{[GeV^2]}$', fontsize=14)
plt.title(r'$Q^2_{p,\mathrm{min}}$ as a function of $y_p$ for $M_N = 10$ GeV', fontsize=16)

plt.grid(True, which="both", linestyle="--")

plt.legend()

# Save the plot as a PDF and JPG
plt.savefig("qmin2p_values_yp.pdf")
plt.savefig("qmin2p_values_yp.jpg")

plt.show()
