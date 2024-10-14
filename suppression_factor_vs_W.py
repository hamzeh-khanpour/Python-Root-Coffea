import numpy as np
import matplotlib.pyplot as plt

# Suppression Factor for Large Photon Virtuality (Exponential Form)
def suppression_factor(Q2, W, c=0.2):
    return np.exp(-Q2 / (c * W**2))

# Parameters
W_values = np.logspace(1.0, 3.0, 303)  # Range of W values from 10 GeV to 1000 GeV
qmax2_values = [1, 10, 50, 100]  # Different values of qmax2 to be plotted

# Plotting the suppression factor as a function of W for different qmax2 values
plt.figure(figsize=(10, 6))
for qmax2 in qmax2_values:
    suppression_values = [suppression_factor(qmax2, W) for W in W_values]
    plt.plot(W_values, suppression_values, label=f'$Q^2_{{\max}}$ = {qmax2} GeV$^2$', linewidth=2)

# Labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Photon-Photon Center-of-Mass Energy $W$ [GeV]', fontsize=14)
plt.ylabel(r'Suppression Factor   $S(Q^2, W)$', fontsize=14)
plt.title(r'$S(Q^2, W) = \exp\left(-\frac{Q^2}{c W^2}\right)$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--")

# Save the plot as a PDF and JPG file
plt.savefig("suppression_factor_vs_W_qmax2_values.pdf")
plt.savefig("suppression_factor_vs_W_qmax2_values.jpg")

# Show the plot
plt.show()



