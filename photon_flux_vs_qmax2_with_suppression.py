import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4  # Electron mass
W = 10.0  # Center-of-mass energy W in GeV

# Minimum Photon Virtuality
def qmin2(mass, y):
    return mass * mass * y * y / (1 - y)

# Suppression Factor for Large Photon Virtuality (Exponential Form)
def suppression_factor(Q2, W, c=0.1):
    return np.exp(-Q2 / (c * W**2))

# Elastic Photon Flux from Electron (with suppression factor)
def flux_y_electron(ye, qmax2, W):
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2(emass, ye)

    # Integration over Q2 from qmin2 to qmax2
    def integrand(Q2):
        y1 = 0.5 * (1.0 + (1.0 - ye) ** 2) / ye
        y2 = (1.0 - ye) / ye
        flux1 = y1 / Q2
        flux2 = y2 / qmax2
        suppression = suppression_factor(Q2, W)  # Apply suppression factor for large virtualities
        return (flux1 - flux2) * suppression

    try:
        result, _ = integrate.quad(integrand, qmin2v, qmax2, epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for electron flux did not converge for ye={ye}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for electron flux: {e}")
        result = 0.0

    return ALPHA2PI * result

# Parameters
qmax2_values = np.logspace(0, 5, 100)  # qmax2 values from 1 to 100000 in log scale
ye_values = [0.0001, 0.001, 0.01, 0.1, 0.2]  # Values of ye to be plotted

# Plotting the photon flux as a function of qmax2 for different ye values
plt.figure(figsize=(10, 6))
for ye in ye_values:
    flux_values = [flux_y_electron(ye, qmax2, W) for qmax2 in qmax2_values]
    plt.plot(qmax2_values, flux_values, label=f'ye = {ye:.4f}', linewidth=2)

# Labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$Q_{\rm max}^2$ [GeV$^2$]', fontsize=14)
plt.ylabel('Photon Flux from Electron', fontsize=14)
plt.title(r'Photon Flux from Electron as a Function of $Q_{\rm max}^2$ with Suppression Factor', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--")

# Save the plot as a PDF file
plt.savefig("photon_flux_vs_qmax2_with_suppression.pdf")
plt.savefig("photon_flux_vs_qmax2_with_suppression.jpg")


# Show the plot
plt.show()
