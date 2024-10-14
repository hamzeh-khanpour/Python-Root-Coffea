import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4  # Electron mass in GeV
qmax2 = 100000.0  # Maximum photon virtuality for electron in GeV^2

# Minimum Photon Virtuality
def qmin2(mass, y):
    return mass * mass * y * y / (1 - y)

# Suppression Factor for Large Photon Virtuality (Exponential Form)
def suppression_factor(Q2, W, c=0.1):
    return np.exp(-Q2 / (c * W**2))

# Elastic Photon Flux from Electron (with full Q2 integration)
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
W_values = np.linspace(10, 1000, 100)  # W values from 10 to 1000 GeV
ye_values = [0.0001, 0.9999]  # Values of ye to be plotted

# Calculate flux for different values of ye
flux_results = {ye: [flux_y_electron(ye, qmax2, W) for W in W_values] for ye in ye_values}

# Plotting the flux as a function of W for different values of ye
plt.figure(figsize=(10, 6))

for ye, flux in flux_results.items():
    plt.plot(W_values, flux, linewidth=2, label=f'ye = {ye}')

plt.xlabel(r'Photon-Photon Center-of-Mass Energy $W$ [GeV]', fontsize=14)
plt.ylabel(r'Photon Flux from Electron $S_{\gamma/e}$', fontsize=14)
plt.title('Photon Flux from Electron as a Function of $W$', fontsize=16)
plt.grid(True)
plt.legend(title='Electron Longitudinal Momentum Fraction (ye)', fontsize=12)

# Save the plot as a PDF file
plt.savefig("flux_y_electron_plot.pdf")

# Show the plot
plt.show()
