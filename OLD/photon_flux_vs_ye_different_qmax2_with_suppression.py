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
def suppression_factor(Q2, W, c=0.2):
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
qmax2_values = [1, 10, 100, 1000, 10000, 100000]  # Different values of qmax2 to be plotted
ye_values = np.logspace(-4, -0.7, 100)  # ye values from 0.0001 to approximately 0.2 in log scale

# Plotting the photon flux as a function of ye for different qmax2 values
plt.figure(figsize=(10, 6))
for qmax2 in qmax2_values:
    flux_values = [flux_y_electron(ye, qmax2, W) for ye in ye_values]
    plt.plot(ye_values, flux_values, label=f'$Q_{{\mathrm{{max}}}}^2 = {qmax2}\ \mathrm{{GeV}}^2$', linewidth=2)

# Labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$y_e$', fontsize=14)
plt.ylabel('Photon Flux from Electron', fontsize=14)
plt.title(r'Photon Flux from Electron as a Function of $y_e$ for Different $Q_{\rm max}^2$ with Suppression Factor', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--")

# Save the plot as a PDF file
plt.savefig("photon_flux_vs_ye_different_qmax2_with_suppression.pdf")
plt.savefig("photon_flux_vs_ye_different_qmax2_with_suppression.jpg")


# Show the plot
plt.show()
