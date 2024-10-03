import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
alpha = 1 / 137  # Fine-structure constant
M_p = 0.938  # Proton mass in GeV
mu_p = 2.792  # Proton magnetic moment
m_tau = 1.77686  # Tau mass in GeV
m_e = 0.000511  # Electron mass in GeV
E_electron = 50  # Electron beam energy at the LHeC in GeV
E_proton = 7000  # Proton beam energy in GeV
s_ep = 4 * E_electron * E_proton  # CM energy squared of the ep system

# Photon-photon cross section for tau pair production
def cs_tautau(W):
    if W <= 2 * m_tau:
        return 0  # below tau production threshold
    beta_tau = np.sqrt(1 - (2 * m_tau / W)**2)
    cs = (4 * np.pi * alpha**2) / W**2 * (beta_tau * (3 - beta_tau**2) / 2) * 1e9  # in pb
    return cs

# Proton form factors for elastic scattering (dipole approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71)**(-2)

def G_M(Q2):
    return mu_p * G_E(Q2)

# Small cutoff to avoid division by zero
epsilon = 1e-8

# Electron Q2_min based on y_e and electron mass
def Q2_min_electron(y_e):
    if y_e >= 1 - epsilon:
        y_e = 1 - epsilon
    return (m_e**2 * y_e**2) / (1 - y_e)

# Proton Q2_min based on y_p and masses of proton and pion
def Q2_min_proton(y_p):
    if y_p >= 1 - epsilon:
        y_p = 1 - epsilon
    mMin2 = (M_p + m_tau)**2
    return max(0, (mMin2 / (1 - y_p)) - M_p**2)

# Elastic photon flux for the proton
def phi_p(y, Q2_min, Q2_max):
    def integrand(Q2):
        return (G_E(Q2)**2 + (Q2 / (4 * M_p**2)) * G_M(Q2)**2) / Q2
    flux, _ = quad(integrand, Q2_min, Q2_max, limit=1000)
    return (alpha / np.pi) * flux / y

# Elastic photon flux for the electron
def phi_e(y, Q2_min, Q2_max):
    def integrand(Q2):
        return ((1 - y) * (1 - Q2_min / Q2) + y**2 / 2) / Q2
    flux, _ = quad(integrand, Q2_min, Q2_max, limit=1000)
    return (alpha / np.pi) * flux / y

# Exact photon-photon center-of-mass energy including virtualities
def exact_W(y_e, y_p, Q2_e, Q2_p):
    return np.sqrt(y_e * y_p * s_ep - Q2_e - Q2_p)

# Luminosity spectrum calculation (elastic case)
def S_gamma_gamma(W, s_ep, Q2_max_e=100000, Q2_max_p=10):
    luminosity = 0
    y_e_min = W**2 / s_ep  # Minimum y for the electron
    y_e_values = np.linspace(y_e_min, 1 - epsilon, 1000)

    for y_e in y_e_values:
        y_p = W**2 / (y_e * s_ep)

        # Calculate Q2_min for electron and proton based on y
        Q2_min_e = Q2_min_electron(y_e)
        Q2_min_p = Q2_min_proton(y_p)

        W_exact = exact_W(y_e, y_p, Q2_min_e, Q2_min_p)  # Use the exact formula

        if W_exact >= 2 * m_tau:  # Tau production threshold
            luminosity += phi_e(y_e, Q2_min_e, Q2_max_e) * phi_p(y_p, Q2_min_p, Q2_max_p) / y_e

    return 2 * W / s_ep * luminosity

# Parameters for photon virtualities
Q2_max_e = 100000  # Maximal photon virtuality for electron (GeV^2)
Q2_max_p = 10  # Maximal photon virtuality for proton (GeV^2)

# Generate and plot the elastic photon-photon luminosity spectrum
W_values = np.linspace(10, 1000, 100)  # Photon-photon CM energy in GeV
luminosity_spectrum = [S_gamma_gamma(W, s_ep, Q2_max_e, Q2_max_p) for W in W_values]
cross_section = [cs_tautau(W) for W in W_values]

# Plotting the Luminosity Spectrum
plt.figure(figsize=(8, 6))
plt.plot(W_values, luminosity_spectrum, label=r'Elastic Luminosity Spectrum')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Photon-Photon CM Energy $W_{\gamma\gamma}$ (GeV)')
plt.ylabel(r'Luminosity Spectrum $S_{\gamma\gamma}(W)$')
plt.title('Elastic Photon-Photon Luminosity Spectrum at the LHeC')
plt.grid(True)
plt.legend()
plt.savefig("luminosity_spectrum.pdf")
plt.show()

# Plotting the Total Tau-Tau Cross Section
plt.figure(figsize=(8, 6))
plt.plot(W_values, cross_section, label=r'Total $\tau^+\tau^-$ Cross Section')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Photon-Photon CM Energy $W_{\gamma\gamma}$ (GeV)')
plt.ylabel(r'Cross Section $\sigma_{\tau^+\tau^-}(W)$ [pb]')
plt.title(r'Total Cross Section for $\tau^+\tau^-$ Production')
plt.grid(True)
plt.legend()
plt.savefig("tau_tau_cross_section.pdf")
plt.show()

# Print the luminosity spectrum and cross section at W = 10 GeV
W_10GeV = 10.0
index_W_10GeV = np.argmin(np.abs(W_values - W_10GeV))
luminosity_at_10GeV = luminosity_spectrum[index_W_10GeV]
cross_section_at_10GeV = cross_section[index_W_10GeV]

print(f"Luminosity Spectrum at W = 10 GeV: {luminosity_at_10GeV}")
print(f"Total Tau Tau Cross Section at W = 10 GeV: {cross_section_at_10GeV}")

# Save values at W = 10 GeV in a text file
with open("tau_tau_cross_section_at_10GeV.txt", "w") as file:
    file.write(f"Luminosity Spectrum at W = 10 GeV: {luminosity_at_10GeV}\n")
    file.write(f"Total Tau Tau Cross Section at W = 10 GeV: {cross_section_at_10GeV}\n")
