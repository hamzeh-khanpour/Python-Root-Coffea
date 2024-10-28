import numpy as np

# Given parameters
y_e = 0.9
y_p = 0.9
E_e = 50  # GeV
E_p = 7000  # GeV

# Define a range of values for Q_e^2 (virtuality of the photon)
Q_e2_values = np.linspace(0, 100000, 100)  # from 0 to 10 GeV^2, 100 points

# Define functions for the two W^2 expressions

# First Formula: Derived Previously
def W2_derived(Q_e2):
    return 4 * E_e * E_p * y_e * y_p - 2 * Q_e2

# Second Formula: Formula from the Screenshot
def W2_screenshot(Q_e2):
    term1 = -Q_e2
    term2 = 2 * y_e * y_p * E_e * E_p
    term3 = 2 * y_p * E_p * np.sqrt((y_e * E_e)**2 + Q_e2) * (1 - Q_e2 / (2 * E_e**2 * (1 - y_e)))
    return term1 + term2 + term3

# Calculate W^2 for both formulas over the range of Q_e^2 values
W2_derived_values = [W2_derived(Q_e2) for Q_e2 in Q_e2_values]
W2_screenshot_values = [W2_screenshot(Q_e2) for Q_e2 in Q_e2_values]

# Compare the two formulas
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(Q_e2_values, W2_derived_values, label=r'$W^2_{\text{derived}}$', color='b')
plt.plot(Q_e2_values, W2_screenshot_values, label=r'$W^2_{\text{screenshot}}$', color='r', linestyle='--')
plt.xlabel(r'$Q_e^2$ (GeV$^2$)')
plt.ylabel(r'$W^2$ (GeV$^2$)')
plt.title('Comparison of $W^2$ Calculations')


plt.legend()
plt.grid(True)

plt.show()
