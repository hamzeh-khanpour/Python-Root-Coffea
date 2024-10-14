import numpy as np
import matplotlib.pyplot as plt

# Define suppression factor
def suppression_factor(Q2, W, c):
    return np.exp(-Q2 / (c * W**2))

# Function to evaluate integrated luminosity or cross-section (mock version for illustration)
def integrated_cross_section(W, c):
    Q2_values = np.linspace(0, 100000, 100)  # Range of Q2 values for integration
    return np.sum(suppression_factor(Q2_values, W, c))

# Parameters for testing
W_example = 100  # Center-of-mass energy (GeV)
c_values = np.linspace(0.01, 1.0, 20)  # Values of c to test
results = []

for c in c_values:
    result = integrated_cross_section(W_example, c)
    results.append(result)

# Plotting the effect of c on the integrated cross-section
plt.figure(figsize=(10, 6))
plt.plot(c_values, results, 'o-', linewidth=2)
plt.xlabel('Suppression Factor Parameter (c)', fontsize=14)
plt.ylabel('Integrated Cross-Section (Arbitrary Units)', fontsize=14)
plt.title('Effect of Suppression Factor Parameter (c) on Cross-Section', fontsize=16)
plt.grid(True)
plt.show()
