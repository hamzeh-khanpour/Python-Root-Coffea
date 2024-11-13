import numpy as np
import matplotlib.pyplot as plt

# Define suppression factor function
def suppression_factor(Q2_W2_ratio, c):
    return np.exp(-Q2_W2_ratio / c)

# Parameters
Q2_W2_ratios = np.linspace(0, 5, 100)  # Q2/W^2 ratios from 0 to 5
c_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Different values of c to plot

# Plotting the suppression factor for different values of c
plt.figure(figsize=(10, 6))

for c in c_values:
    suppression_values = suppression_factor(Q2_W2_ratios, c)
    plt.plot(Q2_W2_ratios, suppression_values, linewidth=2, label=f'c = {c}')

plt.xlabel(r'$Q^2 / W^2$', fontsize=14)
plt.ylabel('Suppression Factor', fontsize=14)
plt.title(r'$S(Q, W) = \exp\left(-\frac{Q^2}{c W^2}\right)$', fontsize=16)
plt.grid(True)
plt.legend(title='Suppression Factor Parameter', fontsize=12)

# Save the plot as a PDF file
plt.savefig("suppression_factor_plot.pdf")
plt.savefig("suppression_factor_plot.jpg")

# Show the plot
plt.show()



