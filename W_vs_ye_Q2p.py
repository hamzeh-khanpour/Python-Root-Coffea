import numpy as np
import matplotlib.pyplot as plt

# Given constants
E_e = 50  # GeV
E_p = 7000  # GeV
y_p = 0.001
Q_p2 = 100  # GeV^2, as specified in the instructions

y_e_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
Q_e2_values = np.linspace(0, 10000, 500)  # Q_e^2 values from 0 to 10000 GeV^2

# Define line styles for each y_e value
line_styles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), 
               (0, (5, 2)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (1, 10)), 
               (0, (3, 10, 1, 10)), (0, (5, 5)), (0, (1, 1))]

# Function to calculate W^2 according to the provided formula
def calculate_W2(Q_e2, y_e, E_e, E_p, y_p, Q_p2):
    term1 = -Q_e2
    term2 = -Q_p2
    term3 = 2 * (y_e * E_e * y_p * E_p)
    term4 = 2 * np.sqrt(y_e**2 * E_e**2 + Q_e2) * np.sqrt(y_p**2 * E_p**2 + Q_p2)
    return term1 + term2 + term3 - term4

# Prepare to save W^2 values to a text file
output_data = []

# Plotting W^2 as a function of Q_e^2 for selected y_e values
plt.figure(figsize=(10, 6))

for i, y_e in enumerate(y_e_values):
    W2_values = [calculate_W2(Q_e2, y_e, E_e, E_p, y_p, Q_p2) for Q_e2 in Q_e2_values]
    W_values = np.sqrt(W2_values)  # Convert W^2 to W
    plt.plot(Q_e2_values, W2_values, label=f'$y_e = {y_e}$', linestyle=line_styles[i % len(line_styles)])

    # Append data for each y_e to output_data list
    for Q_e2, W2 in zip(Q_e2_values, W2_values):
        output_data.append([y_e, Q_e2, W2])

# Convert output data to a NumPy array and save to file
output_data = np.array(output_data)
np.savetxt("W2_yp_Q2p0_001.txt", output_data, header="y_e Q_e^2 W^2", fmt="%0.6e", delimiter='\t')

# Plot labels and settings
plt.xlabel(r'$Q_e^2$ (GeV$^2$)')
plt.ylabel(r'$W^2$ (GeV$^2$)')
plt.title(r'$W^2$ as a function of $Q_e^2$ for different $y_e$ values')

# Add text with E_e, E_p, and y_p values
plt.text(0.3, 0.9, f'$E_e$ = {E_e} GeV\n$E_p$ = {E_p} GeV\n$y_p$ = {y_p}\n$Q_p^2$ = {Q_p2} GeV$^2$', 
         transform=plt.gca().transAxes, fontsize=12, color='blue', ha='left', va='center')

#plt.xscale('log')
#plt.yscale('log')

plt.ylim(1, 1000000)

plt.legend()
plt.grid(True)

# Save the plot as a PDF and JPG
plt.savefig("W_yp_Q2p0_001.pdf")
plt.savefig("W_yp_Q2p0_001.jpg") 

plt.show()
