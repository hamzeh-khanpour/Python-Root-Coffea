import numpy as np
import matplotlib.pyplot as plt


# LaTeX settings for colored text
#plt.rcParams["text.usetex"] = True
#plt.rcParams['text.latex.preamble'] = r'\usepackage{xcolor}'

# Given constants
E_e = 50  # GeV
E_p = 7000  # GeV
y_p = 0.01
Q_e2 = 100  # GeV^2, fixed value

# Center-of-mass energy squared
s = 4 * E_e * E_p

# y_e values for the plot
y_e_values = np.linspace(0.0001, 0.99, 500)  # y_e values from 0.001 to 0.99

# Function to calculate W^2 using the exact formula (Equation C.9)
def calculate_W2_exact(Q_e2, y_e, E_e, E_p, y_p):
    term1 = -Q_e2
    term2 = 4 * y_e * y_p * E_e * E_p
    term3 = 2 * y_p * E_p * np.sqrt((y_e * E_e) ** 2 + Q_e2)
    term4 = (1 - Q_e2 / (2 * E_e ** 2 * (1 - y_e))) ** 2.0  # Note the square here for correct calculation
    return term1 + term2   #  + term3 * term4

# Calculate W values using the exact formula for each y_e
W_exact_values = [np.sqrt(calculate_W2_exact(Q_e2, y_e, E_e, E_p, y_p)) if calculate_W2_exact(Q_e2, y_e, E_e, E_p, y_p) > 0 else 0 for y_e in y_e_values]

# Calculate W values using the simple formula for each y_e
W_simple_values = [np.sqrt(y_e * y_p * s) for y_e in y_e_values]

# Plotting W (exact vs simple) as a function of y_e
plt.figure(figsize=(10, 6))

plt.plot(y_e_values, W_exact_values, label=f'W Formula ($W = \sqrt{{-Q_e^2 + y_e y_p s}}$ with $Q_e^2 = {Q_e2}$ GeV$^2$)', linestyle='solid', color='blue')
plt.plot(y_e_values, W_simple_values, label=f'W Formula ($W = \sqrt{{y_e y_p s}}$)', linestyle='dashed', color='red')


plt.xlabel(r'$y_e$')
plt.ylabel(r'$W$ (GeV)')
plt.title(r'Comparison of $W$ as a function of $y_e$)')

plt.text(0.6, 0.5, f'$E_e$ = {E_e} GeV\n$E_p$ = {E_p} GeV\n$y_p$ = {y_p}', 
         transform=plt.gca().transAxes, fontsize=12, color='blue', ha='left', va='center')

plt.grid(True)
plt.legend()
plt.ylim(0, max(max(W_exact_values), max(W_simple_values)) * 1.1)  # Set y-limit to slightly above the maximum W

# Save the plot as a PDF and JPG
plt.savefig("W_Comparison_vs_ye_y_p_0_1_New_Qe2_100.pdf")
plt.savefig("W_Comparison_vs_ye_y_p_0_1_New_Qe2_100.jpg") 

plt.show()
