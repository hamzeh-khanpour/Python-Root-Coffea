import matplotlib.pyplot as plt
import numpy as np

# Data from Table S3
# Columns: z, Mπ+ (proton target), δMπ+ (uncertainty), W (GeV)
data = {
    "z_0.325": {"W": [2.2, 2.5, 2.9, 3.2], "Mπ+": [1.0138, 0.8554, 0.6476, 0.5756], "δMπ+": [0.0269, 0.0230, 0.0246, 0.0205]},
    "z_0.375": {"W": [2.2, 2.5, 2.9, 3.2], "Mπ+": [0.7735, 0.6438, 0.4555, 0.3718], "δMπ+": [0.0116, 0.0100, 0.0097, 0.0078]},
    "z_0.425": {"W": [2.2, 2.5, 2.9, 3.2], "Mπ+": [0.5826, 0.4746, 0.3272, 0.2492], "δMπ+": [0.0077, 0.0065, 0.0056, 0.0046]},
    "z_0.475": {"W": [2.2, 2.5, 2.9, 3.2], "Mπ+": [0.4529, 0.3659, 0.2379, 0.1882], "δMπ+": [0.0070, 0.0057, 0.0043, 0.0038]},
    "z_0.525": {"W": [2.2, 2.5, 2.9, 3.2], "Mπ+": [0.3648, 0.2748, 0.1647, 0.1441], "δMπ+": [0.0059, 0.0047, 0.0032, 0.0029]},
}

# Convert W to W^2 (since the W values are given in GeV)
for key, values in data.items():
    data[key]["W^2"] = np.array(values["W"]) ** 2

# Plotting the data
plt.figure(figsize=(8, 6))

# Plot each z bin
for key, values in data.items():
    plt.errorbar(values["W^2"], values["Mπ+"], yerr=values["δMπ+"], fmt='o-', label=f'{key}', capsize=3)

# Add labels and title
plt.xlabel(r'$W^2$ (GeV$^2$)', fontsize=14)
plt.ylabel(r'$M_{\pi^+}$', fontsize=14)
plt.title(r'Charged Pion Multiplicities $M_{\pi^+}$ vs $W^2$', fontsize=16)
plt.legend(title=r'$z$-bin')
plt.grid(True)
plt.show()
