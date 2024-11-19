import math

def photon_photon_cm_energy(Q_e, Q_p, y_e, y_p, E_e, E_p):
    # Calculate the photon-photon center of mass energy squared (W^2)
    W_squared = (
        -Q_e**2 - Q_p**2 + 2 * (y_e * y_p * E_e * E_p - math.sqrt(y_e**2 * E_e**2 + Q_e**2) * math.sqrt(y_p**2 * E_p**2 + Q_p**2))
    )
    return W_squared

# Example usage:
Q_e = 1.0  # Virtuality of photon from electron (GeV^2)
Q_p = 0.5  # Virtuality of photon from proton (GeV^2)
y_e = 0.8  # Fraction of energy carried by photon from electron
y_p = 0.7  # Fraction of energy carried by photon from proton
E_e = 50  # Energy of incoming electron (GeV)
E_p = 7000  # Energy of incoming proton (GeV)

W_squared = photon_photon_cm_energy(Q_e, Q_p, y_e, y_p, E_e, E_p)
print(f"Photon-photon center of mass energy squared (W^2): {W_squared} GeV^2")
