# Elastic Photon-Photon Luminosity Spectrum at LHeC --- Updated Version

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass
pmass = 0.938272081    # Proton mass



q2emax = 100.0  # Maximum photon virtuality for electron in GeV^2 (matching your settings)
q2pmax = 100.0  # Maximum photon virtuality for proton in GeV^2 (matching your settings)



# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)



def G_M(Q2):
    return 7.78 * G_E(Q2)



# Minimum Photon Virtuality
def qmin2(mass, y):
    return (mass * y)**2 / (1 - y)



# Photon Flux from Electron
def flux_y_electron(ye, Q2e):
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2(emass, ye)
    if Q2e < qmin2v or Q2e > q2emax:
        return 0.0

    flux = ALPHA2PI / (ye * Q2e) * (
        (1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye**2
    )
    return flux



# Photon Flux from Proton
def flux_y_proton(yp, Q2p):
    if yp <= 0 or yp >= 1:
        return 0.0
    qmin2v = qmin2(pmass, yp)
    if Q2p < qmin2v or Q2p > q2pmax:
        return 0.0

    gE2 = G_E(Q2p)
    gM2 = G_M(Q2p)
    formE = (4 * pmass**2 * gE2 + Q2p * gM2) / (4 * pmass**2 + Q2p)
    formM = gM2

    flux = ALPHA2PI / (yp * Q2p) * (
        (1 - yp) * formE + 0.5 * yp**2 * formM
    )
    return flux



# Elastic Photon-Photon Luminosity Spectrum Calculation at Given W (Exact formula for W^2)
def flux_el_yy_atW(W, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared

    # Integration over ye from ye_min to ye_max (which is 1)
    ye_min = W**2 / s_cms

    def integrand(ye):
        Q2e_min = qmin2(emass, ye)
        Q2e_max = q2emax

        # Integration over Q2_e from Q2e_min to Q2e_max
        def Q2e_integrand(Q2e):
            yp_min = (W**2) / (s_cms)       # yp_min = (W**2 + Q2e) / (s_cms * ye)
            yp_max = 1.0

            # Integration over yp from yp_min to yp_max (which is 1)
            def proton_integrand(yp):
                Q2p_min = qmin2(pmass, yp)
                Q2p_max = q2pmax

                # Integration over Q2_p from Q2p_min to Q2p_max
                def Q2p_integrand(Q2p):
                    # Exact treatment of W^2 = ye * yp * s - Qe^2 - Qp^2
                    W2_exact = ye * yp * s_cms - Q2e - Q2p
                    if W2_exact <= 0.0:
                        return 0.0

                    # Calculate photon fluxes
                    flux_e = flux_y_electron(ye, Q2e)
                    flux_p = flux_y_proton(yp, Q2p)

                    return flux_e * flux_p

                try:
                    result_Q2p, _ = integrate.quad(Q2p_integrand, Q2p_min, Q2p_max, epsrel=1e-4)
                except integrate.IntegrationWarning:
                    print(f"Warning: Integration for proton flux Q2 did not converge for yp={yp}")
                    result_Q2p = 0.0
                except Exception as e:
                    print(f"Error during integration for proton flux Q2: {e}")
                    result_Q2p = 0.0

                return result_Q2p

            try:
                result_yp, _ = integrate.quad(proton_integrand, yp_min, yp_max, epsrel=1e-4)
            except integrate.IntegrationWarning:
                print(f"Warning: Integration for proton flux did not converge for ye={ye}")
                result_yp = 0.0
            except Exception as e:
                print(f"Error during integration for proton flux: {e}")
                result_yp = 0.0

            return result_yp

        try:
            result_Q2e, _ = integrate.quad(Q2e_integrand, Q2e_min, Q2e_max, epsrel=1e-4)
        except integrate.IntegrationWarning:
            print(f"Warning: Integration for electron flux Q2 did not converge for ye={ye}")
            result_Q2e = 0.0
        except Exception as e:
            print(f"Error during integration for electron flux Q2: {e}")
            result_Q2e = 0.0

        return result_Q2e

    try:
        result_ye, _ = integrate.quad(integrand, ye_min, 1.0, epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for elastic luminosity did not converge for W={W}")
        result_ye = 0.0
    except Exception as e:
        print(f"Error during integration for elastic luminosity: {e}")
        result_ye = 0.0

    return result_ye





# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV

W_values = np.logspace(1.0, 3.0, 101)  # Range of W values from 10 GeV to 1000 GeV



# Calculate the Elastic Photon-Photon Luminosity Spectrum
luminosity_values = [flux_el_yy_atW(W, eEbeam, pEbeam) for W in W_values]


# Plot the Results
plt.figure(figsize=(10, 8))
plt.xlim(10.0, 1000.0)
plt.ylim(1.e-7, 1.e-1)


plt.loglog(W_values, luminosity_values, linestyle='solid', linewidth=2, label='Elastic')


plt.xlabel(r"$W$ [GeV]", fontsize=18)


plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
plt.title("Elastic Photon-Photon Luminosity Spectrum at LHeC", fontsize=20)
plt.grid(True, which="both", linestyle="--")
plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)


plt.savefig("elastic_photon_luminosity_spectrum.pdf")
plt.savefig("elastic_photon_luminosity_spectrum.jpg")


plt.show()

