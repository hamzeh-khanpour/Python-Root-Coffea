import numpy as np
import vegas
import matplotlib.pyplot as plt

# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / np.pi  # Fine structure constant divided by pi
emass = 0.000511  # Electron mass in GeV
pmass = 0.938272  # Proton mass in GeV
q2emax = 100000.0  # Maximum photon virtuality for electron in GeV^2
q2pmax = 100000.0  # Maximum photon virtuality for proton in GeV^2
hbarc2 = 0.389  # Conversion factor for cross-section to pb

# Sigma_{gamma_gamma} for tau tau
def cs_tautau_w_condition_Hamzeh(wvalue):
    mtau = 1.77686
    alpha2 = (1.0 / 137.0)**2

    beta = np.sqrt(np.where(1.0 - 4.0 * mtau**2 / wvalue**2 >= 0.0, 1.0 - 4.0 * mtau**2 / wvalue**2, np.nan))
    cs = np.where(wvalue > 2 * mtau, (4.0 * np.pi * alpha2 * hbarc2 / wvalue**2) * beta *
                  ((3.0 - beta**4) / (2.0 * beta) * np.log((1.0 + beta) / (1.0 - beta)) - 2.0 + beta**2), 0.0)
    return cs

# Photon Fluxes

def flux_y_pl(ye, q2max):
    if ye <= 0 or ye >= 1:
        return 0.0
    flux = ALPHA2PI / ye * (1 - ye + 0.5 * ye**2)
    return flux

def flux_y_dipole(yp, pmass, qmax2):
    if yp <= 0 or yp >= 1:
        return 0.0
    flux = ALPHA2PI / yp * (1 - yp)
    return flux

# Photon-Photon Flux Product
def flux_yy_atye(w, Y, qmax2e, qmax2p, eEbeam, pEbeam):
    yp = w * np.exp(Y) / (2.0 * pEbeam)
    ye = w * np.exp(-Y) / (2.0 * eEbeam)

    if yp <= 0.0 or yp >= 1.0 or ye <= 0.0 or ye >= 1.0:
        return 0.0

    flux_prod = cs_tautau_w_condition_Hamzeh(w) * flux_y_dipole(yp, pmass, qmax2p) * w * flux_y_pl(ye, emass, qmax2e)
    return flux_prod

# Differential Cross-Section dSigma/dY
def differential_cross_section_dY(Y, eEbeam, pEbeam, W_min, W_max):
    def vegas_integrand(x):
        W = W_min + x[0] * (W_max - W_min)
        flux = flux_yy_atye(W, Y, q2emax, q2pmax, eEbeam, pEbeam)
        return flux * (W_max - W_min)

    integrator = vegas.Integrator([[0, 1]])
    result = integrator(vegas_integrand, nitn=10, neval=10000)
    return result.mean if result.Q > 0.1 else 0.0

# Main Calculation
if __name__ == "__main__":
    eEbeam = 20.0  # Electron beam energy in GeV
    pEbeam = 7000.0  # Proton beam energy in GeV
    s_cms = 4.0 * eEbeam * pEbeam
    W_min = 10.0
    W_max = np.sqrt(s_cms)

    # Rapidities
    Y_values = np.linspace(-2.5, 2.5, 100)
    dSigma_dY = []

    for Y in Y_values:
        dSigma_dY.append(differential_cross_section_dY(Y, eEbeam, pEbeam, W_min, W_max))

    # Save results
    np.savetxt("dSigma_dY_tau_tau.txt", np.column_stack((Y_values, dSigma_dY)), header="Y dSigma/dY [pb]")

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(Y_values, dSigma_dY, label=r"$\frac{d\sigma}{dY}$ for $\tau^+\tau^-", linewidth=2)
    plt.xlabel(r"$Y$", fontsize=16)
    plt.ylabel(r"$\frac{d\sigma}{dY}$ [pb]", fontsize=16)
    plt.title("Differential Cross-Section for $\tau^+\tau^-$", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.savefig("dSigma_dY_tau_tau.pdf")
    plt.show()
