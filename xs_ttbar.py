# The `integrated_t_tbar_cross_section` function computes the integrated tau-tau production cross-section 
# for a given threshold energy `W0` and electron/proton beam energies `eEbeam` and `pEbeam`.
#   - It defines an integrand combining tau-tau cross-section and the interpolated S_yy.
#   - If S_yy is zero at a given W, it skips that W value to avoid unnecessary computations.
#   - The result is integrated over W from W0 up to the maximum value set by `sqrt(s_cms)`.
#   - Integration warnings are caught, ensuring stable results.
#   - This function returns the integrated cross-section result in picobarns (pb).
# Hamzeh Khanpour, Laurent Forthomme, and Krzysztof Piotrzkowski -- November 2024


import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Load photon luminosity data from the text file
#data = np.loadtxt('Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_10_using_vegas_tagged_elastic.txt', comments='#')
#data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_10_q2emax_100000_q2pmax_10_using_vegas.txt', comments='#')
#data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_50_q2emax_100000_q2pmax_1000_using_vegas.txt', comments='#')
data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_300_q2emax_100000_q2pmax_100000_using_vegas.txt', comments='#')

W_data = data[:, 0]
S_yy_data = data[:, 1]


# Create an interpolation function for S_yy
S_yy_interp = interp1d(W_data, S_yy_data, kind='linear', bounds_error=False, fill_value=0.0)


# Tau-Tau Production Cross-Section Calculation at Given W
def cs_ttbar_w_condition_Hamzeh(wvalue):  # Eq.62 of Physics Reports 364 (2002) 359-450
    mtop = 172.50
    Qf = 2.0/3.0
    Nc = 3.0
    hbarc2 =  0.389
    alpha2 = (1.0/137.0)*(1.0/137.0)

    # Element-wise calculation of beta using np.where
    beta = np.sqrt(np.where(1.0 - 4.0 * mtop * mtop / wvalue**2.0 >= 0.0, 1.0 - 4.0 * mtop * mtop / wvalue**2.0, np.nan))

    # Element-wise calculation of cs using np.where
    cs = np.where(wvalue > mtop, ( 4.0 * np.pi * alpha2 * Qf**4.0 * Nc * hbarc2 ) / wvalue**2.0 * (beta) * \
             ( (3.0 - (beta**4.0))/(2.0 * beta) * np.log((1.0 + beta)/(1.0 - beta)) - 2.0 + beta**2.0 ), 0.0) * 1e9

    return cs



# Integrated Tau-Tau Production Cross-Section from W_0 to sqrt(s_cms)
def integrated_t_tbar_cross_section(W0, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared

    def integrand(W):
        # Get the tau-tau cross-section and S_yy value
        t_tbar_cross_section = cs_ttbar_w_condition_Hamzeh(W)
        S_yy_value = S_yy_interp(W)

        # Skip integration if S_yy is zero to avoid contributing zero values
        if S_yy_value == 0.0:
            #print(f"Skipping W={W} due to S_yy=0")
            return 0.0

        return t_tbar_cross_section * S_yy_value     # to calculates the integrated t_tbar cross-section

    try:
        result, _ = integrate.quad(integrand, W0, np.sqrt(s_cms), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for t_tbar production cross-section did not converge for W_0={W0}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for t_tbar production cross-section: {e}")
        result = 0.0
    return result



################################################################################


# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
mtop = 172.50  # Top quark mass in GeV
W0_value = 2*mtop  # GeV

# Calculate Integrated t_tbar Production Cross-Section at W_0 = 10 GeV
integrated_cross_section_value = integrated_t_tbar_cross_section(W0_value, eEbeam, pEbeam)
print(f"Integrated inelastic t_tbar Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")

    
################################################################################


