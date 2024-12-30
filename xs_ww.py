# The `integrated_ww_cross_section` function computes the integrated mu-mu production cross-section 
# for a given threshold energy `W0` and electron/proton beam energies `eEbeam` and `pEbeam`.
#   - It defines an integrand combining mu-mu cross-section and the interpolated S_yy.
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


# mu-mu Production Cross-Section Calculation at Given W
def cs_ww_w(wvalue):

    re = 2.8179403262e-15 * 137.0 / 128.0
    me = 0.510998950e-3
    mw = 80.379
    hbarc2 =  0.389
    alpha2 = (1.0/128.0)*(1.0/128.0)

    #if wvalue > 2.0 * mw:
        #cs = (19.0/2.0) * np.pi * re * re * me * me / mw / mw \
             #* np.sqrt(wvalue * wvalue - 4.0 * mw * mw) / wvalue
    #elif wvalue > 300.0:
        #cs = 8.0 * np.pi * re * re * me * me / mw / mw
    #else:
        #cs = 0.0

    if wvalue > 2.0 * mw:
        cs = (19.0/2.0) * np.pi * hbarc2 * alpha2  / mw / mw \
             * np.sqrt(wvalue * wvalue - 4.0 * mw * mw) / wvalue         * 1e9
    elif wvalue > 300.0:
        cs = 8.0 * np.pi * hbarc2 * alpha2  / mw / mw                    * 1e9
    else:
        cs = 0.0

    return cs



# Integrated mu-mu Production Cross-Section from W_0 to sqrt(s_cms)
def integrated_ww_cross_section(W0, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared

    def integrand(W):
        # Get the mu-mu cross-section and S_yy value
        ww_cross_section = cs_ww_w(W)
        S_yy_value = S_yy_interp(W)

        # Skip integration if S_yy is zero to avoid contributing zero values
        if S_yy_value == 0.0:
            #print(f"Skipping W={W} due to S_yy=0")
            return 0.0

        return ww_cross_section * S_yy_value     # to calculates the integrated mu-mu cross-section

    try:
        result, _ = integrate.quad(integrand, W0, np.sqrt(s_cms), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for ww production cross-section did not converge for W_0={W0}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for ww production cross-section: {e}")
        result = 0.0
    return result



################################################################################


# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
mww = 80.379  # w mass in GeV 
W0_value = 2*mww  # GeV

# Calculate Integrated ww Production Cross-Section at W_0 = 10 GeV
integrated_cross_section_value = integrated_ww_cross_section(W0_value, eEbeam, pEbeam)
print(f"Integrated inelastic ww Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")

    
################################################################################


