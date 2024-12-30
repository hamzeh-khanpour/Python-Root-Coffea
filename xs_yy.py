# The `integrated_yy_cross_section` function computes the integrated y-y production cross-section
# for a given threshold energy `W0` and electron/proton beam energies `eEbeam` and `pEbeam`.
#   - It defines an integrand combining y-y cross-section and the interpolated S_yy.
#   - If S_yy is zero at a given W, it skips that W value to avoid unnecessary computations.
#   - The result is integrated over W from W0 up to the maximum value set by `sqrt(s_cms)`.
#   - Integration warnings are caught, ensuring stable results.
#   - This function returns the integrated cross-section result in picobarns (pb).
# Hamzeh Khanpour, Laurent Forthomme, and Krzysztof Piotrzkowski -- November 2024


from scipy.integrate import quad
import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import ggMatrixElements  # Import your photon-photon matrix element module

# Load photon luminosity data from the text file
data = np.loadtxt('Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_10_using_vegas_FCC_he.txt', comments='#')
#data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_10_q2emax_100000_q2pmax_10_using_vegas_FCC_he.txt', comments='#')
#data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_50_q2emax_100000_q2pmax_1000_using_vegas_FCC_hetxt', comments='#')
#data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_300_q2emax_100000_q2pmax_100000_using_vegas_FCC_he.txt', comments='#')

W_data = data[:, 0]
S_yy_data = data[:, 1]


# Create an interpolation function for S_yy
S_yy_interp = interp1d(W_data, S_yy_data, kind='linear', bounds_error=False, fill_value=0.0)


# Function to calculate the gamma gamma production cross-section
##################################################################

# Photon-photon cross-section σ(γγ→γγ) using the matrix element

# Constants
alpha  = 1 / 137  # Fine-structure constant
hbarc2 = 0.389  # Conversion factor to pb

# Mandelstam variables
def t_min(W):
    return -W**2

def t_max(W):
    return 0


# Differential cross-section for gamma-gamma -> gamma-gamma using ggMatrixElements
def diff_cs_gg_to_gg(s, t):
    # Calculate the squared matrix element using ggMatrixElements
    sqme = ggMatrixElements.sqme_sm(s, t, False)  # s, t, exclude loops = False
    return sqme / (16. * np.pi * s**2.)  # The prefactor for 2-to-2 scattering



# Total cross-section for gamma-gamma -> gamma-gamma as a function of W
def cs_gg_to_gg_w(W):
    s = W**2.0                # s = W^2
    t_min_value = t_min(W)
    t_max_value = t_max(W)



# Numerical integration over t
    def integrand(t, s):
        return diff_cs_gg_to_gg(s, t)

    result, _ = quad(integrand, t_min_value, t_max_value, args=(s,))
    return result * hbarc2 * 1e9  # Convert to pb



# Integrated y-y Production Cross-Section from W_0 to sqrt(s_cms)
def integrated_yy_cross_section(W0, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared

    def integrand(W):
        # Get the y-y cross-section and S_yy value
        yy_cross_section = cs_gg_to_gg_w(W)
        S_yy_value = S_yy_interp(W)

        # Skip integration if S_yy is zero to avoid contributing zero values
        if S_yy_value == 0.0:
            #print(f"Skipping W={W} due to S_yy=0")
            return 0.0

        return yy_cross_section * S_yy_value     # to calculates the integrated y-y cross-section

    try:
        result, _ = integrate.quad(integrand, W0, np.sqrt(s_cms), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for yy production cross-section did not converge for W_0={W0}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for yy production cross-section: {e}")
        result = 0.0
    return result



################################################################################


# Parameters
eEbeam = 60.0  # Electron beam energy in GeV
pEbeam = 50000.0  # Proton beam energy in GeV
W0_value = 10.0  # GeV

# Calculate Integrated yy Production Cross-Section at W_0 = 10 GeV
integrated_cross_section_value = integrated_yy_cross_section(W0_value, eEbeam, pEbeam)
print(f"Integrated inelastic yy Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")


################################################################################


