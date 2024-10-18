# Elastic and Inelastic Photon-Photon Luminosity Spectrum at LHeC --- Hamzeh Khanpour October 2024

import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.integrate as integ



# Constants in GeV
ALPHA2PI = 7.2973525693e-3 / math.pi  # Fine structure constant divided by pi
emass = 5.1099895e-4   # Electron mass
pmass = 0.938272081    # Proton mass
pi0mass = 0.1349768    # Pion mass

q2emax = 10.0  # Maximum photon virtuality for electron in GeV^2
q2pmax = 10.0  # Maximum photon virtuality for proton in GeV^2
mNmax = 10.0  # Upper mass limit for inelastic interactions

# ALLM parameters -- arXiv:hep-ph/9712415
Mass2_0 = 0.31985
Mass2_P = 49.457
Mass2_R = 0.15052
Q2_0    = 0.52544
Lambda2 = 0.06527

Ccp = (0.28067, 0.22291,  2.1979)
Cap = (-0.0808, -0.44812, 1.1709)
Cbp = (0.36292, 1.8917,   1.8439)

Ccr = (0.80107, 0.97307, 3.4942)
Car = (0.58400, 0.37888, 2.6063)
Cbr = (0.01147, 3.7582,  0.49338)

# ALLM Form Factors Functions
# --------------------------------------------------------------


# ALLM Form Factors Functions
# --------------------------------------------------------------

def tvalue(Q2):
    return math.log \
           ((math.log((Q2 + Q2_0) / Lambda2) / math.log(Q2_0 / Lambda2)))

def xP(xbj, Q2):
    if xbj == 0:
        print("xbj zero")
        return -1.
    xPinv = 1. + Q2 / (Q2 + Mass2_P) * (1. / xbj - 1.)
    return 1. / xPinv

def xR(xbj, Q2):    
    if xbj == 0:
        print("xbj zero")
        return -1.
    xPinv = 1. + Q2 / (Q2 + Mass2_R) * (1. / xbj - 1.)
    return 1. / xPinv

def type1(tval, tuple1):
    return tuple1[0] + tuple1[1] * (tval ** tuple1[2])

def type2(tval, tuple1):
    return tuple1[0] +\
           (tuple1[0] - tuple1[1]) * (1. / (1. + tval ** tuple1[2]) - 1.)

def aP(tval):
    return type2(tval, Cap)

def bP(tval):
    return type1(tval, Cbp)

def cP(tval):
    return type2(tval, Ccp)

def aR(tval):
    return type1(tval, Car)

def bR(tval):
    return type1(tval, Cbr)

def cR(tval):
    return type1(tval, Ccr)

def allm_f2P(xbj, Q2):
    tval = tvalue(Q2) 
    return cP(tval) * (xP(xbj, Q2) ** aP(tval)) * ((1. - xbj) ** bP(tval))

def allm_f2R(xbj, Q2):
    tval = tvalue(Q2) 
    return cR(tval) * (xR(xbj, Q2) ** aR(tval)) * ((1. - xbj) ** bR(tval))

def allm_f2(xbj, Q2):
    return Q2 / (Q2 + Mass2_0) * (allm_f2P(xbj, Q2) + allm_f2R(xbj, Q2))


# --------------------------------------------------------------



def allm_f2divx_mN(mN, Q2, yp):

    A2 = mN*mN - pmass * pmass                                                            # Hamzeh
    mqdiff = mN*mN - pmass * pmass + Q2                                                   # Hamzeh
    
    if mqdiff < 0:
        print('mN*mN, Q2:', mN*mN, Q2)
        return 0.
    
    xbj = Q2 / mqdiff

    minM2 = (pmass + pi0mass) * (pmass + pi0mass)
    
    qmin2 = (mN*mN / (1.0 - yp) - pmass * pmass) * yp
    
    if xbj < 0:
        print('xbj: ', xbj)
        return 0.
    else:
        # 27 Jul 2021: adding Qmin2
        if qmin2 < Q2:
            return allm_f2(xbj, Q2) / Q2**0.0 * 2.0 * mN * mqdiff           # Hamzeh: It should be Q2**2.0 in Syy200.py
        else:
            return 0.

def allm_formM_mN2(Q2, yp, mMin2, mNmax):
    return integ.quad(allm_f2divx_mN, mMin2, mNmax, args=(Q2, yp),
                      epsrel=1.e-2)



# --------------------------------------------------------------



def allm_xf2_mN(mN, Q2, yp):
    
    A2 = mN*mN - pmass * pmass                                                           # Hamzeh
    mqdiff = mN*mN - pmass * pmass + Q2

    if mqdiff < 0:
        print('mN*mN, Q2:', mN*mN, Q2)
        return 0.

    xbj = Q2 / mqdiff

    minM2 = (pmass + pi0mass) * (pmass + pi0mass)
    
    qmin2 = (mN*mN / (1.0 - yp) - pmass * pmass) * yp

    if xbj < 0:
        print('xbj: ', xbj)
        return 0.
    else:
    
        if qmin2 < Q2:
            return allm_f2(xbj, Q2) / Q2**0.0  * 2.0 * mN *  (1.0/mqdiff) * ( 1.0 - qmin2 / Q2 )
        else:
            return 0.

def allm_formE_qmin2(Q2, yp, mMin2, mNmax):
    return integ.quad(allm_xf2_mN, mMin2, mNmax, args=(Q2, yp),
                      epsrel=1.e-2)


# --------------------------------------------------------------


# --------------------------------------------------------------

# Minimum Photon Virtuality
def qmin2(mass, y):
    return mass * mass * y * y / (1 - y)

# Elastic Form Factors (Dipole Approximation)
def G_E(Q2):
    return (1 + Q2 / 0.71) ** (-4)

def G_M(Q2):
    return 7.78 * G_E(Q2)

# Elastic Photon Flux from Electron
def flux_y_electron(ye, qmax2):
    if ye <= 0 or ye >= 1:
        return 0.0
    qmin2v = qmin2(emass, ye)
    y1 = 0.5 * (1.0 + (1.0 - ye) ** 2) / ye
    y2 = (1.0 - ye) / ye
    flux1 = y1 * math.log(qmax2 / qmin2v)
    flux2 = y2 * (1.0 - qmin2v / qmax2)
    return ALPHA2PI * (flux1 - flux2)

# Elastic Photon Flux from Proton
def flux_y_proton(yp, qmax2):
    if yp <= 0 or yp >= 1:
        return 0.0
    qmin2v = qmin2(pmass, yp)

    def integrand(lnQ2):
        Q2 = np.exp(lnQ2)
        gE2 = G_E(Q2)
        gM2 = G_M(Q2)
        formE = (4 * pmass ** 2 * gE2 + Q2 * gM2) / (4 * pmass ** 2 + Q2)
        formM = gM2
        flux_tmp = (1 - yp) * (1 - qmin2v / Q2) * formE + 0.5 * yp ** 2 * formM
        return flux_tmp * ALPHA2PI / (yp * Q2) * Q2  # Corrected integrand for change of variable

    try:
        result, _ = integrate.quad(integrand, math.log(qmin2v), math.log(qmax2), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for proton flux did not converge for yp={yp}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for proton flux: {e}")
        result = 0.0
    return result

# Inelastic Photon Flux from Proton
def flux_y_inel(yp, mMin2, qmax2, mNmax):
    if yp <= 0 or yp >= 1:
        return 0.0
    mMin2 = (pmass + pi0mass) * (pmass + pi0mass)  # Proton mass plus pion mass for minimum threshold
    
    qmin2v = (mMin2 / (1 - yp) - pmass * pmass) * yp

    def integrand(lnQ2):
        Q2 = np.exp(lnQ2)
        formE = allm_formE_qmin2(Q2, yp, mMin2, mNmax)[0]
        formMq2 = allm_formM_mN2(Q2, yp, mMin2, mNmax)[0]
        eps = 1e-12
        formMNew = formMq2 / (Q2 * Q2 + eps)
        formENew = formE
        flux_tmp = (1 - yp) * formENew + 0.5 * yp ** 2 * formMNew
        return flux_tmp * ALPHA2PI / yp

    try:
        # Split the integration range to improve convergence
        mid_lnQ2 = (math.log(qmin2v) + math.log(qmax2)) / 2
        result_1, _ = integrate.quad(integrand, math.log(qmin2v), mid_lnQ2, epsrel=1e-1)
        result_2, _ = integrate.quad(integrand, mid_lnQ2, math.log(qmax2), epsrel=1e-1)
        result = result_1 + result_2
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for inelastic proton flux did not converge for yp={yp}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for inelastic proton flux: {e}")
        result = 0.0

    return result

# Elastic Photon-Photon Luminosity Spectrum Calculation at Given W
def flux_el_yy_atW(W, eEbeam, pEbeam, qmax2e, qmax2p):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared
    ymin = W * W / s_cms

    def integrand(ye):
        yp = W * W / (s_cms * ye)
        if yp <= 0.0 or yp >= 1.0:
            return 0.0
        return flux_y_proton(yp, qmax2p) * yp * flux_y_electron(ye, qmax2e)

    try:
        result, _ = integrate.quad(integrand, ymin, 1.0, epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for elastic luminosity did not converge for W={W}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for elastic luminosity: {e}")
        result = 0.0
    return result * 2.0 / W

# Inelastic Photon-Photon Luminosity Spectrum Calculation at Given W
def flux_inel_yy_atW(W, eEbeam, pEbeam, qmax2e, qmax2p, mNmax):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared
    ymin = W * W / s_cms

    def integrand(ye):
        yp = W * W / (s_cms * ye)
        if yp <= 0.0 or yp >= 1.0:
            return 0.0
        return flux_y_inel(yp, pmass + pi0mass, qmax2p, mNmax) * yp * flux_y_electron(ye, qmax2e)

    try:
        result, _ = integrate.quad(integrand, ymin, 1.0, epsrel=1e-1)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for inelastic luminosity did not converge for W={W}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for inelastic luminosity: {e}")
        result = 0.0
    return result * 2.0 / W

# Parameters
eEbeam = 50.0  # Electron beam energy in GeV
pEbeam = 7000.0  # Proton beam energy in GeV
W_values = np.logspace(1.0, 3.0, 101)  # Range of W values from 10 GeV to 1000 GeV

# Calculate the Elastic and Inelastic Photon-Photon Luminosity Spectrum
elastic_luminosity_values = [flux_el_yy_atW(W, eEbeam, pEbeam, q2emax, q2pmax) for W in W_values]
elasticluminosity_at_W10 = flux_el_yy_atW(10.0, eEbeam, pEbeam, q2emax, q2pmax)

inelastic_luminosity_values = [flux_inel_yy_atW(W, eEbeam, pEbeam, q2emax, q2pmax, mNmax) for W in W_values]
inelasticluminosity_at_W10 = flux_el_yy_atW(10.0, eEbeam, pEbeam, q2emax, q2pmax)

print(f"Elastic Photon-Photon Luminosity Spectrum at W = 10.0 GeV: {elasticluminosity_at_W10:.6e} GeV^-1")
print(f"Inelastic Photon-Photon Luminosity Spectrum at W = 10.0 GeV: {inelasticluminosity_at_W10:.6e} GeV^-1")

# Plot the Results
plt.figure(figsize=(10, 8))
plt.xlim(10.0, 1000.0)
plt.ylim(1.e-7, 1.e-1)
plt.loglog(W_values, elastic_luminosity_values, linestyle='solid', linewidth=2, label='Elastic')
plt.loglog(W_values, inelastic_luminosity_values, linestyle='dashed', linewidth=2, label='Inelastic')
plt.xlabel(r"$W$ [GeV]", fontsize=18)
plt.ylabel(r"$S_{\gamma\gamma}$ [GeV$^{-1}$]", fontsize=18)
plt.title("Elastic and Inelastic Photon-Photon Luminosity Spectrum at LHeC", fontsize=20)
plt.grid(True, which="both", linestyle="--")
plt.legend(title=r'$Q^2_e < 10^5 \, \mathrm{GeV}^2, \, Q^2_p < 10^5 \, \mathrm{GeV}^2$', fontsize=14)
plt.savefig("photon_luminosity_spectrum.pdf")
plt.savefig("photon_luminosity_spectrum.jpg")
plt.show()
