import numpy as np

import scipy.constants as cst
from astropy import units as u

"""
Re-doing calculations from Paladini et al 2015, excluding region B
Calculations are for Te and ne in hot gas in RCW 49 based on H109alpha line
"""


# Continuum temperature, mJy/beam
Tc = 61. # Based on their own result of Te > 9740
# Line temperature, upper limit, mJy/beam
Tl = 3 * 2.6
# delta V km s-1
dV_regB = 8.2
dV = 10.8
nu_GHz = 5.
line_of_sight_distance_pc = 6.1*2. # Paladini used 1 pc, but ~17 pc seems closer to reality
# line_of_sight_distance_pc = 1.

theta_beam = 7.3 # arcsec at 5 GHz, from Table 3 in Paladini (maj == min)


def convert_mJybeam_to_Kelvin(mJybeam, full_eq=True):
    if full_eq:
        # Equations below 3G4 from
        # https://www.cv.nrao.edu/course/astr534/Interferometers2.html
        # Checked with their numbers and got their result
        theta = theta_beam*u.arcsec.to(u.rad)
        beam_sr = np.pi * theta**2 / (4 * np.log(2))
        S_per_sr = mJybeam * 1e-3 * 1e-26 / beam_sr
        c2kv2 = (cst.c**2) / (2 * cst.k * (nu_GHz*1e9)**2)
        return S_per_sr * c2kv2
    else:
        # Convert a brightness temperature from mJy/beam to Kelvin
        # Using the last equation from
        # https://science.nrao.edu/facilities/vla/proposing/TBconv
        coefficient = 1.222e3 # bunch of unit conversions and constants
        return coefficient * mJybeam / ((nu_GHz * theta_beam)**2)



def calc_Te(deltaV):
    # From Caswell and Haynes 1987 originally, first equation
    # 5 GHz specific, Ne=NH (no He), LTE only
    # Approximation (with above) based on Shaver et al 1983 Eq 1
    # This equation reproduced in Paladini 2015 Eq 1
    # deltaV in km s-1
    return 10150. * (Tl * deltaV / Tc)**(-0.87)

def calc_tauC(Te):
    Tc_K = convert_mJybeam_to_Kelvin(Tc)
    print(f"--- Tc = {Tc_K:.3f} K")
    # return  Tc_K / Te
    return -1 * np.log(1 - Tc_K/Te)


"""
All the different Gaunt factor / EM equations
"""

def calc_EM_Paladini(tauC, Te):
    # From Paladini 2015
    # tauC is continuum
    # This equation resembles Draine Eq 10.23 (in the exponents)
    return (tauC * Te**1.35 * nu_GHz**2.1) / 8.24e-2


def calc_gFF_Abitbol(Te):
    # From Abitbol 2018
    first = np.log(0.04955 / nu_GHz)
    second = 1.5 * np.log(Te)
    return first + second


def calc_gFF_Draine(Te):
    # Draine 2011 (book) Eq 10.9
    # Kept a close eye on the units
    Zi = 1. # charge per ion, assume H only
    inside_exp = (np.sqrt(3)/np.pi) * np.log(Zi * nu_GHz * (Te/1e4)**(-3./2))
    inside_log = np.exp(5.960 - inside_exp)
    return np.log(inside_log + np.e)


gFF_methods = {'Draine': calc_gFF_Draine, 'Abitbol': calc_gFF_Abitbol}


def calc_gFF(Te, method='Draine'):
    return gFF_methods[method](Te)


def calc_EM_Abitbol(tauFF, Te):
    # From Abitbol 2018
    # tauFF is continuum, assumed to be free-free
    top = tauFF * Te**.15 * nu_GHz**2
    bottom = 0.0314 * calc_gFF(Te, method='Abitbol')
    return top / bottom


def calc_EM_Draine(tauFF, Te):
    # Based on Draine Eq 10.22, appears to be identical to Planck 2014
    # Solved for EM and units shuffled around
    # The coefficient equals 5.468e-2, coefficient in Planck 2014 Eq 4
    coefficient = 1.772 * 3.086 * 1e-2 # 1.772 from Draine, 3.086 from cm/pc, includes T4->T
    top = tauFF * Te**(1.5) * nu_GHz**2.
    bottom = calc_gFF(Te, method='Draine') * coefficient
    # Return value is EM [pc cm-6]
    return top / bottom


EM_methods = {'Draine': calc_EM_Draine, 'Abitbol': calc_EM_Abitbol, 'Paladini': calc_EM_Paladini}


def calc_EM(tauFF, Te, method='Draine'):
    # Returns pc cm-6 (definitely if you use Draine)
    return EM_methods[method](tauFF, Te)


"""
And finally, a very simple and hopefully correct expression for electron density
"""

def calc_ne(EM):
    # From Paladini 2015 Eq 5
    # EM should be in pc cm-6
    # distance should be in pc
    # Returns cm-3
    return np.sqrt(EM/line_of_sight_distance_pc)


# def calc_ne_full(Te):
#     tauC = calc_tauC(Te)
#     EM = calc_EM(tauC, Te)
#     return calc_ne(EM)

print(f"Paladini et al 2015 used delta V = {dV_regB:.1f} km/s")
Te_regB = calc_Te(dV_regB)
tauC_regB = calc_tauC(Te_regB)
EM_regB = calc_EM(tauC_regB, Te_regB)
ne_regB = calc_ne(EM_regB)
print(f"They found:")
print(f"Te = {Te_regB:.1f} K")
print(f"tauC = {tauC_regB:.4f}")
print(f"EM = {EM_regB:.1E} pc cm-6")
print(f"ne = {ne_regB:.3f} cm-3")
print()
print(f"Excluding region B, delta V = {dV:.1f} km/s")
Te = calc_Te(dV)
tauC = calc_tauC(Te)
EM = calc_EM(tauC, Te)
ne = calc_ne(EM)
print(f"I find:")
print(f"Te = {Te:.1f} K")
print(f"tauC = {tauC:.4f}")
print(f"Gaunt = {calc_gFF(Te):.2f} (If Abitbol: {calc_gFF(Te, method='Abitbol'):.2f})")
print(f"EM = {EM:.1E} pc cm-6")
print(f"ne = {ne:.3f} cm-3")

