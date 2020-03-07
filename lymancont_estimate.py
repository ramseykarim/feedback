import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst

"""
Reverse engineering some values from Hjellming 1968
This paper derives an expression for Lyman continuum flux based on
radio continuum flux density
This was used in Whiteoak & Uchida 1997 to estimate the required
Ly cont. flux for the WR20b bubble and compare that to the Ly cont.
flux for a WN7 like WR20b

The issue is that Whiteoak & Uchida assumed a distance of 2.3 kpc, which
has more recently been more or less overruled by a popular distance of ~4.2 kpc
(see Cantat-Gaudin via Townsley, or Zeidler)
Furthermore, the WN7 classification for WR20b is more popularly cited as WN6ha
(see probably a Rauw paper)
"""
__author__ = "Ramsey Karim"

# ALL UNITS CGS

a = 3e-39
def b(T, nu):
    return (17.72 + np.log(T**(3./2) / nu)) / np.sqrt(T)

# Hjellming 1968 parameter assumptions
T_Hj = 6000. # 6 kK
nu_Hj = 1400.e6 # 1400 MHz

b_Hj = b(T_Hj, nu_Hj)
numerical_factor_Hj = 1.014e-47
xssum = a * b_Hj / numerical_factor_Hj
def numerical_factor(T, nu):
    return a * b(T, nu) / xssum

# Full equation from Hjellming
def D2Snu(T, nu, LLy):    
    return numerical_factor(T, nu) * LLy


# Refactoring equation for LLy
def solve_LLy(D, S, T, nu):
    # D in kpc, S in Jy? ("flux units")
    D2S = D*D*S
    nf = numerical_factor(T, nu)
    return D2S / nf

def solve_LLy_Hj(D, S, *args):
    # Same units as above
    D2S = D*D*S
    return D2S / numerical_factor_Hj


"""
Some equations from Schraml & Mezger 1969 for useful properties
Most of these are from pg 23 of the PDF (291 of the paper)
"""
def theta_Gauss(ang_diameter_arcmin):
    # Gaussian half-power width of the HII region from its "spherical" angular diameter
    # Returns arcmin
    return ang_diameter_arcmin / 1.471


def a_factor(nu_GHz, Te):
    # Mezger and Henderson 1967 Appendix (A.2)
    # frequency must be in GHz
    # electron temperature in K
    #### I CAN CALCULATE MY OWN VALUE BASED ON DRAINE TO COMPARE
    gff = np.log(0.04995 / nu_GHz) + 1.5*np.log(Te) # Gaunt factor
    return 0.366 * (nu_GHz**0.1) * (Te ** -0.15) * gff


def calc_Ne(Te, nu_GHz, S, D, ang_diameter_arcmin):
    # Te in Kelvin
    # nu in GHz
    # S in Jy (flux units)
    # D in kpc
    # angular diameter in arcminutes
    thG = theta_Gauss(ang_diameter_arcmin)
    return 98.152 * a_factor(nu_GHz, Te)**(-0.5) * (nu_GHz**0.05) * (Te**0.175) * (S**0.5) * (D**-0.5) * (thG**-1.5)


def calc_MHII(Te, nu_GHz, S, D, ang_diameter_arcmin):
    # Same units as above
    He_mod = 1. # 1 + (NHe+ / NH+)
    thG = theta_Gauss(ang_diameter_arcmin)
    return 0.09954 * a_factor(nu_GHz, Te)**(-0.5) * (nu_GHz**0.05) * (Te**0.175) * (S**0.5) * (D**2.5) * (thG**1.5) / He_mod


def calc_EM(Te, nu_GHz, S, ang_diameter_arcmin):
    thG = theta_Gauss(ang_diameter_arcmin)
    return 4122.5 * a_factor(nu_GHz, Te)**(-1) * (nu_GHz**0.1) * (Te**0.35) * S * (thG**-2)

nu_GHz = 0.843

regions = {"Ring A": (110., np.nan), "Ring B": (34., 1.5e49),
    "Total": (210., 9.5e49)}

# Directly from WU97: 7.3 and 4.1. I think Ring B is larger, ~5
regions_ang_diam = {"Ring A": 7.3, "Ring B": 5, "Total": np.nan}

selected_region = "Ring B"

S843, WUf = regions[selected_region] # Jy
ang_diam = regions_ang_diam[selected_region]

def tryD(d, T=8000.):
    print(f"Trying d={d:.1f} kpc")
    RCWLLy_me = solve_LLy(d, S843, T, nu_GHz*1e9)
    RCWLLy_Hj = solve_LLy_Hj(d, S843, T, nu_GHz*1e9)
    print(f"{RCWLLy_me:.2E} ({np.log10(RCWLLy_me):.2f})  from my method")
    print(f"{RCWLLy_Hj:.2E} ({np.log10(RCWLLy_Hj):.2f}) from Hjellming")

if __name__ == "__main__":

    WU_dist = 2.3
    accepted_dist = 4.2

    T_try = 7600.
    print(f"In {selected_region}, Whiteoak and Uchida 1997 found {WUf:.2E} ({np.log10(WUf):.2f}) Lyman continuum photons s-1")
    tryD(WU_dist, T=T_try)
    tryD(accepted_dist, T=T_try)

    print()
    print("Using equations from Scraml & Mezger 1969:")
    d = accepted_dist
    ne = calc_Ne(T_try, nu_GHz, S843, d, ang_diam)
    M = calc_MHII(T_try, nu_GHz, S843, d, ang_diam)
    EM = calc_EM(T_try, nu_GHz, S843, ang_diam)
    print(f"n_e = {ne:.2f}")
    print(f"M_HII = {M:.2f}")
    print(f"EM = {EM:.1E}")
