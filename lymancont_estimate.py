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

S843 = 34. # Jy

def tryD(d, T=8000.):
    print(f"Trying d={d:.1f} kpc")
    RCWLLy_me = solve_LLy(d, S843, T, 843.e6)
    RCWLLy_Hj = solve_LLy_Hj(d, S843, T, 843.e6)
    print(f"{RCWLLy_me:.2E} from my method")
    print(f"{RCWLLy_Hj:.2E} from Hjellming")

if __name__ == "__main__":
    T_try = 8e3
    print("Whiteoak and Uchida 1997 found 1.5 +/- 0.2 E+49 Lyman continuum photons s-1")
    tryD(2.3, T=T_try)
    tryD(4.2, T=T_try)
