"""
Check Maitraiyee's calculations, per her request
As response to Rolf's comments on the RCW 49 paper (near submission!)
Created: December 2, 2020
"""

import numpy as np
from scipy import constants as cst
from astropy import units as u

def get_constant(name):
    return cst.value(name) * u.Unit(cst.unit(name))

nu_13CO = 330587.965 * u.MHz
T_bg = 2.73 * u.K

h = get_constant('Planck constant')
k = get_constant('Boltzmann constant')

hv_k = (h * nu_13CO / k).to(u.K)
print("h = ", h)
print("k = ", k)
print("nu = ", nu_13CO)
print("h nu / k = ", hv_k)
print()
print("T_bg = ", T_bg)
bg_term = 1./(np.exp((hv_k / T_bg).decompose()) - 1)
print("background term: ", bg_term)


T_13CO = 2.2084 * u.K
T_ex = 14.5 * u.K
print("T_13CO = ", T_13CO)
print("T_ex = ", T_ex)
interior_A = (hv_k/T_13CO).decompose()
interior_B = (1./(np.exp((hv_k/T_ex).decompose()) - 1)) - bg_term
interior_inverted = 1./(interior_A * interior_B)
tau = -1. * np.log(1 - interior_inverted)
print("tau = ", tau.decompose())
