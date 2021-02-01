"""
Estimates for HII region quantities
Just want to check some stuff while I (procrastinate) make my presentation for
UJC
Created: October 27, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np

import astropy.units as u

def RSO(Q0_49, nH_2, T_4):
    coeff = 9.77e18
    coeff *= (Q0_49)**(1./3)
    coeff *= (nH_2)**(-2./3)
    coeff *= (T_4)**(0.28)
    return coeff * u.cm


def tioniz(Q0_49, nH_2, T_4):
    top = (4.*np.pi/3) * RSO(Q0_49, nH_2, T_4)**3. * (nH_2*100 / u.cm**3)
    bottom = Q0_49*1e49/u.s
    return (top/bottom).to(u.yr)

def tioniz_book(Q0_49, nH_2, T_4):
    return 1.22e3*u.yr / nH_2

Q0_O75V = (10**(49 - 48.61))
Q0_O3V = (10**0.64)
Q0s = [Q0_O75V, Q0_O3V]

nHs = [0.03/100, 0.3/100, 0.03, 0.1, 1, 3]

Ts = [0.001, 1]


args = Q0_O75V, nHs[1], Ts[1]

x = RSO(*args)
print(f"{x.to(u.pc):.2f}")

x = tioniz(*args)
if x > 1e4*u.yr:
    print(f"{x:.2E}")
else:
    print(f"{x:.1f}")

x = tioniz_book(*args)
if x > 1e4*u.yr:
    print(f"{x:.2E}")
else:
    print(f"{x:.1f}")
