"""
I am making several calculations regarding the cluster's IMF and total mass
Papers to reference are Ascenso 2007, Rauw 2007 & 2011, and Zeidler 2017
Created: June 30, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib.pyplot as plt

from . import misc_utils

nsamp = 1000

sample_gamma = lambda g, ge: np.random.normal(loc=g, scale=ge, size=(nsamp, 1))

gamma_A07 = 1.2
gamma_err_A07 = 0.16
gamma_samples_A07 = sample_gamma(gamma_A07, gamma_err_A07)

# gamma_Z17 = 1.03 # MS only; huge error bar
# gamma_err_Z17 = 0.22
gamma_Z17 = 1.46
gamma_err_Z17 = 0.06
gamma_samples_Z17 = sample_gamma(gamma_Z17, gamma_err_Z17)

def printout_error(samples):
    med = np.median(samples)
    err = np.array(misc_utils.flquantiles(samples.flatten(), 6)) - med
    print(med, err)



def calculate_xi0(Mtot, M1, M2, gamma):
    """
    Calc the IMF normalization factor xi_0
    Masses should be in solar masses (or at least self-consistent units)
    """
    top = (gamma - 1.) * Mtot
    pwr = lambda m : m**(1. - gamma)
    bottom = pwr(M1) - pwr(M2)
    return top/bottom

def calculate_Mtot(xi0, M1, M2, gamma):
    first = xi0 / (gamma - 1.)
    pwr = lambda m : m**(1. - gamma)
    second = pwr(M1) - pwr(M2)
    return first * second

xi0_samples_A07 = calculate_xi0(2809., 0.8, 11., gamma_samples_A07).T
xi0_samples_Z17 = calculate_xi0(2809., 0.8, 11., gamma_samples_Z17).T

print("xi_0 A07")
printout_error(xi0_samples_A07)
print("xi_0 Z17")
printout_error(xi0_samples_Z17)

print("A07 mass [10, 80]")
mtot_OB_samples = calculate_Mtot(xi0_samples_A07, 10, 80., gamma_samples_A07)
printout_error(mtot_OB_samples)
print("A07 mass [0.65, 80]")
mtot_OB_samples = calculate_Mtot(xi0_samples_A07, 0.65, 80., gamma_samples_A07)
printout_error(mtot_OB_samples)
print("A07 mass [1, 120]")
mtot_OB_samples = calculate_Mtot(xi0_samples_A07, 1, 120., gamma_samples_A07)
printout_error(mtot_OB_samples)
print("A07 mass [1, 80]")
mtot_OB_samples = calculate_Mtot(xi0_samples_A07, 1, 80., gamma_samples_A07)
printout_error(mtot_OB_samples)

print("A07 mass [1, 100]")
mtot_OB_samples = calculate_Mtot(xi0_samples_A07, 1, 100., gamma_samples_A07)
printout_error(mtot_OB_samples)

print()

print("Z17 mass [10, 80]")
mtot_OB_samples = calculate_Mtot(xi0_samples_Z17, 10, 80., gamma_samples_Z17)
printout_error(mtot_OB_samples)
print("Z17 mass [0.65, 80]")
mtot_OB_samples = calculate_Mtot(xi0_samples_Z17, 0.65, 80., gamma_samples_Z17)
printout_error(mtot_OB_samples)
print("Z17 mass [1, 120]")
mtot_OB_samples = calculate_Mtot(xi0_samples_Z17, 1, 120., gamma_samples_Z17)
printout_error(mtot_OB_samples)
print("Z17 mass [1, 80]")
mtot_OB_samples = calculate_Mtot(xi0_samples_Z17, 1, 80., gamma_samples_Z17)
printout_error(mtot_OB_samples)

print("Z17 mass [1, 100]")
mtot_OB_samples = calculate_Mtot(xi0_samples_Z17, 1, 100., gamma_samples_Z17)
printout_error(mtot_OB_samples)


"""
# Let's say we have everything above 10 solar masses, and that was like 1300
# Take upper limit to be 80 for WR20a components
mtot_OB = calculate_Mtot(xi0_A07, 10., 80., gamma_A07)

gamma_samples = np.random.normal(loc=gamma_A07, scale=gamma_err_A07, size=10000)

# mtot_OB_samples = calculate_Mtot(xi0_A07, 10., 80., gamma_samples)
mtot_OB_samples = calculate_Mtot(xi0_A07, .65, 80., gamma_samples)

mtot_OB_med = np.median(mtot_OB_samples)
mtot_OB_err = np.array(misc_utils.flquantiles(mtot_OB_samples, 6)) - mtot_OB_med
print(mtot_OB_med, mtot_OB_err)

plt.subplot(121)
plt.hist(gamma_samples)
plt.subplot(122)
plt.hist(mtot_OB_samples, bins=28)
plt.show()
"""
