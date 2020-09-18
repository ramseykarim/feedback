"""
Toy model to see what radiation pressure looks like when taking into account
the absorption curve of dust.
On recommendation of Lee to make more toy models.
See also: calculations in molecular_cloud_momentum.py

Created: August 29, 2020
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from scipy import constants as cst

from . import dust_mass as dust_mass_functions
from . import catalog
from .mantipython.physics.greybody import B as mpy_B

def setup_dust():
    """
    1/(1-a) = (ext/abs)
    abs = ext(1-a)
    abs - ext = -ext*a
    1 - (abs/ext) = a

    a = 0: all absorption, black body
    a = 1: all scattering
    """
    Rv = 3.1
    d = dust_mass_functions.Draine_data(Rv)
    wl = dust_mass_functions.get_wl(d) * u.micron
    kabs = dust_mass_functions.get_k(d) #* u.cm * u.cm / u.g
    Cext = dust_mass_functions.get_C(d) #* u.cm * u.cm
    a = dust_mass_functions.get_albedo(d)
    R = dust_mass_functions.Draine_gastodust(Rv)
    # Get the gas-to-H ratio
    kabs160 = dust_mass_functions.get_val_at(160., wl.to_value(), kabs)
    Cabs160 = dust_mass_functions.convert_ktoC(kabs160, R)
    Cext160 = dust_mass_functions.get_val_at(160., wl.to_value(), Cext)
    # Finish doing this later
    return {'kabs': kabs, 'Cext': Cext, 'Rv': Rv, 'wl': wl, 'a': a, 'R': R}

# I also need to pull in a stellar spectrum
# So i need to interpolate the dust absorption
# first just use a black body at like 25 kK, early B star

def planck_function(wl, T, rstar, Rstar):
    """
    wl is Quantity, T is in K
    rstar: star radius
    Rstar: distance to star
    both radii as Quantity
    """
    nu = wl.to(u.Hz, equivalencies=u.spectral()).to_value()
    Bv = mpy_B(nu, T) * u.MJy / u.sr
    """
    There's something about solid angle I need to deal with but I'm not certain what it is
    Momentum is directional, so solid angle definitely matters
    I think I figured out the solid angle thing, I just need:
        rstar: star radius
        Rstar: distance to star
    see notes from 8-31-20
    """
    solid_angle_of_star = (np.pi * rstar**2 / Rstar**2).decompose() * u.sr
    Fv = Bv * solid_angle_of_star
    # Energy / area / frequency / time (some kind of flux)
    return Fv


# what was I going to write here? column density to absorption?
def tau(Cext, N_H):
    """
    Cext in Draine units (cm^2/H)
    N(H) is H nucleons per cm^2 (H/cm^2)
    """
    return Cext * N_H


def assume_albedo_0():
    dust_info = setup_dust()
    Cext = dust_info['Cext']
    wl = dust_info['wl']
    e_axis = wl.to(u.eV, equivalencies=u.spectral()).to_value()
    N_H = 1e21 # cm-2
    t = tau(Cext, N_H)
    plt.subplot(131)
    absorption_fraction = (1 - np.exp(-t))
    plt.plot(e_axis, absorption_fraction)
    plt.xscale('log')
    # plt.yscale('log')
    # plt.xlim([1, 25])

    plt.subplot(133)
    Fv = planck_function(wl, 25e3, 6.2*u.solRad, 5*u.pc)
    Pv = (Fv / (cst.c * u.m / u.s)).to(u.dyn / (u.cm*u.cm) / u.Hz)
    plt.plot(e_axis, Pv.to_value())
    plt.xscale('log')
    # plt.xlim([1, 25])

    plt.subplot(132)
    plt.plot(e_axis, (Pv.to_value()*absorption_fraction))
    plt.xscale('log')

    total_pressure = np.trapz(Pv.to_value(), x=wl.to(u.Hz, equivalencies=u.spectral()).to_value())
    print(total_pressure)


    # plt.show()

assume_albedo_0()
