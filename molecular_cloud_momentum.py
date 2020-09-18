"""
Following a suggestion by Lee, I am making very rough calculations regarding
the acceleration of molecular clouds

Created: August 20, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as cst
import astropy.units as u


def estimate_total_acceleration():
    """
    Total shell acceleration
    We know the momentum transfer rate from stellar winds (see below)
    And we know the shell is ~1.2 x 10^5 solar masses
    How much acceleration is that? What's the average velocity?
    How much acceleration is it if it's even heavier, by a factor of 2 maybe,
    to account for the molecular clouds that appear to be swept up?
    """
    # Momentum flux
    OB_mvflux = 5.6e29 * u.dyn
    WR_mvflux = 2.83e29 * u.dyn
    tot_mvflux = 8.51e29 * u.dyn

    shell_mass = 1.2e5 * u.solMass

    wind_acceleration = (tot_mvflux / shell_mass).to(u.km / u.s / u.Myr)
    age = 2 * u.Myr
    def print_acc(acceleration):
        print(f"Acceleration: {acceleration:.3f}")
        print(f"Current velocity: {(acceleration*age).to(u.km/u.s):3f}")
        print(f"Average velocity: {(acceleration*age/2).to(u.km/u.s):.3f}")
        print(f"Distance over lifetime: {(acceleration*age*age/2).to(u.pc):.3f}")
        print()

    print("WIND ONLY:")
    print_acc(wind_acceleration)

    star_lum = 3.76E+06 * u.solLum # FUV
    star_lum = 1.12E+07 * u.solLum # Total
    fuv_mv = (star_lum / (cst.c * u.m / u.s)).to(u.dyn)
    fuv_acceleration = (fuv_mv / shell_mass).to(u.km / u.s / u.Myr)
    print(f"FUV momentum flux: {fuv_mv:.2E}")
    print("RADIATION ONLY:")
    print_acc(fuv_acceleration)
    print("WIND + RADIATION:")
    print_acc(wind_acceleration + fuv_acceleration)



def rough_estimate_1000():
    """
    (also see Aug 20, 2020 in my notes)
    The setup for my estimate is:
    Take the -3.7 km/s molecular cloud on the west side of RCW 49
    The cloud is probably about 1000 solar masses, but could be
    between 10^2 and 10^4

    The cloud can be approximated by a circle of 1.21 arcminute radius
    The outer shell of RCW 49 can be approximted by a sphere of 6 pc radius
    Take 4.16 kpc as the distance to RCW 49

    From my catalog work, I found momentum flux values of:
    OB: (5.6 \pm 0.4) * 10^29 dyn
    WR: (2.83 \pm 0.7) * 10^29 dyn
    Both: (8.51 \pm 0.75) * 10^29 dyn
    """
    distance_los = 4.16 * u.kpc

    # How much solid angle does the cloud subtend on the outer shell?
    cloud_radius = 1.21*u.arcmin
    cloud_radius = (cloud_radius * (distance_los / u.radian)).to(u.pc)
    cloud_area = np.pi * cloud_radius * cloud_radius
    shell_radius = 6 * u.pc
    shell_area = np.pi * shell_radius * shell_radius
    solid_angle_ratio = (cloud_area / shell_area).decompose()
    print(f"Cloud at outer shell subtends {solid_angle_ratio*4*np.pi*u.sr:.2f}")

    OB_mvflux = 5.6e29 * u.dyn
    WR_mvflux = 2.83e29 * u.dyn
    tot_mvflux = 8.51e29 * u.dyn

    cloud_mass = 1e4 * u.solMass

    shell_mass = 1.2e5 * u.solMass
    shell_mass_frac = shell_mass * solid_angle_ratio
    print(f"In addition to the {cloud_mass:.2E} molecular cloud, there is {shell_mass_frac:.2E} of shell mass in this solid angle")

    mvflux_frac = tot_mvflux * solid_angle_ratio
    acceleration = (mvflux_frac / (shell_mass_frac + cloud_mass)).to(u.km / u.s / u.Myr)
    print(acceleration)


estimate_total_acceleration()
