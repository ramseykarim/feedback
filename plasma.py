import numpy as np
from astropy import units as u
import scipy.constants as cst

"""
Using L. Townsley's 2019 RCW 49 X-ray emitting plasma fit params
with equations/notes from Townsley 2003 & 2011 to find electron density
"""


"""
Constants
"""

cm3_to_pc3 = u.cm.to(u.pc) ** 3 # ~ 3e-56

# Bolometric X-ray emissivity for cosmic abundance plasma at kT = 0.7 keV
# (Townsley 2003, based on CHIANTI)
emissivity = 5e-23 # erg cm3 s-1
# My own value from Figure 1, Schmitt & Ness 2004 (Coronal abundances...)
# emissivity = 0.3 * 1e-23 # erg cm3 s-1 ; this is even worse

"""
RCW 49 parameters
"""

# Townsley 2019 fit
kT = 0.27 # keV
logSEM = 55.9 # cm-3 pc-2
logSBtc = 32.88 # erg s-1 pc-2
A_outer = 16.5 # pc2
A_inner = 6.0 # pc2

"""
Equations
The following equations assume that electron density is constant
throughout the bubble.
"""

def electron_density_SEM(r):
    # r is bubble radius in pc
    # Returns cm-3
    SEM = 10.**logSEM
    return np.sqrt(SEM * cm3_to_pc3 / (2 * r))

def electron_density_L(r):
    # r same as above
    # Returns same as above
    SBtc = 10.**logSBtc
    return np.sqrt(cm3_to_pc3 * SBtc / (2 * r * emissivity))


def plasma_temperature():
    # Returns Kelvin
    return kT * 1000. / (cst.k * u.J.to(u.eV))


def thermal_pressure(r):
    # Assuming a mean molecular weight per electron of 0.62
    # (fully ionized cosmic abundance plasma, Townsley 2003)
    # Returns P/k in K cm-3
    return plasma_temperature() * 2.2 * electron_density_SEM(r)


def sphere_volume(r):
    # Volume of a sphere
    # r in pc, returns cm3
    return (4./3) * np.pi * r**3 / cm3_to_pc3


def thermal_energy(r):
    # Takes volume to be the entire spherical bubble
    # Returns ergs
    return (3./2) * thermal_pressure(r)*(cst.k * u.J.to(u.erg)) * sphere_volume(r)


def mass(r):
    # Assumes mu 0.62 again, and fully ionized
    # Returns solar masses
    return 0.62 * u.M_p.to(u.solMass) * electron_density_SEM(r) * sphere_volume(r)


if __name__ == "__main__":
    print("Physical properties of plasma in RCW 49")

    # FEEDBACK estimate of physical bubble radius
    bubble_radius = 8.5 # pc, based on ~7 min radius bubble at 4.21 kpc (Townsley's distance)
    # Based on Maitraiyee's bubble estimates
    bubble_radius_ang = 300 * u.arcsec
    bubble_radius = 4.21e3 * u.pc * bubble_radius_ang.to(u.rad).to_value()
    """
    Updated 4/13/20: new-and-improved assumptions
    Wd2-outer is 16.5 pc2, inner is 6.0 pc2. Townsley used contours to make these regions,
    so the outline of Wd2-outer should indicate the region of peak intensity.
    Thus, this region should be 16.5+6.0=22.5 pc2 in total.
    It's roughly circular, though not exactly (one moderate bump). We can
    approximate it as a circle.
    Using the projected circular area pi r^2 = 22.5 pc2, we will find r.
    This r will provide the volume. Assume full bubble and we can cut it down later.
    """
    # New method described above.
    A_total = (A_outer + A_inner) * u.pc*u.pc
    print(f"Total Wd2 peak emission region area: {A_total:.2f}")
    bubble_radius = np.sqrt(A_total / np.pi)

    print(f"Assuming {bubble_radius:.1f} radius bubble of constant density,\n fully ionized, cosmic abundance plasma.\n Assumes nH = ne for electron density.")
    bubble_radius = bubble_radius.to_value()
    print()

    print("Electron density:")
    print(f"From SEM (surface emission measure):  {electron_density_SEM(bubble_radius):.2E} cm-3")
    print(f"From SBtc (total surface brightness): {electron_density_L(bubble_radius):.2E} cm-3")

    print(f"This should be approximately unity: {10**(logSBtc - logSEM) / emissivity}")

    print("Since it's not unity, we use the SEM value.")
    print(f"Electron density = {electron_density_SEM(bubble_radius):.2f} cm-3")
    print()

    print(f"Plasma temperature = {plasma_temperature() / 1e6 : .2f} MK")
    print()

    print(f"Thermal pressure = {thermal_pressure(bubble_radius):.2E} K cm-3")
    print()

    fullsph = "(full sphere)"
    therm_E = thermal_energy(bubble_radius)
    print(f"Thermal energy = {therm_E:.2E} ergs {fullsph}")
    age = 2.e6*u.year
    print(f"Average wind power over {age:.1E} = {therm_E/(age.to(u.s).to_value()):.2E} ergs s-1")
    print()

    print(f"Mass of plasma = {mass(bubble_radius):.2f} solar masses {fullsph}")
