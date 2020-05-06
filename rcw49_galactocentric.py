"""
Calculating the galactocentric distance to RCW 49
Maitraiyee needed this number to make some other calculations
Created: unsure, can check emails
"""

import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u

l = 284.3 # RCW 49 galactic longitude
b = -0.3 # RCW 49 galactic latitude
d_rcw49 = 4.16 # RCW 49 heliocentric distance
R0 = 8. # Galactocentric distance of the Sun

# # Via astropy only
# rcw49_coord = SkyCoord(l=l*u.deg, b=b*u.deg, distance=d*u.kpc, frame='galactic')
# print(rcw49_coord.galactocentric.galcen_distance)


# Via Brand and Blitz 1993
cos = lambda x: np.cos(np.deg2rad(x))

def galactocentric(d):
    Rsq = (d * cos(b))**2 + R0**2 - 2*R0*d*cos(b)*cos(l)
    R = np.sqrt(Rsq)
    print(f"{R:.2f} kpc")

print("2 kpc: ", end="")
galactocentric(2)
print("4.16 kpc: ", end="")
galactocentric(4.16)
print("8 kpc: ", end="")
galactocentric(8)
