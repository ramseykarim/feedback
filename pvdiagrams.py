import numpy as np
import matplotlib.pyplot as plt
import sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube
import pvextractor


"""
Testing SpectralCube and pvextractor and making PV diagrams
Created: March 24th, 2020 (in the midst of the quarantine)
"""
__author__ = "Ramsey Karim"


fn = "../ancillary_data/sofia/rcw49-cii.fits"

with fits.open(fn) as hdul:
    h = hdul[0].header
    w = WCS(h)
    wflat = WCS(h, naxis=2)
    data = hdul[0].data * u.K / (u.m / u.s)
cube = SpectralCube(data=data, wcs=w)
subcube = cube.spectral_slab(-20*u.km/u.s, +20*u.km/u.s)
del data
mom0 = subcube.moment(order=0)

wr20b = SkyCoord(156.07666638433972*u.deg, -57.80826909707589*u.deg)

# coords = SkyCoord([p0, p1], unit=(u.hourangle, u.deg))
# p = pvextractor.Path(coords, width=1*u.arcmin)
angle = 0.
count = 0
plt.figure(figsize=(6, 4.5))
while angle < 360.:
    p = pvextractor.PathFromCenter(center=wr20b, length=10*u.arcmin, angle=angle*u.deg, width=1*u.arcsec)
    sl = pvextractor.extract_pv_slice(subcube, p)
    ax2 = plt.subplot(122, projection=WCS(sl.header))
    plt.imshow(sl.data, origin='lower')
    # ax.set_aspect(.1)
    ax1 = plt.subplot(121, projection=wflat)
    plt.imshow(mom0.to_value(), origin='lower', cmap='plasma')
    # ax1.plot(*(x.to(u.deg).to_value() for x in (p._coords.ra, p._coords.dec)), transform=ax1.get_transform('world'), linewidth=2, color='green')
    c0, c1 = p._coords[0], p._coords[1]
    ax1.arrow(*(x.to(u.deg).to_value() for x in (c0.ra, c0.dec)),
        *(x.to(u.deg).to_value() for x in ((c1.ra - c0.ra), (c1.dec - c0.dec))),
        color='green', transform=ax1.get_transform('world'), length_includes_head=True,
        width=0.003,
    )
    sys.stdout.write(f"\r{count:04d}")
    sys.stdout.flush()
    # plt.savefig(f"./figures/pv_anim/wr20b/im{count:04d}.png")
    # plt.clf()
    count += 1
    angle += 360.
print()
