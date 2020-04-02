import numpy as np
import matplotlib.pyplot as plt
import sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
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
wr20b_bubble = SkyCoord("10:24:18.7061 -57:48:14.937", unit=(u.hourangle, u.deg), frame=FK5)
wr20b_bubble_radius = 1.35967*u.deg

# coords = SkyCoord([p0, p1], unit=(u.hourangle, u.deg))
# p = pvextractor.Path(coords, width=1*u.arcmin)


def cosine_pv(displacement_along_path, radius, expansion_velocity,
    path_angular_displacement=0.0, base_velocity=0.0, n_points=50):
    """
    Return (x_array, y_array) for a cosine curve, ready to overlay on a PV diagram.
    This will return exactly one period of a cosine curve, peaking in the center.
    If path_angular_displacement is not 0, there will be some zeros on either
    side of the returned cosine indicating the unseen full radius of the shell.
    The PV diagram is defined along a spatial "path".
    :param displacement_along_path: (presumably angular) distance ALONG path at
        which the center of the cosine occurs.
    :param radius: the full radius of the sphere. Only equal to the width of
         the returned cosine if path_angular_displacement is 0.0.
    :param expansion_velocity: the peak expansion velocity if the expanding
        sphere is viewed at its CENTER. The expansion_velocity is only equal to
        the peak expansion velocity of this cosine curve if
        path_angular_displacement is 0.0.
    :param path_angular_displacement: the angle (deg) from center at which the path
        cuts across the sphere. Angle of 0 implies the path crosses the actual
        center of the sphere, while -90 or 90 implies that the path lies tangent
        to the visible sphere, grazing the limb brightened edge.
        Should be -90 <= path_angular_displacement <= 90 or equivalent.
    :param base_velocity: the velocity of the center of the expanding sphere.
        Added to the y_value at the end.
    :param n_points: number of points in each return array
    """
    # Sample n_points
    x_array = np.linspace(displacement_along_path - radius, displacement_along_path + radius, n_points)
    # Radius is modified by the path_angular_displacement since the path is off-axis
    # See pg. 83 in notebook (3/30/2020) for explanation
    modified_radius = radius * np.sin(np.pi/2 - np.deg2rad(path_angular_displacement))
    # This radius becomes +/- pi/2 from displacement_along_path
    cosine_thru_center = np.cos((x_array-displacement_along_path)*np.pi/2 / modified_radius)
    # Modify with path_angular_displacement, converted to radians
    y_array =  np.cos(np.deg2rad(path_angular_displacement)) * cosine_thru_center
    y_array[np.isnan(y_array)] = 0.
    return x_array, y_array * expansion_velocity + base_velocity


def rotate_around(center):
    angle = 0.
    count = 0
    plt.figure(figsize=(6, 4.5))
    while angle < 360.:
        p = pvextractor.PathFromCenter(center=center, length=5*u.arcmin, angle=angle*u.deg, width=2*u.arcsec)

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
        plt.savefig(f"./figures/pv_anim/wr20b_incoming/im{count:04d}.png")
        plt.clf()
        count += 1
        angle += 10.
    print()

def horizontal_series_thru(center):
    series_length = 10*u.arcmin
    stepwidth = 0.25*u.arcmin/np.cos(center.dec)
    start_ra = center.ra - (series_length/2.)/np.cos(center.dec)
    end_ra = center.ra + (series_length/2.)/np.cos(center.dec)
    current_position = SkyCoord(start_ra, center.dec)
    angle = 0.
    count = 0
    plt.figure(figsize=(6, 4.5))
    while current_position.ra < end_ra:
        p = pvextractor.PathFromCenter(center=current_position, length=5*u.arcmin, angle=angle*u.deg, width=2*u.arcsec)
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
        plt.savefig(f"./figures/pv_anim/wr20b_horiz/im{count:04d}.png")
        plt.clf()
        count += 1
        current_position = SkyCoord(current_position.ra + stepwidth, current_position.dec)
    print()

def vertical_series_thru(center):
    series_length = 10*u.arcmin
    stepwidth = 0.10*u.arcmin
    start_dec = center.dec - series_length/2.
    end_dec = center.dec + series_length/2.
    current_position = SkyCoord(center.ra, start_dec)
    angle = 270.
    count = 0
    plt.figure(figsize=(6, 4.5))
    while current_position.dec < end_dec:
        p = pvextractor.PathFromCenter(center=current_position, length=5*u.arcmin, angle=angle*u.deg, width=4*u.arcsec)
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
        plt.savefig(f"./figures/pv_anim/wr20b_vert_finer/im{count:04d}.png")
        plt.clf()
        count += 1
        current_position = SkyCoord(current_position.ra, current_position.dec + stepwidth)
    print()


def overlay_cosine():
    center = wr20b
    # Horizontal length of EACH HORIZONTAL PV CUT
    pathlength = 5*u.arcmin
    # Vertical length of the SEIRES OF HORIZONTAL CUTS
    series_length = 10*u.arcmin
    stepwidth = 0.25*u.arcmin
    start_dec = center.dec - series_length/2.
    end_dec = center.dec + series_length/2.

    n_image = 16
    current_position = SkyCoord(center.ra, start_dec + n_image*stepwidth, frame=FK5)
    dec_displacement_from_bubble_center = (wr20b_bubble.dec - current_position.dec).deg
    ang_displacement_from_bubble_center = 90 - np.rad2deg(np.arccos(dec_displacement_from_bubble_center / wr20b_bubble_radius.to_value()))
    print(ang_displacement_from_bubble_center) # TODO: what is going on with this results? plot circle and see if it makes sense

    angle = 270.
    count = 16
    plt.figure(figsize=(6, 4.5))
    p = pvextractor.PathFromCenter(center=current_position, length=pathlength, angle=angle*u.deg, width=2*u.arcsec)

    c0, c1 = p._coords[0], p._coords[1]
    ra_displacement_bubble_center_from_left_edge = (c0.ra - wr20b_bubble.ra).deg * np.cos(current_position.dec)
    print(ra_displacement_bubble_center_from_left_edge, dec_displacement_from_bubble_center)
    return
    sl = pvextractor.extract_pv_slice(subcube, p)
    ax2 = plt.subplot(122, projection=WCS(sl.header))
    plt.imshow(sl.data, origin='lower')

    cosx, cosy = cosine_pv(0.032, 0.025, 5700., base_velocity=8000.)
    ax2.plot(cosx, cosy, color='r', linewidth=4, transform=ax2.get_transform('world'), alpha=0.3)
    # cosx, cosy = cosine_pv(0.032, 0.025, 5700., base_velocity=8000.)
    # ax2.plot(cosx, cosy, color='r', linewidth=4, transform=ax2.get_transform('world'), alpha=0.3)

    # ax.set_aspect(.1)
    ax1 = plt.subplot(121, projection=wflat)
    plt.imshow(mom0.to_value(), origin='lower', cmap='plasma')
    ax1.plot([wr20b.ra.to_value()], [wr20b.dec.to_value()], color='blue', marker='*', markersize=8, transform=ax1.get_transform('world'), alpha=0.3)
    # ax1.plot(*(x.to(u.deg).to_value() for x in (p._coords.ra, p._coords.dec)), transform=ax1.get_transform('world'), linewidth=2, color='green')
    ax1.arrow(*(x.to(u.deg).to_value() for x in (c0.ra, c0.dec)),
        *(x.to(u.deg).to_value() for x in ((c1.ra - c0.ra), (c1.dec - c0.dec))),
        color='green', transform=ax1.get_transform('world'), length_includes_head=True,
        width=0.003,
    )
    plt.show()



overlay_cosine()
