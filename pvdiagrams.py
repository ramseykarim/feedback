import numpy as np
import matplotlib.pyplot as plt
import sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.nddata.utils import Cutout2D
from spectral_cube import SpectralCube
import pvextractor

from . import misc_utils
from . import catalog


"""
Testing SpectralCube and pvextractor and making PV diagrams
Created: March 24th, 2020 (in the midst of the quarantine)
"""
__author__ = "Ramsey Karim"


# RCW 49
fn = f"{catalog.utils.ancillary_data_path}sofia/rcw49-cii.fits"
fn = f"{catalog.utils.ancillary_data_path}apex/apexCO/RCW49_12CO.fits"
# M16
fn = f"{catalog.utils.m16_data_path}apex/M16_12CO3-2.fits"
fn = f"{catalog.utils.m16_data_path}sofia/M16_CII_U.fits"
fn = f"{catalog.utils.m16_data_path}bima/M16_12CO1-0_7x4.fits"

with fits.open(fn) as hdul:
    h = hdul[0].header
    w = WCS(h)
    wflat = WCS(h, naxis=2)
    data = hdul[0].data * u.K / (u.m / u.s)
cube = SpectralCube(data=data, wcs=w)

subcube = cube.spectral_slab(-30*u.km/u.s, +30*u.km/u.s)
del data
mom0 = subcube.moment(order=0)



"""
Star positions, bubble centers, etc.
"""
wr20b = SkyCoord(156.07666638433972*u.deg, -57.80826909707589*u.deg)
wr20b_bubble = SkyCoord("10:24:18.7061 -57:48:14.937", unit=(u.hourangle, u.deg), frame=FK5)
wr20a = SkyCoord("10:23:58.0545 -57:45:48.862", unit=(u.hourangle, u.deg), frame=FK5) # approx from ds9, not SIMBAD or anything
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


def rotate_around(center, length=5, width=0.1):
    angle = 0.
    count = 0
    plt.figure(figsize=(12, 9))
    while angle < 180.:
        p = pvextractor.PathFromCenter(center=center, length=length*u.arcmin, angle=angle*u.deg, width=width*u.arcmin)

        sl = pvextractor.extract_pv_slice(subcube, p)
        ax2 = plt.subplot(122, projection=WCS(sl.header))
        plt.imshow(sl.data, origin='lower')
        # ax.set_aspect(.1)
        ax1 = plt.subplot(121, projection=wflat)
        plt.imshow(mom0.to_value(), origin='lower', cmap='plasma')
        plt.colorbar()
        # ax1.plot(*(x.to(u.deg).to_value() for x in (p._coords.ra, p._coords.dec)), transform=ax1.get_transform('world'), linewidth=2, color='green')
        c0, c1 = p._coords[0], p._coords[1]
        ax1.arrow(*(x.to(u.deg).to_value() for x in (c0.ra, c0.dec)),
            *(x.to(u.deg).to_value() for x in ((c1.ra - c0.ra), (c1.dec - c0.dec))),
            color='green', transform=ax1.get_transform('world'), length_includes_head=True,
            width=linewidth_from_data_units((width*u.arcmin).to(u.deg).to_value(), ax1, reference='y'),
        )
        sys.stdout.write(f"\r{count:04d}")
        sys.stdout.flush()
        plt.savefig(f"./figures/pv_anim/rcw49_rot/im{count:04d}.png")
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


def overlay_cosine(center, bubble_radius):
    """
    overlay_cosine(wr20b, wr20b_bubble_radius)
    """
    # Horizontal length of EACH HORIZONTAL PV CUT
    pathlength = 5*u.arcmin
    # Vertical length of the SEIRES OF HORIZONTAL CUTS
    series_length = 10*u.arcmin
    stepwidth = 0.25*u.arcmin
    start_dec = center.dec - series_length/2.
    end_dec = center.dec + series_length/2.

    n_image = 16
    current_position = SkyCoord(center.ra, start_dec + n_image*stepwidth, frame=FK5)
    dec_displacement_from_bubble_center = (center.dec - current_position.dec).deg
    ang_displacement_from_bubble_center = 90 - np.rad2deg(np.arccos(dec_displacement_from_bubble_center / bubble_radius.to_value()))
    print(ang_displacement_from_bubble_center) # TODO: what is going on with this results? plot circle and see if it makes sense

    angle = 270.
    count = 16
    plt.figure(figsize=(6, 4.5))
    p = pvextractor.PathFromCenter(center=current_position, length=pathlength, angle=angle*u.deg, width=2*u.arcsec)

    c0, c1 = p._coords[0], p._coords[1]
    ra_displacement_bubble_center_from_left_edge = (c0.ra - center.ra).deg * np.cos(current_position.dec)
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
    ax1.plot([center.ra.to_value()], [center.dec.to_value()], color='blue', marker='*', markersize=8, transform=ax1.get_transform('world'), alpha=0.3)
    # ax1.plot(*(x.to(u.deg).to_value() for x in (p._coords.ra, p._coords.dec)), transform=ax1.get_transform('world'), linewidth=2, color='green')
    ax1.arrow(*(x.to(u.deg).to_value() for x in (c0.ra, c0.dec)),
        *(x.to(u.deg).to_value() for x in ((c1.ra - c0.ra), (c1.dec - c0.dec))),
        color='green', transform=ax1.get_transform('world'), length_includes_head=True,
        width=0.003,
    )
    plt.show()



def along_pillar_12CO(pillar_top, pillar_base, coord_to_mark):
    """
    RCW 49:
    pillar_top = "10:24:09.3382 -57:48:54.070"
    pillar_base = "10:24:27.4320 -57:50:35.824"
    coord_to_mark = wr20b
    """
    pillar_coords = SkyCoord([pillar_top, pillar_base], unit=(u.hourangle, u.deg), frame=FK5)
    p = pvextractor.Path(pillar_coords, width=50*u.arcsec)
    c0, c1 = p._coords[0], p._coords[1]

    sl = pvextractor.extract_pv_slice(subcube, p)
    ax2 = plt.subplot(122, projection=WCS(sl.header))
    ax2.imshow(sl.data, origin='lower', aspect=.5)
    ax2.coords[1].set_format_unit(u.km/u.s)
    plt.contour(sl.data, cmap='autumn_r', linewidths=0.5, levels=[10., 15., 20., 25., 30., 35.,])
    plt.ylabel("Velocity (km/s)")

    mom0_cutout = Cutout2D(mom0.to_value(), coord_to_mark, [10.*u.arcmin, 10.*u.arcmin], wcs=wflat)
    ax1 = plt.subplot(121, projection=mom0_cutout.wcs)
    plt.imshow(mom0_cutout.data, origin='lower', cmap='plasma')
    ax1.plot([coord_to_mark.ra.to_value()], [coord_to_mark.dec.to_value()], color='white', marker='*', markersize=8, transform=ax1.get_transform('world'), alpha=0.3)
    # ax1.plot(*(x.to(u.deg).to_value() for x in (p._coords.ra, p._coords.dec)), transform=ax1.get_transform('world'), linewidth=2, color='green')
    ax1.arrow(*(x.to(u.deg).to_value() for x in (c0.ra, c0.dec)),
        *(x.to(u.deg).to_value() for x in ((c1.ra - c0.ra), (c1.dec - c0.dec))), # this is somehow correct; no cos(dec) term is necessary.
        color='green', transform=ax1.get_transform('world'), length_includes_head=True,
        width=0.003,
    )

    plt.show()


def vertical_cut_thru_entire_structure(pct):
    # Pct is the percentage of the full path that this cut will be at
    # Start here (center of horizontal/constant-Dec line)
    top_position = SkyCoord("10:24:09.7327 -57:39:14.350", unit=(u.hourangle, u.deg), frame=FK5)
    # End here (center of horizontal line)
    bottom_position = SkyCoord("10:24:09.7327 -57:55:05.338", unit=(u.hourangle, u.deg), frame=FK5)

    stepwidth = 0.25*u.arcmin
    start_dec = top_position.dec
    # Moving down (south) in declination
    end_dec = bottom_position.dec
    stop_dec = start_dec + (end_dec - start_dec)*pct/100.
    current_position = top_position
    angle = 270.
    count = 0
    plt.figure(figsize=(12, 9))
    while current_position.dec > stop_dec:
        count += 1
        current_position = SkyCoord(current_position.ra, current_position.dec - stepwidth)

    p = pvextractor.PathFromCenter(center=current_position, length=14*u.arcmin, angle=angle*u.deg, width=10*u.arcmin)
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
        width=linewidth_from_data_units((0.25*u.arcmin).to(u.deg).to_value(), ax1, reference='y'),
    )
    plt.show()




def vertical_series_thru_entire_structure():
    # Start here (center of horizontal/constant-Dec line)
    top_position = SkyCoord("10:24:09.7327 -57:39:14.350", unit=(u.hourangle, u.deg), frame=FK5)
    # End here (center of horizontal line)
    bottom_position = SkyCoord("10:24:09.7327 -57:55:05.338", unit=(u.hourangle, u.deg), frame=FK5)

    stepwidth = 0.25*u.arcmin
    start_dec = top_position.dec
    # Moving down (south) in declination
    end_dec = bottom_position.dec
    current_position = top_position
    angle = 270.
    count = 0
    plt.figure(figsize=(12, 9))
    while current_position.dec > end_dec:
        p = pvextractor.PathFromCenter(center=current_position, length=14*u.arcmin, angle=angle*u.deg, width=0.5*u.arcmin)
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
            width=linewidth_from_data_units((0.25*u.arcmin).to(u.deg).to_value(), ax1, reference='y'),
        )
        sys.stdout.write(f"\r{count:04d}")
        sys.stdout.flush()
        plt.savefig(f"./figures/pv_anim/rcw49_vert/im_cii_{count:04d}.png")
        plt.clf()
        count += 1
        current_position = SkyCoord(current_position.ra, current_position.dec - stepwidth)
    print()


"""
Some code I stole directly from StackOverflow
"""

def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    (SO user Felix, https://stackoverflow.com/a/35501485)
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)


def moment_1_image():
    """
    Not a PV diagram, just wanted to follow up on Marc's idea of making moment1
    maps to detect filaments
    Use the spectral_slab method of cube to take small (few km/s wide) slices
    and then make moment 1 images (mean velocity)
    Could also make moment 2 images to see what those even look like
    """
    pillars_in_cii = { # clearest in CII
        "major-southern": (4., 13.5),
    }
    pillars_in_co = { # clearest in CO
        "broad-interior": (-10., 7.),
        "focus-interior-1": (-10., -4.),
        "focus-interior-2": (-4., 4.),
        "focus-interior-3": (4., 7.),
    }
    vlims = (-10, 0)
    vlims = tuple(v*u.km/u.s for v in vlims)
    subcube_pillar = cube.spectral_slab(*vlims)
    cutout_width = 20.*u.arcmin
    mom0 = subcube_pillar.moment(order=0).to_value()
    mom0_cut = Cutout2D(mom0, wr20a, [cutout_width]*2, wcs=wflat)
    mom0, mom0w = mom0_cut.data, mom0_cut.wcs

    mom1 = subcube_pillar.moment(order=1).to(u.km/u.s).to_value()
    mom1_cut = Cutout2D(mom1, wr20a, [cutout_width]*2, wcs=wflat)
    mom1, mom1w = mom1_cut.data, mom1_cut.wcs

    plt.figure(figsize=(8, 4))
    plt.subplot(121, projection=mom0w)
    img_lims = misc_utils.flquantiles(mom0[np.isfinite(mom0)].ravel(), 16)
    plt.imshow(mom0, origin='lower')
    plt.colorbar(label="Temperature (K)")
    plt.title("Moment 0")

    plt.subplot(122, projection=mom1w)
    plt.imshow(mom1, origin='lower', vmin=vlims[0].to_value(), vmax=vlims[1].to_value(), cmap='jet')
    plt.colorbar(label="Mean velocity (km/s)")
    plt.title("Moment 1")

    plt.show()


if __name__ == "__main__":
    moment_1_image()
