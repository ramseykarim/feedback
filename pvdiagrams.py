import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 11}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cm
import matplotlib.patches as mpatches
import matplotlib.transforms as mpl_transforms
import sys
import os

from math import ceil

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.nddata.utils import Cutout2D
from spectral_cube import SpectralCube
import pvextractor
import regions
from reproject import reproject_interp

from . import misc_utils
from . import catalog
from . import cube_utils


"""
Testing SpectralCube and pvextractor and making PV diagrams
Created: March 24th, 2020 (in the midst of the quarantine)
It looks like I made heavy edits around July 22, Aug 1, and Aug 27, 2020.
I am making more edits on October 8, 2020 (still quarantine obvi)
"""
__author__ = "Ramsey Karim"

# Useful units
km_s = u.km / u.s

# RCW 49
fn = f"{catalog.utils.ancillary_data_path}apex/apexCO/RCW49_12CO.fits"
fn = f"{catalog.utils.ancillary_data_path}sofia/rcw49-cii.fits"
# M16
filenames = ["apex/M16_12CO3-2.fits", "apex/M16_13CO3-2.fits",
    "bima/M16_12CO1-0_7x4.fits",
    "sofia/M16_CII_U.fits",
    ]
fn = filenames[3]

"""
### July 22, 2020: I just learned that the units should be K, not K/(km/s)
### I'll have to address this more globally later, but I'll start treating it
### properly in new code now

cube = SpectralCube(data=data, wcs=w)

subcube = cube.spectral_slab(17*km_s, 31*km_s)
del data
mom0 = subcube.moment(order=0)
"""


"""
Star positions, bubble centers, etc.
"""
wr20b = SkyCoord(156.07666638433972*u.deg, -57.80826909707589*u.deg)
wr20b_bubble = SkyCoord("10:24:18.7061 -57:48:14.937", unit=(u.hourangle, u.deg), frame=FK5)
wr20a = SkyCoord("10:23:58.0545 -57:45:48.862", unit=(u.hourangle, u.deg), frame=FK5) # approx from ds9, not SIMBAD or anything
wr20b_bubble_radius = 1.35967*u.deg

m16_bima_center = SkyCoord("18:18:51.2900 -13:50:00.890", unit=(u.hourangle, u.deg), frame=FK5)
m16_marc_pillar2_center = SkyCoord('18:18:51.5 -13:50:26.3', unit=(u.hourangle, u.deg), frame=FK5)
m16_marc_pillar2_kwargs = dict(angle=-47*u.degree, width=5*u.arcsec, length=150*u.arcsec)
m16_pillar1_coords = tuple(SkyCoord(x, unit=(u.hourangle, u.deg), frame=FK5) for x in ("18:19:00.5191 -13:51:16.046", "18:18:49.8401 -13:48:29.025"))
m16_pillar1_kwargs = dict(width=15*u.arcsec)

m16_allpillars_series_endpoints = tuple(SkyCoord(x, unit=(u.hourangle, u.deg), frame=FK5) for x in ("18:18:58.4575 -13:49:35.577", "18:18:48.0482 -13:51:28.726"))
m16_allpillars_series_kwargs = dict(pvpath_angle=-50*u.degree, pvpath_width=10*u.arcsec, pvpath_length=4.6*u.arcmin)

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
PLOT PATH AND REFERENCE IMAGE
"""

def prepare_subcube(cube, vlims):
    """
    Parse vlims argument
    Apply velocity limits to a cube
    :param cube: CubeData
    :param vlims: (lo, hi) velocity limits in km/s (but just number, not quantity)
    """
    vlims = tuple(v*u.km/u.s for v in vlims)
    return cube.data.spectral_slab(*vlims)


def prepare_reference_image(spectr_cube_obj, flat_wcs, center_coord, size=None):
    """
    :param spectr_cube_obj: SpectralCube object
    :param flat_wcs: 2D image WCS of spectr_cube_obj
    :param center_coord: coordinate for the center of the field
    :param size: box size, as length-2 sequence of Angle or Quantity, or
        as single Angle or Quantity. If single, then square box.
        If sequence, then X/Y order is whatever Cutout2D wants.
    :returns: np.ndarray image, WCS object from Cutout2D
    :::::: FUTURE ::::::::
    Maybe I should have an option to give a WCS definition to regrid to.
    Cutout2D has a rounding problem, it likes pixels. Works good for one image,
    but if I am directly comparing multiple images, then not so good.
    """
    if size is None:
        raise RuntimeError("prepare_reference_image: Size cannot be None.")
    elif not isinstance(size, tuple):
        size = [size, size]
    mom0_cutout = Cutout2D(spectr_cube_obj.moment(order=0).to_value(), center_coord, size, wcs=flat_wcs, mode='partial', fill_value=np.nan)
    return mom0_cutout.data, mom0_cutout.wcs



def plot_path(cube, subcube, path,
    center_coord=None, coord_to_mark=None,
    img_to_plot=None, img_stretch='linear', pv_stretch='linear',
    img_lims=None, pv_lims=None,
    img_subplot=(121,), pv_subplot=(122,),
    fig=None, show=True,
    contours=False):
    """
    :param cube: CubeData
    :param subcube: a SpectralCube instance, already trimmed to the correct
        velocity limits, OR the vlims tuple, which can be
        passed to prepare_subcube() with cube
    :param path: Path instance for the PV slice
    :param center_coord: center of the moment 0 image to plot. If None,
        approximates a center using the average of the Path endpoints.
        Useful if you are repeating this function with different Paths but
        want the reference image to stay still. If img_to_plot is specified,
        then center_coord is ignored.
    :param coord_to_mark: (optional) coordinate to place a marker at
        could be used for a nearby star. SkyCoord.
    :param img_to_plot: if not None, should give a tuple of:
        1) image to show
        2) WCS object for that image
        Useful if this is reused, don't have to
        recalculate the Cutout2D
    :param img_stretch: the visual stretch for the accompanying spatial image.
        If string, must be 'linear', 'log', or 'arcsinh'. Can also be
        callable, in which case it'scalled on the 2D array of numbers
    :param pv_stretch: the visual stretch for the PV diagram. Same rules as
        img_stretch
    :param img_lims: vlims for the image. If None, lets imshow do its own thing.
        If not None, should be a tuple of numbers: (vmin, vmax)
    :param pv_lims: vlims for the pv diagram. If None, lets imshow do its own
        thing. If not None, should be a tuple of numbers: (vmin, vmax)
    :param img_subplot: Number indicator of img axis.
        subplot number/tuple for img. Has to be tuple,
        even if only one number. Args unpacked into plt.subplot()
    :param pv_subplot_number: same as img_subplot, for pv plot
    :param fig: figure object to use
    :param show: whether or not to show the plot
    :param contours: None if no contours, else pass in the spacing between
        contours (like sigma)
    :returns: the figure used
    """
    c0, c1 = SkyCoord(path._coords[0]), SkyCoord(path._coords[1])
    # print(c0.to_string(style='hmsdms', sep=':'))
    # print(c1.to_string(style='hmsdms', sep=':'))
    length = c0.separation(c1)
    pv_stretch = misc_utils.check_stretch(pv_stretch)

    if fig is None:
        fig = plt.figure(figsize=(9.5, 4))
    else:
        plt.figure(fig.number)

    # PV diagram
    if isinstance(subcube, tuple):
        # subcube is vlims
        subcube = prepare_subcube(cube, subcube)
    sl = pvextractor.extract_pv_slice(subcube, path)
    ax2 = plt.subplot(*pv_subplot, projection=WCS(sl.header))

    # Figure out visual pv limits
    if pv_lims is not None:
        pv_vlim_kwargs = dict(vmin=pv_stretch(pv_lims[0]), vmax=pv_stretch(pv_lims[1]))
    else:
        pv_vlim_kwargs = dict()


    im = ax2.imshow(pv_stretch(sl.data), origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), **pv_vlim_kwargs)
    fig.colorbar(im, ax=ax2, label='T (K)')

    if contours:
        levels = np.arange(pv_lims[0], pv_lims[1]*3, contours)
        ax2.contour(sl.data, colors='k', linewidths=0.5, levels=levels)

    ax2.coords[1].set_format_unit(u.km/u.s)
    cube.help_plot_pv(ax2)
    ax2.set_ylabel("Velocity (km/s)")
    ax2.coords[0].set_format_unit(u.arcsec)
    ax2.coords[0].set_major_formatter('x')
    ax2.set_xlabel("Displacement (\")")
    ax2.set_title(f"PV diagram")

    # Moment 0 image
    img_stretch = misc_utils.check_stretch(img_stretch)
    if img_to_plot is None:
        if center_coord is None:
            center_coord = catalog.utils.coordinate_midpoint(c0, c1)
        img_to_plot, cutout_wcs = prepare_reference_image(subcube, cube.wcs_flat, center_coord, size=length*3)
    else:
        img_to_plot, cutout_wcs = img_to_plot
        img_to_plot = img_stretch(img_to_plot)
    # Figure out axis
    ax1 = plt.subplot(*img_subplot, projection=cutout_wcs)
    # Figure out visual image limits
    if img_lims is not None:
        img_vlim_kwargs = dict(vmin=img_stretch(img_lims[0]), vmax=img_stretch(img_lims[1]))
    else:
        img_vlim_kwargs = dict()
    # Show the image
    ax1.imshow(img_to_plot, origin='lower', cmap='plasma', **img_vlim_kwargs)
    if coord_to_mark is not None:
        ax1.plot([coord_to_mark.ra.to_value()], [coord_to_mark.dec.to_value()], color='white', marker='*', markersize=8, transform=ax1.get_transform('world'), alpha=0.3)
    # ax1.plot(*(x.to(u.deg).to_value() for x in (p._coords.ra, p._coords.dec)), transform=ax1.get_transform('world'), linewidth=2, color='green')
    ax1.arrow(*(x.to(u.deg).to_value() for x in (c0.ra, c0.dec)),
        *(x.to(u.deg).to_value() for x in ((c1.ra - c0.ra), (c1.dec - c0.dec))), # this is somehow correct; no cos(dec) term is necessary.
        color='white', transform=ax1.get_transform('world'), length_includes_head=True,
        width=(path.width.to(u.deg).to_value() if path.width else 0.001), fill=(path.width is None), lw=0.6, alpha=0.7,
    )
    ax1.set_xlabel("RA")
    ax1.set_ylabel("Dec")
    ax1.set_title(f"{cube.name()} integrated intensity (within PV plot limits)", fontsize=8)

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.05)

    if show:
        plt.show()
    # Return the figure
    return fig

"""
SINGLE PATH FUNCTIONS
"""

def path_from_description(coord_A, coord_B, width=None):
    """
    :param coord_A: EITHER:
        1) coordinate beyond the top of the pillar. string
        2) center coord. SkyCoord
        3) Path
    :param coord_B: EITHER:
        1) coordinate below the base of the pillar. string
        2) dict of kwargs for at least angle and length, others are fine too
        3) ignored, but something must be here since it's positional
    :param width: width of the pv cut. Default is 15 arcseconds. Quantity.
        If 'width' is a key in coord_B, then this is ignored
    """
    if width is None:
        # Can still be overriden by coord_B
        width = 15*u.arcsec
    if (isinstance(coord_A, str) and isinstance(coord_B, str)) or (isinstance(coord_A, SkyCoord) and isinstance(coord_B, SkyCoord)):
        if isinstance(coord_A, str):
            pillar_coords = SkyCoord([coord_A, coord_B], unit=(u.hourangle, u.deg), frame=FK5)
        else:
            pillar_coords = SkyCoord([coord_A, coord_B])
        p = pvextractor.Path(pillar_coords, width=width)
    elif isinstance(coord_A, SkyCoord) and isinstance(coord_B, dict):
        coord_B = coord_B.copy()
        if 'width' not in coord_B:
            coord_B['width'] = width
        else:
            width = coord_B['width']
        p = pvextractor.PathFromCenter(coord_A, **coord_B)
    elif isinstance(coord_A, pvextractor.Path):
        p = coord_A
    else:
        raise RuntimeError(f"Incorrect input:\ncoord_A = {coord_A}\ncoord_B = {coord_B}\nTry again!")
    return p


def path_from_ds9(reg_file_name, index, width=None):
    """
    :param reg_file_name: the filename of a DS9 region file containing at least
        one Vector or Line
    :param index: the 0-indexed index of the desired Line or Vector in the
        region file. The index SHOULD NOT COUNT regions that aren't Lines
        or Vectors. Think index of [x for x in reg_list if line_or_vector(x)].
        If index is None, returns a list of all Line or Vector Paths
    :param width: Path width attribute to be passed directly to Path.width
    :returns: either Path or list of Paths
    """
    # Get list of Paths
    path_list = pvextractor.paths_from_regfile(reg_file_name)
    if not path_list:
        # Added May 11, 2023; still using this!
        # No lines or vectors; return a null value in accordance with how many were requested
        if index is None:
            return path_list
        else:
            return None
    if index is None:
        # Assign width to all Paths; you can always go change it
        if width is not None:
            for p in path_list:
                p.width = width
        return path_list
    else:
        # Get the path from the list of Lines and Vectors
        p = path_list[index]
        # Assign the width
        p.width = width
        return p


"""
SERIES OF PATHS FUNCTIONS
"""

def linear_series_from_description(start_center, end_center, pvpath_length,
    pvpath_angle, pvpath_width=None, points_not_paths=False, n_steps=30):
    """
    Create a linear series of parallel Paths. Generator function, so generates
    lazily.
    :param start_center: SkyCoord for the center of the first PV path
    :param end_center: SkyCoord for the center of the last PV path.
        There is no absolute guarantee that the last PV path will pass exactly
        through this point, but it will be close, since if you
        specify pvpath_width, and the separation between the start and
        end center points is not divisible by that width, then you will not
        hit the end point. The behavior in this case is to overshoot the
        end point with the last PV path.
    :param pvpath_length: angle Quantity or Angle, the length of each PV path
    :param pvpath_angle: angle Quantity or Angle, the angle of the PV path on
        sky, with 0 being a path extending towards the West (sky), and positive
        angles moving SOUTHWEST (sky).
        Note that a Northward path should be an angle of 270 degrees.
    :param pvpath_width: angle Quantity or Angle, the width of each PV path.
        The velocity spectrum is averged across this width.
        The width also sets the step size along the series; step size will be
        0.5 * width.
        If pvpath_width is None, then Path uses the "None" convention of
        interpolation along the path. Step size then defaults to whatever makes
        the series 30 steps long.
    :param points_not_paths: option to return the center point of the PV path,
        rathern than the pvextractor PathFromCenter object itself, in
        the generator.
        If points_not_paths is True, then all the keyword
    :param n_steps: if pvpath_width is None, take n_steps from start to finish
    :returns: three useful things:
        1) the center coordinate of the path
        2) the longest characteristic length of the system (the max of
            the series length and the PV path length)
        3) Generator that returns a finite number of Path objects. The generator
            function has already been called and is ready to loop over
    """
    # Get position angle and separation of series
    series_position_angle = start_center.position_angle(end_center)
    series_length = start_center.separation(end_center)
    # Get midpoint of series
    center_coord = catalog.utils.coordinate_midpoint(start_center, end_center)
    # Get the stepwidth, somehow
    if pvpath_width is None:
        # Default n_steps
        stepwidth = series_length / float(n_steps - 1)
    elif not hasattr(pvpath_width, 'unit'):
        # Assume we gave it a number of steps...
        n_steps = pvpath_width # Override n_steps
        stepwidth = series_length / float(n_steps - 1)
        # Set width to None to be safe
        pvpath_width = None
    else:
        stepwidth = 0.5 * pvpath_width
        # We want to overshoot the end point, and also to include endpoints.
        n_steps = int(ceil(series_length / stepwidth)) + 1
    print(f"Preparing series of {n_steps} steps over length of {series_length.to(u.arcmin):.2f} at a position angle of {series_position_angle.to(u.deg):.2f}")
    if not points_not_paths:
        print(f"Each Path has a length of {pvpath_length.to(u.arcmin):.2f}, width of {(pvpath_width if pvpath_width else 0.*u.arcsec).to(u.arcsec):.2f}, and position angle of {pvpath_angle.to(u.deg):.2f}")
    # Gather the PathFromCenter kwargs
    path_kwargs = dict(length=pvpath_length, width=pvpath_width, angle=pvpath_angle)
    def path_generator():
        # Get an Angle quantity of 0 length
        separation_along_series = series_length * 0.
        # Cycle through all n_steps and increment the separation_along_series
        for current_step_number in range(n_steps):
            # SkyCoord.directional_offset_by easily returns the coordinate we want
            current_center = start_center.directional_offset_by(series_position_angle, separation_along_series)
            separation_along_series += stepwidth
            if points_not_paths:
                # Useful for other applications of needing this series
                yield current_center
            else:
                yield pvextractor.PathFromCenter(center=current_center, **path_kwargs)
    return center_coord, max([x for x in (series_length, pvpath_length) if x is not None]), path_generator()


def linear_series_from_ds9(reg_file_name, pvpath_length=None, pvpath_angle=None, pvpath_width=None, **kwargs):
    """
    Creates a linear series of parallel Paths. Generator function, so generates
    lazily. Sources the
    :param reg_file_name: filename of ds9 .reg file. Must contain at least
        one Vector or Line region. If pvpath_length and pvpath_angle are left
        as None, then must contain at least two Vectors or Lines.
        Any more than one or two, depending on the case, are ignored.
    :param pvpath_length: (optional) angle Quantity or Angle,
        the length of each PV path. If not present, there must be a second
        Line or Vector in the .reg file.
    :param pvpath_angle: (optional) angle Quantity or Angle,
        the angle of the PV path on sky, with 0 being a path extending
        towards the West (sky), and positive angles moving SOUTHWEST (sky).
        Note that a Northward path should be an angle of 270 degrees.
        If not present, there must be a second Line or Vector in the .reg file.
    :param pvpath_width: angle Quantity or Angle, the width of each PV path.
        The velocity spectrum is averged across this width.
        The width also sets the step size along the series; step size will be
        0.5 * width.
        If pvpath_width is None, then Path uses the "None" convention of
        interpolation along the path. Step size then defaults to whatever makes
        the series 30 steps long.
    :returns: Generator that returns a finite number of Path objects

    September 9, 2020: I kinda messed around with some stuff, added in that
        points_not_paths bit, but I don't think it's necessary. I can feel
        an impending code refactor coming (post thesis proposal?)
        so I'll wait till then to fix it. But just so you know.
    """
    # regions module doesn't support Vectors, so we'll just use pvextractor's
    # region parser
    path_list = path_from_ds9(reg_file_name, None)
    series_coords = [SkyCoord(x) for x in path_list.pop(0)._coords]
    points_not_paths = False # by default
    if path_list and ((pvpath_angle is None) or (pvpath_length is None)):
        # If there is a second path_list element
        # Get the angle and length from the second line or vector in the file
        template_path_coords = [SkyCoord(x) for x in path_list.pop(0)._coords]
        pvpath_length = template_path_coords[0].separation(template_path_coords[1])
        pvpath_angle = template_path_coords[0].position_angle(template_path_coords[1])
    else:
        # Toggle points_not_paths
        pvpath_angle = pvpath_length = None
        points_not_paths = True
    # With the information gathered, use linear_series_from_description to
    # finish the job
    return linear_series_from_description(series_coords[0], series_coords[1],
        pvpath_length, pvpath_angle, pvpath_width=pvpath_width,
        points_not_paths=points_not_paths, **kwargs)


def run_plot_and_save_series(cube, vlims, center_coord, length_scale, path_generator, savename, **plot_kwargs):
    """
    Save a series of images created from a series of Paths
    :param cube: a CubeData instance for plot_path()
    :param vlims: tuple (lo, hi) velocity limits for plot_path()
    :param center_coord: SkyCoord for the center of the reference image
    :param length_scale: characteristic length to use as side length of
        square reference image. Quantity or Angle
    :param path_generator: the Generator result of one of the
        linear_series_from_ functions
    :param savename: some .png filename. Will insert _{step number : 03d}
        right before the .png part.
    :param plot_kwargs: keyword arguments to be passed to plot_path()
    """
    # Create a savename generator
    savename = os.path.abspath(savename)
    save_dir, save_filename = os.path.split(savename)
    save_filename_stub = save_filename.replace('.png', '')
    save_filename_generator = lambda n: os.path.join(save_dir, f"{save_filename_stub}{n:03d}.png")
    # Set up the subcube
    subcube = prepare_subcube(cube, vlims)
    # Create the reference image
    img_to_plot_info = prepare_reference_image(subcube, cube.wcs_flat,
        center_coord, size=length_scale)
    # Initialize None figure, to be replaced by the actual figure
    fig = None
    # Loop through paths and save images
    for n, p in enumerate(path_generator):
        # Write the number to terminal
        print(p._coords)
        sys.stdout.write(f"{n:03d}\n")
        sys.stdout.flush()
        # Plot and save the figure (only novel on first iteration)
        fig = plot_path(cube, subcube, p, img_to_plot=img_to_plot_info, fig=fig, show=False, **plot_kwargs)
        plt.savefig(save_filename_generator(n))
        # Clear the last figure and increment n
        plt.clf()
        n += 1



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


def along_pillar(cube, vlims, coord_A, coord_B, width=None, **kwargs):
    """
    RCW 49:
    pillar_top = "10:24:09.3382 -57:48:54.070"
    pillar_base = "10:24:27.4320 -57:50:35.824"
    coord_to_mark = wr20b
    """
    # Use path_from_description
    p = path_from_description(coord_A, coord_B, width=width)
    # Use general Path plotting function
    plot_path(cube, vlims, p, **kwargs)


def moment_1_image(cube, vlims, focus_coord, cutout_width=None):
    """
    Not a PV diagram, just wanted to follow up on Marc's idea of making moment1
    maps to detect filaments
    Use the spectral_slab method of cube to take small (few km/s wide) slices
    and then make moment 1 images (mean velocity)
    Could also make moment 2 images to see what those even look like
    :param cube: CubeData instance
    :param vlims: (lo, hi) velocity limits in km/s (but just number, not quantity)
    :param focus_coord: the center of the 2D image to zoom in on
    :param cutout_width: (optional) width of the square cutout image.
        Default is 20 arcmin. Cutoud2D is set to "trim".
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
    vlims = tuple(v*u.km/u.s for v in vlims)
    subcube_pillar = cube.data.spectral_slab(*vlims)
    cutout_width = 20.*u.arcmin
    mom0 = subcube_pillar.moment(order=0).to_value()
    mom0_cut = Cutout2D(mom0, focus_coord, [cutout_width]*2, wcs=cube.wcs_flat)
    mom0, mom0w = mom0_cut.data, mom0_cut.wcs

    mom1 = subcube_pillar.moment(order=1).to(u.km/u.s).to_value()
    mom1_cut = Cutout2D(mom1, focus_coord, [cutout_width]*2, wcs=cube.wcs_flat)
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

def run_all_data():
    subplot_size = (3, 2)
    subplot_number = 1
    fig = plt.figure(figsize=(5, 7))
    for fn in filenames:
        cube = cube_utils.CubeData(fn)
        # moment_1_image(cube, m16_bima_center, (17, 31))
        along_pillar(cube, (17, 31), m16_marc_pillar_center, m16_marc_pillar_kwargs,
            img_subplot_number=(*subplot_size, subplot_number),
            pv_subplot_number=(*subplot_size, subplot_number + 1),
            fig=fig, show=False)
        del cube
        subplot_number += 2
    plt.show()


def run_series_from_ds9(filename):
    path_info = linear_series_from_ds9(catalog.utils.search_for_file("catalogs/pillar_series.reg"), pvpath_width=m16_allpillars_series_kwargs['pvpath_width'])
    cube = cube_utils.CubeData(filename)
    run_plot_and_save_series(cube, (17, 31), *path_info, f"{catalog.utils.figures_path}pv_anim/m16_pillars/img_{cube.filename_stub()}.png")
    del cube


def run_vectors_from_ds9(filename):
    path_list = path_from_ds9(catalog.utils.search_for_file("catalogs/pillar_vectors.reg"), None, width=m16_allpillars_series_kwargs['pvpath_width'])
    cube = cube_utils.CubeData(filename)
    n = 0
    for p in path_list:
        plot_path(cube, (17, 31), p, show=False)
        plt.savefig(f"/home/ramsey/Pictures/8-27-20-work/pillar_{cube.filename_stub()}_pv_{n:1d}.png")
        plt.clf()
        n += 1
    del cube



if __name__ == "__main__":
    print("nothing here")
    # for i in range(len(filenames)):
    # for i in [3,]:
    #     cube = cube_utils.CubeData(filenames[i])
    #     along_pillar(cube, (17, 31), *m16_pillar1_coords, **m16_pillar1_kwargs, show=True)
    #     savename = cube.filename_stub() + "_pv.png"
    #     print(savename)
        # plt.savefig("/home/ramsey/Pictures/8-01-20-work/" + savename)
