import numpy as np
import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 6}
matplotlib.rc('font', **font)
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
from . import cube_utils


"""
Testing SpectralCube and pvextractor and making PV diagrams
Created: March 24th, 2020 (in the midst of the quarantine)
"""
__author__ = "Ramsey Karim"

# Useful units
km_s = u.km / u.s

# RCW 49
fn = f"{catalog.utils.ancillary_data_path}apex/apexCO/RCW49_12CO.fits"
fn = f"{catalog.utils.ancillary_data_path}sofia/rcw49-cii.fits"
# M16
filenames = [f"{catalog.utils.m16_data_path}apex/M16_12CO3-2.fits",
    "bima/M16_12CO1-0_7x4.fits",
    "sofia/M16_CII_U.fits"]
fn = filenames[1]

# I should move this to the name==main block
# with fits.open(fn) as hdul:
#     h = hdul[0].header
#     w = WCS(h)
#     wflat = WCS(h, naxis=2)
#     # MEMORY PROBLEM WITH APEX M16 HERE
#     data = hdul[0].data * (u.K / (u.m / u.s))

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
m16_marc_pillar_center = SkyCoord('18:18:51.5 -13:50:26.3', unit=(u.hourangle, u.deg), frame=FK5)
m16_marc_pillar_kwargs = dict(angle=-47*u.degree, width=5*u.arcsec, length=150*u.arcsec)


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

def check_stretch(stretch):
    """
    Sanitize the visual stretch command, raise a RuntimeError if it's not valid
    :param stretch: either a string key to the valid_stretches dictionary
        defined here, or a callable function that can operate on numbers
    """
    valid_stretches = {'linear': lambda x: x, 'log': np.log10, 'arcsinh': np.arcsinh}
    if stretch in valid_stretches:
        return valid_stretches[stretch]
    elif callable(stretch):
        try:
            stretch(np.ones((2, 2), dtype=np.float64))
        except:
            raise RuntimeError(f"Your stretch function doesn't work right.")
        else:
            return stretch
    else:
        raise RuntimeError(f"Not a valid stretch: {stretch}")


def along_pillar(cube, vlims, coord_A, coord_B, width=None, coord_to_mark=None,
    img_stretch='linear', pv_stretch='linear',
    img_subplot_number=(121,), pv_subplot_number=(122,),
    fig=None, show=True):
    """
    RCW 49:
    pillar_top = "10:24:09.3382 -57:48:54.070"
    pillar_base = "10:24:27.4320 -57:50:35.824"
    coord_to_mark = wr20b
    :param cube: CubeData
    :param vlims: (lo, hi) velocity limits in km/s (but just number, not quantity)
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
    :param coord_to_mark: (optional) coordinate to place a marker at
        could be used for a nearby star. SkyCoord.
    :param img_stretch: the visual stretch for the accompanying spatial image.
        If string, must be 'linear', 'log', or 'arcsinh'. Can also be
        callable, in which case it'scalled on the 2D array of numbers
    :param pv_stretch: the visual stretch for the PV diagram. Same rules as
        img_stretch
    :param img_subplot_number: subplot number/tuple for img. Has to be tuple,
        even if only one number. Args unpacked into plt.subplot()
    :param pv_subplot_number: same as img_subplot_number, for pv plot
    :param fig: figure object to use
    :param show: whether or not to show the plot
    """
    if width is None:
        # Can still be overriden by coord_B
        width = 15*u.arcsec
    if (isinstance(coord_A, str) and isinstance(coord_B, str)) or (isinstance(coord_A, SkyCoord) and isinstance(coord_B, SkyCoord)):
        if isinstance(coord_A, str):
            pillar_coords = SkyCoord([coord_A, coord_B], unit=(u.hourangle, u.deg), frame=FK5)
        else:
            pillar_coords = SkyCoord([coord_A, coord_B])
        center_coord = SkyCoord(np.mean(pillar_coords.ra), np.mean(pillar_coords.dec))
        p = pvextractor.Path(pillar_coords, width=width)
    elif isinstance(coord_A, SkyCoord) and isinstance(coord_B, dict):
        coord_B = coord_B.copy()
        if 'width' not in coord_B:
            coord_B['width'] = width
        else:
            width = coord_B['width']
        p = pvextractor.PathFromCenter(coord_A, **coord_B)
        center_coord = coord_A
    elif isinstance(coord_A, pvextractor.Path):
        p = coord_A
    else:
        raise RuntimeError(f"Incorrect input:\ncoord_A = {coord_A}\ncoord_B = {coord_B}\nTry again!")
    c0, c1 = SkyCoord(p._coords[0]), SkyCoord(p._coords[1])
    # print(c0.to_string(style='hmsdms', sep=':'))
    # print(c1.to_string(style='hmsdms', sep=':'))
    length = c0.separation(c1)
    img_stretch = check_stretch(img_stretch)
    pv_stretch = check_stretch(pv_stretch)

    if fig is None:
        fig = plt.figure(figsize=(8, 4))
    else:
        plt.figure(fig.number)

    # PV diagram
    vlims = tuple(v*u.km/u.s for v in vlims)
    subcube = cube.data.spectral_slab(*vlims)
    sl = pvextractor.extract_pv_slice(subcube, p)
    ax2 = plt.subplot(*pv_subplot_number, projection=WCS(sl.header))
    ax2.imshow(pv_stretch(sl.data), origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]))
    # plt.contour(sl.data, cmap='autumn_r', linewidths=0.5, levels=[10., 15., 20., 25., 30., 35.,])
    ax2.coords[1].set_format_unit(u.km/u.s)
    cube.help_plot_pv(ax2)
    ax2.set_ylabel("Velocity (km/s)")
    ax2.coords[0].set_format_unit(u.arcsec)
    ax2.coords[0].set_major_formatter('x')
    ax2.set_xlabel("Displacement (\")")
    ax2.set_title(f"PV diagram")


    # Moment 0 image
    mom0_cutout = Cutout2D(subcube.moment(order=0).to_value(), center_coord, [length*3, length*3], wcs=cube.wcs_flat, mode='partial', fill_value=np.nan)
    ax1 = plt.subplot(*img_subplot_number, projection=mom0_cutout.wcs)
    plt.imshow(img_stretch(mom0_cutout.data), origin='lower', cmap='plasma')
    if coord_to_mark is not None:
        ax1.plot([coord_to_mark.ra.to_value()], [coord_to_mark.dec.to_value()], color='white', marker='*', markersize=8, transform=ax1.get_transform('world'), alpha=0.3)
    # ax1.plot(*(x.to(u.deg).to_value() for x in (p._coords.ra, p._coords.dec)), transform=ax1.get_transform('world'), linewidth=2, color='green')
    ax1.arrow(*(x.to(u.deg).to_value() for x in (c0.ra, c0.dec)),
        *(x.to(u.deg).to_value() for x in ((c1.ra - c0.ra), (c1.dec - c0.dec))), # this is somehow correct; no cos(dec) term is necessary.
        color='white', transform=ax1.get_transform('world'), length_includes_head=True,
        width=width.to(u.deg).to_value(), fill=False, lw=0.5, alpha=0.6,
    )
    ax1.set_xlabel("RA")
    ax1.set_ylabel("Dec")
    ax1.set_title(f"{cube.name()} integrated intensity (within PV plot limits)")

    if show:
        plt.show()




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


if __name__ == "__main__":
    cube = cube_utils.CubeData(filenames[2])
    along_pillar(cube, (19, 24), m16_marc_pillar_center, m16_marc_pillar_kwargs)
