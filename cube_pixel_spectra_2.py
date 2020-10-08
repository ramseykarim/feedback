import numpy as np
import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 7}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import sys
import warnings

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
# from astropy.modeling import models, fitting
# from astropy import convolution

import regions
from math import ceil

from . import misc_utils
from . import catalog
from . import cube_utils
from . import pvdiagrams
from . import crosscut
# This is where I "started" this task, some functions might be useful
from . import cube_pixel_spectra as cps1

"""
A sequel to cube_pixel_spectra.py. Looking closer at the M16 line data,
fitting things to the spectra, etc.
The goal remains the same, I'm just still working on it.

Created: September 25, 2020
    (while listening to the Front Bottoms on a pleasant, rainy Friday evening)
"""
__author__ = "Ramsey Karim"


def cutout_subcube(length_scale_mult=2):
    """
    This is just the first few lines of cps1.area_fit_attempt
    I think this subcube will be useful, so I'll keep using it.
    This is the same grid that I fit all those Gaussians to.
    length_scale_mult was 2 in that grid; I can change it here
    """
    warnings.filterwarnings("ignore")

    reg_index = 0
    global_center_coord, length_scale, location_generator = pvdiagrams.linear_series_from_description(
        *crosscut.coords_from_region(catalog.utils.search_for_file("catalogs/m16_lines_of_interest.reg"), index=reg_index),
        None, None, pvpath_width=10*u.arcsec, points_not_paths=True
    )

    img, w = crosscut.DataLayer("CII", "sofia/M16_CII_U.fits", cube=True, alpha=0.7, vlims=(5, 40)).load()
    img_cutout = Cutout2D(img, global_center_coord, [length_scale*length_scale_mult]*2, wcs=w, mode='partial', fill_value=np.nan)

    fn = cps1.filenames[3]
    cube = cube_utils.CubeData(fn)
    spectral_axis = cube.data.spectral_axis.to(u.km/u.s).to_value()
    cube_name = cube.name()

    # Make a subcube using those slices. This has good WCS (even though it doesn't look like it)
    subcube = cube.data[:, img_cutout.slices_original[0], img_cutout.slices_original[1]]
    return subcube


def mask_above_half_power(cube, xpower=2):
    """
    Per Lee's recommendation in our Sept 25, 2020 Friday meeting
    For each spatial pixel, I will mask out spectral pixels that are below
    the half-power (or 1/3, or something) level of that spectrum.
    Then, I can make a moment 0 or 1 or whatever image with that.

    SpectralCube offers pretty good masking capability, so I'll make the most
    of that.

    xpower is something to DIVIDE the max by for the  mask. If you want below
    half power masked out, then xpower = 2. If you want below 1/3, then it's 3
    """
    kms = u.km/u.s
    kkms = u.K*u.km/u.s
    moment = 2
    moment_units = [kkms, kms, kms*kms][moment]
    half_power = np.max(cube, axis=0)/xpower
    mask = (cube > half_power) & (cube > 6.5*u.K)
    masked_cube = cube.with_mask(mask)
    fig = plt.figure(figsize=(10, 8))
    ax0 = plt.subplot2grid((2, 2), (0, 0), fig=fig, projection=cube.wcs, slices=('x','y', 50))
    ax1 = plt.subplot2grid((2, 2), (0, 1), fig=fig, projection=cube.wcs, slices=('x','y', 50))
    ax2 = plt.subplot2grid((2, 2), (1, 0), fig=fig)
    ax3 = plt.subplot2grid((2, 2), (1, 1), fig=fig)
    img = cube.moment(order=moment).to(moment_units).to_value()
    im = ax0.imshow((img), origin='lower')
    fig.colorbar(im, ax=ax0)
    img = masked_cube.moment(order=moment).to(moment_units).to_value()
    im = ax1.imshow((img), origin='lower', vmax=4)
    fig.colorbar(im, ax=ax1)
    ax2.plot(cube.spectral_axis.to_value(), cube.mean(axis=(1, 2)).to(u.K).to_value(), linewidth=0.7)
    ax3.plot(masked_cube.spectral_axis.to_value(), masked_cube.mean(axis=(1, 2)).to(u.K).to_value(), linewidth=0.7)
    ax3.set_xlim(ax2.get_xlim())
    ax2.set_title("Mean spectrum, unmasked")
    ax3.set_title(f"Mean spectrum, masked above 1/{xpower} power")
    plt.show()


if __name__ == "__main__":
    subcube = cutout_subcube(length_scale_mult=8)
    mask_above_half_power(subcube, xpower=2)
