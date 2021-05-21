"""
A designated place to plot MORE nice M16 images

Created: May 9, 2021
Starting from investigating the "systematic" 25-26 km/s gas and the CO(3-2)
absorption thing

This is meant as a continuation of m16_pictures.py so that that file doesn't
get too long
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 13}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import os

from math import ceil
from scipy import signal

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5

from photutils import centroids

# Lord forgive me
from . import crosscut
pvdiagrams = crosscut.pvdiagrams
misc_utils = pvdiagrams.misc_utils
catalog = pvdiagrams.catalog
cube_utils = pvdiagrams.cube_utils
reproject_interp = pvdiagrams.reproject_interp
pvextractor = pvdiagrams.pvextractor
regions = pvdiagrams.regions
SpectralCube = pvdiagrams.SpectralCube
Cutout2D = pvdiagrams.Cutout2D

from . import cube_pixel_spectra as cps1
from . import cube_pixel_spectra_2 as cps2

mpl_cm = pvdiagrams.mpl_cm
mpl_colors = pvdiagrams.mpl_colors
mpl_transforms = pvdiagrams.mpl_transforms
mpatches = pvdiagrams.mpatches

make_vel_stub = lambda x : f"[{x[0].to_value():.1f}, {x[1].to_value():.1f}] {x[0].unit}"
kms = u.km/u.s

def cii_systematic_emission_2():
    """
    May 9, 2021
    I copied and pasted this from m16_pictures.cii_systematic_emission
    I made a second region collection, systematicvelocity_samples_2.reg,
    which is only circles so I can grab their centers
    I want to just put text numbers on each center instead of drawing the
    circles to keep the colors in the diagram to a minimum
    """
    fn = catalog.utils.search_for_file("apex/M16_12CO3-2_truncated.fits"); line_name = "12CO3-2"
    fn = catalog.utils.search_for_file("apex/M16_13CO3-2_truncated.fits"); line_name = "13CO3-2"
    fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits"); line_name = "CII"
    kms = u.km/u.s
    cube = SpectralCube.read(fn)
    cube._unit = u.K
    cube = cube.with_spectral_unit(kms)

    sysvel_limits = (25*kms, 26*kms)
    sysvel_stub = make_vel_stub(sysvel_limits)
    reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/systematicvelocity_samples_2.reg"))

    fig = plt.figure(figsize=(18, 10))

    ax_img = plt.subplot2grid((2, 3), (0, 0))
    mom0 = cube.spectral_slab(*sysvel_limits).moment0()
    im = ax_img.imshow(np.arcsinh(mom0.to_value()), origin='lower', vmin=1.5, vmax=5, cmap='Greys')
    fig.colorbar(im, ax=ax_img)
    ax_img.set_title(f"{line_name} moment 0 {sysvel_stub}")

    ax_img2 = plt.subplot2grid((2, 3), (1, 0))
    im = ax_img2.imshow(np.arcsinh(mom0.to_value()), origin='lower', vmin=1.5, vmax=5, cmap='Greys')
    fig.colorbar(im, ax=ax_img2)
    ax_img2.set_title(f"{line_name} moment 0 {sysvel_stub}")

    ax1 = plt.subplot2grid((2, 3), (0, 1), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2, 3), (1, 1), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)
    axes = (ax1, ax2, ax3, ax4)
    def add_spectrum(img_ax, spec_ax, reg, idx):
        spectrum = cube.subcube_from_regions([reg]).mean(axis=(1, 2))
        pixreg = reg.to_pixel(mom0.wcs)
        pix_center = pixreg.center.xy
        pix_radius = pixreg.radius
        img_ax.text(pix_center[0], pix_center[1]+pix_radius, f"{idx}", color='r', fontsize=11, ha='center', va='bottom')
        p = spec_ax.plot(cube.spectral_axis.to_value(), spectrum.to_value(), label=f"{idx}")
        pixreg.plot(ax=img_ax, color=p[0].get_c())

    for i, reg in enumerate(reg_list):
        if i % 4 == 0:
            add_spectrum(ax_img, ax1, reg, i)
        elif i % 4 == 1:
            add_spectrum(ax_img2, ax2, reg, i)
        elif i % 4 == 2:
            add_spectrum(ax_img, ax3, reg, i)
        else:
            add_spectrum(ax_img2, ax4, reg, i)
        i += 1

    spec_ylim = (-1, 50) # CII
    # spec_ylim = (-3, 25) # 12CO(3-2)
    # spec_ylim = (-1, 10) # 13CO(3-2)
    # spec_xlim = (15, 40) # OLD, Marc suggested to widen it
    spec_xlim = (-5, 55)
    for ax in axes:
        ax.legend()
        ax.set_ylim(spec_ylim)
        ax.set_xlim(spec_xlim)
        ax.axvspan(*(svl.to_value() for svl in sysvel_limits), color='k', alpha=0.1)
        ax.axhline(0, color='k', alpha=0.2)

    ax2.set_xlabel("velocity (km/s)")
    ax4.set_xlabel("velocity (km/s)")
    ax1.set_ylabel(f"{line_name} intensity (K)")
    ax2.set_ylabel(f"{line_name} intensity (K)")
    ax3.set_title(f"{line_name} spectra averaged over selected positions")
    # plt.show()
    fig.savefig(f"/home/ramsey/Pictures/2021-05-20-work/selected_spectra_{line_name}.png")


def minimum_valid_cutout(img):
    """
    Isolate a sub-array from an array which has a padding of unnecessary or
    invalid values. This function is general-purpose and can be thought of
    as 1) the opposite of the np.pad function or 2) similar to the Cutout2D
    function in astropy.
    You pass in a boolean array where there are True values surrounded by
    False values. The True vales should be confined to a small space, though
    they don't necessarily have to cover a perfect rectangle or be uniform.
    This function will return the array slices necessary to capture all the True
    values and discard as many False edge values as possible.
    The worst case scenario for this function is an isolated True value far
    from the main cluster of True values; try to avoid that.
    :param img: a boolean array where False values are unnecessary.
        Ideally, the True values are limited to a single small rectangular
        (aligned with array axes) region in which there are few, if any, False
        values.
    :returns: tuple(slice, slice) which can be applied to the original image
        to obtain the valid subregion. Similar to the "slices" attribute
        from astropy's Cutout2D.
    """
    return tuple(slice(np.min(x), np.max(x)+1) for x in np.where(img))


def correlate_to_find_offset():
    """
    May 12, 2021
    Correlating the BIMA and APEX 3-2 mom0 images between 20-27
    right now, do the native resolutions of each (don't match yet)
    """
    # co10_img, co10_hdr = fits.getdata(catalog.utils.search_for_file("bima/M16.BIMA.13co.mom0.fits"), header=True)
    co10_img, co10_hdr = fits.getdata(catalog.utils.search_for_file("bima/M16_12CO1-0_7x4_mom0.fits"), header=True)
    # I cannot believe this stuff is necessary
    co10_img = np.squeeze(co10_img)
    for k in list(co10_hdr.keys()):
        if k != 'HISTORY' and k[-1] == '3':
            del co10_hdr[k]
    co10_hdr['NAXIS'] = 2
    # /end data wrangling
    # co32_img = reproject_interp(catalog.utils.search_for_file("apex/M16_12CO3-2_mom0.fits"), co10_hdr, return_footprint=False)
    co32_img, co32_hdr = fits.getdata(catalog.utils.search_for_file("apex/M16_12CO3-2_mom0.fits"), header=True)
    co10_img, fp = reproject_interp((co10_img, co10_hdr), co32_hdr, return_footprint=True)
    min_cutout_sl = minimum_valid_cutout(fp > 0.5)
    co10_img = co10_img[min_cutout_sl]
    # co10_img = co32_img[min_cutout_sl] #### THIS ONE MAKES 10 A COPY OF 32
    co32_img = co32_img[min_cutout_sl]
    # print(co10_img.shape)
    # print(co32_img.shape)
    # co10_img = co10_img[]
    # co32_img = co32_img[]
    # print(co10_img.shape)
    # print(co32_img.shape)
    # return
    fig = plt.figure(figsize=(10, 15))
    ax1 = plt.subplot(131)
    ax1.imshow(co10_img, origin='lower')
    ax2 = plt.subplot(132)
    ax2.imshow(co32_img, origin='lower')


    corr = signal.correlate2d(co32_img, co10_img, boundary='fill', mode='same')
    ax3 = plt.subplot(133)
    # ax3.imshow(fp, origin='lower', vmin=-1, vmax=1)
    corr_sl = tuple(slice(x//2 - 5, x//2 + 5) for x in corr.shape)
    ax3.imshow(corr, origin='lower')
    mask = np.zeros_like(corr).astype(bool)
    mask[corr_sl] = True
    corr[~mask] = np.nan
    x1, y1 = centroids.centroid_com(corr)
    x2, y2 = centroids.centroid_1dg(corr)
    x3, y3 = centroids.centroid_2dg(corr)
    print("Centroids:")
    print(x1, y1)
    print(x2, y2)
    print(x3, y3)
    ax3.plot(x1, y1, color='black', ms=5, mew=2, marker='+')
    ax3.plot(x2, y2, color='white', ms=5, mew=2, marker='+')
    ax3.plot(x3, y3, color='red', ms=5, mew=2, marker='+')
    """
    The correlation produces results! I need to quantify these and obtain
    the direction and magnitude (in sky angle) of the offset
    The direction appears to be due North-South, from my quick glance, and about
    half an APEX 3-2 pixel
    """
    plt.show()


if __name__ == "__main__":
    cii_systematic_emission_2()
