"""
A designated place for even MORE nice M16 images
Descended from m16_investigation and m16_pictures

Created: July 9, 2021
Starting with investigating the "light feature" over P2 and repeating some
transverse PV diagrams
I hope the paper is within reach. Maybe august?? Rise and grind

m16_investigation got too long so I'm starting this one. I estimate that I'll
need maybe just one more to wrap up M16. After that, I should see if I can
consolidate any of this into reusable packages. Though tbh I should wait for
one more region to make sure everything truly is reusable. Tho I do have RCW49..
"""
__author__ = "Ramsey Karim"

# All imports dumped from the last file, m16_investigation
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 13}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import os
import glob

from math import ceil
from scipy import signal
from scipy.interpolate import UnivariateSpline

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

# Let us begin with a rewrite of m16_investigation.compare_carma_to_sofia_pv
def easy_pv():
    """
    July 9, 2021
    Despite the name of this function, this may not be so easy
    I just want a somewhat general function to make PVs from lines or vector
    regions and from any cube. Let's see how I do!

    The first use case will be on Pillar 2 to examine the transverse gradient
    right where the light feature is
    But I suppose I can test it on whatever if it's supposed to be general
    And remember the VOPT VRAD thing..
    """
    # This PV will be in color and perhaps also in contour
    main_pv_fn = "sofia/M16_CII_U.fits"
    # The rest of them will be in contours
    extra_pv_fns = []

    all_pv_fns = [main_pv_fn] + extra_pv_fns
    # Generator to save memory
    all_cubes = (cube_utils.CubeData(fn) for fn in all_pv_fns)
    # Is that necessary? Can it just wait for the loop?

    """
    Todo: get regions together
    Write the main loop through the cubes
    Decide how you'll pick the reference image (HST? always nice)
    Maybe overlay other regions on the reference (like the polygons to
    point out where the light feature is)

    I'll return to P2 soon, I need to just focus on P1 for the poster I think
    Though this could also work well for the poster if I run with "are these
    all just threads/tails and not gradients?"

    I'm going to reference pvdiagrams_2.m16_pv_again2 and pvdiagrams_2.try_reproject_pv
    """
    # Make this an argument eventually
    reg_filename = catalog.utils.search_for_file("catalogs/pillar1_threads_pv.reg")
    path_list = pvdiagrams.path_from_ds9(reg_filename, None)



def oi_image():
    """
    July 20, 2021
    Make OI integrated intensity image from Nicola's cube
    """
    vel_lims = (20*kms, 27*kms)
    cii_levels = [60, 90, 120, 150, 180, 210]
    cii_cube = cps2.cutout_subcube(length_scale_mult=6)
    cii_mom0 = cii_cube.spectral_slab(*vel_lims).moment0()
    # plt.imshow(cii_mom0.to_value(), origin='lower')
    # plt.contour(cii_mom0.to_value(), levels=cii_levels, colors='k')
    # plt.show()
    # return

    fn = catalog.utils.search_for_file("sofia/m16_OI_63.fits")
    cube = cube_utils.CubeData(fn)
    cube = cube.data.with_spectral_unit(kms)
    mom0 = cube.spectral_slab(*vel_lims).moment0()

    cii_mom0_reproj = reproject_interp((cii_mom0.to_value(), cii_mom0.wcs), mom0.wcs, shape_out=mom0.shape, return_footprint=False)
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, projection=mom0.wcs)
    im = ax.imshow(mom0.to_value(), origin='lower', vmin=0, vmax=50)
    fig.colorbar(im, ax=ax, label='[OI] integrated intensity (K km/s)')
    ax.contour(cii_mom0_reproj, levels=cii_levels, colors='k')
    for coord in ax.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    ax.set_title("[OI] 63 $\mu$m with [CII] 158 $\mu$m in contours")
    ax.text(0.03, 0.9, "Integrated intensity between\n20-27 km/s", transform=ax.transAxes, fontsize=13)
    ax.text(0.03, 0.7, "[CII] integrated intensity\ncontours at 60, 90, 120,\n150, 180, 210 K km/s", transform=ax.transAxes, fontsize=13)
    # OI beam
    patch = cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.06]), misc_utils.get_pixel_scale(mom0.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('grey')
    ax.add_artist(patch)
    ax.text(0.9, 0.06, "OI", ha='center', va='center', transform=ax.transAxes, fontsize=12)
    # CII beam
    patch = cii_cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.8, 0.06]), misc_utils.get_pixel_scale(mom0.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('grey')
    ax.add_artist(patch)
    ax.text(0.8, 0.06, "CII", ha='center', va='center', transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig("/home/ramsey/Pictures/2021-07-20-work/oi.png")
    # plt.show()


def simple_mom0():
    """
    July 20, 2021
    Moment 0 images for the presentation at the Stuttgart conference
    HCO+, CII, CO1-0
    """
    # cube = cps2.cutout_subcube(length_scale_mult=6, reg_index=2)
    # vlims = dict(vmin=0, vmax=200)
    # levels = list(np.linspace(60, 225, 8))

    # cube = cube_utils.CubeData("carma/M16.ALL.hcop.sdi.cm.subpv.fits")
    # cube.convert_to_K()
    # cube = cube.data
    # vlims = dict(vmin=0, vmax=40)
    # levels = list(np.linspace(4, 42, 5))

    cube = cube_utils.CubeData("bima/M16_12CO1-0_7x4.fits")
    cube.convert_to_K()
    cube = cube.data
    vlims = dict(vmin=0, vmax=300)
    levels = list(np.linspace(80, 300, 5))

    cube = cube.with_spectral_unit(kms).spectral_slab(20*kms, 27*kms)
    mom0 = cube.moment0()
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, projection=mom0.wcs)
    im = ax.imshow(mom0.data, origin='lower', cmap='cividis', **vlims)
    fig.colorbar(im, ax=ax, label='Integrated intensity (K km/s)')
    ax.contour(mom0.data, levels=levels, colors='k')
    for coord in ax.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    # ax.set_title("upGREAT [CII] 158$\mu$m, integrated between 20$-$27 km/s")
    # ax.set_title("HCO+, integrated between 20$-$27 km/s")
    ax.set_title("$^{12}$CO (J=1$-$0), integrated between 20$-$27 km/s")
    patch = cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.06]), misc_utils.get_pixel_scale(mom0.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('grey')
    ax.add_artist(patch)
    # plt.show(); return
    plt.tight_layout()
    plt.savefig("/home/ramsey/Pictures/2021-07-20-work/co10.png")






if __name__ == "__main__":
    cube = simple_mom0()
