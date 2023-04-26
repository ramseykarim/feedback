"""
A designated place for M16 entire region / bubble related procedures and images
Descended from m16_investigation and m16_pictures and m16_deepdive


Created: April 25, 2023
Just circulated the M16 pillars paper and jumping into the next project before
my keyboard gets cold.
Starting with some moment 0 and PV diagrams of the entire region in CII and
CO(3-2). Maybe even the CO(1-0) that I found online
"""
__author__ = "Ramsey Karim"

# All imports dumped from the last file, m16_deepdive
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 13}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import sys
import os
import glob
import datetime
import time

# from math import ceil
# from scipy import signal
# from scipy.interpolate import UnivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
# from astropy.table import Table, QTable
from astropy import constants as const

# import pandas as pd
# from io import StringIO

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

from .mantipython.physics import greybody, dust, instrument


make_vel_stub = lambda x : f"[{x[0].to_value():.1f}, {x[1].to_value():.1f}] {x[0].unit}"
kms = u.km/u.s
marcs_colors = ['#377eb8', '#ff7f00','#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


def try_manual_pv_slice():
    """
    August 25, 2023
    I wanna see if I can just plot array slices for PV diagrams since pvextractor takes a while on large/many slices
    Ok yes this works, WCS likes to throw lots of warnings about it but it is ultimately fine

    I can expand this out to PV movies along RA and Dec (non-aligned paths will need to be done with pvextractor still)
    but this will take more time than I have tonight before the meeting tomorrow
    """
    cube_obj = cube_utils.CubeData('cii').convert_to_K().convert_to_kms()
    pv_slice = cube_obj.data[:, 200, :]
    print(pv_slice)
    print(pv_slice.shape)
    ax = plt.subplot(111, projection=pv_slice.wcs)
    plt.imshow(pv_slice.to_value(), origin='lower')
    plt.show()


def m16_large_moment0():
    """
    August 25, 2023
    Make some big moment0 maps of CII and CO3-2 to showcase the interesting
    features and places for further study
    """
    # Use the big CII map. Needs some cleaning but generally much larger than the original I was using before.

    # line_stub = 'cii'
    # line_stub = '12co32'
    line_stub = 'c18o10'

    filenames = {
        'cii': "sofia/m16_CII_final_15_5_0p5_clean.fits",
        '12co10': "purplemountain/G17co12.fits", '13co10': "purplemountain/G17co13.fits",
        'c18o10': "purplemountain/G17c18o.fits",
    }
    if line_stub in filenames:
        # Use the custom filename rather than the default
        filename = filenames[line_stub]
    else:
        # Use default filename from cube_utils (many of these are centered around Pillars)
        filename = line_stub

    velocity_intervals = [ # what I have for now.. will add later
        (10, 17), (17, 21), (23.5, 27), (27, 33), (30, 35),
    ]

    cube_obj = cube_utils.CubeData(filename).convert_to_K().convert_to_kms()

    # Can always remove this loop
    for i in range(len(velocity_intervals)):

        vel_lims = tuple(x*kms for x in velocity_intervals[i])
        mom0 = cube_obj.data.spectral_slab(*vel_lims).moment0()

        fig = plt.figure()
        ax = plt.subplot(111, projection=cube_obj.wcs_flat)
        im = ax.imshow(mom0.to_value(), origin='lower', vmin=0, cmap='plasma')
        fig.colorbar(im, ax=ax, label=f"{cube_utils.cubenames[line_stub]} {mom0.unit.to_string('latex_inline')}")
        vel_stub_simple = ".".join(f"{x.to_value():.1f}" for x in vel_lims)
        savename = f"/home/ramsey/Pictures/2023-04-25/mom0_{line_stub}_{vel_stub_simple}.png"
        if not os.path.exists(os.path.dirname(savename)):
            os.makedirs(os.path.dirname(savename))
            print("Created", os.path.dirname(savename))
        fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=make_vel_stub(vel_lims),
            file=__file__, func='m16_large_moment0'))

if __name__ == "__main__":
    m16_large_moment0()
