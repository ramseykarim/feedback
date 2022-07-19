"""
June 29, 2022
Modeled on the code in velocity3d_rcw49.py, but in a separate file because
a lot of things in that file are just so old. I think I didn't even know about
SpectralCube at that point, so it's just deeply old and I'd rather start over
here for M16.

The purpose of this file is to do the same exact thing for M16 that I did for
RCW49 in that previous file. I want a volume rendering of the pillars.
Maybe later, the entire CII region. For now, just pillars.
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
import datetime

from math import ceil

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5

from reproject import mosaicking

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
marcs_colors = ['#377eb8', '#ff7f00','#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

kms = u.km/u.s


from mayavi import mlab


def plot_cube(line_stub, length_scale_mult=None,
    reg_filename='catalogs/central_line_across_shelf.reg'):
    """
    June 29, 2022
    Plot a cube in 3d
    Modeled on velocity3d_rcw49.plot_cube, but uses the stuff I've built since
    for M16.

    :param line_stub: 'cii', '12co10', etc. Keys to the dicts in cube_utils
    :param length_scale_mult: sent to cps2.cutout_subcube, controls the size of
        the subcube
    """
    cube = cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line_stub],
        length_scale_mult=length_scale_mult, reg_filename=reg_filename)
    velocity_axis = cube.world[:, 0, 0][0]
    dec_axis = cube.world[0, :, 0][1]
    ra_axis = cube.world[0, 0, :][2]

    # Right now, we have V, Dec, RA
    # We want RA, V, Dec (for visualization reasons)
    print("before swap", cube.shape)
    data = np.moveaxis(cube.unmasked_data[:], 2, 0)
    print("after swap ", data.shape)

    ragrid, vgrid, dgrid = np.meshgrid(ra_axis.to_value(), velocity_axis.to_value()/150, dec_axis.to_value(), indexing='ij')

    src = mlab.pipeline.scalar_field(ragrid, vgrid, dgrid, data.to_value())
    contour_levels = [10, 20, 30, 40]
    mlab.pipeline.iso_surface(src, contours=contour_levels, opacity=0.2, vmin=contour_levels[0], vmax=contour_levels[-1])



if __name__ == "__main__":
    mlab.figure (bgcolor=tuple([0.2]*3), fgcolor=tuple([0.93]*3), size=(800, 700))
    plot_cube('cii', length_scale_mult=3)
    mlab.show()
