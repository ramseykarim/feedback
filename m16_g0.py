"""
For exploring the stars and radiation field of M16 / NGC 6611

Created: June 16, 2022
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
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
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
from astropy.table import Table, QTable
from astropy import constants as const

import pandas as pd
from io import StringIO

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

from . import queries as vizier_queries

mpl_cm = pvdiagrams.mpl_cm
mpl_colors = pvdiagrams.mpl_colors
mpl_transforms = pvdiagrams.mpl_transforms
mpatches = pvdiagrams.mpatches

make_vel_stub = lambda x : f"[{x[0].to_value():.1f}, {x[1].to_value():.1f}] {x[0].unit}"
kms = u.km/u.s
marcs_colors = ['#377eb8', '#ff7f00','#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


def get_pillar_image(w_to_project=None, shape_to_project=None, line='cii'):
    """
    June 16, 2022
    Get a quick moment image of the pillars to overlay (probably as contours)
    on something else. Optionally reproject the moment image to the argument
    WCS and shape.
    """
    if line == 'cii':
        line_kwargs = dict()
    else:
        line_kwargs = dict(data_filename=cube_utils.cubefilenames[line])
    cube = cps2.cutout_subcube(**line_kwargs, length_scale_mult=6)
    mom0 = cube.spectral_slab(19*kms, 27*kms).moment0()
    if w_to_project is not None:
        mom0_reproj = reproject_interp((mom0.to_value(), mom0.wcs), w_to_project, shape_out=shape_to_project, return_footprint=False)
        result = mom0_reproj
    else:
        result = mom0.to_value()
    return result




def compare_O5s_to_morestars():
    """
    June 16, 2022
    Compare the G0 map from just the two O5s to the map from all stars within
    like 4 arcmin. Goal here is to get an idea of how incorrect the assumption
    is that only the brightest stars matter

    Also compare >4.5 log10 G0 to entire cluster core
    """

    # g0_bright_map_name = f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/g0_hillenbrand_stars_O5s.fits"
    g0_bright_map_name = f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/g0_hillenbrand_stars_fuvgt5.0_ltxarcmin.fits"
    g0_4arcmin_map_name = f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/g0_hillenbrand_stars_ltxarcmin.fits"


    g0_maps, hdrs = list(zip(*[fits.getdata(x, header=True) for x in [g0_bright_map_name, g0_4arcmin_map_name]]))

    w = WCS(hdrs[0])
    map_ratio = g0_maps[1]/g0_maps[0]
    fig = plt.figure()
    ax = plt.subplot(111, projection=w)
    im = ax.imshow(map_ratio, vmin=1, vmax=3)
    fig.colorbar(im, ax=ax, label='ratio of $G_0$ values')
    ax.set_title("Ratio of entire cluster core to log10(G0)>5.0")

    pillar_mom0 = get_pillar_image(w, map_ratio.shape, line='hcop')
    ax.contour(pillar_mom0, colors='k', linewidths=0.9, levels=5, alpha=0.7)

    fig.savefig("/home/ramsey/Pictures/2022-06-16/g0_ratio_clustercore_to_fuvgt5.0.png",
        metadata=catalog.utils.create_png_metadata(title='hcop 19-27 overlaid for reference',
            file=__file__, func="compare_O5s_to_morestars"))


def estimate_los_distance_required_for_star_to_not_matter():
    """
    June 16, 2022
    There are a couple stars within an arcminute of the pillars, and I want to
    know what their l.o.s. distance would have to be for them to not matter so
    much for the total G0 on the pillars
    """
    los_distance = 2.*u.kpc
    radiation_field_1d = 1.6e-3 * u.erg / (u.cm**2 * u.s)

    approx_linear_distances = {401: 1*u.arcmin, 351: 0.6*u.arcmin, 367: 0.3*u.arcmin}
    stellar_types = {401: 'O8.5V', 351: 'B1V', 367: 'O9.5V'}
    log10_FUVsolLum_vals = {401: 4.57, 351: 4.06, 367: 4.35}

    def angular_to_physical_distance(angular_distance):
        return angular_distance*(los_distance/(1*u.rad)).decompose()

    def convert_log10fuv_to_g0(index, radial_physical):
        fuv_lum = (10**log10_FUVsolLum_vals[index])*u.solLum
        projected_physical = angular_to_physical_distance(approx_linear_distances[index])
        squared_physical_distance = projected_physical**2 + radial_physical**2
        result = (fuv_lum / (4*np.pi*squared_physical_distance)) / radiation_field_1d
        return result.decompose()

    approx_cluster_size = [2, 4, 6]*u.arcmin
    print("approximate cluster size: ", angular_to_physical_distance(approx_cluster_size).to(u.pc))

    r_array = np.arange(0, 4, 0.1)*u.pc

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    for idx in stellar_types:
        label = f"{stellar_types[idx]} ({idx})"
        ax.plot(r_array, convert_log10fuv_to_g0(idx, r_array), label=label)
    ax.legend()
    ax.set_ylim(0, 2000)
    ax.set_ylabel("$G_0$ (Habing units)")
    ax.set_xlabel("Line-of-sight distance component from pillars (pc)")

    fig.savefig("/home/ramsey/Pictures/2022-06-16/los_star_distance_g0_estimate.png",
        metadata=catalog.utils.create_png_metadata(title='assuming 401: 1\', 351: 0.6\', 367: 0.3\'',
            file=__file__, func='estimate_los_distance_required_for_star_to_not_matter'))



def estimate_pushing_around_gas_with_stars():
    """
    July 6, 2022
    Do we have the momentum to move the threads over the lifespan of the pillars?
    Depends on: momentum transfer from stars, pillar mass, pillar and star lifespans
    """
    # Make sure the right thing is returned from m16_stars (change the True/False flags to arguments)
    star_info_dict = vizier_queries.m16_stars()
    thread_masses = [7, 2] * u.solMass # East, West
    thread_collecting_area_fractions = np.array([2594., 11272.]) # 1/these. Assume threads are cylinders, did (pi r^2) / (4pi R^2)
    thread_velocity = 1*kms # order of magnitude
    mvflux = star_info_dict['mvflux'][0]
    timescale = (thread_masses*thread_velocity / mvflux).decompose() * thread_collecting_area_fractions
    print()
    print("TIMESCALE TO PUSH")
    print(timescale.to(u.Myr))


if __name__ == "__main__":
    estimate_pushing_around_gas_with_stars()
