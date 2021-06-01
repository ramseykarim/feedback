"""
Opening up crosscut_2 to avoid circular imports in crosscut.py
Created: November 12, 2020
Original purpose to make cross cuts along and across various moment images of
the pillars
Part of that is to check for systematic pointing offsets between CII, CO, others
"""

import sys
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 10}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib import patches

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D

from . import crosscut
pvdiagrams = crosscut.pvdiagrams
misc_utils = pvdiagrams.misc_utils
catalog = pvdiagrams.catalog
cube_utils = pvdiagrams.cube_utils

from . import cube_pixel_spectra as cps1
from . import cube_pixel_spectra_2 as cps2

mpl_cm = pvdiagrams.mpl_cm
mpl_colors = pvdiagrams.mpl_colors
mpatches = pvdiagrams.mpatches

if __name__ != "__main__":
    # Just a reminder
    raise RuntimeError("Where the hell are you importing this to, dependencies are already on thin ice")


def cut_across_m16_pillar():
    """
    Nov 12, 2020
    Cut across the pillars using the same cuts as the m16_pv_again2 function
        in pvdiagrams.py
    But this time, crosscuts instead of PV
    Do this with moment0 images in CII, CO 3-2 and 1-0, HST, Spitzer, others
    """

    colors, path_list, path_name, vlims, grid_shape, region_name = crosscut.setup_paths(1)

    fig = plt.figure(figsize=(8, 7))
    cco = crosscut.CrossCut(path_list[0], vlims=vlims, log=False)
    cco.setup_figure(fig=fig, xcut_axis=plt.subplot2grid(grid_shape, (0, 1)))
    layers = [
        crosscut.DataLayer("[CII]", cube_utils.CubeData("sofia/M16_CII_U.fits"), cube=True, alpha=0.7),
        crosscut.DataLayer("12CO(1-0)", cube_utils.CubeData("bima/M16_12CO1-0_7x4.fits"), cube=True, alpha=0.7),
        crosscut.DataLayer("12CO(3-2)", cube_utils.CubeData("apex/M16_12CO3-2_truncated.fits"), cube=True, alpha=0.7),
        crosscut.DataLayer("8 um", "spitzer/SPITZER_I4_mosaic.fits"),
        crosscut.DataLayer("F657N", "hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits", alpha=0.2)
    ]
    cco.add_data_layer(*layers)
    cco.update_plot(norm=True)
    cco.switch_axes('xcut')
    plt.ylabel("Normalized intensity")
    plt.title(f"Cross cut $-$ {path_name[0]}")
    cco.plot_image("12CO(1-0)", stretch='linear', line_color=colors[0], subplot_number=121)
    for i, p in enumerate(path_list):
        if i == 0:
            continue
        cco2 = crosscut.CrossCut(p, vlims=vlims, log=False)
        cco2.setup_figure(fig=fig, xcut_axis=plt.subplot2grid(grid_shape, (i, 1)))
        cco2.add_data_layer(*layers)
        cco2.update_plot(norm=True, legend=False)
        cco2.switch_axes('xcut')
        plt.title(f"Cross cut $-$ {path_name[i]}")
        cco2.plot_image(cco, line_color=colors[i])
    cco2.switch_axes('xcut')
    plt.xlabel("Distance along cross-cut (arcseconds)")
    cco.switch_axes('img')
    plt.legend(handles=[mpatches.Patch(color=colors[i], label=path_name[i]) for i in range(4)])
    plt.show()


def cut_across_m16_pillars_again():
    """
    Cut across all pillars, like the Jan ~26, 2021 PV in pvdiagrams_2.try_reproject_pv
    """

    colors, path_list, path_name, vlims, grid_shape, region_name = crosscut.setup_paths(2, select=5)

    fig = plt.figure(figsize=(17, 7))
    cco = crosscut.CrossCut(path_list[0], vlims=vlims, log=False)
    cco.setup_figure(fig=fig, xcut_axis=plt.subplot2grid(grid_shape, (0, 1)))
    fx = lambda x: x - x[0]
    layers = [
        crosscut.DataLayer("[CII]", cube_utils.CubeData("sofia/M16_CII_U_APEXbeam.fits"), cube=True, alpha=0.9, color='r', linestyle='-.', linewidth=1.2),
        crosscut.DataLayer("$^{12}$CO (1$-$0)", cube_utils.CubeData("bima/M16_12CO1-0_APEXbeam.fits"), cube=True, alpha=0.7, color='b', linewidth=1.2),
        crosscut.DataLayer("$^{12}$CO (3$-$2)", cube_utils.CubeData("apex/M16_12CO3-2_truncated.fits"), cube=True, alpha=0.7, color='g', linewidth=1.2),
        crosscut.DataLayer("$^{13}$CO (3$-$2)", cube_utils.CubeData("apex/M16_13CO3-2_truncated.fits"), cube=True, alpha=0.7, color='LimeGreen', linewidth=1),
        crosscut.DataLayer("8 $\mu$m", "spitzer/SPITZER_I4_mosaic.fits", linestyle='--', norm=90, offset=fx, alpha=0.4, linewidth=1),
        # crosscut.DataLayer("F657N", "hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits", alpha=0.3, linestyle='--', norm=90, offset=fx, linewidth=1),
    ]
    cco.add_data_layer(*layers)
    cco.update_plot(norm=True)
    cco.switch_axes('xcut')
    plt.ylabel("Normalized intensity")
    plt.title(f"Cross cut $-$ {path_name[0]}")
    # cco.plot_image("F657N", stretch='linear', line_color=colors[0], subplot_number=121, vlims=(0.1, 0.7), cmap='Greys', cutout=False)
    cco.plot_image("8 $\mu$m", stretch='arcsinh', line_color=colors[0], subplot_number=121, vlims=(80, 400), cmap='Greys', cutout=False)
    for i, p in enumerate(path_list):
        cco2 = crosscut.CrossCut(p, vlims=vlims, log=False)
        cco2.setup_figure(fig=fig, xcut_axis=plt.subplot2grid(grid_shape, (i, 1)))
        cco2.add_data_layer(*layers)
        cco2.update_plot(norm=False, legend=True)
        cco2.switch_axes('xcut')
        plt.title(f"Cross cut $-$ {path_name[i]}")
        cco2.plot_image(cco, line_color=colors[i])
    cco2.switch_axes('xcut')
    plt.xlabel("Distance along cross-cut (arcseconds)")
    cco.switch_axes('img')
    plt.xlabel(" ")
    plt.ylabel(" ")
    for axis_name in ('x', 'y'):
        plt.gca().tick_params(axis=axis_name, labelbottom=False, labelleft=False)
    # plt.legend(handles=[mpatches.Patch(color=colors[i], label=path_name[i]) for i in range(len(path_name))])
    plt.tight_layout()
    plt.show()
    # fig.savefig("/home/ramsey/Pictures/2021-03-01-work/across_all_xcut_5_redo.png")


if __name__ == "__main__":
    cut_across_m16_pillars_again()
