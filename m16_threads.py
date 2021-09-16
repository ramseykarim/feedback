"""
Created: September 16, 2021
Purpose: to make figures about the two "threads" in Pillar 1 in M16
I didn't want to completely rewrite some otherwise reusable functions
in other files (m16_investigation, m16_deepdive) to do this, so I'll
do the whole thing in here

The general goal, following the 2021-09-15 meeting with Marc, Xander,
and Lee, is:
1) Look at PV diagrams across & along the pillars
    Comment on differences b/w CO(1-0), CII, HCO+, anything else
2) Pick out a few regions along these PV cuts and take spectra.
    Background subtract the CII spectra (see m16_investigation work)
3) With old spectra & new spectra, compare and see if anything changes

These images should also be really great for telling what the threads
are. I can probably repeat this process for PV cuts along the pillars
in addition to across (intended use case).
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
marcs_colors = ['#377eb8', '#ff7f00','#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


def pv(selected_region=0, mol_idx=False):
    """
    September 16, 2021
        Follows from m16_investigation.compare_carma_to_sofia_pv (copy+paste+edit)
        Right now (check 2021-07-15 images) the contour-on-contour PVs look great
        I just want to see CO(1-0) and HCO+ in the same image
        Can see where it goes from there
    """
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    # get all the filenames at once
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcn"), *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits"]
    colors = [marcs_colors[0],] + [marcs_colors[1], marcs_colors[6],]*3
    names = ['[CII]', 'HCN', 'HCN (CII beam)', 'HCO+', 'HCO+ (CII beam)', '12CO(1-0)', '12CO(1-0) (CII beam)']
    short_names = ['cii', 'hcn', 'hcnCONV', 'hcop', 'hcopCONV', 'co10', 'co10CONV']
    levels = [list(np.linspace(15, 40, 9))] + [list(np.linspace(2, 9, 8))]*2 + [list(np.linspace(2, 7, 7))]*2 + [list(np.linspace(25, 80, 8))]*2
    # not always necessary but
    img_for_background = 0

    if mol_idx is False:
        mol_idx = 1 # 1 is hcn, 3 is hcop, 5 is co10
    # mol_idx is an arg, starting at 1 for the molecules
    trim_lists = lambda l : [l[0]] + l[mol_idx:mol_idx+2]
    # trim all lists so we can just loop through them and only get one molecular line
    # if we use multiple lines at once, remove these calls
    cube_filenames = trim_lists(cube_filenames)
    colors = trim_lists(colors)
    names = trim_lists(names)
    levels = trim_lists(levels)

    # set up the vectors
    reg_filename = catalog.utils.search_for_file("catalogs/p1_threads_pathsandcircles.reg")
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=10*u.arcsec)
    vel_limits = np.array([22, 28])*u.km/u.s

    fig = plt.figure(figsize=(14, 7))
    # The two PV slice axes, which will be stacked on the right hand side
    ax_sl_unconv, ax_sl_conv = None, None
    # Stuff from the PV slices
    sl_grid_wcs, sl_grid_shape, sl_grid_header = None, None, None
    # For contours; first one will be CII, second will be unconv, third conv
    contour_args_list, contour_kwargs_list = [], []
    for i, c_fn in enumerate(cube_filenames):
        cube = cube_utils.CubeData(c_fn)
        cube.convert_to_K()
        # should actually use cps2 cutout as wrapper for this and save
        # the one that will make the reference image
        sl = pvextractor.extract_pv_slice(cube.data.spectral_slab(*vel_limits), pv_path)
        if ax_sl_unconv is None:
            sl_grid_wcs = WCS(sl.header)
            sl_grid_header = sl.header
            ax_sl_unconv = plt.subplot2grid((2, 2), (0, 1), colspan=1, projection=sl_grid_wcs)
            ax_sl_conv = plt.subplot2grid((2, 2), (1, 1), colspan=1, projection=sl_grid_wcs)
            sl_grid_shape = sl.data.shape
            contour_args = (sl.data,) # marc suggested that one of the layers should be an image instead of all contours
        else:
            sl.header['CTYPE2'] = 'VRAD' # thanks to the note in m16_pictures.single_parallel_pillar_pvs
            contour_args = (reproject_interp((sl.data, sl.header), sl_grid_header, return_footprint=False),)
        contour_args_list.append(contour_args)
        contour_kwargs_list.append(dict(linewidths=1.2, colors=colors[i], levels=levels[i]))

    print("GRID SHAPE", sl_grid_shape)
    # ax_sl.imshow(*contour_args_list.pop(img_for_background), origin='lower', cmap='viridis')
    ax_sl_unconv.imshow(*contour_args_list[img_for_background], origin='lower', cmap='viridis')
    for i, ax_sl in enumerate([ax_sl_unconv, ax_sl_conv]):
        ax_sl.contour(*contour_args_list[i+1], **contour_kwargs_list[i+1])
        ax_sl.contour(*contour_args_list[0], **contour_kwargs_list[0])
        # ax_sl.imshow(*contour_args_list[i+1], origin='lower', cmap='viridis')

        ax_sl.coords[1].set_format_unit(u.km/u.s)
        ax_sl.coords[0].set_format_unit(u.arcsec)
        ax_sl.coords[0].set_major_formatter('x')
        if i == 1:
            ax_sl.set_xlabel("Offset, from E to W (arcseconds)")
        else:
            ax_sl.set_xlabel(" ")
        ax_sl.set_ylabel("V (km/s)")
        ax_sl.tick_params(axis='x', direction='in')
        ax_sl.tick_params(axis='y', direction='in')

        handles = [mpatches.Patch(color=c, label=n) for ii, n, c in zip(range(len(cube_filenames)), names, colors) if (2-i)!=ii]
        ax_sl.legend(handles=handles, loc='lower right')

    ax_sl_unconv.set_title(f"PV Diagrams")
    # vel_limits = [(24.4, 25.5), (24.8, 25.7), (23.7, 25.8), (25., 27.)][selected_region]
    # cube = cube_utils.CubeData(cube_filenames[0]) # can also use cps2 cutout function
    # img = cube.data.spectral_slab(*(v*u.km/u.s for v in vel_limits)).moment0().to(u.K * u.km / u.s)

    img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    w = WCS(hdr)
    vlims = dict(vmin=0.1, vmax=0.8)

    ax_img = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection=w)
    ax_img.imshow(img, origin='lower', **vlims, cmap='Greys_r')

    ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color='red', linestyle='-', lw=3, transform=ax_img.get_transform('world'))
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    # plt.tight_layout()
    plt.show()
    # fig.savefig(f"/home/rkarim/Pictures/2021-09-16-work/pv_{selected_region}_{short_names[mol_idx]}.png")




if __name__ == "__main__":
    pv(mol_idx=3)


