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


def pv(selected_region=0):
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
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits"]
    colors = [marcs_colors[0],] + [marcs_colors[1], marcs_colors[6],]*2
    names = ['[CII]', 'HCO+', 'HCO+ (CII beam)', '12CO(1-0)', '12CO(1-0) (CII beam)']
    short_names = ['cii', 'hcop', 'hcopCONV', 'co10', 'co10CONV']
    levels = [list(np.linspace(15, 40, 9))] + [list(np.linspace(2, 7, 7))]*2 + [list(np.linspace(25, 80, 8))]*2

    # set up the vectors
    reg_filename = catalog.utils.search_for_file("catalogs/p1_threads_pathsandcircles.reg")
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=10*u.arcsec)
    vel_limits = np.array([22, 28])*u.km/u.s

    fig = plt.figure(figsize=(14, 7))

    # Get slices on their native grids; first one will be CII; then unconv,conv for hcop and again for co10
    sl_list_nativegrid = []
    for i, c_fn in enumerate(cube_filenames):
        cube = cube_utils.CubeData(c_fn).convert_to_K().data
        sl = pvextractor.extract_pv_slice(cube.spectral_slab(*vel_limits), pv_path)
        sl.header['CTYPE2'] = 'VRAD' # thanks to the note in m16_pictures.single_parallel_pillar_pvs
        sl_list_nativegrid.append(sl)
    # identify cii slice
    cii_sl = sl_list_nativegrid[0]

    # now loop thru the two lines
    for i, molecule in enumerate(["hcop", "co10"]):
        mol_idx = short_names.index(molecule)
        # get the unconv and conv slices (they are next to each other in the list)
        slices = sl_list_nativegrid[mol_idx:mol_idx+2]
        ax_sl_unconv = plt.subplot2grid((2, 3), (0, 1+i), projection=WCS(slices[0].header))
        ax_sl_conv = plt.subplot2grid((2, 3), (1, 1+i), projection=WCS(slices[1].header))
        axes = [ax_sl_unconv, ax_sl_conv]
        # reproject CII to unconv (conv should have same WCS)
        cii_reproj = reproject_interp((cii_sl.data, cii_sl.header), slices[0].header, return_footprint=False)
        # plot the molecule slices as images on each of axes
        for j, ax, sl in zip(range(len(axes)), axes, slices):
            ax.imshow(sl.data, origin='lower', cmap='Greys')
            ax.contour(sl.data, linewidths=1.2, colors=colors[mol_idx+j])
            ax.contour(cii_reproj, linewidths=1.2, colors=colors[0])
            ax.coords[1].set_format_unit(u.km/u.s)
            ax.coords[0].set_format_unit(u.arcsec)
            ax.coords[0].set_major_formatter('x')
            if j == 1:
                ax.set_xlabel("Offset, from E to W (arcseconds)")
            else:
                ax.set_xlabel(" ")
                ax.set_title(f"PV Diagrams (Greyscale: {names[mol_idx+j]})")
            ax.set_ylabel("V (km/s)")
            ax.tick_params(axis='x', direction='in')
            ax.tick_params(axis='y', direction='in')
            handles = [mpatches.Patch(color=c, label=n) for ii, n, c in zip(range(len(cube_filenames)), names, colors) if ii in (0, mol_idx+j)]
            ax.legend(handles=handles, loc='lower right')


    img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    w = WCS(hdr)
    vlims = dict(vmin=0.1, vmax=0.8)

    ax_img = plt.subplot2grid((2, 3), (0, 0), rowspan=2, projection=w)
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
    pv()


