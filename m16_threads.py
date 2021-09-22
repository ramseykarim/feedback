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
    smooth = False
    if smooth:
        smooth_stub = ".SMOOTH"
    else:
        smooth_stub = ""
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, f"carma/M16.ALL.*subpv{smooth_stub}.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, f"carma/M16.ALL.*subpv.SOFIAbeam{smooth_stub}.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    # get all the filenames at once
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits"]
    colors = [marcs_colors[0],] + [marcs_colors[1], marcs_colors[6],]*2
    names = ['[CII]', 'HCO+', 'HCO+ (CII beam)', '12CO(1-0)', '12CO(1-0) (CII beam)']
    short_names = ['cii', 'hcop', 'hcopCONV', 'co10', 'co10CONV']
    levels = [list(np.linspace(15, 40, 9))] + [list(np.linspace(2, 7, 7))]*2 + [list(np.linspace(25, 80, 8))]*2
    onesigma = [1,] + [0.3,]*2 + [5,]*2

    # set up the vectors
    reg_filename = catalog.utils.search_for_file("catalogs/p1_threads_pathsandcircles.reg")
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=10*u.arcsec)
    vel_limits = np.array([20, 28])*u.km/u.s

    fig = plt.figure(figsize=(14, 9))

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
            ax.imshow(sl.data, origin='lower', cmap='Greys', aspect=((3./4)*sl.shape[1]/sl.shape[0]))
            ax.contour(sl.data, linewidths=1.2, levels=np.linspace(onesigma[mol_idx+j], np.nanmax(sl.data), 6), colors=colors[mol_idx+j])
            ax.contour(cii_reproj, linewidths=1.2, levels=np.linspace(onesigma[0], np.nanmax(cii_reproj), 8), colors=colors[0])
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
            if molecule == 'co10':
                ax.invert_yaxis()


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
    plt.tight_layout()
    plt.show()
    # fig.savefig(f"/home/rkarim/Pictures/2021-09-21-work/pv_{selected_region}{smooth_stub}.png")


def sample_spectra(selected_region=0):
    """
    Created: September 21, 2021
    Figure out how to make selected_region 0 or 1 and use the correct regions
    """
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    # get all the filenames at once
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits"]
    colors = [marcs_colors[0],] + [marcs_colors[1], marcs_colors[6]]
    names = ['[CII]', 'HCO+', 'HCO+ (CII beam)', '$^{12}$CO(1-0)', '$^{12}$CO(1-0) (CII beam)']
    short_names = ['cii', 'hcop', 'hcopCONV', 'co10', 'co10CONV']

    co = 18
    if co == 12:
        co_stub = ""
    elif co == 13:
        co_stub = "_13co10"
        cube_filenames = cube_filenames[:-2] + ["bima/M16.BIMA.13co1-0.fits", "bima/M16.BIMA.13co1-0.SOFIAbeam.fits"]
        names = names[:-2] + ["$^{13}$CO(1-0)", "$^{13}$CO(1-0) (CII beam)"]
        short_names = short_names[:-2] + ['13co10', '13co10CONV']
    elif co == 18:
        co_stub = "_c18o10"
        cube_filenames = cube_filenames[:-2] + ["bima/M16.BIMA.c18o.cm.SMOOTH.fits", "bima/M16.BIMA.c18o.cm.SOFIAbeam.SMOOTH.fits"]
        names = names[:-2] + ["C$^{18}$O(1-0) (smooth)", "C$^{18}$O(1-0) (CII beam, smooth)"]
        short_names = short_names[:-2] + ['c18o10', 'c18o10CONV']


    # set up regions
    reg_filename = catalog.utils.search_for_file("catalogs/p1_threads_pathsandcircles.reg")
    # set up the vector
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=10*u.arcsec)
    circle_reg_list = regions.read_ds9(reg_filename) # only reads 4 circles, doesn't "see" vectors
    # set up the circles
    circle_reg_list = circle_reg_list[selected_region*2:(selected_region+1)*2] # index the correct 2 of them
    # set up background sample circle
    bg_reg_filename = catalog.utils.search_for_file("catalogs/pillar_background_sample_multiple.reg")
    bg_reg = [regions.read_ds9(bg_reg_filename)[0]]

    fig = plt.figure(figsize=(15, 10))

    ax_spec_cii = plt.subplot2grid((3, 3), (0, 1), colspan=2)
    ax_spec_hcop = plt.subplot2grid((3, 3), (1, 1), colspan=2)
    ax_spec_co10 = plt.subplot2grid((3, 3), (2, 1), colspan=2)
    ax_spec_list = [ax_spec_cii, ax_spec_hcop, ax_spec_co10]
    # Get spectra; first one will be CII; then unconv,conv for hcop and again for co10
    # each entry is a list of 2 (left,right circles)
    spectra_lists = []
    # identify cii spectra
    cii_spectra = None
    cii_bg_spectra = None
    for i, c_fn in enumerate(cube_filenames):
        if i in [1, 3]:
            continue
        cube = cube_utils.CubeData(c_fn).convert_to_K().data
        spectra = []
        for reg in circle_reg_list:
            spectra.append(cube.subcube_from_regions([reg]).mean(axis=(1, 2)))
        print(short_names[i], np.std(spectra[0][:20] - np.mean(spectra[0][:20])))
        print(short_names[i], np.std(spectra[1][:20] - np.mean(spectra[1][:20])))
        ax_spec = ax_spec_list[i//2]
        # ax_spec = ax_spec_list[0] if i == 0 else ax_spec_list[(i+1)//2] # if I want to use the unconv versions
        ax_spec.plot(cube.spectral_axis.to(kms).to_value(), spectra[0], color=colors[i//2], linestyle='-', label=f"(NE) {names[i]}")
        ax_spec.plot(cube.spectral_axis.to(kms).to_value(), spectra[1], color=colors[i//2], linestyle='--', label=f"(SW) {names[i]}")
        if i == 0:
            cii_spectra = spectra
            cii_bg_spectra = cube.subcube_from_regions(bg_reg).mean(axis=(1, 2))
            ax_spec.plot(cube.spectral_axis.to(kms).to_value(), cii_bg_spectra, color=colors[i//2], linestyle=':', label=f"BG {names[i]}")
            ax_spec.plot(cube.spectral_axis.to(kms).to_value(), spectra[0]-cii_bg_spectra, color=marcs_colors[3], linestyle='-', label=f"(NE BGsub) {names[i]}")
            ax_spec.plot(cube.spectral_axis.to(kms).to_value(), spectra[1]-cii_bg_spectra, color=marcs_colors[3], linestyle='--', label=f"(SW BGsub) {names[i]}")
    for i, ax_spec in enumerate(ax_spec_list):
        if i == 0:
            ax_spec.set_title("Spectra")
        if i == 2:
            ax_spec.set_xlabel("V (km/s)")
        ax_spec.set_ylabel("T (K)")
        ax_spec.legend()
        ax_spec.set_xlim([18, 30])
        ax_spec.axhline(0, color='k', alpha=0.3)

    # load reference image (HST)
    img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    w = WCS(hdr)
    vlims = dict(vmin=0.1, vmax=0.8)
    # plot reference image
    ax_img = plt.subplot2grid((3, 3), (0, 0), rowspan=3, projection=w)
    ax_img.imshow(img, origin='lower', **vlims, cmap='Greys_r')
    ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color='red', linestyle='-', lw=3, transform=ax_img.get_transform('world'), alpha=0.3)
    # plot circles over reference
    for i, reg in enumerate(circle_reg_list):
        pixreg = reg.to_pixel(w)
        pixreg.plot(ax=ax_img, color='red', linestyle=['-', '--'][i])
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')

    plt.tight_layout()
    plt.show()
    # fig.savefig(f"/home/rkarim/Pictures/2021-09-21-work/sample_spectra_{selected_region}{co_stub}.png")


def sample_spectra_2(selected_region=0):
    """
    Created September 21, 2021
    Sequel to sample_spectra(); that function was to compare CII to HCO+ and CO(1-0)
    The key to that one (and why I am keeping it as is) is that they should be at the same
    resolution: the CII beam.
    This new function, sample_spectra_2, will compare HCO+, HCN, 13CO, and C18O (smooth)
    at their native resolutions to see if they all show the same general pattern
    I'll use the same general format as the above function and the same regions
    """
    smooth = True
    if smooth:
        smooth_stub = ".SMOOTH"
    else:
        smooth_stub = ""
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, f"carma/M16.ALL.*subpv{smooth_stub}.fits"))
    get_mol = lambda mol : [f for f in filepaths if mol in f].pop()
    # get all the filenames at once
    cube_filenames = [get_mol("hcop"), get_mol("hcn"), "bima/M16_12CO1-0_7x4.fits", "bima/M16.BIMA.13co1-0.fits", "bima/M16.BIMA.c18o.cm.SMOOTH.fits"]
    colors = marcs_colors
    multipliers = [1, 1, 1, 3, 10]
    names = ['HCO+', 'HCN', '$^{12}$CO(1-0)', "$^{13}$CO(1-0)" + f" x{multipliers[3]}", "C$^{18}$O(1-0) (smooth)" + f" x{multipliers[4]}"]
    short_names = ['hcop', 'hcn', '12co10', '13co10', 'c18o10']

    # set up regions
    reg_filename = catalog.utils.search_for_file("catalogs/p1_threads_pathsandcircles.reg")
    # set up the vector
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=10*u.arcsec)
    circle_reg_list = regions.read_ds9(reg_filename) # only reads 4 circles, doesn't "see" vectors
    # set up the circles
    circle_reg_list = circle_reg_list[selected_region*2:(selected_region+1)*2] # index the correct 2 of them

    fig = plt.figure(figsize=(15, 10))

    ax_spec_carma = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    ax_spec_bima = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    ax_spec_list = [ax_spec_carma, ax_spec_bima]
    # Plot spectra
    for i, c_fn in enumerate(cube_filenames):
        cube = cube_utils.CubeData(c_fn).convert_to_K().data
        spectral_axis = cube.spectral_axis.to(kms).to_value()
        ax_spec = ax_spec_list[int(i < 2)]
        for reg, linestyle, reg_label in zip(circle_reg_list, ('-', '--'), ('NE', 'SW')):
            spectrum = cube.subcube_from_regions([reg]).mean(axis=(1, 2))
            ax_spec.plot(spectral_axis, spectrum * multipliers[i], color=colors[i], linestyle=linestyle, label=f"({reg_label}) {names[i]}")
    # Mess with axes
    for i, ax_spec in enumerate(ax_spec_list):
        if i == 0:
            ax_spec.set_title("Spectra")
        else:
            ax_spec.set_xlabel("V (km/s)")
        ax_spec.set_ylabel("T (K)")
        ax_spec.legend()
        ax_spec.set_xlim([18, 30])
        ax_spec.axhline(0, color='k', alpha=0.3)

    # load reference image (HST)
    img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    w = WCS(hdr)
    vlims = dict(vmin=0.1, vmax=0.8)
    # plot reference image
    ax_img = plt.subplot2grid((3, 3), (0, 0), rowspan=3, projection=w)
    ax_img.imshow(img, origin='lower', **vlims, cmap='Greys_r')
    ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color='red', linestyle='-', lw=3, transform=ax_img.get_transform('world'), alpha=0.3)
    # plot circles over reference
    for i, reg in enumerate(circle_reg_list):
        pixreg = reg.to_pixel(w)
        pixreg.plot(ax=ax_img, color='red', linestyle=['-', '--'][i])
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')

    plt.tight_layout()
    plt.show()
    # fig.savefig(f"/home/rkarim/Pictures/2021-09-21-work/molecular_spectra_{selected_region}{smooth_stub}.png")


def investigate_cii_background():
    """
    Created: September 21, 2021 (but pretty close to Sept 22 midnight)
    This is pretty similar to what I was doing in m16_investigation (scale_background())
    but I know more what I'm looking for here.
    Following my 2021/09/21 convo with Marc and Xander, I want to check if the CII background
    is pretty consistent all around the pillars, thus justifying the use of background
    subtraction from one region.
    The other background samples have contamination from other features in critical velocity
    ranges, so they can't all be averaged together.
    But, they can be compared by eye to see if the same general background pattern persists.
    Furthermore, they can reveal whether or not there is any gradient in the background
    that might contribute to the velocity shift between the two threads or elsewhere across
    Pillar 1.
    """
    # get the CII cube
    cube = cube_utils.CubeData("sofia/M16_CII_U.fits").convert_to_K().data
    spectral_axis = cube.spectral_axis.to(kms).to_value()
    bg_reg_filename = catalog.utils.search_for_file("catalogs/pillar_background_sample_multiple_3.reg")
    bg_reg_list = regions.read_ds9(bg_reg_filename)
    bg_reg_labels = ['Nominal background', 'BG sample 2', 'BG sample 3']

    fig = plt.figure(figsize=(14, 7))
    ax_spec = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    # plot averaged spectrum from each region
    for i, reg in enumerate(bg_reg_list):
        spectrum = cube.subcube_from_regions([reg]).mean(axis=(1, 2))
        ax_spec.plot(spectral_axis, spectrum, color=marcs_colors[i], label=bg_reg_labels[i])
    # mess with axis
    ax_spec.set_ylabel("T (K)")
    ax_spec.set_xlabel("V (km/s)")
    ax_spec.set_title("[CII] background")
    ax_spec.axhline(0, color='k', alpha=0.3)
    ax_spec.legend()

    # load reference image (HST)
    img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    w = WCS(hdr)
    vlims = dict(vmin=0.1, vmax=0.8)
    # plot reference image
    ax_img = plt.subplot2grid((1, 3), (0, 0), projection=w)
    ax_img.imshow(img, origin='lower', **vlims, cmap='Greys_r')
    # plot circles over reference
    for i, reg in enumerate(bg_reg_list):
        pixreg = reg.to_pixel(w)
        pixreg.plot(ax=ax_img, color=marcs_colors[i])
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')

    plt.tight_layout()
    # plt.show()
    fig.savefig("/home/rkarim/Pictures/2021-09-21-work/cii_background_3.png")


def check_mom1():
    """
    Created September 21, 2021 (though its past midnight sept 22 now)
    Seeing weird things in the previous function about CII background,
    so quickly checking the mom1 map just in case
    """
    vel_limits = (18*kms, 40*kms)
    cube = cps2.cutout_subcube(length_scale_mult=15).spectral_slab(*vel_limits)
    mom1 = cube.moment1()
    mom0 = cube.moment0()
    fig, ax = plt.subplots(subplot_kw=dict(projection=mom1.wcs), figsize=(11, 8))
    im = ax.imshow(mom1.to_value(), origin='lower', vmin=22, vmax=30, cmap='jet')
    fig.colorbar(im, ax=ax)
    ax.contour(mom0.to_value(), colors='k', lws=1.2, levels=10)
    ax.set_title(f"[CII] moments; 1 in color, 0 in contour {make_vel_stub(vel_limits)}")
    ax.set_xlabel("RA"); ax.set_ylabel("Dec")

    # plot circles over reference
    bg_reg_filename = catalog.utils.search_for_file("catalogs/pillar_background_sample_multiple_2.reg")
    bg_reg_list = regions.read_ds9(bg_reg_filename)
    for i, reg in enumerate(bg_reg_list):
        pixreg = reg.to_pixel(mom1.wcs)
        pixreg.plot(ax=ax, lw=2, color='SlateGray')

    patch = cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.06]), misc_utils.get_pixel_scale(mom1.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('k')
    ax.add_artist(patch)

    plt.show()
    # plt.tight_layout()
    # fig.savefig("/home/rkarim/Pictures/2021-09-21-work/cii_mom1.png")



if __name__ == "__main__":
    investigate_cii_background()


