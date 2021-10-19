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
    smooth = True
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
    reg_filename = catalog.utils.search_for_file("catalogs/pillar1_threads_pv_v4.reg")
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=10*u.arcsec)
    vel_limits = np.array([20, 28])*u.km/u.s

    fig = plt.figure(figsize=(16, 9))

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
    plt.subplots_adjust(left=0.05, right=0.95)
    # plt.show()
    fig.savefig(f"/home/rkarim/Pictures/2021-10-12-work/pv_{selected_region}{smooth_stub}.png")


def sample_spectra(selected_region=0):
    """
    Created: September 21, 2021
    Figure out how to make selected_region 0 or 1 and use the correct regions
    2021-10-12: switched from averaging over circles to just using a single spectrum along the line
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

    co = 12
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
    reg_filename = catalog.utils.search_for_file("catalogs/pillar1_threads_pv_v5.reg")
    # set up the vector; width=None
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=None)
    point_reg_list = regions.read_ds9(reg_filename) # only reads 4 points, doesn't "see" vectors
    # set up the POINTS
    point_reg_list = point_reg_list[selected_region*2:(selected_region+1)*2] # index the correct 2 of them
    # set up background sample circle
    bg_reg_filename = catalog.utils.search_for_file("catalogs/pillar_background_sample_multiple_4.reg")
    bg_reg_all = regions.read_ds9(bg_reg_filename)
    selected_bg = (0, 3)
    bg_reg_selected = [bg_reg_all[i] for i in selected_bg]
    bg_reg_labels = [f'BG sample {i+1}' for i in selected_bg]

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
        cube = cube_utils.CubeData(c_fn).convert_to_K()
        cube_wcs_flat = cube.wcs_flat
        cube = cube.data # get SpectralCube from CubeData
        spectra = []
        spectral_axis = cube.spectral_axis.to(kms).to_value()
        for reg in point_reg_list:
            # spectra.append(cube.subcube_from_regions([reg]).mean(axis=(1, 2))) # for circle regions
            point_center = reg.to_pixel(cube_wcs_flat).center.xy
            spectra.append(cube[:, round(point_center[1]), round(point_center[0])])
        print(short_names[i], np.std(spectra[0][:20] - np.mean(spectra[0][:20])))
        print(short_names[i], np.std(spectra[1][:20] - np.mean(spectra[1][:20])))
        ax_spec = ax_spec_list[i//2]
        # ax_spec = ax_spec_list[0] if i == 0 else ax_spec_list[(i+1)//2] # if I want to use the unconv versions
        ax_spec.plot(spectral_axis, spectra[0], color=colors[0], linestyle=('-' if i else '--'), label=f"NE {names[i]}")
        ax_spec.plot(spectral_axis, spectra[1], color=colors[1], linestyle=('-' if i else '--'), label=f"SW {names[i]}")
        if i == 0:
            cii_spectra = spectra
            # now im subtracting different BG spectra from each side of the pillar
            cii_bg_spectra = [cube.subcube_from_regions([bg_reg]).mean(axis=(1, 2)) for bg_reg in bg_reg_selected]
            ax_spec.plot(spectral_axis, cii_bg_spectra[0], color=colors[0], linestyle=':', label=f"BG (for NE) {names[i]}")
            ax_spec.plot(spectral_axis, cii_bg_spectra[1], color=colors[1], linestyle=':', label=f"BG (for SW) {names[i]}")
            cii_bgsub = [spectra[j]-cii_bg_spectra[j] for j in range(2)]
            ax_spec.plot(spectral_axis, cii_bgsub[0], color=colors[0], linestyle='-', label=f"NE BGsub {names[i]}")
            ax_spec.plot(spectral_axis, cii_bgsub[1], color=colors[1], linestyle='-', label=f"SW BGsub {names[i]}")
            spectra = cii_bgsub
        # spec_mask = (spectral_axis > 20) & (spectral_axis < 27.5)
        # mean_velocities = [(np.sum(spectra[j][spec_mask]*spectral_axis[spec_mask])/np.sum(spectra[j][spec_mask])).to_value() for j in range(2)]
        # peak_velocities = [spectral_axis[spec_mask][np.argmax(spectra[j][spec_mask])] for j in range(2)]
        # for j in range(2):
        #     ax_spec.axvline(mean_velocities[j], color=colors[j], linestyle=':', alpha=0.4, label=f'Mean velocity: {mean_velocities[j]:.2f} km/s (vertical line)')
        #     ax_spec.axvline(peak_velocities[j], color=colors[j], linestyle='-', alpha=0.4, label=f'Peak velocity: {peak_velocities[j]:.2f} km/s (vertical line)')
    for i, ax_spec in enumerate(ax_spec_list):
        if i == 0:
            ax_spec.set_title("Spectra")
        if i == 2:
            ax_spec.set_xlabel("V (km/s)")
        ax_spec.set_ylabel("T (K)")
        ax_spec.legend()
        ax_spec.set_xlim([10, 40])
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
    from matplotlib.lines import Line2D
    for i, reg in enumerate(point_reg_list):
        pixreg = reg.to_pixel(w)
        artist = Line2D([pixreg.center.xy[0]], [pixreg.center.xy[1]], marker='x', markersize=8, color=colors[i])
        ax_img.add_artist(artist)
    # plot background samples
    for i, reg in enumerate(bg_reg_selected):
        pixreg = reg.to_pixel(w)
        pixreg.plot(ax=ax_img, color=colors[i], linestyle=':')

    # axis stuff
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')

    plt.tight_layout()
    # plt.show()
    fig.savefig(f"/home/rkarim/Pictures/2021-10-12-work/sample_spectra_{selected_region}{co_stub}_BG{selected_bg[0]+1}-{selected_bg[1]+1}.png")


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
    idx = 4
    bg_reg_filename = catalog.utils.search_for_file(f"catalogs/pillar_background_sample_multiple_{idx}.reg")
    bg_reg_list = regions.read_ds9(bg_reg_filename)
    bg_reg_labels = [f'BG sample {i}' for i in range(1, 5)]

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
    ax_spec.set_ylim([-1, 12])

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
    fig.savefig(f"/home/rkarim/Pictures/2021-09-28-work/cii_background_{idx}.png")


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


def verify_background_doesnt_change_results(selected_region=0):
    """
    Created: September 29, 2021
    Try a bunch of background subtraction combinations and see if the blue excess or
    the peak shift changes
    This is all for CII since CII is the only one with a noticeable background
    """
    cube_filename = ["sofia/M16_CII_U.fits", *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits"]
    colors = [marcs_colors[0],] + [marcs_colors[1], marcs_colors[6]]
    names = ['[CII]', 'HCO+', 'HCO+ (CII beam)', '$^{12}$CO(1-0)', '$^{12}$CO(1-0) (CII beam)']
    short_names = ['cii', 'hcop', 'hcopCONV', 'co10', 'co10CONV']

    co = 12
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
    bg_reg_filename = catalog.utils.search_for_file("catalogs/pillar_background_sample_multiple_4.reg")
    bg_reg_all = regions.read_ds9(bg_reg_filename)
    selected_bg = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)]
    bg_reg_selected = [tuple(bg_reg_all[i] for i in tup) for tup in selected_bg]
    bg_reg_labels = [tuple(f'BG sample {i+1}' for i in tup) for tup in selected_bg]

    fig = plt.figure(figsize=(15, 10))

    ax_blue = plt.subplot2grid((1, 3), (0, 1))
    ax_peak = plt.subplot2grid((1, 3), (0, 2))
    ax_list = [ax_blue, ax_peak]

    blue_excess_vel_lim = (22*kms, 24*kms)

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
        ax_spec.plot(cube.spectral_axis.to(kms).to_value(), spectra[0], color=colors[i//2], linestyle='-', label=f"NE {names[i]}")
        ax_spec.plot(cube.spectral_axis.to(kms).to_value(), spectra[1], color=colors[i//2], linestyle='--', label=f"SW {names[i]}")
        if i == 0:
            # now im subtracting different BG spectra from each side of the pillar
            cii_spectra = spectra
            cii_bg_spectra = [cube.subcube_from_regions([bg_reg]).mean(axis=(1, 2)) for bg_reg in bg_reg_all]
            blue_excess_mask = (cii_spectra.spectral_axis >= blue_excess_vel_lim[0]) & (cii_spectra.spectral_axis <= blue_excess_vel_lim[1])
            print(blue_excess_mask)
            for j, bg_idx_tup in enumerate(selected_bg):
                # the indices contained in bg_idx_tup can be used to index cii_bg_spectra
                bg_reg_tup = bg_reg_selected[j]
                bg_label_tup = bg_reg_labels[j]
                cii_spectra_bgsub = [spec - bg for spec, bg in zip(cii_spectra, [cii_bg_spectra[a] for a in bg_idx_tup])]
                blue_excess_value = np.sum(cii_spectra_bgsub[0][blue_excess_mask]) - np.sum(cii_spectra_bgsub[1][blue_excess_mask])
                peak_shift_value = cii_spectra.spectral_axis[cii_spectra_bgsub[0].argmax()] - cii_spectra.spectral_axis[cii_spectra_bgsub[1].argmax()]

                ax_blue.plot([0], [blue_excess_value], color='grey', marker='x')
                ax_peak.plot([0], [peak_shift_value], color='grey', marker='x')
        else:
            blue_excess_mask = (TODO) ###################################################################################################################### LEFT OF HERE
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
    ax_img = plt.subplot2grid((1, 3), (0, 0), projection=w)
    ax_img.imshow(img, origin='lower', **vlims, cmap='Greys_r')
    ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color='red', linestyle='-', lw=3, transform=ax_img.get_transform('world'), alpha=0.3)
    # plot circles over reference
    for i, reg in enumerate(circle_reg_list):
        pixreg = reg.to_pixel(w)
        pixreg.plot(ax=ax_img, color='red', linestyle=['-', '--'][i])
    # plot background samples
    for i, reg in enumerate(bg_reg_selected):
        pixreg = reg.to_pixel(w)
        pixreg.plot(ax=ax_img, color='grey', linestyle=['-', '--'][i])
    # axis stuff
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')

    plt.tight_layout()
    # plt.show()
    fig.savefig(f"/home/rkarim/Pictures/2021-09-28-work/sample_spectra_{selected_region}{co_stub}_BG{selected_bg[0]+1}-{selected_bg[1]+1}.png")


def center_and_radius_from_reg(reg, wcs, override_radius=None):
    """
    Get the pixel center and radius from a SkyRegion given an image wcs. Option to override the
    SkyRegion's radius before it is converted to a PixelRegion. Original SkyRegion is not modified.
    :param reg: SkyRegion to work with. Function assumes that it has .center and .radius attributes
    :param wcs: image WCS for converting reg from SkyRegion to PixelRegion
    :param override_radius: angular Quantity to override .radius attribute of SkyRegion
        If None, doesn't alter radius
    :returns: center, radius in pixel units
        Center is int (i, j) tuple. Radius is single int
        All 3 values are rounded and converted to int
    """
    # Make deep copy of Region reg so as not to alter original
    reg = reg.copy()
    # Override the radius if we're doing that
    if override_radius is not None:
        reg.radius = override_radius
    # Convert to PixelRegion
    pixreg = reg.to_pixel(wcs)
    # Get the center and radius
    pix_center = pixreg.center # has .x and .y attributes that correspond to j and i
    pix_radius = pixreg.radius
    # Routine to round and convert to int
    clean_pixel = lambda x : int(round(x))
    # Return: (i, j), radius
    return (clean_pixel(pix_center.y), clean_pixel(pix_center.x)), clean_pixel(pix_radius)


def argmax(arr):
    """
    Simple array argmax
    :param arr: n-D array
    returns: tuple(arr(i)) for each coordinate i in ijk order
        If the peak value is unique, each arr(i) is a scalar and can be
        easily converted to an int. If the peak value is not unique, then
        the arr(i) may be multi-valued. However, if this is observed data,
        then the non-unique-peak case will be extraordinarily low.
    """
    return np.where(arr == np.max(arr))


def find_peak(img, guess_center=None, guess_radius=None):
    """
    :param img: 2D array from which to find peak location
        Not written for or tested with 3+ D array
    :param guess_center: int (i, j) tuple center for search box
    :param guess_radius: int radius for search. The search box
        is centered at the guess_center and extends guess_radius
        in each direction, so it's a square of side length
        2*guess_radius
    :raise ValueError: if the peak value is not unique within the
        search box
    :returns: int (i, j) tuple peak location
    """
    # Define square search box with width = 2*radius
    guess_i_lo = guess_center[0] - guess_radius
    guess_i_hi = guess_center[0] + guess_radius
    guess_j_lo = guess_center[1] - guess_radius
    guess_j_hi = guess_center[1] + guess_radius
    # Turn into slices for img array
    box_slices = (slice(guess_i_lo, guess_i_hi), slice(guess_j_lo, guess_j_hi))
    # Get the peak within the search box
    img_peak_loc = argmax(img[box_slices])
    # Translate back to original img pixel coordinates
    img_peak_loc = (img_peak_loc[0] + guess_i_lo, img_peak_loc[1] + guess_j_lo)
    # Return int (i, j) tuple, with an implicit assert that the peak is unique
    try:
        return tuple(int(x) for x in img_peak_loc)
    except TypeError as e:
        raise ValueError(f"Peak value is not unique; this should be a very rare occurance! There are {len(img_peak_loc[0])} pixels with the maximum value within the search box.") from e


def emission_peak_spectra(check_peak=True):
    """
    Created: October 5, 2021 (moms bday is tomorrow)
    Compare the spectra at the emission peaks of CO10 and CII (and maybe other lines)
    """

    if check_peak:
        co10_cube = cps2.cutout_subcube(data_filename="bima/M16_12CO1-0_14x14.fits", length_scale_mult=4)
        cii_cube = cps2.cutout_subcube(length_scale_mult=4)

        full_power_loc_list = []
        full_power_val_list = []
        mom0s = []
        for cube in [co10_cube, cii_cube]:
            full_power_idx = cube.argmax(axis=0)
            full_power_loc = cube.spectral_axis[full_power_idx].to_value()
            full_power_val = cube.max(axis=0).to_value()
            full_power_loc_list.append(full_power_loc)
            full_power_val_list.append(full_power_val)
            mom0s.append(cube.moment0())

        reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/pillar1_emissionpeaks.reg"))

        ax_cii = plt.subplot(121, projection=mom0s[1].wcs)
        ax_co10 = plt.subplot(122, projection=mom0s[0].wcs)
   
        for j, img, mom0, ax in zip(range(2), full_power_val_list, mom0s, [ax_co10, ax_cii]):
            # plot the moment 0 img
            ax.imshow(img, origin='lower', cmap='viridis')
            # overlay BOTH emission peak regions
            for i, reg in enumerate(reg_list):
                pixreg = reg.to_pixel(mom0.wcs)
                pixreg.plot(ax=ax, color=marcs_colors[i])
            # hide the coordinates
            for coord in ax.coords:
                coord.set_ticks_visible(False)
                coord.set_ticklabel_visible(False)
                coord.set_axislabel('')
            # find the actual peaks
            """
            ## ORIGINAL PROCESS
            guess_region = reg_list[j]
            guess_region.radius = 20*u.arcsec
            guess_region = guess_region.to_pixel(mom0.wcs)
            guess_peak = guess_region.center
            guess_peak = (int(round(guess_peak.y)), int(round(guess_peak.x)))
            box_width = int(round(guess_region.radius)) # half width actually
            guess_box = [[guess_peak[1] - box_width, guess_peak[1] + box_width], [guess_peak[0] - box_width, guess_peak[0] + box_width]]
            ax.plot([guess_box[0][0], guess_box[0][1], guess_box[0][1], guess_box[0][0], guess_box[0][0]],
                [guess_box[1][0], guess_box[1][0], guess_box[1][1], guess_box[1][1], guess_box[1][0]], color='k')
            peak_pixel = argmax(img[guess_box[1][0]:guess_box[1][1], guess_box[0][0]:guess_box[0][1]])
            peak_pixel = [peak_pixel[0]+guess_box[1][0], peak_pixel[1]+guess_box[0][0]]
            """
            guess_center, guess_radius = center_and_radius_from_reg(reg_list[j], mom0.wcs, override_radius=20*u.arcsec)
            peak_pixel = find_peak(img, guess_center, guess_radius)
            ax.plot(peak_pixel[1], peak_pixel[0], 'x', color=marcs_colors[2])


        cii_max_loc = argmax(full_power_val_list[1])
        co10_max_loc= argmax(full_power_val_list[0])
        print(cii_max_loc, co10_max_loc) # I LEFT OFF HERE
        plt.savefig("/home/rkarim/Pictures/2021-10-18-work/emissionpeaks_DEBUG.png")
        return

    # this only runs if check_peak is False (I just don't want the entire function indented that far)
    reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/pillar1_emissionpeaks.reg"))
    co10_cube = cps2.cutout_subcube(data_filename="bima/M16_12CO1-0_14x14.fits", length_scale_mult=4)
    cii_cube = cps2.cutout_subcube(length_scale_mult=4)
    cube_list = [co10_cube, cii_cube]
    names = ["$^{12}$CO (1$-$0)", "[CII]"]
    # setup figure
    fig = plt.figure(figsize=(12, 8))
    ax_spec = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax_spec.set_title("Spectra within selected regions")
    ax_spec.set_xlabel("Velocity (km/s)"); ax_spec.set_ylabel("Intensity (K)")
    # ok now grab the spectra and plot them
    spectra = []
    for i, reg in enumerate(reg_list):
        spectrum = cube_list[i].subcube_from_regions([reg]).mean(axis=(1, 2))
        spectra.append(spectrum)
        ax_spec.plot(cube_list[i].spectral_axis.to(kms).to_value(), spectrum / np.max(spectrum), color=marcs_colors[i], label=names[i])

        spec2 = cube_list[i].subcube_from_regions([reg_list[1-i]]).mean(axis=(1, 2))
        ax_spec.plot(cube_list[i].spectral_axis.to(kms).to_value(), spec2 / np.max(spec2), color=marcs_colors[1-i], label=names[i]+" ALT", linestyle=':')
    ax_spec.set_xlim([15, 35])
    # Plot the moment0s for reference; use 23-28 km/s
    vel_limits = (23*kms, 28*kms)
    # make moment0s
    co10_mom0 = co10_cube.spectral_slab(*vel_limits).moment0().to(u.K*kms)
    cii_mom0 = cii_cube.spectral_slab(*vel_limits).moment0().to(u.K*kms)
    # make axes
    ax_cii = plt.subplot2grid((2, 2), (0, 0), projection=cii_mom0.wcs)
    ax_co10 = plt.subplot2grid((2, 2), (0, 1), projection=co10_mom0.wcs)
    # title axes
    ax_cii.set_title(names[1] + " " + make_vel_stub(vel_limits))
    ax_co10.set_title(names[0] + " " + make_vel_stub(vel_limits))
    for mom0, ax in zip([cii_mom0, co10_mom0], [ax_cii, ax_co10]):
        # plot the moment 0 img
        ax.imshow(mom0.to_value(), origin='lower', cmap='Greys_r')
        # overlay BOTH emission peak regions
        for i, reg in enumerate(reg_list):
            pixreg = reg.to_pixel(mom0.wcs)
            pixreg.plot(ax=ax, color=marcs_colors[i])
        # hide the coordinates
        for coord in ax.coords:
            coord.set_ticks_visible(False)
            coord.set_ticklabel_visible(False)
            coord.set_axislabel('')


    bg_reg_all = regions.read_ds9(catalog.utils.search_for_file("catalogs/pillar_background_sample_multiple_4.reg"))
    selected_bg = bg_reg_all[0]
    cii_bg_spectrum = cii_cube.subcube_from_regions([selected_bg]).mean(axis=(1, 2))
    cii_spectrum_bgsub = spectra[1] - cii_bg_spectrum
    ax_spec.plot(cii_cube.spectral_axis.to(kms).to_value(), cii_spectrum_bgsub/np.max(cii_spectrum_bgsub), color=marcs_colors[1], linestyle='--', label='[CII] BGSUB')

    pixreg = selected_bg.to_pixel(cii_mom0.wcs)
    pixreg.plot(ax=ax_cii, color='w', linestyle='--')
    pix_center = pixreg.center.xy
    ax_cii.text(*pix_center, "Background", color='w', fontsize=8, ha='center', va='center')

    ax_spec.legend()
    fig.savefig("/home/rkarim/Pictures/2021-10-05-work/emissionpeaks.png")



def channel_maps_again(*cube_idxs, vel_start=24.5, vel_stop=25.5, grid_shape=None, figsize=None, idx_for_img=None):
    """
    Created: October 11, 2021
    Copied largely from m16_investigation.thin_channel_maps_rb
    Repurposed to make improved channel maps of CII on CO / HCO+
    And also grid them like in m16_pictures.m16_channel_maps
    :param cube_idxs: The first argument in cube_idxs should have the largest footprint
    :param idx_for_img: If None, all contours. If int, index of cube_idxs argument tuple that will be imshow'd
    """
    kms = u.km/u.s
    # copied from the crosscut version
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcn"), *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits", "bima/M16.BIMA.13co1-0.fits", "bima/M16.BIMA.13co1-0.SOFIAbeam.fits", "bima/M16.BIMA.c18o.cm.SMOOTH.fits", "bima/M16.BIMA.c18o.cm.SOFIAbeam.SMOOTH.fits"]
    names = ['CII', 'HCN', 'HCN (CII beam)', 'HCO+', 'HCO+ (CII beam)', '$^{12}$CO(1-0)', '$^{12}$CO(1-0) (CII beam)', "$^{13}$CO(1-0)", "$^{13}$CO(1-0) (CII beam)", "C$^{18}$O(1-0) (smooth)", "C$^{18}$O(1-0) (CII beam, smooth)"]
    short_names = ['cii', 'hcn', 'hcnCONV', 'hcop', 'hcopCONV', 'co10', 'co10CONV', '13co10', '13co10CONV', 'c18o10', 'c18o10CONV']


    if idx_for_img is None:
        colors = [marcs_colors[i] for i in [1, 0, 2]]
        default_text_color = 'k'
    else:
        colors = [marcs_colors[i] for i in [1, 2, 3]]
        default_text_color = 'k'

    contour_levels = [
        np.arange(15, 100, 7.5), # cii
        # np.arange(0.5, 20, 1), np.arange(0.5, 20, 2), # hcn
        # np.arange(0.5, 20, 1), np.arange(0.5, 20, 2), # hcop
        ] + [[0.5, 1.5, 3, 6, 10, 15, 20],]*4 + [
        np.arange(15, 130, 15), np.arange(15, 130, 15), # co10/CONV
        np.arange(5, 60, 5), np.arange(3, 60, 3), # 13co
        np.arange(0.5, 3, 0.5), np.arange(0.5, 3, 0.5), #c18o
    ]
    img_vlims = [
        (0, 50), # cii
        (0, 15), (0, 15), # hcn
        (0, 15), (0, 15), # hcop
        (0, 60), (0, 50), # co10/CONV
        (0, 20), (0, 20), # 13co
        (0, 2.5), (0, 2.5), # c18o
    ]

    def check_idx(idx):
        if not isinstance(idx, int):
            print(idx, end=": ")
            idx = short_names.index(idx)
            print(idx)
        return idx

    cube_idxs = [check_idx(idx) for idx in cube_idxs]

    # c1_idx = 0
    # c2_idx = 2
    unique_label = "-".join([short_names[idx] for idx in cube_idxs]) + ("_img_" if idx_for_img is not None else "")
    cubes = [cube_utils.CubeData(cube_filenames[idx]).convert_to_K().data for idx in cube_idxs]
    vel_start *= kms
    vel_stop *= kms
    channel_width = 0.5*kms

    def make_moment_series(cube):
        return cube_utils.make_moment_series(cube, (vel_start, vel_stop), channel_width)

    cube_moments = [make_moment_series(cube) for cube in cubes]
    """
    cube_moments has an entry for each line (CII, CO10, etc)
    each entry of cube_moments is a list of moment 0 info tuples: (v0, v1, moment)
    """
    # assert len(cube_moments[0]) == 20 # good opportunity for a quick check
    if grid_shape is None:
        grid_shape = (5, 5)
    fig = plt.figure(figsize=figsize)
    ax, im = None, None
    min_cutout_sl = None

    for i in range(len(cube_moments[0])):
        # Iterate through each channel and plot all lines' moments
        v_left, v_right, img1_raw = cube_moments[0][i]
        additional_imgs = []
        footprints = []
        for j in range(1, len(cube_moments)):
            # Iterate through distinct molecular/atomic lines (CII, CO10, etc)
            additional_img_raw = cube_moments[j][i][2]
            additional_img, fp = reproject_interp((additional_img_raw.to_value(), additional_img_raw.wcs), img1_raw.wcs, shape_out=img1_raw.shape, return_footprint=True)
            additional_imgs.append(additional_img)
            footprints.append(fp > 0.5)
        if min_cutout_sl is None:
            min_cutout_sl = misc_utils.minimum_valid_cutout(np.all(footprints, axis=0))
        img1 = img1_raw.to_value()[min_cutout_sl]
        all_imgs = [img[min_cutout_sl] for img in additional_imgs]
        all_imgs.insert(0, img1)
        del additional_imgs

        ax = plt.subplot2grid(grid_shape, (i//grid_shape[1], i%grid_shape[1])) # can't do projection=wcs easily since we used slices

        if idx_for_img is not None:
            vmin = img_vlims[cube_idxs[idx_for_img]][0]
            vmax = img_vlims[cube_idxs[idx_for_img]][1] # alternatively, None for floating vmax
            im = ax.imshow(all_imgs[idx_for_img], origin='lower', cmap='Blues', vmin=vmin, vmax=vmax)
        else:
            ax.imshow(np.zeros_like(all_imgs[0]), origin='lower', vmin=0, vmax=1, cmap='Greys')

        color_idx = 0
        text_x = 0.97
        text_y = 0.90
        text_y_step = 0.05
        text_kwargs = dict(fontsize=12, transform=ax.transAxes, ha='right')
        for j in range(len(cube_moments)):
            if j != idx_for_img:
                ax.contour(all_imgs[j], colors=colors[color_idx], levels=contour_levels[cube_idxs[j]])
                if i == 0:
                    ax.text(text_x, text_y, f"{names[cube_idxs[j]]}", c=colors[color_idx], **text_kwargs)
                color_idx += 1
            else:
                if i == 0:
                    ax.text(text_x, text_y, f"{names[cube_idxs[j]]} (color)", c=default_text_color, **text_kwargs)
            text_y = text_y - text_y_step
        vel_str = make_vel_stub((v_left, v_right))
        ax.text(text_x, 0.95, vel_str, color=default_text_color, **text_kwargs)
        for axis_name in ('x', 'y'):
            ax.tick_params(axis=axis_name, direction='in')
            ax.tick_params(axis=axis_name, labelbottom=False, labelleft=False)
    plt.tight_layout(h_pad=0, w_pad=0, pad=1.01)
    if idx_for_img is not None:
        insetcax = inset_axes(ax, width="5%", height="60%", loc='lower right', bbox_to_anchor=(0, 0.01, 0.97, 1), bbox_transform=ax.transAxes)
        cbar = fig.colorbar(im, cax=insetcax, orientation='vertical')
        insetcax.tick_params(axis='y', colors=default_text_color)
        insetcax.yaxis.set_ticks_position('left')

    fig.savefig(f"/home/rkarim/Pictures/2021-10-18-work/contouroverlay_{unique_label}_channelmaps.png")
    # plt.show()


def m16_pv_again2(selected_set=0, line_stub='cii'):
    """
    did not find anything interesting in those parallel cuts. try across??

    July 12, 2021 update: I might edit this (I will push to github first)
    for use in the upcoming Future of Airborne astro conference.
    The only thing is, I don't think I need multuple paths overlaid?
    Can just reference this function and write it into m16_deepdive.easy_pv
    with each pv in a different subplot and all the paths on the HST

    # colors = ['MediumOrchid', 'LimeGreen', 'DarkOrange', 'MediumBlue']

    pillar1_threads_pv.reg:
    6 total, 4 in sequene and 2 in diff sequence
    See 2021-07-13 pictures and slides 2-3 in google drive
    4 are across P1, and 2 are down each thread (these are the ones I want)

    2021-10-12: moved from pvdiagrams_2.py to m16_threads.py

    :param selected_set: this is either 0 or 1, "0" is first 3 (transverse), "1" is last 3 (vertical, threads)
        "2" will select only the East and West vertical paths (not the Center)
    :param line_stub: indicator of which atomic/molecular line to use. Must be in the list "line_stubs"
    """
    reg_filename = catalog.utils.search_for_file("catalogs/pillar1_threads_pv_v4.reg")
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=None) # no width needed, just use the inherent beam dilation

    fig = plt.figure(figsize=(15, 7))
    if selected_set == 0:
        path_list = path_list[:3]
        path_name = ['North', 'Center', 'South']
        direction_stub = "east to west"
    elif selected_set == 1:
        path_list = path_list[3:]
        path_name = ['East', 'Center', 'West']
        direction_stub = "south to north"
    elif selected_set == 2:
        path_list = [path_list[3], path_list[5]]
        path_name = ['East', 'West']
        direction_stub = "south to north"
    cmap = mpl_cm.get_cmap('viridis')
    colors = [mpl_colors.to_hex(cmap(x)) for x in np.linspace(0., 0.75, len(path_list))]
    colors = marcs_colors
    axes_sl = []
    handles = []
    reg_index = 1
    # data_filename=f"apex/M16_12CO3-2_truncated.fits",
    line_stubs = ['cii', 'hcop', 'hcn', 'nh2p', 'cs']
    line_names = ['CII', 'HCO+', 'HCN', 'NH2+', 'CS']
    if line_stub == "cii":
        line_fn = None
        levels = np.arange(10, 401, 10)
    else:
        line_fn = f"carma/M16.ALL.{line_stub}.sdi.cm.subpv.SMOOTH.fits"
        if selected_set > 0:
            levels = np.sinh(np.linspace(np.arcsinh(1), np.arcsinh(61), 10))
        else:
            levels = np.arange(1, 61, 1)
    line_name = line_names[line_stubs.index(line_stub)]
    vel_lims = (22*u.km/u.s, 28*u.km/u.s)
    subcube = cps2.cutout_subcube(reg_filename=reg_filename, reg_index=reg_index, length_scale_mult=2, data_filename=line_fn).spectral_slab(*vel_lims)
    # if line_fn is not None and "SMOOTH" not in line_fn:
    # # if line_fn is None or "SMOOTH" not in line_fn:
    #     print("SMOOTHING")
    #     subcube = cps2.smooth(subcube)
    cargs_list, ckwargs_list = [], []
    for idx in range(len(path_list)):
        path = path_list[idx]
        sl = pvextractor.extract_pv_slice(subcube, path)
        if idx == 0:
            sl_wcs = WCS(sl.header)
            ax_sl = plt.subplot2grid((1, 3), (0, 1), colspan=2, projection=sl_wcs)
            ax_sl.imshow(np.zeros_like(sl.data), origin='lower', vmin=0, vmax=1, cmap='Greys', aspect=sl.data.shape[1]*0.7/sl.data.shape[0])
            axes_sl.append(ax_sl)
            # im = ax_sl.imshow(sl.data, origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap='Greys')
            ax_sl.coords[1].set_format_unit(u.km/u.s)
            ax_sl.coords[1].set_major_formatter('x.xx')
            ax_sl.coords[0].set_format_unit(u.arcsec)
            ax_sl.coords[0].set_major_formatter('x.xx')
            ax_sl.set_xlabel(f"Offset, from {direction_stub} (arcseconds)")
            ax_sl.set_ylabel("Velocity (km/s)")
            ax_sl.set_title(f"{line_name} PV diagrams")
        contour_args = (sl.data,)
        cargs_list.append(contour_args)
        contour_kwargs = dict(linewidths=1.2, colors=[colors[idx]], alpha=1)
        ckwargs_list.append(contour_kwargs)
    # Figure out contour levels automatically, if not already specified
    if levels is None:
        global_sl_max = max([np.max(x) for (x,) in cargs_list])
        print("GLOBAL MAX",global_sl_max)
        levels = np.linspace(0, global_sl_max, 15)[4:-1:2]
    print("LEVELS",levels)
    for idx in range(len(path_list)):
        contour_args = cargs_list[idx]
        contour_kwargs = ckwargs_list[idx]
        ax_sl.contour(*contour_args, **contour_kwargs, levels=levels)
        handles.append(mpatches.Patch(color=colors[idx], label=path_name[idx]))
    ax_sl.legend(handles=handles)

    img_select = 'sofia'
    if img_select == 'sofia':
        img = subcube.moment0().to(u.K * u.km / u.s)
        w = img.wcs
        img = img.to_value()
        # vlims = dict(vmin=45, vmax=200)
        vlims = dict(vmin=None, vmax=None)
    elif img_select == 'hst':
        img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
        w = WCS(hdr)
        vlims = dict(vmin=0.1, vmax=0.7)
    else:
        raise NotImplementedError
    ax_img = plt.subplot2grid((1, 3), (0, 0), projection=w)
    im = ax_img.imshow(img, origin='lower', **vlims, cmap='Greys_r')
    for idx, p in enumerate(path_list):
        ax_img.plot([c.ra.deg for c in p._coords], [c.dec.deg for c in p._coords], color=colors[idx], transform=ax_img.get_transform('world'), label=path_name[idx])
    ax_img.set_title(f"{line_name} integrated intensity {make_vel_stub(vel_lims)}")
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    # Plot the beam on the image
    patch = subcube.beam.ellipse_to_plot(*(ax_img.transAxes + ax_img.transData.inverted()).transform([0.1, 0.9]), misc_utils.get_pixel_scale(w))
    patch.set_alpha(0.5)
    patch.set_facecolor('grey')
    patch.set_edgecolor('k')
    ax_img.add_artist(patch)
    # Plot the beam as a line in the PV slice
    beam_size_mean = np.sqrt(subcube.beam.major*subcube.beam.minor).to(u.deg).to_value()
    beamtransform = mpl_transforms.blended_transform_factory(ax_sl.get_transform('world'), ax_sl.transAxes)
    x_offset = 5*u.arcsec.to(u.deg)
    # Plot the beam in degrees in the x coord and axes in the y coord
    ax_sl.plot([x_offset, x_offset + beam_size_mean], [0.1, 0.1], transform=beamtransform, color='k', marker='|', alpha=0.5)
    plt.subplots_adjust(left=0.05, right=0.95)
    fig.savefig(f"/home/rkarim/Pictures/2021-10-12-work/pillar_series{selected_set}_{line_stub}_PVs.png")
    # plt.show()


def dissect_sample_spectra():
    """
    Created: October 12, 2021
    Copied sample_spectra() and going to put some Gaussian fitting in here to make a point
    Also going to take larger samples of the threads since I want to show that this is an average thing
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

    co = 12
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
    reg_filename = catalog.utils.search_for_file("catalogs/pillar1_threads_pv_v5_withboxes.reg")
    # set up the BOXES
    box_reg_list = regions.read_ds9(reg_filename)[-2:] # only reads 4 points, doesn't "see" vectors
    # set up background sample circle
    bg_reg_filename = catalog.utils.search_for_file("catalogs/pillar_background_sample_multiple_4.reg")
    bg_reg_all = regions.read_ds9(bg_reg_filename)
    selected_bg = (0, 3)
    bg_reg_selected = [bg_reg_all[i] for i in selected_bg]
    bg_reg_labels = [f'BG sample {i+1}' for i in selected_bg]

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
        cube = cube_utils.CubeData(c_fn).convert_to_K()
        cube_wcs_flat = cube.wcs_flat
        cube = cube.data # get SpectralCube from CubeData
        spectra = []
        spectral_axis = cube.spectral_axis.to(kms).to_value()
        for reg in box_reg_list:
            spectra.append(cube.subcube_from_regions([reg]).mean(axis=(1, 2))) # for boxes
        print(short_names[i], np.std(spectra[0][:20] - np.mean(spectra[0][:20])))
        print(short_names[i], np.std(spectra[1][:20] - np.mean(spectra[1][:20])))
        ax_spec = ax_spec_list[i//2]
        # ax_spec = ax_spec_list[0] if i == 0 else ax_spec_list[(i+1)//2] # if I want to use the unconv versions
        if i == 0:
            cii_spectra = spectra
            # now im subtracting different BG spectra from each side of the pillar
            cii_bg_spectra = [cube.subcube_from_regions([bg_reg]).mean(axis=(1, 2)) for bg_reg in bg_reg_selected]
            cii_bgsub = [spectra[j]-cii_bg_spectra[j] for j in range(2)]
            ax_spec.plot(spectral_axis, cii_bgsub[0], color=colors[0], linestyle='-', label=f"NE BGsub {names[i]}")
            ax_spec.plot(spectral_axis, cii_bgsub[1], color=colors[1], linestyle='-', label=f"SW BGsub {names[i]}")
            spectra = cii_bgsub
        else:
            ax_spec.plot(spectral_axis, spectra[0], color=colors[0], linestyle=('-' if i else '--'), label=f"NE {names[i]}")
            ax_spec.plot(spectral_axis, spectra[1], color=colors[1], linestyle=('-' if i else '--'), label=f"SW {names[i]}")

        spec_mask = (spectral_axis > 23) & (spectral_axis < 27.5)
        mean_velocities = [(np.sum(spectra[j][spec_mask]*spectral_axis[spec_mask])/np.sum(spectra[j][spec_mask])).to_value() for j in range(2)]
        # peak_velocities = [spectral_axis[spec_mask][np.argmax(spectra[j][spec_mask])] for j in range(2)]
        for j in range(2):
            ax_spec.axvline(mean_velocities[j], color=colors[j], linestyle='-', alpha=0.4, label=f'Mean velocity: {mean_velocities[j]:.2f} km/s (vertical line)')
            # ax_spec.axvline(peak_velocities[j], color=colors[j], linestyle='-', alpha=0.4, label=f'Peak velocity: {peak_velocities[j]:.2f} km/s (vertical line)')
        ax_spec.axvline(25.75, color=colors[0], linestyle=':', alpha=0.4)
        ax_spec.axvline(25, color=colors[1], linestyle=':', alpha=0.4)
    for i, ax_spec in enumerate(ax_spec_list):
        if i == 0:
            ax_spec.set_title("Spectra")
        if i == 2:
            ax_spec.set_xlabel("V (km/s)")
        ax_spec.set_ylabel("T (K)")
        ax_spec.legend()
        ax_spec.set_xlim([10, 40])
        ax_spec.axhline(0, color='k', alpha=0.3)

    # load reference image (HST)
    img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    w = WCS(hdr)
    vlims = dict(vmin=0.1, vmax=0.8)
    # plot reference image
    ax_img = plt.subplot2grid((3, 3), (0, 0), rowspan=3, projection=w)
    ax_img.imshow(img, origin='lower', **vlims, cmap='Greys_r')

    ### I don't think I need this anymore # ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color='red', linestyle='-', lw=3, transform=ax_img.get_transform('world'), alpha=0.3)

    # plot circles over reference
    from matplotlib.lines import Line2D
    for i, reg in enumerate(box_reg_list):
        pixreg = reg.to_pixel(w)
        pixreg.plot(ax=ax_img, color=colors[i])
    # plot background samples
    for i, reg in enumerate(bg_reg_selected):
        pixreg = reg.to_pixel(w)
        pixreg.plot(ax=ax_img, color=colors[i], linestyle=':')

    # axis stuff
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')

    plt.tight_layout()
    # plt.show()
    fig.savefig(f"/home/rkarim/Pictures/2021-10-12-work/thread_box_spectra{co_stub}_BG{selected_bg[0]+1}-{selected_bg[1]+1}_meanvelocities.png")



def check_blue_excess_in_cii():
    """
    October 12 (13 now, 12:25AM) 2021
    Just following up on a hunch; dissect_sample_spectra() shows that CII has a blue excess, and I want to compare a moment0 image from 20-24 or so
    to the HCO+ between like 20-30
    """
    hcop_cube = cps2.cutout_subcube(data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.fits", reg_index=0, length_scale_mult=4)
    cii_cube = cube_utils.CubeData("sofia/M16_CII_U.fits").convert_to_K().data.with_spectral_unit(kms)
    # hcop_cube = cube_utils.CubeData("carma/M16.ALL.hcop.sdi.cm.subpv.fits").convert_to_K().data.with_spectral_unit(kms)
    cii_moment = cii_cube.spectral_slab(20*kms, 23*kms).moment0()
    hcop_moment = hcop_cube.spectral_slab(24*kms, 26*kms).moment0()
    cii_reproj = reproject_interp((cii_moment.to_value(), cii_moment.wcs), hcop_moment.wcs, shape_out=hcop_moment.shape, return_footprint=False)
    img1 = np.arcsinh(hcop_moment.to_value())
    img2 = np.arcsinh(cii_reproj)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection=hcop_moment.wcs)
    plt.imshow(img1, origin='lower', vmin=0, cmap='Greys_r')
    plt.contour(img2, cmap='plasma', levels=np.linspace(np.arcsinh(7), np.arcsinh(120), 35), linewidths=0.6, vmin=np.arcsinh(5), vmax=np.arcsinh(15))

    y = 0.9
    dy = 0.052
    x = 0.15
    patch = cii_cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([x, y]), misc_utils.get_pixel_scale(hcop_moment.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('k')
    ax.add_artist(patch)
    ax.text(x, y+dy, "[CII]", transform=ax.transAxes, ha='center', color='w')

    x = 0.25
    patch = hcop_cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([x, y]), misc_utils.get_pixel_scale(hcop_moment.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('k')
    ax.add_artist(patch)
    ax.text(x, y+dy, "HCO+", transform=ax.transAxes, ha='center', color='w')

    # axis stuff
    ax.set_title("[CII] [20-24] km/s blue excess contours over HCO+ [24-26] km/s greyscale")
    for coord in ax.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')

    plt.savefig("/home/rkarim/Pictures/2021-10-12-work/blue_excess.png")


if __name__ == "__main__":
    # vel_lims = dict(vel_start=21.5, vel_stop=22.5)
    # vel_lims = dict(vel_start=19.5, vel_stop=27.5) # production
    # vel_lims = dict(vel_start=24.5, vel_stop=25.5) # testing
    # channel_maps_again('c18o10', 'hcn', **vel_lims, grid_shape=(4, 4), figsize=(20, 20), idx_for_img=None)

    emission_peak_spectra()
