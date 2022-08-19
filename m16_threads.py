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


def pv(selected_region=0, pillar=1):
    """
    September 16, 2021
        Follows from m16_investigation.compare_carma_to_sofia_pv (copy+paste+edit)
        Right now (check 2021-07-15 images) the contour-on-contour PVs look great
        I just want to see CO(1-0) and HCO+ in the same image
        Can see where it goes from there
    Revisited January 12 and 25, 2022. Prepping paper.
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
    names = ['[CII]', 'HCO+(1-0)', 'HCO+(1-0) (CII beam)', '12CO(1-0)', '12CO(1-0) (CII beam)']
    short_names = ['cii', 'hcop', 'hcopCONV', 'co10', 'co10CONV']
    # levels = [list(np.linspace(15, 40, 9))] + [list(np.linspace(2, 7, 7))]*2 + [list(np.linspace(25, 80, 8))]*2
    onesigma = [1,] + [0.31, 0.15] + [6.2, 2.6]

    # set up the vectors
    if pillar == 1:
        reg_filename_short = "catalogs/pillar1_threads_pv_v5_withboxes.reg" # Pillar 1
        vel_limits = np.array([20, 28])*u.km/u.s # Pillar 1
    elif pillar == 2:
        reg_filename_short = "catalogs/pillar2_across.reg" # Pillar 2
        vel_limits = np.array([18, 26])*u.km/u.s # Pillar 2
    pillar_stub = f"p{pillar}"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=10*u.arcsec)

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
            ax.contour(sl.data, linewidths=1.2, levels=np.arange(onesigma[mol_idx+j], np.nanmax(sl.data), onesigma[mol_idx+j]*2), colors=colors[mol_idx+j])
            ax.contour(cii_reproj, linewidths=1.2, levels=np.arange(onesigma[0], np.nanmax(cii_reproj), onesigma[0]*2), colors=colors[0])
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
    # fig.savefig(f"/home/rkarim/Pictures/2021-10-12-work/pv_{selected_region}{smooth_stub}.png")
    # 2021-01-12, 2022-01-25, 2022-02-01, 2022-02-23
    fig.savefig(f"/home/ramsey/Pictures/2022-05-23/pv_{pillar_stub}_{selected_region}{smooth_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=f"PV from {reg_filename_short}, 0th contour = 3sig, steps of 2sig",
            file=__file__, func='pv'))


def sample_spectra(selected_region=0):
    """
    Created: September 21, 2021
    Figure out how to make selected_region 0 or 1 and use the correct regions
    2021-10-12: switched from averaging over circles to just using a single spectrum along the line
    2022-01-14: dressing it up a little for the paper
    2022-02-22: added beams
    """
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    # get all the filenames at once
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits"]
    colors = [marcs_colors[0],] + [marcs_colors[1], marcs_colors[6]]
    names = ['[CII]', 'HCO+(1-0)', 'HCO+(1-0) (CII beam)', '$^{12}$CO(1-0)', '$^{12}$CO(1-0) (CII beam)']
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
    reg_filename_short = "catalogs/pillar1_threads_pv_v5.reg"
    # reg_filename_short = "catalogs/pillar1_pointsofinterest_v2.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    # set up the vector; width=None
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=None)
    point_reg_list = regions.Regions.read(reg_filename) # only reads 4 points, doesn't "see" vectors
    # set up the POINTS
    point_reg_list = point_reg_list[selected_region*2:(selected_region+1)*2] # index the correct 2 of them

    fig = plt.figure(figsize=(16, 8))

    ax_spec_cii = plt.subplot2grid((3, 3), (0, 1), colspan=2)
    ax_spec_hcop = plt.subplot2grid((3, 3), (1, 1), colspan=2)
    ax_spec_co10 = plt.subplot2grid((3, 3), (2, 1), colspan=2)
    ax_spec_list = [ax_spec_cii, ax_spec_hcop, ax_spec_co10]
    # Get spectra; first one will be CII; then unconv,conv for hcop and again for co10
    # each entry is a list of 2 (left,right circles)
    spectra_lists = []
    beams = []
    # identify cii spectra
    cii_spectra = None
    cii_bg_spectrum = cps2.get_cii_background()
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

        if i == 0:
            spectra = [s - cii_bg_spectrum for s in spectra]

        ax_spec.plot(spectral_axis, spectra[0], color=colors[0], linestyle='-', label=f"NE {names[i]}")
        ax_spec.plot(spectral_axis, spectra[1], color=colors[1], linestyle='-', label=f"SW {names[i]}")

        beams.append((cube.beam, names[i]))


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
    # ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color='red', linestyle='-', lw=3, transform=ax_img.get_transform('world'), alpha=0.3)
    # plot circles over reference
    from matplotlib.lines import Line2D
    for i, reg in enumerate(point_reg_list):
        pixreg = reg.to_pixel(w)
        artist = Line2D([pixreg.center.xy[0]], [pixreg.center.xy[1]], marker='x', markersize=8, color=colors[i])
        ax_img.add_artist(artist)

    text_x = 0.88
    text_y = 0.95
    text_y_step = 0.07
    for beam, name in beams:
        # Plot the beam
        beam_x = text_x + 0.04
        beam_y = text_y
        patch = beam.ellipse_to_plot(*(ax_img.transAxes + ax_img.transData.inverted()).transform([beam_x, beam_y]), misc_utils.get_pixel_scale(w))
        patch.set_alpha(0.7)
        patch.set_facecolor('LightGray')
        patch.set_edgecolor('LightGray')
        ax_img.add_artist(patch)
        ax_img.text(text_x, text_y, f"{name} beam", transform=ax_img.transAxes, c='w', fontsize=12, ha='right', va='center')
        text_y = text_y - text_y_step
        break ############### for this function, all at CII beam

    # axis stuff
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')

    for i, ax in enumerate(ax_spec_list):
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
        if i < 2:
            ax.xaxis.set_ticklabels([])


    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    # plt.show()
    # fig.savefig(f"/home/rkarim/Pictures/2021-10-12-work/sample_spectra_{selected_region}{co_stub}_BG{selected_bg[0]+1}-{selected_bg[1]+1}.png")
    # 2021-01-14, 2022-01-25
    fig.savefig(f"/home/ramsey/Pictures/2022-02-22/sample_spectra_{selected_region}{co_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=f'regs from {reg_filename_short}, standard cii BGsub',
            file=__file__, func='sample_spectra'))


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
    names = ['HCO+(1-0)', 'HCN', '$^{12}$CO(1-0)', "$^{13}$CO(1-0)" + f" x{multipliers[3]}", "C$^{18}$O(1-0) (smooth)" + f" x{multipliers[4]}"]
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
            blue_excess_mask = ... # TODO ## LEFT OF HERE # but I don't remember what i was doing...
            """
            Revewied this Jan 19, 2022: sadly I no longer remember what I was really trying to do here...
            """
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
    Find the array index of the peak value in a (2D) array
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


def translate_array_index(idx, source_wcs, target_wcs):
    """
    Translate an array index from one image to another using both of their WCS
    Basically, do source_array_index -> world -> target_array_index
    :param idx: int (i, j) tuple array index from the source image
    :param source_wcs: WCS object for the image to which idx refers
    :param target_wcs: WCS object for the image to which we desire the translated index
    """
    return target_wcs.world_to_array_index(source_wcs.array_index_to_world(*idx))


def emission_peak_spectra(check_peak=True):
    """
    Created: October 5, 2021 (moms bday is tomorrow)
    Compare the spectra at the emission peaks of CO10 and CII (and maybe other lines)
    """

    if check_peak:
        # co10_cube = cps2.cutout_subcube(data_filename="bima/M16_12CO1-0_14x14.fits", length_scale_mult=4)
        co10_cube = cps2.cutout_subcube(data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.SOFIAbeam.regrid.fits", length_scale_mult=4)
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
        other_peak = []

        point_regions_to_save = []

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
            guess_center, guess_radius = center_and_radius_from_reg(reg_list[j], mom0.wcs, override_radius=30*u.arcsec)
            peak_pixel = find_peak(img, guess_center, guess_radius) # returned in (i, j) order
            ax.plot(peak_pixel[1], peak_pixel[0], 'x', color=marcs_colors[2])
            other_peak.append(translate_array_index(peak_pixel, mom0.wcs, mom0s[1-j].wcs))
            peak_skycoord = mom0.wcs.array_index_to_world(*peak_pixel)
            point_regions_to_save.append(regions.PointSkyRegion(center=peak_skycoord))
        for j, ax in enumerate([ax_co10, ax_cii]):
            ax.plot(other_peak[1-j][1], other_peak[1-j][0], '+', color=marcs_colors[3])
        ax_cii.plot([0], [0], 'x', color='r')
        ax_co10.plot([1], [1], 'x', color='r')
        plt.show()
        # plt.savefig("/home/rkarim/Pictures/2021-10-19-work/emissionpeaks_DEBUG.png")
        regions.Regions(point_regions_to_save).write(catalog.utils.search_for_file("catalogs/pillar1_emissionpeaks.reg").replace('.reg', '.hcopregrid.moreprecise.reg'))
        return

    # this only runs if check_peak is False (I just don't want the entire function indented that far)
    reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/pillar1_emissionpeaks.reg"))
    # co10_cube = cps2.cutout_subcube(data_filename="bima/M16_12CO1-0_14x14.fits", length_scale_mult=4)
    co10_cube = cps2.cutout_subcube(data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.SOFIAbeam.regrid.fits", length_scale_mult=4)
    cii_cube = cps2.cutout_subcube(length_scale_mult=4)
    cube_list = [co10_cube, cii_cube]
    full_power_val_list = [cube.max(axis=0) for cube in cube_list]
    # names = ["$^{12}$CO (1$-$0)", "[CII]"]
    names = ["HCO+", "[CII]"]
    # setup figure
    fig = plt.figure(figsize=(16, 10))
    ax_spec_co10 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    ax_spec_co10.set_title(f"Spectra at {names[0]} peak location (further South)")
    ax_spec_cii = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    ax_spec_cii.set_title(f"Spectra at {names[1]} peak location (further North)")
    axes = [ax_spec_co10, ax_spec_cii]
    # ok now grab the spectra and plot them
    # this now involves locating the peak temperature using the original region as a guess location
    # Iterate first to find both peaks and translate to both pixel grids
    peak_list = []
    other_peak_list = []
    for i, reg in enumerate(reg_list):
        # Find the location of the peak value for this cube
        peak_pixel = find_peak(full_power_val_list[i].to_value(), *center_and_radius_from_reg(reg, full_power_val_list[i].wcs, override_radius=20*u.arcsec))
        peak_list.append(peak_pixel)
        # Translate this to the other cube's WCS (so this is this cube's peak on the other cube's grid)
        other_peak_pixel = translate_array_index(peak_pixel, full_power_val_list[i].wcs, full_power_val_list[1-i].wcs)
        other_peak_list.append(other_peak_pixel)
    # Iterate again now that we have all the indices we need
    peak_spectra = []
    other_peak_spectra = []
    spectral_axes_values = []
    for i in range(len(reg_list)):
        # Get the spectrum of this cube at this cube's peak
        peak_pixel = peak_list[i]
        peak_spectrum = cube_list[i][:, peak_pixel[0], peak_pixel[1]]
        peak_spectra.append(peak_spectrum)
        # Get the spectrum of this cube at the OTHER cube's peak location
        # Other list is reversed
        other_peak_pixel = other_peak_list[1-i]
        other_peak_spectrum = cube_list[i][:, other_peak_pixel[0], other_peak_pixel[1]]
        other_peak_spectra.append(other_peak_spectrum)


        spectral_axis = cube_list[i].spectral_axis.to(kms).to_value()
        spectral_axes_values.append(spectral_axis)
        # if it's CII (i==1) use ":" linestyle because we still need to do background subtraction
        # also add a label to the unsubtracted CII
        no_bgsub_txt = (" (unsubtracted)" if i else "")
        # Plot this cube's peak spectrum on this Axes
        axes[i].plot(spectral_axis, peak_spectrum / np.max(peak_spectrum), color=marcs_colors[i], label=names[i]+no_bgsub_txt, linestyle=(':' if i else '-'))
        # Plot this cube's spectrum at the other cube's peak location ON THE OTHER AXES
        axes[1-i].plot(spectral_axis, other_peak_spectrum / np.max(other_peak_spectrum), color=marcs_colors[i], label=names[i]+no_bgsub_txt, linestyle=(':' if i else '-'))
        # Some axes stuff
        axes[i].set_xlim([10, 40])
        for v in [23, 24, 25, 26]:
            axes[i].axvline(v, color='grey', alpha=0.15)
        for v in [23.5, 24.5, 25.5]:
            axes[i].axvline(v, color='grey', alpha=0.15, linestyle='--')
        # Only do X label for CO10. If CII, that's the top plot so don't label the X axis
        if i == 0:
            axes[i].set_xlabel("Velocity (km/s)")
        axes[i].set_ylabel("Intensity (normalized)")

    # Plot the moment0s for reference; use 23-28 km/s
    vel_limits = (23*kms, 28*kms)
    # make moment0s
    co10_mom0 = co10_cube.spectral_slab(*vel_limits).moment0().to(u.K*kms)
    cii_mom0 = cii_cube.spectral_slab(*vel_limits).moment0().to(u.K*kms)
    # make axes
    ax_cii = plt.subplot2grid((2, 3), (0, 0), projection=cii_mom0.wcs)
    ax_co10 = plt.subplot2grid((2, 3), (1, 0), projection=co10_mom0.wcs)
    # title axes
    ax_cii.set_title(names[1] + " " + make_vel_stub(vel_limits))
    ax_co10.set_title(names[0] + " " + make_vel_stub(vel_limits))
    img_axes = [ax_co10, ax_cii]
    markers = ['o', 'x']
    for i_reversed, mom0, ax in zip(range(2), [cii_mom0, co10_mom0], img_axes[::-1]):
        i = 1 - i_reversed
        # plot the moment 0 img
        ax.imshow(mom0.to_value(), origin='lower', cmap='Greys_r')
        # plot this cube's peak location
        peak_pixel = peak_list[i]
        ax.plot(peak_pixel[1], peak_pixel[0], markers[i], color='r', label=names[i]+' peak location (this row)')
        # plot this cube's peak location on the OTHER AXES (this is better for consistent legend order)
        other_peak_pixel = other_peak_list[i]
        img_axes[1-i].plot(other_peak_pixel[1], other_peak_pixel[0], markers[i], color='r', label=names[i]+' peak location')
        # hide the coordinates
        for coord in ax.coords:
            coord.set_ticks_visible(False)
            coord.set_ticklabel_visible(False)
            coord.set_axislabel('')

    bg_reg_all = regions.read_ds9(catalog.utils.search_for_file("catalogs/pillar_background_sample_multiple_4.reg"))
    selected_bg = bg_reg_all[0]
    cii_bg_spectrum = cii_cube.subcube_from_regions([selected_bg]).mean(axis=(1, 2))
    cii_peak_spectrum_bgsub = peak_spectra[1] - cii_bg_spectrum
    cii_other_peak_spectrum_bgsub = other_peak_spectra[1] - cii_bg_spectrum

    ax_spec_cii.plot(spectral_axes_values[1], cii_peak_spectrum_bgsub/np.max(cii_peak_spectrum_bgsub), color=marcs_colors[1], linestyle='-', label=names[1]+' with subtraction')
    ax_spec_co10.plot(spectral_axes_values[1], cii_other_peak_spectrum_bgsub/np.max(cii_other_peak_spectrum_bgsub), color=marcs_colors[1], linestyle='-', label=names[1]+' with subtraction')
    ax_spec_cii.plot(spectral_axes_values[1], 0.2*cii_bg_spectrum/np.max(cii_bg_spectrum), color='grey', alpha=0.7, linestyle=':', label=names[1]+' Background')

    pixreg = selected_bg.to_pixel(cii_mom0.wcs)
    pixreg.plot(ax=ax_cii, color='w', linestyle='--')
    pix_center = pixreg.center.xy
    ax_cii.text(*pix_center, "Background", color='w', fontsize=8, ha='center', va='center')
    # Final touches on the img axes
    for ax in img_axes:
        ax.legend(loc='lower center')
    # Final touches on the spectrum axes
    for ax in axes:
        ax.legend()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.96)
    ## Previous work was October 19, 2021
    fig.savefig("/home/ramsey/Pictures/2021-12-09-work/emissionpeaks_cii-hcopregrid.png")



def channel_maps_again(*cube_idxs, vel_start=24.5, vel_stop=25.5, grid_shape=None, figsize=None, idx_for_img=None, level_scaling='log', check_levels=False, **cutout_subcube_kwargs):
    """
    Created: October 11, 2021
    Copied largely from m16_investigation.thin_channel_maps_rb
    Repurposed to make improved channel maps of CII on CO / HCO+
    And also grid them like in m16_pictures.m16_channel_maps
    TODO: remake these and contour a set xsigma above noise. so you have to check the noise actually this time
    :param cube_idxs: The first argument in cube_idxs should have the largest footprint
    :param idx_for_img: If None, all contours. If int, index of cube_idxs argument tuple that will be imshow'd
    :param level_scaling: string, 'linear' or 'log', for how contour levels are distributed
    :param check_levels: bool, if True just print out the noise multipliers for the contours and do nothing else
    """
    kms = u.km/u.s
    # copied from the crosscut version
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    short_names = list(cube_utils.cubefilenames.keys())
    print(short_names)
    cube_filenames = [cube_utils.cubefilenames[k] for k in short_names]
    names = [cube_utils.cubenames[k] for k in short_names]
    # cube_filenames = ["sofia/M16_CII_U.fits",
    #     *get_both_mol("hcn"), *get_both_mol("hcop"),
    #     "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits",
    #     "bima/M16.BIMA.13co1-0.fits", "bima/M16.BIMA.13co1-0.SOFIAbeam.fits",
    #     "bima/M16.BIMA.c18o.cm.SMOOTH.fits", "bima/M16.BIMA.c18o.cm.SOFIAbeam.SMOOTH.fits"]
    # names = ['CII',
    #     'HCN', 'HCN (CII beam)',
    #     'HCO+', 'HCO+ (CII beam)',
    #     '$^{12}$CO(1-0)', '$^{12}$CO(1-0) (CII beam)',
    #     "$^{13}$CO(1-0)", "$^{13}$CO(1-0) (CII beam)",
    #     "C$^{18}$O(1-0) (smooth)", "C$^{18}$O(1-0) (CII beam, smooth)"]
    # short_names = ['cii',
    #     'hcn', 'hcnCONV',
    #     'hcop', 'hcopCONV',
    #     '12co10', '12co10CONV',
    #     '13co10', '13co10CONV',
    #     'c18o10', 'c18o10CONV']


    if idx_for_img is None:
        colors = [marcs_colors[i] for i in [1, 0, 2]]
        default_text_color = 'k'
    else:
        # colors = [marcs_colors[i] for i in [1, 5, 2]]
        colors = ['cyan', 'white']
        # colors = ['k', 'white'] + [marcs_colors[i] for i in [0, 1, 5, 2]]
        default_text_color = 'w'

    """
    I originally wrote these down here, but I moved them first to
    m16_investigation.overlaid_contours_for_offset() and then to
    cube_utils.onesigmas so that I can make sure I'm always using the
    same noise across all functions
    """
    onesigmas = [cube_utils.onesigmas[k] for k in short_names]

    zeroth_contour_sigma = 5
    """
    Have to account for the number of channels in each bin for each line;
    this is a function that can be used once we do that.
    The channel counting happens in make_moment_series, and the noise calc
    happens on the fly right before the contour command
    """
    if level_scaling == 'log':
        contour_stretch_base = 1.822
        contour_stretch_coeff = 6 #3
        contour_levels_multipliers = [zeroth_contour_sigma] + [zeroth_contour_sigma + int(round(contour_stretch_coeff * contour_stretch_base**i)) for i in range(10)]
    elif level_scaling == 'linear':
        contour_sigma_step = 10
        contour_levels_multipliers = [zeroth_contour_sigma + contour_sigma_step*i for i in range(10)]
    print("<CONTOURS AT (xsigma)>")
    print(contour_levels_multipliers)
    print('<end CONTOURS>')
    if check_levels:
        return
    def make_contour_levels(sigma):
        # sigma is 1sigma noise level after accounting for moment
        return [sigma*n for n in contour_levels_multipliers]

    img_vlims = [
        (0, 50), # cii
        (0, 15), (0, 15), # hcn
        (0, 15), (0, 7), # hcop
        (0, 5), (0, 5), # n2hp
        (0, 15), (0, 15), # cs
        (0, 65), (0, 50), # co10/CONV
        (0, 20), (0, 20), # 13co
        (0, 2.5), (0, 2.5), # c18o
        (0, 15), # co65
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
    # cubes = [cube_utils.CubeData(cube_filenames[idx]).convert_to_K().data for idx in cube_idxs]

    ############### length_scale_mult=2.5, reg_filename="catalogs/p1_IDgradients_thru_head.reg", reg_index=2 ####### these are good for focusing on p1a
    cubes = [cps2.cutout_subcube(data_filename=cube_filenames[idx], **cutout_subcube_kwargs) for idx in cube_idxs]
    vel_start *= kms
    vel_stop *= kms
    channel_width = 0.5*kms
    # Use these to create histogram bins using arange, which is exactly how make_moment_series works
    bins = np.arange(vel_start.to_value(), vel_stop.to_value(), channel_width.to_value())

    def make_moment_series(cube):
        return cube_utils.make_moment_series(cube, (vel_start, vel_stop), channel_width, return_nchannels=True)

    cube_moments, channels_per_moment = (zip(*(make_moment_series(cube) for cube in cubes)))
    """
    cube_moments has an entry for each line (CII, CO10, etc)
    each entry of cube_moments is a list of [moment 0 info tuples: (v0, v1, moment)]
    length of list is number of moments

    channels_per_moment has an entry for each line, same as cube_moments
    each entry of channels_per_moment is an array of ints. int indicates the number
        of channel maps that went into creating that moment. for noise purposes
    """
    #### DEBUG 2022-02-01
    # # number of lines
    # print(len(cube_moments))
    # # number of moments
    # print(len(cube_moments[0]))
    # # should be 3: vlo, vhi, moment0_img
    # print(len(cube_moments[0][0]))
    # # size of moment image
    # print(cube_moments[0][0][2].shape)
    # print('='*40)
    # print(channels_per_moment)
    # print('='*40)
    # return cube_moments, channels_per_moment

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
        """
        Reproject everything to the first line's grid
        """
        for j in range(1, len(cube_moments)):
            # Iterate through distinct molecular/atomic lines (CII, CO10, etc)
            additional_img_raw = cube_moments[j][i][2]
            additional_img, fp = reproject_interp((additional_img_raw.to_value(), additional_img_raw.wcs), img1_raw.wcs, shape_out=img1_raw.shape, return_footprint=True)
            additional_imgs.append(additional_img)
            footprints.append(fp > 0.5)
        """
        Isolate the area where all the data is valid (get rid of NaN edges from reproject)
        """
        if min_cutout_sl is None:
            min_cutout_sl = misc_utils.minimum_valid_cutout(np.all(footprints, axis=0))
        img1 = img1_raw.to_value()[min_cutout_sl]
        all_imgs = [img[min_cutout_sl] for img in additional_imgs]
        all_imgs.insert(0, img1)
        del additional_imgs

        ax = plt.subplot2grid(grid_shape, (i//grid_shape[1], i%grid_shape[1])) # can't do projection=wcs easily since we used slices

        """
        Plot background image
        """
        cmap = 'cividis'
        if idx_for_img is not None:
            vmin = img_vlims[cube_idxs[idx_for_img]][0]
            vmax = img_vlims[cube_idxs[idx_for_img]][1] # alternatively, None for floating vmax
            im = ax.imshow(all_imgs[idx_for_img], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax) # Blues
        else:
            ax.imshow(np.zeros_like(all_imgs[0]), origin='lower', vmin=0, vmax=1, cmap='Greys')

        """
        Loop through and plot contours, and add the label text to first Axes
        """
        color_idx = 0
        text_x = 0.88
        text_y = 0.2
        text_y_step = 0.07
        text_kwargs = dict(fontsize=12, transform=ax.transAxes, ha='right', va='center')
        for j in range(len(cube_moments)):
            if j != idx_for_img:
                # Calculate the noise associated with this moment 0 image
                nchannels = channels_per_moment[j][i]
                noise_1sig_channel = onesigmas[cube_idxs[j]]
                cube_dv = np.abs(np.diff(cubes[j].spectral_axis[:2])[0].to(kms).to_value())
                # This calculation is based on the work I did in my notebook on 1/25/2022
                # Moment 0 = integral( I_nu )d_nu
                # The error on a sum is x_err*sqrt(N) if all N elements have the same error x_err
                # Moment 0 error is then x_err*sqrt(N)*d_nu
                noise_1sig_moment = noise_1sig_channel * cube_dv * np.sqrt(nchannels)
                levels = make_contour_levels(noise_1sig_moment)
                # Linestyles: make the first one dashed, everything else solid
                linestyles = ['-'] + ['-']*(len(levels) - 1) # I need to deal with the dashes/negative contours
                ax.contour(all_imgs[j], colors=colors[color_idx], levels=levels, linestyles=linestyles, linewidths=1, alpha=0.9)
                if i == 0:
                    # (colors[color_idx] if not (colors[color_idx] in ('k', 'black') and ) else ...)
                    if (colors[color_idx] in ('k', 'black')) and (idx_for_img is not None) and (cmap in ('magma', 'inferno', 'cividis')):
                        # This complex case statement basically means: if you're about to throw black text on a black background, don't.
                        # Magma and inferno have black/very dark as their low-value colors
                        txt_color = 'white'
                        contour_stub = '(black contours)'
                    else:
                        txt_color = colors[color_idx]
                        contour_stub = '(contours)'
                    ax.text(text_x, text_y, f"{names[cube_idxs[j]]} {contour_stub}", c=txt_color, **text_kwargs)
                color_idx += 1
            else:
                if i == 0:
                    # Text on the first image
                    ax.text(text_x, text_y, f"{names[cube_idxs[j]]} (color)", c=default_text_color, **text_kwargs)

            beam_x = text_x + 0.06 # use 0.04 unless it's super zoomed in
            beam_y = text_y
            if i == 0: # just for this run; otherwise, just do i == 0
                # Plot the beam on the first image
                patch = cubes[j].beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([beam_x, beam_y]), misc_utils.get_pixel_scale(img1_raw.wcs))
                patch.set_alpha(0.5)
                patch.set_facecolor('LightGray')
                patch.set_edgecolor('LightGray')
                ax.add_artist(patch)

            text_y = text_y - text_y_step

        """
        Add velocity text to each Axes
        """
        vel_str = make_vel_stub((v_left, v_right))
        ax.text(text_x, 0.95, vel_str, color=default_text_color, **text_kwargs)
        for axis_name in ('x', 'y'):
            ax.tick_params(axis=axis_name, direction='in')
            ax.tick_params(axis=axis_name, labelbottom=False, labelleft=False)
    plt.tight_layout(h_pad=0, w_pad=0, pad=1.01)
    """
    Add colorbar to last Axes
    """
    if idx_for_img is not None:
        insetcax = inset_axes(ax, width="5%", height="60%", loc='lower right', bbox_to_anchor=(0, 0.01, 0.97, 1), bbox_transform=ax.transAxes)
        cbar = fig.colorbar(im, cax=insetcax, orientation='vertical')
        insetcax.tick_params(axis='y', colors=default_text_color)
        insetcax.yaxis.set_ticks_position('left')
    plt.subplots_adjust(hspace=0, wspace=0)
    # 2021-10-18, 2022-01-24, 2022-02-01, 2022-02-15, 2022-02-22, 2022-03-24, 2022-03-29, 2022-05-10
    fig.savefig(f"/home/ramsey/Pictures/2022-08-11/contouroverlay_{unique_label}_channelmaps_{level_scaling.upper()}.png",
        metadata=catalog.utils.create_png_metadata(title=f'contours {contour_levels_multipliers} xsigma',
            file=__file__, func='channel_maps_again'))
    # plt.show()


def m16_pv_again2(selected_set=0, line_stub='cii', pillar=1):
    """
    did not find anything interesting in those parallel cuts. try across??

    July 12, 2021 update: I might edit this (I will push to github first)
    for use in the upcoming Future of Airborne astro conference (Stuttgart?).
    The only thing is, I don't think I need multuple paths overlaid?
    Can just reference this function and write it into m16_deepdive.easy_pv
    with each pv in a different subplot and all the paths on the HST

    # colors = ['MediumOrchid', 'LimeGreen', 'DarkOrange', 'MediumBlue']

    pillar1_threads_pv.reg:
    6 total, 4 in sequene and 2 in diff sequence
    See 2021-07-13 pictures and slides 2-3 in google drive
    4 are across P1, and 2 are down each thread (these are the ones I want)

    2021-10-12: moved from pvdiagrams_2.py to m16_threads.py
    2022-01-11: revisiting this so I can make the figure on my laptop and put it
    in the paper
    2022-01-25: making all the contours have meaning in terms of noise

    :param selected_set: this is either 0 or 1, "0" is first 3 (transverse), "1" is last 3 (vertical, threads)
        "2" will select only the East and West vertical paths (not the Center)
    :param line_stub: indicator of which atomic/molecular line to use. Must be in the list "line_stubs"
    :param pillar: select pillar
    """
    if pillar == 1:
        reg_filename_short = "catalogs/pillar1_threads_pv_v5_withboxes.reg"
        vel_lims = (22*u.km/u.s, 28*u.km/u.s)
    elif pillar == 2:
        reg_filename_short = "catalogs/pillar2_across.reg"
        vel_lims = (19*u.km/u.s, 25*u.km/u.s)
    pillar_stub = f"p{pillar}"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=None) # no width needed, just use the inherent beam dilation

    fig = plt.figure(figsize=(15, 7))
    if pillar == 1:
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
    elif pillar == 2:
        path_name = ['N', 'Mid-N', 'Mid-S', 'S']
        direction_stub = "east to west"
    cmap = mpl_cm.get_cmap('viridis')
    colors = [mpl_colors.to_hex(cmap(x)) for x in np.linspace(0., 0.75, len(path_list))]
    colors = marcs_colors
    axes_sl = []
    handles = []
    if pillar == 1:
        reg_index = 1
    elif pillar == 2:
        # These aren't necessarily supposed to be the same number
        # It's just coincidence that reg 2 works for pillar 2
        reg_index = 2
    # data_filename=f"apex/M16_12CO3-2_truncated.fits",
    line_stubs = ['cii', 'hcop', 'hcn', 'nh2p', 'cs', '12co10']
    line_names = ['CII', 'HCO+', 'HCN', 'NH2+', 'CS', '$^{12}$CO (1$-$0)']
    if pillar == 1:
        if line_stub == "cii":
            line_fn = None
            levels = np.arange(10, 401, 10) # 1 is 1sigma
        elif line_stub == '12co10':
            line_fn = "bima/M16_12CO1-0_7x4.Kkms.fits"
            levels = np.arange(20, 401, 20) # 5 is 1sigma
        else:
            line_fn = f"carma/M16.ALL.{line_stub}.sdi.cm.subpv.SMOOTH.fits"
            hcop_smooth_noise = 0.3
            # hcop_noise = 0.55
            if selected_set > 0:
                levels = np.sinh(np.linspace(np.arcsinh(hcop_smooth_noise*3), np.arcsinh(61), 10))
            else:
                levels = np.arange(hcop_smooth_noise*3, 61, hcop_smooth_noise*3)
    if pillar == 2:
        if line_stub == 'cii':
            line_fn = None
            cii_noise = cube_utils.onesigmas['cii']
            levels = np.arange(cii_noise*5, 200, cii_noise*5)
        elif line_stub == '12co10':
            line_fn = "bima/M16_12CO1-0_7x4.Kkms.fits"
            co10_noise = cube_utils.onesigmas['12co10']
            levels = np.arange(co10_noise*3, 200, co10_noise*3)
        else:
            line_fn = f"carma/M16.ALL.{line_stub}.sdi.cm.subpv.SMOOTH.fits"
            hcop_smooth_noise = cube_utils.onesigmas['hcopCONV']
            levels = np.arange(hcop_smooth_noise*3, 61, hcop_smooth_noise*3)
    line_name = line_names[line_stubs.index(line_stub)]
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
            if line_stub == '12co10':
                ax_sl.invert_yaxis()
            ax_sl.get_ylim()
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
    # fig.savefig(f"/home/rkarim/Pictures/2021-10-12-work/pillar_series{selected_set}_{line_stub}_PVs.png")
    # 2021-01-12, 2022-01-25, 2022-02-24
    fig.savefig(f"/home/ramsey/Pictures/2022-05-23/pillar_{pillar_stub}_series{selected_set}_{line_stub}SMOOTH_3sig_PVs.png",
        metadata=catalog.utils.create_png_metadata(title=f"PV from {reg_filename_short}, contours are 0th=2sig arcsinh hcopSMOOTH",
            file=__file__, func='m16_pv_again2'))
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


def make_spectrum_series_across_threads():
    """
    January 11, 2022
    I want to try that "series" plot type again, from pvdiagrams.py
    I was working with this type of plot briefly in Fall 2020, but never really
    used it after that. I think it would be a good demonstration of how the
    spectrum evolves from east to west across the two threads, in a way that the
    PV diagram can't really capture
    """
    path_info = pvdiagrams.linear_series_from_ds9(catalog.utils.search_for_file("catalogs/p1_threads_pathsandpoints.reg"), pvpath_width=pvdiagrams.m16_allpillars_series_kwargs['pvpath_width'])
    cube = cube_utils.CubeData("carma/M16.ALL.hcop.sdi.cm.subpv.fits")
    pvdiagrams.run_plot_and_save_series(cube, (17, 30), *path_info,
        "/home/ramsey/Pictures/2022-01-11-work/threadseries/test1.png")
    del cube


def figure_for_hcop_linewidths():
    """
    January 12, 2022
    A figure to show how I got the average single-component linewidths for HCO+
    at native resolution
    """
    # Get cube
    cube = cps2.cutout_subcube(length_scale_mult=2.5, data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.fits")
    # Get regions and convert to pixel coords
    sky_regions = regions.Regions.read(catalog.utils.search_for_file("catalogs/p1_hcop_linewidth_samples.reg"))
    pixel_coords = [tuple(round(x) for x in reg.to_pixel(cube[0, :, :].wcs).center.xy[::-1]) for reg in sky_regions]

    # Set up axes
    fig = plt.figure(figsize=(10, 10))
    ax_img = plt.subplot2grid((2, 2), (0, 0), projection=cube[0, :, :].wcs)
    ax_spec0 = plt.subplot2grid((2, 2), (0, 1))
    ax_spec1 = plt.subplot2grid((2, 2), (1, 0))
    ax_spec2 = plt.subplot2grid((2, 2), (1, 1))
    ax_spec_list = [ax_spec0, ax_spec1, ax_spec2]
    # Make moment 0 to plot
    vel_lims = (19*kms, 27*kms)
    mom0 = cube.spectral_slab(*vel_lims).moment0()
    ax_img.imshow(mom0.to_value(), origin='lower', vmin=0, cmap='plasma')
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    # Initialize things for fitting
    # Make template model for fitting
    g0 = cps2.models.Gaussian1D(amplitude=10, mean=24.5, stddev=0.46,
        bounds={'amplitude': (0, None), 'mean': (20, 30), 'stddev': (0.3, 1.3)})
    g1 = g0.copy()
    g1.mean = 25.5
    g_init = g0
    # cps2.fix_std(g_init)
    # Fitter
    fitter = cps2.fitting.LevMarLSQFitter(calc_uncertainties=True)
    # Spectral axis
    spectral_axis = cube.spectral_axis.to_value()
    # Noise
    noise = 0.546
    weights = np.full(spectral_axis.size, 1.0/noise)
    # Loop through the 3 pixels and plot things
    for idx, (i, j) in enumerate(pixel_coords):
        # Label the point on the reference image
        ax_img.plot([j], [i], 'o', markersize=5, color='w')
        ax_img.text(j+10, i+10, str(idx+1), color='w', fontsize=12, ha='center', va='center')
        ax_spec_list[idx].text(0.9, 0.9, str(idx+1), color='k', fontsize=14, ha='center', va='center', transform=ax_spec_list[idx].transAxes)
        # Extract, fit, and plot spectrum
        spectrum = cube[:, i, j].to_value()
        g_fit = fitter(g_init, spectral_axis, spectrum, weights=weights)
        cps2.plot_noise_and_vlims(ax_spec_list[idx], noise, None)
        cps2.plot_everything_about_models(ax_spec_list[idx], spectral_axis, spectrum, g_fit, noise=noise, dof=(spectral_axis.size - 3))
        ax_spec_list[idx].set_xlabel("Velocity (km/s)")
        ax_spec_list[idx].set_ylabel("HCO+ line intensity (K)")
        ax_spec_list[idx].set_ylim([-2, 12])
    plt.tight_layout()
    fig.savefig("/home/ramsey/Pictures/2022-01-11-work/hcop_single_component_linewidths.png",
        metadata={'Author': "Ramsey Karim", 'Title': "HCO+ single component linewidths",
            'Source': f'{os.path.basename(__file__).replace(".py", "")}.figure_for_hcop_linewidths'})


def test_moment_noise_thing():
    """
    February 1, 2022
    Checking if a moment that only includes 1 channel is different than a channel map
    It should be, since the moment uses the channel width in the calculation
    """
    cube = cps2.cutout_subcube(length_scale_mult=4)
    vel_lims = (24.5*kms, 25.0*kms)
    channel_map_0 = cube[cube.closest_spectral_channel(vel_lims[0]), :, :]
    channel_map_1 = cube[cube.closest_spectral_channel(vel_lims[1]), :, :]
    subcube = cube.spectral_slab(*vel_lims)
    moment0 = subcube.moment0()
    mom0_by_hand = (channel_map_0 + channel_map_1)*(cube.spectral_axis[1]-cube.spectral_axis[0])
    result = mom0_by_hand / moment0
    plt.subplot(131)
    plt.imshow(moment0.to_value(), origin='lower')
    plt.title("spectral_cube")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(result.decompose().to_value(), origin='lower')
    plt.title("by hand / code")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(moment0.to_value(), origin='lower')
    plt.title("by hand")
    plt.colorbar()
    plt.show()


def highlight_threads_moment0():
    """
    Feb 22, 2022 (2/22/2022)
    Similar to m16_deepdive.simple_mom0(), which highlighted certain regions
    around the pillars in different lines.
    This one (for the Arrowhead conference, and maybe the paper later) is
    intented to highlight the threads and the cap in a couple different lines.
    I want an image background (cii?) maybe with leading contours, and then
    molecular lines (hcop?) over it in contour
    Can do background in blues with default blue contours as assists, and orange
    (default) for second line
    """
    # first_line = 'cii'
    # second_line = 'hcop'

    first_line = '13co10'
    second_line = 'cii'

    carma_filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    carma_conv_filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    filenames = {'cii': catalog.utils.search_for_file("sofia/M16_CII_U.fits"),
        'hcop': get_mol('hcop', carma_filepaths), 'hcopCONV': get_mol('hcop', carma_conv_filepaths),
        '12co10': catalog.utils.search_for_file('bima/M16_12CO1-0_7x4.fits'), '12co10CONV': catalog.utils.search_for_file('bima/M16_12CO1-0_14x14.fits'),
        '13co10': catalog.utils.search_for_file('bima/M16_12CO1-0_14x14.fits'), '13co10CONV': catalog.utils.search_for_file('bima/M16.BIMA.13co1-0.SOFIAbeam.fits')}

    names = {'cii': 'CII',
        'hcn': 'HCN', 'hcnCONV': 'HCN (CII beam)',
        'hcop': 'HCO+', 'hcopCONV': 'HCO+ (CII beam)',
        '12co10': '$^{12}$CO(1-0)', '12co10CONV': '$^{12}$CO(1-0) (CII beam)',
        '13co10': "$^{13}$CO(1-0)", '13co10CONV': "$^{13}$CO(1-0) (CII beam)",
        'c18o10': "C$^{18}$O(1-0) (smooth)", 'c18o10CONV': "C$^{18}$O(1-0) (CII beam, smooth)"}

    contour_levels_multipliers_dict = {
        'cii': list(range(10, 101, 10)),
        'hcop': [3, 5, 10, 15, 20, 30, 50, 70, 90],
    }
    contour_levels_multipliers_dict['hcopCONV'] = contour_levels_multipliers_dict['hcop']
    contour_levels_multipliers_dict['13co10'] = contour_levels_multipliers_dict['hcop']

    image_cube = cps2.cutout_subcube(data_filename=filenames[first_line],
        length_scale_mult=2, reg_filename="catalogs/p1_IDgradients_thru_head.reg", reg_index=2)
    wcs_flat = image_cube[0, :, :].wcs

    contour_cube = cps2.cutout_subcube(data_filename=filenames[second_line],
        length_scale_mult=None)

    vel_lims_list = [(22, 23.5), (24, 24.5), (25.5, 26)]
    region_names = ['Blue cap', 'Western thread', 'Eastern thread']
    vel_lims = lambda idx : tuple(x*kms for x in vel_lims_list[idx])

    fig = plt.figure(figsize=(15, 6))
    ax_1 = plt.subplot2grid((1, 3), (0, 0), projection=wcs_flat)
    ax_2 = plt.subplot2grid((1, 3), (0, 1), projection=wcs_flat)
    ax_3 = plt.subplot2grid((1, 3), (0, 2), projection=wcs_flat)
    axes = [ax_1, ax_2, ax_3]

    for i in range(len(region_names)):
        # Loop through the three regions
        # Replicate the cube_utils.make_moment_series functionality to get
        # nchannels for each moment 0 map
        current_vel_lims = vel_lims(i)
        spectral_slab = image_cube.spectral_slab(*current_vel_lims)
        # Noise
        nchannels = spectral_slab.shape[0]
        channel_noise = cube_utils.onesigmas[first_line]
        cube_dv = np.abs(np.diff(image_cube.spectral_axis[:2]))[0].to(kms).to_value()
        moment_noise = channel_noise * cube_dv * np.sqrt(nchannels)
        # Make and plot moment
        mom0 = spectral_slab.moment0().to(u.K*kms)
        img_data = mom0.to_value()
        im = axes[i].imshow(img_data/moment_noise, origin='lower', cmap='Blues', vmin=0)
        # Colorbar
        insetcax = inset_axes(axes[i], width='90%', height='5%', loc='lower center',
            bbox_to_anchor=(0, 1.01, 1, 1), bbox_transform=axes[i].transAxes, borderpad=0)
        cbar = fig.colorbar(im, cax=insetcax, orientation='horizontal')
        cbar.set_label("Integrated Intensity (K km/s)", labelpad=-45, fontsize=8)
        insetcax.xaxis.set_ticks_position('top')
        for coord in ('x', 'y'):
            axes[i].tick_params(axis=coord, direction='in')
        # Title
        axes[i].text(0.07, 0.92, region_names[i] + ' ' + make_vel_stub(current_vel_lims),
            transform=axes[i].transAxes, fontsize=11)
        if i == 0:
            axes[i].text(0.07, 0.87, names[first_line]+' in color and grey contours',
                transform=axes[i].transAxes, fontsize=11)
            axes[i].text(0.07, 0.82, names[second_line], color=marcs_colors[1],
                transform=axes[i].transAxes, fontsize=11)
        # Contour at 3 sigma
        contour_levels_multipliers = contour_levels_multipliers_dict[first_line]
        levels = [moment_noise*x for x in contour_levels_multipliers]
        linestyles = ['--'] + ['-']*(len(levels) - 1)
        axes[i].contour(img_data, levels=levels, linestyles=linestyles, linewidths=0.75, alpha=0.5, colors='grey')


        # Plot second line in contour
        contour_spectral_slab = contour_cube.spectral_slab(*current_vel_lims)
        # Noise
        nchannels = contour_spectral_slab.shape[0]
        channel_noise = cube_utils.onesigmas[second_line]
        cube_dv = np.abs(np.diff(contour_cube.spectral_axis[:2]))[0].to(kms).to_value()
        moment_noise = channel_noise * cube_dv * np.sqrt(nchannels)
        # Make moment
        contour_mom0_raw = contour_spectral_slab.moment0().to(u.K*kms)
        # Reproject
        contour_map_reproj = reproject_interp((contour_mom0_raw.to_value(), contour_mom0_raw.wcs), wcs_flat, img_data.shape, return_footprint=False)
        # Plot
        contour_levels_multipliers = contour_levels_multipliers_dict[second_line]
        levels = [moment_noise*x for x in contour_levels_multipliers]
        axes[i].contour(contour_map_reproj, levels=levels, linewidths=1, alpha=1, colors=marcs_colors[1], linestyles=linestyles)

        if i > 0:
            axes[i].coords[1].set_ticklabel_visible(False)
            axes[i].coords[1].set_axislabel(' ')
        else:
            axes[i].coords[1].set_axislabel('Dec')
        axes[i].coords[0].set_axislabel("RA")

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(f"/home/ramsey/Pictures/2022-02-22/{first_line}-{second_line}-overlay.png",
        metadata=catalog.utils.create_png_metadata(title=f'cii contours in 10s from 10, {second_line} {contour_levels_multipliers} (sigma)',
            file=__file__, func="highlight_threads_moment0"))


def pv_again(selected_set="across-wide", pillar=1, clip_bright=False, contour='cii', molecular_line='hcop', smooth=True):
    """
    February 23, 2022
    Copied m16_threads.pv() and I want to change the colors and some other stuff
    I want to get rid of the convolved molecular lines and change it to a second
    PV cut across the pillar body.
    This figure should be definitive evidence that there is a double-thread
    structure in the molecular gas
    Selected regions default to (0, 1), which should work for both P1 and P2
    """
    if smooth:
        smooth_stub = ".SMOOTH"
    else:
        smooth_stub = ""

    if pillar == 1:
        if selected_set == "across":
            selected_regions = (0, 1)
        elif selected_set == "across-wide":
            selected_regions = (0, 2)
        elif selected_set == "along":
            selected_regions = (3, 5)
    elif pillar == 2:
        if selected_set == 'N':
            selected_regions = (0, 1)
        elif selected_set == 'S':
            selected_regions = (2, 3)
        elif selected_set == 'baseN':
            selected_regions = (3, 4)
        elif selected_set == 'baseS':
            selected_regions = (5, 6)
    try:
        selected_regions
    except NameError as e:
        raise RuntimeError(f"You forgot to set selected_set ({selected_set}) to something that makes sense for this pillar ({pillar})") from e
    else:
        pass

    try:
        filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, f"carma/M16.ALL.*subpv{smooth_stub}.fits"))
        filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, f"carma/M16.ALL.*subpv.SOFIAbeam{smooth_stub}.fits"))
        get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
        get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
        mol_fn = get_mol(molecular_line, filepaths)
    except IndexError:
        mol_fn = cube_utils.cubefilenames[molecular_line]
    # get all the filenames at once
    cube_filenames = ["sofia/M16_CII_U.fits", mol_fn, "bima/M16_12CO1-0_7x4.fits"]
    colors = [marcs_colors[0], marcs_colors[1], marcs_colors[6],]
    short_names = ['cii', molecular_line, '12co10']
    names = [cube_utils.cubenames[x] for x in short_names]
    # levels = [list(np.linspace(15, 40, 9))] + [list(np.linspace(2, 7, 7))]*2 + [list(np.linspace(25, 80, 8))]*2
    onesigma = [cube_utils.onesigmas[k] for k in short_names]

    # set up the vectors
    if pillar == 1:
        reg_filename_short = "catalogs/pillar1_threads_pv_v6_withboxes.reg" # Pillar 1
        vel_limits = np.array([20, 28])*u.km/u.s # Pillar 1
    elif pillar == 2:
        reg_filename_short = "catalogs/pillar2_across.reg" # Pillar 2
        vel_limits = np.array([19, 25])*u.km/u.s # Pillar 2
    pillar_stub = f"p{pillar}"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    pv_paths = []
    for selected_region in selected_regions:
        pv_paths.append(pvdiagrams.path_from_ds9(reg_filename, selected_region, width=None)) # or 10 arcsec

    fig = plt.figure(figsize=(16, 9))

    # Get slices on their native grids; first two will be CII (reg0, reg1), then hcop, then 12co10 (6 total)
    sl_list_nativegrid = []
    beams = []
    for i, c_fn in enumerate(cube_filenames):
        cube = cube_utils.CubeData(c_fn).convert_to_K().data
        beams.append((cube.beam, names[i]))
        for j in range(len(selected_regions)):
            sl = pvextractor.extract_pv_slice(cube.spectral_slab(*vel_limits), pv_paths[j])
            sl.header['CTYPE2'] = 'VRAD' # thanks to the note in m16_pictures.single_parallel_pillar_pvs
            sl_list_nativegrid.append(sl)

            ##########DEBUG
    #         print(WCS(sl.header).pixel_to_world_values([0], [0]))
    #         print(WCS(sl.header).world_to_pixel_values([0], [26000]))
    # return

    # identify cii slices
    cii_sl_list = sl_list_nativegrid[0:2]

    # now loop thru the two lines
    for i, molecule in enumerate(short_names):
        # This is basically looping across each column of plots (2 total columns)
        if i == 0:
            # Skip CII but preserve the i indexing from the previous loop
            continue
        # get the slide for each region (they are next to each other in the list)
        slices = sl_list_nativegrid[i*2:(i+1)*2]
        # The top axis (reg 0)
        ax_sl_0 = plt.subplot2grid((2, 3), (0, i), projection=WCS(slices[0].header))
        # The bottom axis (reg 1)
        ax_sl_1 = plt.subplot2grid((2, 3), (1, i), projection=WCS(slices[1].header))
        axes = [ax_sl_0, ax_sl_1]
        # plot the molecule slices as images on each of axes
        for j, ax, sl in zip(range(len(axes)), axes, slices):
            # reproject CII to each slice; cannot guarantee that they have same WCS
            # cii_sl_list has two elements, one for each PV path. j indexes the PV paths.
            cii_reproj = reproject_interp((cii_sl_list[j].data, cii_sl_list[j].header), sl.header, return_footprint=False)
            # Save this for later
            sl_wcs = WCS(sl.header)
            # Create levels that are tied to noise

            zeroth_contour_sigma = 3
            contour_sigma_step = 3
            if contour == 'molecular':
                molecular_levels = np.arange(onesigma[i]*zeroth_contour_sigma, np.nanmax(sl.data)*1.2, onesigma[i]*contour_sigma_step)
                molecular_linestyles = ['--'] + ['-']*(len(molecular_levels) - 1)
            elif contour == 'cii':
                cii_levels = np.arange(onesigma[0]*zeroth_contour_sigma, np.nanmax(cii_reproj)*1.2, onesigma[0]*contour_sigma_step)
                cii_linestyles = ['--'] + ['-']*(len(cii_levels) - 1)

            # Plot things
            vlims = dict(vmin=0)
            clipped = False
            if clip_bright and 'along' in selected_set and i == 1:
                vlims['vmax'] = 8
                clipped = True
            im = ax.imshow(sl.data, origin='lower', cmap='magma', aspect=((3./4)*sl.shape[1]/sl.shape[0]), **vlims)

            insetcax = inset_axes(ax, width="80%", height="100%", loc='center', bbox_to_anchor=(1.0, 0, 0.05, 1), bbox_transform=ax.transAxes)
            cbar = fig.colorbar(im, cax=insetcax, orientation='vertical')
            # insetcax.tick_params(axis='y', colors='k')
            # insetcax.yaxis.set_ticks_position('left')


            # No contour on molecular
            if contour == 'molecular':
                ax.contour(sl.data, linewidths=1.2, levels=molecular_levels, colors=marcs_colors[0])
            elif contour == 'cii':
                ax.contour(cii_reproj, linewidths=1.2, levels=cii_levels, colors='white', linestyles=cii_linestyles)
            ax.coords[1].set_format_unit(u.km/u.s)
            ax.coords[0].set_format_unit(u.arcsec)
            ax.coords[0].set_major_formatter('x')
            if j == 1:
                if 'along' in selected_set:
                    ax.set_xlabel("Offset, from S to N (arcseconds)", fontweight='bold')
                elif 'across' in selected_set:
                    ax.set_xlabel("Offset, from E to W (arcseconds)", fontweight='bold')
            else:
                ax.set_xlabel(" ")
                ax.set_title(f"PV Diagrams (Color: {names[i]})", fontweight='bold')
            if i == 1:
                ax.set_ylabel("V (km/s)", fontweight='bold')
            else:
                ax.set_ylabel(" ")
            ax.tick_params(axis='x', direction='in', color='w')
            ax.tick_params(axis='y', direction='in', color='w')
            if i == 1 and j == 0:
                # handles = [mpatches.Patch(color='w', label=names[0])]
                # ax.legend(handles=handles, loc='lower right')
                text_x = 0.2 if 'across' in selected_set else 0.5
                if contour == 'molecular':
                    msg = "Contours show same data"
                elif contour == 'cii':
                    msg = "[C II] in white contours"
                ax.text(text_x, 0.06, msg, transform=ax.transAxes, color='w', fontsize=10)
            if molecule == '12co10':
                ax.invert_yaxis()
            # Make some velocity grid lines for easier visual identification of gradients
            if pillar == 1:
                v_grid_range = (23, 28)
            elif pillar == 2:
                v_grid_range = (20, 25)
            for v in range(*v_grid_range):
                ax.axhline(sl_wcs.world_to_pixel_values(0, v*1000)[1], color='grey', alpha=0.8)

    img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    w = WCS(hdr)
    vlims = dict(vmin=0.1, vmax=0.8)

    ax_img = plt.subplot2grid((2, 3), (0, 0), rowspan=2, projection=w)
    ax_img.imshow(img, origin='lower', **vlims, cmap='Greys_r')

    # Put beam patch on ref figure
    beam_x, beam_y = 0.92, 0.96
    beam_y_step = -0.06
    pad_x = -0.04
    for beam, name in beams:
        patch = beam.ellipse_to_plot(*(ax_img.transAxes + ax_img.transData.inverted()).transform([beam_x, beam_y]), misc_utils.get_pixel_scale(w))
        patch.set_alpha(0.8)
        patch.set_facecolor('white')
        patch.set_edgecolor('grey')
        ax_img.add_artist(patch)
        ax_img.text(beam_x+pad_x, beam_y, name+" beam", transform=ax_img.transAxes, ha='right', va='center', color='w', fontsize=10)
        beam_y += beam_y_step

    if selected_set == "across-wide":
        ax_img.text(0.98, 0.01, "Paths separated by 14\", [C II] beam is 14\"", transform=ax_img.transAxes, ha='right', va='bottom', color='w', fontsize='8')
    elif selected_set == "across":
        ax_img.text(0.98, 0.01, "Paths separated by ~11\", [C II] beam is 14\"", transform=ax_img.transAxes, ha='right', va='bottom', color='w', fontsize='8')

    handles = []
    path_colors = marcs_colors[:2]
    path_labels = ['top row', 'bottom row']
    linestyles = ['-', '--']
    for j, pv_path in enumerate(pv_paths):
        ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color=path_colors[j], linestyle=linestyles[j], lw=3, transform=ax_img.get_transform('world'), label=path_labels[j])
        # handles.append(mpatches.Patch(color=path_colors[j], label=path_labels[j]))
    ax_img.legend(loc='lower left')

    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    plt.subplots_adjust(left=0.05, right=0.95)
    clip_stub = "-clipped" if clipped else ''
    contour_stub = '' if contour == 'cii' else '-molecularcontours'
    molecular_line_stub = '' if molecular_line == 'hcop' else f"-{molecular_line}"
    # plt.show()
    # fig.savefig(f"/home/rkarim/Pictures/2021-10-12-work/pv_{selected_region}{smooth_stub}.png")
    # 2021-01-12, 2022-01-25, 2022-02-01, 2022-02-27, 2022-04-28, 2022-05-23, 2022-07-22
    fig.savefig(f"/home/ramsey/Pictures/2022-08-11/pv_{pillar_stub}_{selected_set}{clip_stub}{molecular_line_stub}{contour_stub}{smooth_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=f"PV from {reg_filename_short}, 0th contour = {zeroth_contour_sigma}sig, steps of {contour_sigma_step}sig",
            file=__file__, func='pv_again'))


if __name__ == "__main__":
    # vel_lims = dict(vel_start=21.5, vel_stop=22.5)
    # vel_lims = dict(vel_start=19.5, vel_stop=27.5) # production at 4,4
    vel_lims = dict(vel_start=20, vel_stop=27.5) # production at 3,5
    # vel_lims = dict(vel_start=24.5, vel_stop=25.5) # testing
    ###
    # fig_params = dict(grid_shape=(4, 4), figsize=(25, 25))
    fig_params = dict(grid_shape=(3, 5), figsize=(25, 15))
    # fig_params = dict(grid_shape=(3, 5), figsize=(25, 13)) # just for the zoomed in version
    ###
    args = channel_maps_again('co65', 'hcop', **vel_lims, **fig_params, idx_for_img=0, level_scaling='log', check_levels=False, length_scale_mult=None)

    # highlight_threads_moment0()


    # pv_again(selected_set='across-wide', clip_bright=True, contour='cii')
    # x = 'across-wide'
    # y = 'along'
    # p = 1
    # pv_again(selected_set=x, pillar=p, clip_bright=False, contour='molecular', smooth=True)
    # pv_again(selected_set=y, pillar=p, clip_bright=False, contour='molecular', smooth=True)
    # pv_again(selected_set=x, pillar=p, clip_bright=False, contour='molecular', smooth=True, molecular_line='co65')
    # pv_again(selected_set=y, pillar=p, clip_bright=False, contour='molecular', smooth=True, molecular_line='co65')

    # gain2(selected_set=2, line_stub='cii') #????
    # sample_spectra(0)

    # figure_for_hcop_linewidths()

    # emission_peak_spectra(check_peak=True)
