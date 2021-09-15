"""
A designated place to plot MORE nice M16 images

Created: May 9, 2021
Starting from investigating the "systematic" 25-26 km/s gas and the CO(3-2)
absorption thing

This is meant as a continuation of m16_pictures.py so that that file doesn't
get too long
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

def cii_systematic_emission_2():
    """
    May 9, 2021
    I copied and pasted this from m16_pictures.cii_systematic_emission
    I made a second region collection, systematicvelocity_samples_2.reg,
    which is only circles so I can grab their centers
    I want to just put text numbers on each center instead of drawing the
    circles to keep the colors in the diagram to a minimum
    """
    fn = catalog.utils.search_for_file("apex/M16_12CO3-2_truncated.fits"); line_name = "12CO3-2"
    fn = catalog.utils.search_for_file("apex/M16_13CO3-2_truncated.fits"); line_name = "13CO3-2"
    fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits"); line_name = "CII"
    kms = u.km/u.s
    cube = SpectralCube.read(fn)
    cube._unit = u.K
    cube = cube.with_spectral_unit(kms)

    sysvel_limits = (25*kms, 26*kms)
    sysvel_stub = make_vel_stub(sysvel_limits)
    reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/systematicvelocity_samples_2.reg"))

    fig = plt.figure(figsize=(18, 10))

    ax_img = plt.subplot2grid((2, 3), (0, 0))
    mom0 = cube.spectral_slab(*sysvel_limits).moment0()
    im = ax_img.imshow(np.arcsinh(mom0.to_value()), origin='lower', vmin=1.5, vmax=5, cmap='Greys')
    fig.colorbar(im, ax=ax_img)
    ax_img.set_title(f"{line_name} moment 0 {sysvel_stub}")

    ax_img2 = plt.subplot2grid((2, 3), (1, 0))
    im = ax_img2.imshow(np.arcsinh(mom0.to_value()), origin='lower', vmin=1.5, vmax=5, cmap='Greys')
    fig.colorbar(im, ax=ax_img2)
    ax_img2.set_title(f"{line_name} moment 0 {sysvel_stub}")

    ax1 = plt.subplot2grid((2, 3), (0, 1), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2, 3), (1, 1), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)
    axes = (ax1, ax2, ax3, ax4)
    def add_spectrum(img_ax, spec_ax, reg, idx):
        spectrum = cube.subcube_from_regions([reg]).mean(axis=(1, 2))
        pixreg = reg.to_pixel(mom0.wcs)
        pix_center = pixreg.center.xy
        pix_radius = pixreg.radius
        img_ax.text(pix_center[0], pix_center[1]+pix_radius, f"{idx}", color='r', fontsize=11, ha='center', va='bottom')
        p = spec_ax.plot(cube.spectral_axis.to_value(), spectrum.to_value(), label=f"{idx}")
        pixreg.plot(ax=img_ax, color=p[0].get_c())

    for i, reg in enumerate(reg_list):
        if i % 4 == 0:
            add_spectrum(ax_img, ax1, reg, i)
        elif i % 4 == 1:
            add_spectrum(ax_img2, ax2, reg, i)
        elif i % 4 == 2:
            add_spectrum(ax_img, ax3, reg, i)
        else:
            add_spectrum(ax_img2, ax4, reg, i)
        i += 1

    spec_ylim = (-1, 50) # CII
    # spec_ylim = (-3, 25) # 12CO(3-2)
    # spec_ylim = (-1, 10) # 13CO(3-2)
    # spec_xlim = (15, 40) # OLD, Marc suggested to widen it
    spec_xlim = (-5, 55)
    for ax in axes:
        ax.legend()
        ax.set_ylim(spec_ylim)
        ax.set_xlim(spec_xlim)
        ax.axvspan(*(svl.to_value() for svl in sysvel_limits), color='k', alpha=0.1)
        ax.axhline(0, color='k', alpha=0.2)

    ax2.set_xlabel("velocity (km/s)")
    ax4.set_xlabel("velocity (km/s)")
    ax1.set_ylabel(f"{line_name} intensity (K)")
    ax2.set_ylabel(f"{line_name} intensity (K)")
    ax3.set_title(f"{line_name} spectra averaged over selected positions")
    plt.show()
    # fig.savefig(f"/home/ramsey/Pictures/2021-05-20-work/selected_spectra_{line_name}.png")


def minimum_valid_cutout(img):
    """
    Isolate a sub-array from an array which has a padding of unnecessary or
    invalid values. This function is general-purpose and can be thought of
    as 1) the opposite of the np.pad function or 2) similar to the Cutout2D
    function in astropy.
    You pass in a boolean array where there are True values surrounded by
    False values. The True vales should be confined to a small space, though
    they don't necessarily have to cover a perfect rectangle or be uniform.
    This function will return the array slices necessary to capture all the True
    values and discard as many False edge values as possible.
    The worst case scenario for this function is an isolated True value far
    from the main cluster of True values; try to avoid that.
    :param img: a boolean array where False values are unnecessary.
        Ideally, the True values are limited to a single small rectangular
        (aligned with array axes) region in which there are few, if any, False
        values.
    :returns: tuple(slice, slice) which can be applied to the original image
        to obtain the valid subregion. Similar to the "slices" attribute
        from astropy's Cutout2D.
    """
    return tuple(slice(np.min(x), np.max(x)+1) for x in np.where(img))


def correlate_to_find_offset():
    """
    May 12, 2021
    Correlating the BIMA and APEX 3-2 mom0 images between 20-27
    right now, do the native resolutions of each (don't match yet)
    """
    # co10_img, co10_hdr = fits.getdata(catalog.utils.search_for_file("bima/M16.BIMA.13co.mom0.fits"), header=True)
    co10_img, co10_hdr = fits.getdata(catalog.utils.search_for_file("bima/M16_12CO1-0_7x4_mom0.fits"), header=True)
    # I cannot believe this stuff is necessary
    co10_img = np.squeeze(co10_img)
    for k in list(co10_hdr.keys()):
        if k != 'HISTORY' and k[-1] == '3':
            del co10_hdr[k]
    co10_hdr['NAXIS'] = 2
    # /end data wrangling
    # co32_img = reproject_interp(catalog.utils.search_for_file("apex/M16_12CO3-2_mom0.fits"), co10_hdr, return_footprint=False)
    co32_img, co32_hdr = fits.getdata(catalog.utils.search_for_file("apex/M16_12CO3-2_mom0.fits"), header=True)
    co10_img, fp = reproject_interp((co10_img, co10_hdr), co32_hdr, return_footprint=True)
    min_cutout_sl = minimum_valid_cutout(fp > 0.5)
    co10_img = co10_img[min_cutout_sl]
    # co10_img = co32_img[min_cutout_sl] #### THIS ONE MAKES 10 A COPY OF 32
    co32_img = co32_img[min_cutout_sl]
    # print(co10_img.shape)
    # print(co32_img.shape)
    # co10_img = co10_img[]
    # co32_img = co32_img[]
    # print(co10_img.shape)
    # print(co32_img.shape)
    # return
    fig = plt.figure(figsize=(10, 15))
    ax1 = plt.subplot(131)
    ax1.imshow(co10_img, origin='lower')
    ax2 = plt.subplot(132)
    ax2.imshow(co32_img, origin='lower')


    corr = signal.correlate2d(co32_img, co10_img, boundary='fill', mode='same')
    ax3 = plt.subplot(133)
    # ax3.imshow(fp, origin='lower', vmin=-1, vmax=1)
    corr_sl = tuple(slice(x//2 - 5, x//2 + 5) for x in corr.shape)
    ax3.imshow(corr, origin='lower')
    mask = np.zeros_like(corr).astype(bool)
    mask[corr_sl] = True
    corr[~mask] = np.nan
    x1, y1 = centroids.centroid_com(corr)
    x2, y2 = centroids.centroid_1dg(corr)
    x3, y3 = centroids.centroid_2dg(corr)
    print("Centroids:")
    print(x1, y1)
    print(x2, y2)
    print(x3, y3)
    ax3.plot(x1, y1, color='black', ms=5, mew=2, marker='+')
    ax3.plot(x2, y2, color='white', ms=5, mew=2, marker='+')
    ax3.plot(x3, y3, color='red', ms=5, mew=2, marker='+')
    """
    The correlation produces results! I need to quantify these and obtain
    the direction and magnitude (in sky angle) of the offset
    The direction appears to be due North-South, from my quick glance, and about
    half an APEX 3-2 pixel
    """
    plt.show()


def convolve_carma_to_sofia():
    """
    May 25, 2021
    I want to investigate those "threads" coming from the tip of Pillar 1
    I want to see if they're visible at SOFIA ~14x14 resolution and compare to
    SOFIA.
    """
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    """
    This contains:
    CS, N2H+, HCO+, HCN
    """
    raise RuntimeError("You already ran this on May 25, 2021")
    fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits")
    cube_cii = SpectralCube.read(fn)
    cube_cii._unit = u.K

    cube_mol = cube_utils.CubeData([f for f in filepaths if 'hcop' in f].pop())
    cube_mol.convert_to_K()
    write_path, original_mol_fn = os.path.split(cube_mol.full_path)
    write_fn = original_mol_fn.replace('.fits', '.SOFIAbeam.fits')
    print(write_path)
    print(original_mol_fn)
    print(write_fn)
    cube_mol = cube_mol.data # gets spectralcube object from cubedata object
    cube_mol = cube_mol.convolve_to(cube_cii.beam)
    cube_mol.write(os.path.join(write_path, write_fn), format='fits')


def compare_carma_to_sofia_crosscut(selected_region=0):
    """
    May 25, 2021
    Following from convolve_carma_to_sofia, I'll use those convolved-up CARMA
    molecular line maps to compare to SOFIA and see if CII has comparatively
    less of a "threaded" structure just below the tip of P1
    I'll also include the unconvolved CARMA map, in each case, to show the
    full extend of the threaded structure

    File: p1_threads.reg
    First path: 24.4-25.5 km/s
    Second path: 24.8-25.7 km/s
    Third path (horns): 23.7-25.8 km/s
    Fourth path (very tip): 25-27 km/s
    (Tested on HCN at native resolution)

    selected_region is an int between 0 and 3 inclusive
    """
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    molecule = 'hcop'
    molecule_name = 'HCO+'
    mol_fn = [f for f in filepaths if molecule in f].pop()
    mol_conv_fn = [f for f in filepaths_conv if molecule in f].pop()

    reg_filename = catalog.utils.search_for_file("catalogs/p1_threads.reg")
    vel_lims = [(24.4, 25.5), (24.8, 25.7), (23.7, 25.8), (25., 27.)][selected_region]
    cc_path = crosscut.coords_from_region(reg_filename, index=selected_region)

    fig = plt.figure(figsize=(17, 7))
    img_ax_f = lambda w: plt.subplot2grid((1, 4), (0, 0), colspan=1, projection=w)
    xcut_ax = plt.subplot2grid((1, 4), (0, 1), colspan=3)
    cco = crosscut.CrossCut(cc_path, vlims=vel_lims)
    cco.setup_figure(fig=fig, xcut_axis=xcut_ax)
    layers = [
        crosscut.DataLayer("[CII] / 4", cube_utils.CubeData("sofia/M16_CII_U.fits"), cube=True, alpha=0.7, f_to_apply=lambda x: x/4),
        crosscut.DataLayer(f"{molecule_name} (CII beam)", cube_utils.CubeData(mol_conv_fn), cube=True, alpha=0.7),
        crosscut.DataLayer(f"{molecule_name}", cube_utils.CubeData(mol_fn), cube=True, alpha=0.7),
    ]
    cco.add_data_layer(*layers)
    cco.update_plot(norm=False)
    cco.switch_axes('xcut')
    plt.ylabel("Intensity")
    plt.title("Cross cut")
    plt.xlabel("Distance along cross-cut (arcseconds)")
    cco.plot_image(layers[2], stretch='linear', subplot_creator=img_ax_f)
    cco.switch_axes('img')
    plt.title(f"{molecule_name} {vel_lims}")
    # plt.show()
    fig.savefig(f"/home/ramsey/Pictures/2021-06-01-work/xc_{selected_region}_{molecule}.png")


def compare_carma_to_sofia_pv(selected_region=0, mol_idx=False):
    """
    May 25, 2021
    Same cuts as compare_carma_to_sofia_crosscut but with PVs now
    Start with 20-30 km/s ranges for everything (for now)
    Following pvdiagrams_2.m16_pv_again and m16_pictures.single_parallel_pillar_pvs
    but this is a little different than either of those because I will only
    plot one slice at a time and will be overlaying several sources of data

    The "galaxy brain" idea here would be to combine the previous function
    with the crosscuts and this one with PV diagrams
    Would look noisy but could be really cool

    selected_region is an int between 0 and 3 inclusive for p1_threads.reg
    Now it's 0-5 inclusive for pillar1_threads_pv.reg


    TODO::::::: swap the CII image for a HC/N/OP image (native resolution)
    so that I can more directly compare CII and HC/ at CII resolution as contours

    Updated June 25, 2021 to include CO10 and potentially follow my TODOs above
    Updated July 13, 2021 to use pillar1_threads_pv.reg instead of p1_threads.reg
        I'm using the new one in pvdiagrams_2.m16_pv_again2 to compare CARMA
        to SOFIA more closely and focus on the threads
    Big update July 13, 2021: I'm going to shove every line in the same image
        That's CII (in all PVs for reference), HCN, HCO+, and 12CO1-0.
        I pushed the prev version to github on July 12 so go look there for the
        single line (+CII) version
    """
    # copied from the crosscut version
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    # get all the filenames at once
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcn"), *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits"]
    colors = [marcs_colors[0],] + [marcs_colors[1], marcs_colors[6],]*3
    names = ['[CII]', 'HCN', 'HCN (CII beam)', 'HCO+', 'HCO+ (CII beam)', '12CO(1-0)', '12CO(1-0) (CII beam)']
    short_names = ['cii', 'hcn', 'hcnCONV', 'hcop', 'hcopCONV', 'co10', 'co10CONV']
    levels = [list(np.linspace(15, 40, 8))] + [list(np.linspace(2, 9, 8))]*2 + [list(np.linspace(2, 7, 7))]*2 + [list(np.linspace(25, 80, 8))]*2
    # We will use all the lines, so I don't drop anything from the list now
    # no we won't!

    if mol_idx is False:
        mol_idx = 1 # 1 is hcn, 3 is hcop, 5 is co10
    # mol_idx is an arg, starting at 1 for the molecules
    trim_lists = lambda l : [l[0]] + l[mol_idx:mol_idx+2]
    # trim all lists so we can just loop through them and only get one molecular line
    cube_filenames = trim_lists(cube_filenames)
    colors = trim_lists(colors)
    names = trim_lists(names)
    levels = trim_lists(levels)

    # also don't need this, use the odd-numbered indices for every background
    # or just don't use background!!!!!
    # img_for_background = 1

    # set up the vectors
    reg_filename = catalog.utils.search_for_file("catalogs/pillar1_threads_pv.reg")
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=None) # Try nonzero width later
    vel_limits = np.array([22, 28])*u.km/u.s
    """
    July 12: Redoing this right now is a bad use of time, I should show Marc or
    someone the images first and then make a pretty image later
    Edit below this line, stack the PVs in a column on the right hand side
    Will need to mess with aspect ratio to get it to look nice

    July 15: I think this is a good idea for exploring different PV slices,
    but for the poster I think I should focus on CO1-0 and HCO+ comparisons with
    CII. I need to make the images less "noisy" and cluttered.
    I am thinking to discard the background color PV image and separate
    the convolved and unconvolved into different subplots on top of each other
    Probably 1/3 reference image and 2/3 subplots (column-wise)
    """

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
    # ax_sl.imshow(*contour_args_list[img_for_background], origin='lower', cmap='viridis')
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
    # plt.show()
    fig.savefig(f"/home/ramsey/Pictures/2021-07-15-work/pv_{selected_region}_{short_names[mol_idx]}.png")


def thin_channel_images_rb(c1_idx, c2_idx, vel_start=24.5, vel_stop=25.5, savefig=True):
    """
    May 27, 2021 (post-MVA appt)
    This will be similar to m16_pictures.make_image_thin_channel_maps
    Following from suggestions from my May 26th M16 meeting, I'll compare thin
    channel maps (like 1 km/s wide) of CO/molecular and CII to see what sorts of
    offsets and gradients I see

    May 28: this function has been really successful and generalized to a few different
    molecular lines and any combination between them. I will extend it to CO 6-5
    and 3-2.

    August 5, 2021: (at Vigilante!)
    I want to re-make these using contours, since I think that's a clearer way
    to convey positional shifts. It may be noisier, but I think it's worthwhile.
    August 10, 2021: gonna add IRAC 4 to the mix, but will need to modify
    this to work with a layer that isn't a cube (......DataLayer?? I wish I had
    time to refactor all this)

    August 14, 2021: (in Sac)
    I want to redo these again with image/contour combinations and better
    contour levels (closer to a couple sigma above the noise)
    """
    kms = u.km/u.s
    # copied from the crosscut version
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcn"), *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits", "spitzer/SPITZER_I4_mosaic.fits"]
    # colors = ['r', 'b', 'cyan', 'violet', 'magenta', 'LimeGreen', 'g']
    names = ['CII', 'HCN', 'HCN (CII beam)', 'HCO+', 'HCO+ (CII beam)', '$^{12}$CO(1-0)', '$^{12}$CO(1-0) (CII beam)', '8$\mu$m']
    short_names = ['cii', 'hcn', 'hcnCONV', 'hcop', 'hcopCONV', 'co10', 'co10CONV', 'irac4']
    vlims_r = [[-0.5, 0], # cii
        [-1, 0], [-0.5, 0], # hcn/CONV
        [-1, 0], [-0.5, 0], #hcop/CONV
        [0, 1], [0, 1],] # co10/CONV

    vlims_g = [[-0.5, 0], # cii
        [-0.5, 0], [-0.5, 1], # hcn/CONV
        [-0.5, 0], [-0.5, 1], #hcop/CONV
        [0, 1], [0, 1],] # co10/CONV

    vlims_b = [[-0.5, 0], # cii
        [-1, 0], [-1, 0], # hcn/CONV
        [-1, -1], [-1, 0], #hcop/CONV
        [0, 0], [0, 0],] # co10/CONV

    contour_levels = [
        np.linspace(10, 80, 11), # cii
        None, None, # hcn
        None, None, # hcop
        np.linspace(10, 130, 10), np.linspace(5, 110, 10), # co10/CONV
        np.linspace(200, 1000, 8), # irac4
    ]
    img_vlims = [
        (0, 80), # cii
        None, None, # hcn
        None, None, # hcop
        (0, 120), (0, 100),
        (50, 500), # irac4
    ]

    if not isinstance(c1_idx, int):
        print(c1_idx, end=': ')
        c1_idx = short_names.index(c1_idx)
        print(c1_idx)
    if not isinstance(c2_idx, int):
        print(c2_idx, end=': ')
        c2_idx = short_names.index(c2_idx)
        print(c2_idx)

    # c1_idx = 0
    # c2_idx = 2
    unique_label = f"{short_names[c1_idx]}-{short_names[c2_idx]}"
    cube1 = cube_utils.CubeData(cube_filenames[c1_idx]).convert_to_K().data # Cube 1 should have larger footprint (CII)
    cube2 = cube_utils.CubeData(cube_filenames[c2_idx]).convert_to_K().data
    vel_start *= kms
    vel_stop *= kms
    channel_width = 1*kms
    current_velocity = vel_start

    if savefig:
        fig = plt.figure(figsize=(12, 12))
    while current_velocity < vel_stop:
        if not savefig:
            fig = plt.figure(figsize=(12, 12))
        vel_lims = (current_velocity, current_velocity + channel_width)
        img1_raw = cube1.spectral_slab(*vel_lims).moment0().to(u.K*kms)
        img2_raw = cube2.spectral_slab(*vel_lims).moment0().to(u.K*kms)
        img2, fp = reproject_interp((img2_raw.to_value(), img2_raw.wcs), img1_raw.wcs, shape_out=img1_raw.shape, return_footprint=True)
        min_cutout_sl = minimum_valid_cutout(fp > 0.5)
        img1 = img1_raw.to_value()[min_cutout_sl]
        img2 = img2[min_cutout_sl]

        fig.clear()
        ax = plt.subplot(111, projection=img1_raw.wcs)

        # im = ax.imshow(img1, origin='lower', cmap='plasma', vmin=img_vlims[c1_idx][0], vmax=img_vlims[c1_idx][1])
        im = ax.imshow(img2, origin='lower', cmap='plasma', vmin=img_vlims[c2_idx][0], vmax=None)
        fig.colorbar(im, ax=ax, label='Integrated intensity (K km/s)')

        # ax.imshow(np.zeros_like(img2), origin='lower', vmin=0, vmax=1, cmap='Greys')
        ax.contour(img1, colors='k', levels=contour_levels[c1_idx])
        # ax.contour(img2, colors='k', levels=contour_levels[c2_idx])

        vel_str = f"{current_velocity.to_value():.1f}$-${(current_velocity + channel_width).to_value():.1f} {current_velocity.unit}"
        ax.text(0.05, 0.90, f"{names[c1_idx]}", transform=ax.transAxes, c=marcs_colors[0], fontsize=25)
        ax.text(0.05, 0.85, f"{names[c2_idx]}", transform=ax.transAxes, c=marcs_colors[1], fontsize=25)
        ax.set_xlabel("RA"), ax.set_ylabel("Dec")
        ax.set_title(f"Channel maps at {vel_str}")
        # ax.legend()
        if savefig:
            fig.savefig(f"/home/ramsey/Pictures/2021-08-14-work/contouroverlay_{unique_label}_{current_velocity.to_value():04.1f}.png")
        current_velocity += channel_width
    if not savefig:
        plt.show()


def overlaid_contours_for_offset():
    """
    August 6, 2021
    This is going to be very similar to the function above, the repurposed RB
    plots that use contours instead. It's going to be contours overlaid (kind of
    like the CO 6-5 comparison plots) but with 13CO 1-0 and 13CO 3-2 against
    the high density tracers (HCO+ probably)
    This is for Rolf to look into the offsets

    Also reference m16_pictures.compare_32_65_10
    Aug 10 2021: added CII

    Aug 29 2021: saving 23-27 km/s mom0s to avoid APEX 3-2 OFF contamination

    Sept 14, 2021: returning to the Aug 14 or so vision (according to the
    images I made back then. more notes would've been nice!). Adjustable moment
    (let the functions decide the vlims and contours) for quick highlighting of
    features. (I haven't implemented this yet, so TODO: update this when I
    do it)
    """
    selected_img = "12co10"
    selected_contours = ["hcop", "12co32",]

    supported_data = ["12co10", "13co10", "12co32", "13co32", "cii", "12co65", "irac4", "hcop"]
    filenames = ["bima/M16_12CO1-0_7x4.mom0.23-27.fits", "bima/M16.BIMA.13co.mom0.fits", # co10
        "apex/M16_12CO3-2.mom0.23-27.fits", "apex/M16_13CO3-2.mom0.23-27.fits", # co32
        "sofia/M16_CII_U.mom0.23-27.fits", "apex/M16_12CO6-5_mom0.fits", # cii, co65
        "spitzer/SPITZER_I4_mosaic.fits", "carma/M16.ALL.hcop.sdi..mom0.23-27.fits"] # irac4, hcop
    labels = ["$^{12}$CO(1-0) 23-27 km/s", "$^{13}$CO(1-0)", "$^{12}$CO(3-2) 23-27 km/s", "$^{13}$CO(3-2) 23-27 km/s", "[CII] 23-27 km/s", "$^{12}$CO(6-5)",
        "8 $\mu$m", "HCO+ 23-27 km/s"]
    short_labels = ["$^{12}$CO(1-0)", "$^{13}$CO(1-0)", "$^{12}$CO(3-2)", "$^{13}$CO(3-2)", "[CII]", "$^{12}$CO(6-5)",
        "8 $\mu$m", "HCO+"]
    vlims = [(None, None),]*6 + [(None, 400), (None, None)]
    contour_levels = [np.linspace(1.5, 14, 7), np.linspace(30, 250, 6), # co10
        np.linspace(18, 105, 7), np.linspace(5, 45, 7), # co32
        np.linspace(20, 140, 7), np.linspace(15, 55, 5), # cii, co65
        np.exp(np.linspace(np.log(100), np.log(1000), 10)), np.linspace(2.5, 42, 5)] # irac4, hcop
    # cube_utils.Beam.from_fits_header(fits.getheader(file))

    colors = ['k', marcs_colors[2], marcs_colors[6]]
    def format_beam_patch(beam_patch):
        beam_patch.set_alpha(0.9)
        beam_patch.set_facecolor('grey')
        beam_patch.set_edgecolor('k')

    handles = []
    img_idx = supported_data.index(selected_img)
    img_data, img_hdr = fits.getdata(catalog.utils.search_for_file(filenames[img_idx]), header=True)
    img_wcs = WCS(img_hdr, naxis=2)
    img_data = np.squeeze(img_data)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection=img_wcs)
    im = ax.imshow(img_data, origin='lower', cmap='cividis', vmin=vlims[img_idx][0], vmax=vlims[img_idx][1])
    ax.set_title("Image: "+labels[img_idx])
    cbar = fig.colorbar(im, ax=ax)
    beam_patch_x = 0.9
    beam_patch_y = 0.08
    img_beam_patch = cube_utils.Beam.from_fits_header(img_hdr).ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([beam_patch_x, beam_patch_y]), misc_utils.get_pixel_scale(img_wcs))
    format_beam_patch(img_beam_patch)
    ax.add_artist(img_beam_patch)
    beam_text_y = beam_patch_y - 0.05
    ax.text(beam_patch_x, beam_text_y, short_labels[img_idx], color='white', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    beam_patch_offset = 0.15
    beam_patch_x -= beam_patch_offset

    for i, selected_contour in enumerate(selected_contours):
        contour_idx = supported_data.index(selected_contour)
        contour_data, contour_hdr = fits.getdata(catalog.utils.search_for_file(filenames[contour_idx]), header=True)
        contour_wcs = WCS(contour_hdr, naxis=2)
        contour_img_reproj = reproject_interp((contour_data, contour_wcs), img_wcs, shape_out=img_data.shape, return_footprint=False)
        # contours
        if len(selected_contours) == 1:
            ax.contour(contour_img_reproj, linewidths=1, cmap='hot', levels=12)
            ax.text(0.95, 0.95, "Contours: "+labels[contour_idx], transform=ax.transAxes, color=('k' if selected_img == 'hcop' else 'white'), fontsize=16, ha='right')
        else:
            ax.contour(contour_img_reproj, linewidths=1, colors=colors[i], levels=12)
            handles.append(mpatches.Patch(color=colors[i], label=labels[contour_idx]))
        # beam
        if not selected_contour == selected_img:
            contour_beam_patch = cube_utils.Beam.from_fits_header(contour_hdr).ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([beam_patch_x, beam_patch_y]), misc_utils.get_pixel_scale(contour_wcs))
            format_beam_patch(contour_beam_patch)
            ax.add_artist(contour_beam_patch)
            ax.text(beam_patch_x, beam_text_y, short_labels[contour_idx], color='white', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            beam_patch_x -= beam_patch_offset

    if len(selected_contours) > 1:
        plt.legend(handles=handles)
    for coord in ax.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    plt.tight_layout()
    contour_stub = "contours_" + "-".join(selected_contours)
    plt.savefig(f"/home/ramsey/Pictures/2021-09-07-work/{selected_img}-{contour_stub}.png")
    # plt.show()



def pillar_sample_spectra(reg_idx):
    """
    June 1, 2021
    In prep for the meeting with Nicola tomorrow, I want to look at spectra of CO 1-0, 3-2, 6-5, and CII towards the head of P1
    I want to see if CII is more redshifted than CO or just broader
    I have a region file called pillar_samples.reg with relevant samples
    The code from cii_systematic_emission_2 can be copied and modified into this file
    """
    reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/pillar_samples.reg"))

    cube_filenames = ["sofia/M16_CII_U_APEXbeam.fits", "apex/M16_12CO3-2_truncated_cutout.fits", "apex/M16_13CO3-2_truncated_cutout.fits",
        "bima/M16_12CO1-0_APEXbeam.fits", "apex/M16_CO6-5_APEXbeam.fits"]
    line_names = ['CII', '12CO(3-2)', '13CO(3-2)',
        '12CO(1-0) * 0.5', '12CO(6-5)']
    multiplier = [1, 1, 1, 0.5, 1]
    kms = u.km/u.s

    reference_filename = catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits")
    reference_name = "HST F657N"
    # reference_filename = catalog.utils.search_for_file("apex/M16_12CO6-5_mom0.fits") # smaller test reference file
    # reference_name = "12CO(6-5)"

    if reg_idx < 3:
        pillar_vel_limits = (22.5*kms, 27.5*kms)
    elif reg_idx == 3:
        pillar_vel_limits = (20*kms, 27.5*kms)
    else:
        pillar_vel_limits = (19*kms, 24*kms)

    pillar_vel_stub = make_vel_stub(pillar_vel_limits)

    fig = plt.figure(figsize=(15, 5)) # originally (18, 10)

    ax_img = plt.subplot2grid((1, 5), (0, 0))
    ref_img, ref_hdr = fits.getdata(reference_filename, header=True)
    ref_wcs = WCS(ref_hdr)
    im = ax_img.imshow(ref_img, origin='lower', cmap='Greys_r', vmin=0.1, vmax=0.6)
    fig.colorbar(im, ax=ax_img)
    ax_img.set_title(reference_name)


    # reg_idx = 0
    reg = reg_list[reg_idx]

    ax_spec = plt.subplot2grid((1, 5), (0, 1), colspan=4, rowspan=1)
    def add_spectrum(cube, spec_ax, reg, idx):
        # holdover from cii_systematic_emission_2() where there were multiple img/spec axes
        subcube = cube.subcube_from_regions([reg])
        spectrum = subcube.mean(axis=(1, 2)) * multiplier[idx]
        if reg_idx < 4:
            mean_vel = np.nanmean(subcube.spectral_slab(*pillar_vel_limits).moment1().to_value())
            mean_stub = f" [Mean: {mean_vel:.2f}]"
        else:
            mean_stub = ""
        p = spec_ax.plot(cube.spectral_axis.to_value(), spectrum.to_value(), label=f"{line_names[idx]}{mean_stub}")
        if reg_idx < 4:
            spec_ax.axvline(mean_vel, color=p[0].get_c(), alpha=0.6)
        return p

    for i, c_fn in enumerate(cube_filenames):
        cube = cube_utils.CubeData(c_fn)
        cube.convert_to_K()
        cube.data = cube.data.with_spectral_unit(kms)
        add_spectrum(cube.data, ax_spec, reg, i)
    pixreg = reg.to_pixel(ref_wcs)
    pix_center = pixreg.center.xy
    pix_radius = pixreg.radius
    # ax_img.text(pix_center[0], pix_center[1]+pix_radius, f"{idx}", color='r', fontsize=11, ha='center', va='bottom')
    if reg_idx < 4:
        ax_spec.text(0.7, 0.6, "Mean velocity taken from\nwithin shaded velocity range", color='k', fontsize=11, ha='left', va='bottom', transform=ax_spec.transAxes)
    pixreg.plot(ax=ax_img, color='r')


    # spec_ylim = (-1, 50) # CII
    # spec_ylim = (-3, 25) # 12CO(3-2)
    # spec_ylim = (-1, 10) # 13CO(3-2)
    # spec_xlim = (15, 40) # OLD, Marc suggested to widen it
    spec_xlim = (15, 45)
    ax_spec.legend()
    # ax_spec.set_ylim(spec_ylim)
    ax_spec.set_xlim(spec_xlim)
    ax_spec.axvspan(*(pvl.to_value() for pvl in pillar_vel_limits), color='grey', alpha=0.05)
    ax_spec.axhline(0, color='k', alpha=0.2)

    ax_spec.set_xlabel("velocity (km/s)")
    ax_spec.set_ylabel("Line intensity (K)")
    ax_spec.set_title("Line spectra (20\" beam) averaged over selected position")
    # plt.show()
    fig.savefig(f"/home/rkarim/Pictures/2021-06-03-work/pillar_samples_{reg_idx}.png")


def multiple_moments():
    # cube = cube_utils.CubeData("sofia/M16_CII_U_APEXbeam.fits").data.spectral_slab(20*kms, 30*kms)
    # vel_lims = (20*kms, 30*kms)
    vel_lims = (23*kms, 27*kms)
    # vel_lims = (19*kms, 24*kms)
    vel_str = f"{vel_lims[0].to_value():.1f}-{vel_lims[1].to_value():.1f}"
    # Pillar1: 0, Pillar2: 2
    cube = cps2.cutout_subcube(length_scale_mult=4., reg_index=0).spectral_slab(*vel_lims)
    fig = plt.figure(figsize=(15, 5))
    # vlims = [(None, None), (24, 26.5), (2, 7)] # 20-30
    vlims = [(None, None), (24.5, 25.5), (1, 1.7)] # 23-27
    # vlims = [(None, 90), (21, 23), (1, 3)] # 19-24
    cargs, ckwargs = None, None
    cargs2, ckwargs2 = None, None
    for i in range(3):
        mom = cube.moment(order=i)
        ax = plt.subplot2grid((1, 3), (0, i))
        im = ax.imshow(mom.to_value(), origin='lower', vmin=vlims[i][0], vmax=vlims[i][1], cmap='nipy_spectral')
        fig.colorbar(im, ax=ax)
        if cargs is None:
            cargs = (mom.to_value(),)
            if any(vlims[0][j] is not None for j in range(2)):
                if vlims[0][0] is not None:
                    cargs[0][cargs[0] < vlims[0][0]] = vlims[0][0]
                if vlims[0][1] is not None:
                    cargs[0][cargs[0] > vlims[0][1]] = vlims[0][1]
            ckwargs = dict(colors='k', linewidths=1)
        elif cargs2 is None:
            cargs2 = (mom.to_value(),)
            cargs2[0][cargs2[0] < vlims[1][0]] = vlims[1][0]
            cargs2[0][cargs2[0] > vlims[1][1]] = vlims[1][1]
            ckwargs2 = dict(colors='cyan', linewidths=1, levels=[24.6, 24.8, 25.35, 25.45])
        ax.contour(*cargs, **ckwargs)
        if cargs2 is not None:
            ax.contour(*cargs2, **ckwargs2)
        ax.set_title(f"[CII] Moment {i} {make_vel_stub(vel_lims)}")
        ax.set_xticks([]), ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(f"/home/ramsey/Pictures/2021-06-03-work/cii_moments_{vel_str}_2.png")
    # plt.show()


def compare_first_moment_to_peak_loc():
    # vel_lims = (20*kms, 30*kms)
    vel_lims = (23*kms, 27*kms)
    # vel_lims = (19*kms, 24*kms)
    vel_str = f"{vel_lims[0].to_value():.1f}-{vel_lims[1].to_value():.1f}"
    cube = cps2.cutout_subcube(length_scale_mult=4., reg_index=0).spectral_slab(*vel_lims)
    mom1 = cube.moment1().to(kms).to_value()
    mom0 = cube.moment0().to_value()
    cargs = (mom0,)
    ckwargs = dict(colors='k', linewidths=1)

    full_power_loc = cube.spectral_axis[cube.argmax(axis=0)].to_value()

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(mom1, origin='lower', vmin=vel_lims[0].to_value(), vmax=vel_lims[1].to_value(), cmap='nipy_spectral')
    plt.colorbar()
    plt.title(f"Moment 1 {vel_str}")
    plt.contour(*cargs, **ckwargs)

    plt.subplot(132)
    plt.imshow(mom1 - full_power_loc, origin='lower', vmin=-1.5, vmax=1.5, cmap='nipy_spectral')
    plt.colorbar()
    plt.title("Moment 1 minus peak location")
    plt.contour(*cargs, **ckwargs)

    plt.subplot(133)
    plt.imshow(full_power_loc, origin='lower', vmin=vel_lims[0].to_value(), vmax=vel_lims[1].to_value(), cmap='nipy_spectral')
    plt.colorbar()
    plt.title(f"Peak location  {vel_str}")
    plt.contour(*cargs, **ckwargs)
    plt.tight_layout()
    plt.savefig(f"/home/ramsey/Pictures/2021-06-07-work/moment1_vs_peak_{vel_str}.png")
    # plt.show()


def estimate_noise(cube, spectrum, pix_radius, ax):
    """
    June 8, 2021
    Helper function for identify_components
    Find the RMS of the positive and negative noise
    """
    # estimate negative noise from -5 to 10 km/s
    sl = slice(cube.closest_spectral_channel(-5*kms), cube.closest_spectral_channel(10*kms))
    noise_spectrum = spectrum[sl]
    pos_noise = noise_spectrum.to_value()[noise_spectrum>0]
    neg_noise = noise_spectrum.to_value()[noise_spectrum<0]
    f = lambda x : np.sum(np.abs(x))
    n1 = list(f(x)/(x.size) for x in (pos_noise, neg_noise))
    n2 = list(f(x)*np.sqrt(pix_radius)/(x.size) for x in (pos_noise, neg_noise))
    ax.plot([pix_radius], [n1[0]], color='r', marker='+')
    ax.plot([pix_radius], [n1[1]], color='r', marker='o')
    ax.plot([pix_radius], [n2[0]], color='k', marker='+')
    ax.plot([pix_radius], [n2[1]], color='k', marker='o')
    """
    The result seems to be:
    Use the second formula, sum(abs(x))*root(radius)/size
    and constrain it to be below 1.3 or 1.4 or so (test this out)
    I am not using RMS because I don't want to take the square root of the
    number of points (afaik)
    """

def scale_background(bg_spectrum, spectrum, pix_radius, hand_picked_scale, debug_plot=False):
    """
    June 8, 2021
    Try to scale the background to the spectrum for subtraction
    Use a negative noise based tolerance for how scaled up the BG should be
    """
    try:
        bg_spectrum = bg_spectrum.to_value()
    except:
        pass
    try:
        spectrum = spectrum.to_value()
    except:
        pass
    negative_noise_tolerance = 1.35 # based on estimate_noise() fidings
    f = lambda x: np.sum(np.abs(x))*pix_radius/x.size
    def rescale_subtract_and_get_noise(scale_factor):
        # parameter is scale factor
        # return the negative noise score
        residual = spectrum - bg_spectrum*scale_factor
        return f(residual[residual<0])
    if debug_plot:
        fig = plt.figure()
        ax = plt.subplot(121)
        ax2 = plt.subplot(122)
    scale_factor = 1.0
    scale_list = []
    noise_list = []
    for i in range(50):
        noise = rescale_subtract_and_get_noise(scale_factor)
        scale_list.append(scale_factor)
        noise_list.append(noise/scale_factor)
        scale_factor += 0.05
    spline = UnivariateSpline(scale_list, noise_list, s=0)
    scale_factor_range = np.linspace(scale_list[0], scale_list[-1], 100)
    interpd_noise = spline(scale_factor_range)
    best_scale_factor = scale_factor_range[np.argmin(interpd_noise)]
    if debug_plot:
        print("SCALE", best_scale_factor)
        ax.plot(scale_list, noise_list, '.')
        ax.plot(scale_factor_range, interpd_noise)
        ax.plot(scale_factor_range, spline.derivative(1)(scale_factor_range))
        ax.axvline(best_scale_factor, alpha=0.5, color='k')
        ax.axvline(hand_picked_scale, alpha=0.5, color='orange')
        ax2.plot(spectrum, color='g', alpha=0.5)
        ax2.plot(spectrum - bg_spectrum*hand_picked_scale, color='orange', alpha=0.5)
        ax2.plot(spectrum - bg_spectrum*best_scale_factor, color='k')
        ax2.axhline(0, color='k', alpha=0.4)



def identify_components(bgsub=False):
    """
    June 8, 2021
    I made a bunch of channel maps and spectrum plots yesterday and it seems like the "red tail"
    of the pillar 1 head is actually present off the pillar too, so it may be some other component
    Now I want to remake those plots in Python and maybe even do some subtraction to see what's going on
    The region file is called catalogs/pillar_samples_components.reg
    """
    reg_fn = "catalogs/pillar_samples_components.reg"
    reg_list = regions.read_ds9(catalog.utils.search_for_file(reg_fn))
    cube_zoom = cps2.cutout_subcube(length_scale_mult=6, reg_index=0)
    cube = cps2.cutout_subcube(length_scale_mult=12, reg_index=2)
    # can use cube.closest_spectral_channel(vel) to find channel

    colors = ['k', 'k', 'r', 'b', 'g', 'k']
    linestyles = ['--', '-', '-', '-', '-', ':']
    names = ["Small Dashed Circle", "Small Solid Circle", "Circle 1", "Circle 2", "Circle 3", "Western Circle"]
    short_names = ["", "", "1 (R)", "2 (B)", "3 (G)", "West"]
    # I hand-edited some of these scale factors
    # bg_scale_factors = [1.47, 2.34, 1.6, 1.22, 1.28, 1.0] # first version
    # bg_scale_factors = [1.47, 1.96, 1.6, 1.22, 1.28, 1.0] # second version
    bg_scale_factors = [1.47, 1.96, 1.4, 1.1, 1.28, 1.0] # third version

    channel = 27*kms
    ch_range = [channel]
    # ch_range = np.arange(35, 40, 0.5)*kms
    for channel in ch_range:
        channel_map = cube[cube.closest_spectral_channel(channel), :, :]
        channel_map_zoom = cube_zoom[cube_zoom.closest_spectral_channel(channel), :, :]

        fig = plt.figure(figsize=(16.5, 8))
        img_ax = plt.subplot2grid((2, 3), (0, 0), projection=channel_map.wcs)
        img_zoom_ax = plt.subplot2grid((2, 3), (1, 0), projection=channel_map_zoom.wcs)
        if bgsub:
            spec_ax = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=2)
            spec_bg_ax = plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=2)
        else:
            spec_ax = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)
        stretch = np.arcsinh
        vlims = dict(vmin=stretch(0), vmax=stretch(30))
        im = img_ax.imshow(stretch(channel_map.to_value()), origin='lower', cmap='nipy_spectral', **vlims)
        fig.colorbar(im, ax=img_ax)
        im = img_zoom_ax.imshow(stretch(channel_map_zoom.to_value()), origin='lower', cmap='nipy_spectral', **vlims)
        fig.colorbar(im, ax=img_zoom_ax)

        if bgsub:
            bg_spectrum = cube.subcube_from_regions([reg_list[-1]]).mean(axis=(1, 2))
        else:
            bg_spectrum = np.zeros(cube.spectral_axis.shape) * u.K


        """
        Try to fit the background
        Done, got the numbers I needed. Leaving this here for posterity
        for i, reg in zip(range(2, 4), reg_list[2:4]):
            pixreg = reg.to_pixel(channel_map.wcs)
            pix_radius = pixreg.radius
            spectrum = cube.subcube_from_regions([reg]).mean(axis=(1, 2))
            scale_background(bg_spectrum, spectrum, pix_radius, bg_scale_factors[i], debug_plot=True)
        plt.show()
        return
        """

        for i, reg in enumerate(reg_list):
            spectrum = cube.subcube_from_regions([reg]).mean(axis=(1, 2))
            spec_ax.plot(cube.spectral_axis.to_value(), spectrum.to_value(), label=names[i], color=colors[i], ls=linestyles[i])
            if bgsub:
                spectrum_bgsub = spectrum - bg_spectrum*bg_scale_factors[i]
                spec_bg_ax.plot(cube.spectral_axis.to_value(), spectrum_bgsub.to_value(), label=names[i], color=colors[i], ls=linestyles[i])
            for map, ax in zip((channel_map, channel_map_zoom), (img_ax, img_zoom_ax)):
                pixreg = reg.to_pixel(map.wcs)
                pix_center = pixreg.center.xy
                pix_radius = pixreg.radius
                ax.text(pix_center[0]-pix_radius/2, pix_center[1]+pix_radius, short_names[i], color='w', fontsize=11, ha='center', va='bottom')
                pixreg.plot(ax=ax, color='w', lw=2, ls=linestyles[i])

            """
            This is how I estimated the negative noise threshold (which I didn't even end up using!!)
            estimate_noise(cube, spectrum, pix_radius, ax2)
            """

        for ax in (img_ax, img_zoom_ax):
            ax.set_xlabel(" ")
            ax.set_ylabel(" ")
        img_ax.tick_params(axis='x', direction='in', labelbottom=False)
        img_ax.tick_params(axis='y', direction='in')
        img_zoom_ax.tick_params(axis='x', direction='in')
        img_zoom_ax.tick_params(axis='y', direction='in')
        spec_ax.axvline(channel.to_value(), color='grey', alpha=0.4, lw=1.5)
        spec_ax.axhline(0, color='grey', alpha=0.4, lw=1.5)
        spec_ax.legend()
        if bgsub:
            spec_bg_ax.axvline(channel.to_value(), color='grey', alpha=0.4, lw=1.5)
            spec_bg_ax.axhline(0, color='grey', alpha=0.4, lw=1.5)
            spec_bg_ax.legend()
        plt.tight_layout()
        # plt.show()
        bgsub_stub = "bgsub_" if bgsub else ""
        fig.savefig(f"/home/ramsey/Pictures/2021-06-08-work/channel_map_{bgsub_stub}{channel.to_value():04.1f}.png")


def identify_components_2():
    """
    June 8, 2021
    Copy of the above function but this time I'm subtracting the brighter "background" spectra
    from the pillar spectra, instead of just that dim Western spectrum
    """
    reg_fn = "catalogs/pillar_samples_components.reg"
    reg_list = regions.read_ds9(catalog.utils.search_for_file(reg_fn))
    cube_zoom = cps2.cutout_subcube(length_scale_mult=6, reg_index=0)
    cube = cps2.cutout_subcube(length_scale_mult=12, reg_index=2)
    # can use cube.closest_spectral_channel(vel) to find channel

    colors = ['k', 'k', 'r', 'b', 'g', 'k']
    linestyles = ['--', '-', '-', '-', '-', ':']
    names = ["Small Dashed Circle", "Small Solid Circle", "Circle 1", "Circle 2", "Circle 3", "Western Circle"]
    short_names = ["", "", "1 (R)", "2 (B)", "3 (G)", "West"]
    bg_scale_factors = [1.47, 1.96, 1.4, 1.1, 1.28, 1.0] # third version

    channel = 24*kms
    ch_range = [channel]
    ch_range = np.arange(20, 31, 0.5)*kms
    for channel in ch_range:
        channel_map = cube[cube.closest_spectral_channel(channel), :, :]
        channel_map_zoom = cube_zoom[cube_zoom.closest_spectral_channel(channel), :, :]

        fig = plt.figure(figsize=(16.5, 8))
        img_ax = plt.subplot2grid((2, 3), (0, 0), projection=channel_map.wcs)
        img_zoom_ax = plt.subplot2grid((2, 3), (1, 0), projection=channel_map_zoom.wcs)
        spec_ax = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=2)
        spec_bg_ax = plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=2)
        stretch = np.arcsinh
        vlims = dict(vmin=stretch(0), vmax=stretch(30))
        im = img_ax.imshow(stretch(channel_map.to_value()), origin='lower', cmap='nipy_spectral', **vlims)
        fig.colorbar(im, ax=img_ax)
        im = img_zoom_ax.imshow(stretch(channel_map_zoom.to_value()), origin='lower', cmap='nipy_spectral', **vlims)
        fig.colorbar(im, ax=img_zoom_ax)

        for i, reg in enumerate(reg_list):
            spectrum = cube.subcube_from_regions([reg]).mean(axis=(1, 2))
            spec_ax.plot(cube.spectral_axis.to_value(), spectrum.to_value(), label=names[i], color=colors[i], ls=linestyles[i])
            if i < 2:
                bg_spectrum = cube.subcube_from_regions([reg_list[i+3]]).mean(axis=(1, 2))
                spectrum_bgsub = spectrum - bg_spectrum
                spec_bg_ax.plot(cube.spectral_axis.to_value(), spectrum_bgsub.to_value(), label=names[i]+" (BG subtracted)", color=colors[i], ls=linestyles[i])
            if i == 2:
                bg_spectrum = cube.subcube_from_regions([reg_list[-1]]).mean(axis=(1, 2))*bg_scale_factors[i]
                spec_bg_ax.plot(cube.spectral_axis.to_value(), (spectrum - bg_spectrum).to_value(), label=names[i]+" (BG subtracted)", color=colors[i], ls=linestyles[i])
            for map, ax in zip((channel_map, channel_map_zoom), (img_ax, img_zoom_ax)):
                pixreg = reg.to_pixel(map.wcs)
                pix_center = pixreg.center.xy
                pix_radius = pixreg.radius
                ax.text(pix_center[0]-pix_radius/2, pix_center[1]+pix_radius, short_names[i], color='w', fontsize=11, ha='center', va='bottom')
                pixreg.plot(ax=ax, color='w', lw=2, ls=linestyles[i])

            """
            This is how I estimated the negative noise threshold (which I didn't even end up using!!)
            estimate_noise(cube, spectrum, pix_radius, ax2)
            """

        for ax in (img_ax, img_zoom_ax):
            ax.set_xlabel(" ")
            ax.set_ylabel(" ")
        img_ax.tick_params(axis='x', direction='in', labelbottom=False)
        img_ax.tick_params(axis='y', direction='in')
        img_zoom_ax.tick_params(axis='x', direction='in')
        img_zoom_ax.tick_params(axis='y', direction='in')
        spec_ax.axvline(channel.to_value(), color='grey', alpha=0.4, lw=1.5)
        spec_ax.axhline(0, color='grey', alpha=0.4, lw=1.5)
        spec_ax.legend()
        spec_bg_ax.axvline(channel.to_value(), color='grey', alpha=0.4, lw=1.5)
        spec_bg_ax.axhline(0, color='grey', alpha=0.4, lw=1.5)
        spec_bg_ax.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(f"/home/ramsey/Pictures/2021-06-08-work/channel_map_extrabgsub_{channel.to_value():04.1f}.png")



def identify_components_with_co():
    """
    June 8, 2021
    Same as identify_components() but instead of overplotting multiple regions,
    it's like pillar_sample_spectra() where I overplot multiple lines
    I will have to pick a region (0 thru 5, 5 is the background) for each of these plots
    I have copied some of the below from pillar_sample_spectra()
    """
    cube_filenames = ["sofia/M16_CII_U_APEXbeam.fits", "apex/M16_12CO3-2_truncated_cutout.fits", "apex/M16_13CO3-2_truncated_cutout.fits",
        "bima/M16_12CO1-0_APEXbeam.fits", "apex/M16_CO6-5_APEXbeam.fits"]
    line_channel_vlims = [(0, 35), (0, 30), (None, None), (None, None), (None, None)]
    line_names = ['CII', '12CO(3-2)', '13CO(3-2)',
        '12CO(1-0) * 0.5', '12CO(6-5)']
    line_colors = ['b', 'orange', 'g', 'r', 'purple']
    multiplier = [1, 1, 1, 0.5, 1]
    kms = u.km/u.s

    #### These are for each region in the pillar_samples_components.reg file
    colors = ['k', 'k', 'r', 'b', 'g', 'k']
    linestyles = ['--', '-', '-', '-', '-', ':']
    names = ["Small Dashed Circle", "Small Solid Circle", "Circle 1", "Circle 2", "Circle 3", "Western Circle"]
    short_names = ["", "", "1 (R)", "2 (B)", "3 (G)", "West"]
    # I hand-edited some of these scale factors
    bg_scale_factors = [1.47, 1.96, 1.4, 1.1, 1.28, 1.0] # third version

    reg_fn = "catalogs/pillar_samples_components.reg"
    reg_list = regions.read_ds9(catalog.utils.search_for_file(reg_fn))
    reg_idx = 0

    ref_idx = 1
    channel = 24*kms
    ref_cube = cps2.cutout_subcube(length_scale_mult=12, data_filename=cube_filenames[ref_idx])
    channel_map = ref_cube[ref_cube.closest_spectral_channel(channel), :, :]
    reference_name = f"{line_names[ref_idx]} {channel.to_value():04.1f} {channel.unit}"

    fig = plt.figure(figsize=(15, 5)) # originally (18, 10)
    ax_img = plt.subplot2grid((1, 3), (0, 0))
    ref_img, ref_wcs = channel_map.to_value(), channel_map.wcs
    stretch = np.arcsinh
    im = ax_img.imshow(stretch(ref_img), origin='lower', cmap='nipy_spectral', vmin=stretch(line_channel_vlims[ref_idx][0]), vmax=stretch(line_channel_vlims[ref_idx][1]))
    # fig.colorbar(im, ax=ax_img)
    ax_img.set_title(reference_name)

    reg = reg_list[reg_idx]

    ax_spec = plt.subplot2grid((1, 3), (0, 1), colspan=2, rowspan=1)
    for i, c_fn in enumerate(cube_filenames):
        cube = cube_utils.CubeData(c_fn)
        cube.convert_to_K()
        cube.data = cube.data.with_spectral_unit(kms)
        subcube = cube.data.subcube_from_regions([reg])
        spectrum = subcube.mean(axis=(1, 2)).to_value() * multiplier[i]
        if line_names[i] == 'CII':
            bg_spectrum = cube.data.subcube_from_regions([reg_list[reg_idx+3]]).mean(axis=(1, 2)).to_value() #* bg_scale_factors[reg_idx]
            ax_spec.plot(cube.data.spectral_axis.to_value(), spectrum, label=f"{line_names[i]}", color=line_colors[i], linestyle='--')
            spectrum -= bg_spectrum
            ax_spec.plot(cube.data.spectral_axis.to_value(), spectrum, label=f"{line_names[i]} (BG subtracted)", color=line_colors[i])
        else:
            ax_spec.plot(cube.data.spectral_axis.to_value(), spectrum, label=f"{line_names[i]}", color=line_colors[i])
    pixreg = reg.to_pixel(ref_wcs)
    pix_center = pixreg.center.xy
    pix_radius = pixreg.radius
    # ax_img.text(pix_center[0], pix_center[1]+pix_radius, f"{idx}", color='r', fontsize=11, ha='center', va='bottom')
    pixreg.plot(ax=ax_img, color='k', lw=2, ls=linestyles[reg_idx])

    spec_xlim = (0, 55)
    ax_spec.legend()
    # ax_spec.set_ylim(spec_ylim)
    ax_spec.set_xlim(spec_xlim)
    # ax_spec.axvspan(*(pvl.to_value() for pvl in pillar_vel_limits), color='grey', alpha=0.05)
    ax_spec.axvline(channel.to_value(), color='k', alpha=0.2)
    ax_spec.axhline(0, color='k', alpha=0.2)

    ax_spec.set_xlabel("velocity (km/s)")
    ax_spec.set_ylabel("Line intensity (K)")
    ax_spec.set_title("Line spectra (20\" beam) averaged over selected position")
    # plt.show()
    fig.savefig(f"/home/ramsey/Pictures/2021-06-08-work/pillar_samples_{reg_idx}.png")


def save_transparent_moment_img():
    """
    June 10, 2021
    Save a transparent-background image of M16 in CII for an overlay with optical data
    Source: https://stackoverflow.com/questions/15857647/how-to-export-plots-from-matplotlib-with-transparent-background

    If I had to pick 3 channels of interest, they would be:
    (17, 22): northern cloud first shows up in blue
    (22, 27): pillars
    (27, 34): northern cloud in red
    """
    vel_lims = [(12, 22), (22, 27), (27, 30)]
    vlims = [(30, 70), (40, 170), (18, 50)]
    selected_region = 0
    cube = cps2.cutout_subcube(length_scale_mult=24.)

    stretch = np.arcsinh
    layers = []
    white_space = None
    nanmask = None
    for i in range(len(vel_lims)):
        mom0 = stretch(cube.spectral_slab(*(v*kms for v in vel_lims[i])).moment0().to_value())
        norm = mpl_colors.Normalize(vmin=stretch(vlims[i][0]), vmax=stretch(vlims[i][1]))
        mom0 = norm(mom0)
        if white_space is None:
            white_space = (mom0 < 0)
            nanmask = np.isnan(mom0)
        else:
            white_space &= (mom0 < 0)
            nanmask |= np.isnan(mom0)
        layers.append(mom0)
    rgb = layers[::-1]
    # rgb.append(mpl_colors.LogNorm(vmin=0.6, vmax=1)(np.sum(rgb, axis=0)))
    rgb.append(np.ones_like(mom0)*0.8)
    rgb = np.stack(rgb, axis=-1)
    rgb[white_space | nanmask, :3] = 1
    rgb[white_space | nanmask, 3] = 0
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.imshow(rgb, origin='lower')
    ax.axis('off')


    for ax_name in ('x', 'y'):
        ax.tick_params(axis=ax_name, direction='in', labelleft=False, labelbottom=False)
    ax.set_ylabel(" ")
    ax.set_xlabel(" ")
    # plt.show()
    plt.savefig("/home/ramsey/Pictures/2021-06-10-work/transparent_pillars_3.png", transparent=True)


def peak_temperature():
    """
    June 28, 2021
    This is re: Marc's suggestion to see the peak temperatures of CII,
    CO, HCN, and HCO+ in his June 25, 2021 email
    I'll start with the convolved-to-SOFIA versions
    """
    cube_filenames = ["sofia/M16_CII_U.fits", "bima/M16_12CO1-0_14x14.fits",
        "carma/M16.ALL.hcn.sdi.cm.subpv.SOFIAbeam.fits", "carma/M16.ALL.hcop.sdi.cm.subpv.SOFIAbeam.fits"]
    names = ["[CII]", "12CO(1-0) (CII beam)", "HCN (CII beam)", "HCO+ (CII beam)"]
    short_names = ["cii", "co10CONV", "hcnCONV", "hcopCONV"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = sum((list(x) for x in axes), [])
    for i in range(4):
        if i == 0:
            cube = cps2.cutout_subcube(length_scale_mult=7., reg_index=2)
        else:
            cube = cube_utils.CubeData(cube_filenames[i])
            cube.convert_to_K()
            cube = cube.data
        cube = cube.with_spectral_unit(kms).spectral_slab(19*kms, 27*kms)
        # full_power_idx = cube.argmax(axis=0)
        # full_power_loc = cube.spectral_axis[full_power_idx].to_value()
        full_power = cube.max(axis=0).to_value()
        ax = axes[i]
        im = ax.imshow(full_power, origin='lower', cmap='nipy_spectral')
        ax.set_title(names[i])
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    # fig.savefig("/home/ramsey/Pictures/2021-06-28-work/peak_temperature.png")


def highlight_threads_regions():
    """
    July 6, 2021
    This follows from Marc's recommendation in our last meeting to highlight
    the structure of P1's head in optical and identify those components
    in velocity space
    """
    reg_fn = catalog.utils.search_for_file("catalogs/pillar1_threads3.reg") # pillar2_samples or pillar1_threads3
    reg_list = regions.read_ds9(reg_fn)
    # cube = cube_utils.CubeData("carma/M16.ALL.hcop.sdi.cm.subpv.fits")
    # cube.convert_to_K()
    # wcs_flat = cube.wcs_flat
    # cube = cube.data.with_spectral_unit(kms)
    line_name = "HCO+"
    line_stub = "hcop"
    line_name = "CII"
    line_stub = "cii"
    cube = cps2.cutout_subcube(length_scale_mult=4, reg_index=0)#, data_filename=f"carma/M16.ALL.{line_stub}.sdi.cm.subpv.fits")
    wcs_flat = cube[0].wcs

    # (23, 24), (24, 25), (25, 26) approximately
    # or 23.5, 24.7, 25.8
    # fourth one is the bulk of the head, wider velocity range
    colors = ['b', 'r', 'g', 'cyan']
    velocities = [23.5, 25.8, 24.7, (23, 26)] # for p1 threads3
    # velocities = [(20.8, 22.1), (21.6, 22.6), (22, 23)] # wide band for p2 samples
    # velocities = [21.5, 22, 22.5] # channels for p2 samples


    # load HST
    hst_img, hst_hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    hst_wcs = WCS(hst_hdr)
    hst_name = "HST F657N"
    # hst_img = cube[100].to_value()
    # hst_wcs = wcs_flat
    hst_vlims = dict(vmin=0.1, vmax=0.8)


    fig = plt.figure(figsize=(20, 9))
    ax1 = plt.subplot2grid((2, 6), (0, 2), projection=wcs_flat)
    ax2 = plt.subplot2grid((2, 6), (0, 3), projection=wcs_flat)
    ax3 = plt.subplot2grid((2, 6), (0, 4), projection=wcs_flat)
    axes = [ax1, ax2, ax3]
    if len(reg_list) > 3:
        ax4 = plt.subplot2grid((2, 6), (0, 5), projection=wcs_flat)
        axes.append(ax4)

    ax_img = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2, projection=hst_wcs)
    ax_img.imshow(hst_img, origin='lower', **hst_vlims, cmap='Greys_r')
    ax_img.set_title(hst_name)
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')

    ax_spec = plt.subplot2grid((2, 6), (1, 2), colspan=4)


    for i in range(len(reg_list)):
        v = velocities[i]
        if isinstance(v, tuple):
            v_tup = tuple(x*kms for x in v)
            subcube_img = cube.spectral_slab(*v_tup).moment0()
            v_stub = make_vel_stub(v_tup)
            del v_tup
            ax_spec.axvspan(*v, color=colors[i], alpha=0.1)
        else:
            idx_csc = cube.closest_spectral_channel(v*kms)
            subcube_img = cube[idx_csc]
            v_csc = cube.spectral_axis[idx_csc]
            v_stub = f"{v_csc.to_value():.1f} {v_csc.unit}"
            del v_csc, idx_csc
            ax_spec.axvline(v, color=colors[i], alpha=0.2)
        axes[i].imshow(subcube_img.to_value(), origin='lower', vmin=0, cmap='nipy_spectral')
        lon, lat = axes[i].coords[0], axes[i].coords[1]
        for coord in axes[i].coords:
            coord.set_ticks_visible(False)
            coord.set_ticklabel_visible(False)
            coord.set_axislabel('')
        axes[i].set_title(f"{line_name} {v_stub}", fontsize=7)

        reg = reg_list[i]
        pixreg = reg.to_pixel(subcube_img.wcs)
        pixreg.plot(ax=axes[i], color=colors[i], linestyle='--')
        pixreg = reg.to_pixel(hst_wcs)
        pixreg.plot(ax=ax_img, color=colors[i], linestyle='--')
        subcube_spec = cube.subcube_from_regions([reg])
        spectrum = subcube_spec.mean(axis=(1, 2))
        ax_spec.plot(cube.spectral_axis.to_value(), spectrum.to_value(), color=colors[i])
    ax_spec.set_xlim([18, 30])
    plt.tight_layout()
    # plt.show()
    fig.savefig(f"/home/ramsey/Pictures/2021-07-13-work/p2_spectra_{line_stub}_0.png")






if __name__ == "__main__":
    # vel_lims = dict(vel_start=21.5, vel_stop=22.5)
    # vel_lims = dict(vel_start=24.5, vel_stop=25.5)
    # vel_lims = dict(vel_start=19.5, vel_stop=27.5)
    # thin_channel_images_rb('cii', 'co10CONV', savefig=1, **vel_lims)


    overlaid_contours_for_offset()

    ### showing the colors
    # plt.subplot(111)
    # for i in range(len(marcs_colors)):
    #     plt.axvline([i], color=marcs_colors[i], lw=6)
    # plt.show()

    ### the code for making all the RGB images
    # # ['cii', 'hcn', 'hcnCONV', 'hcop', 'hcopCONV', 'co10', 'co10CONV']
    # sf = True
    # # gb = True
    # vel_lims = dict(vel_start=24.5, vel_stop=25.5)
    # vel_lims = dict(vel_start=21.5, vel_stop=23.5)
    # vel_lims = dict(vel_start=17.5, vel_stop=30.5)
    # for gb in (True, False):
    #     thin_channel_images_rb('cii', 'co10CONV', gb, savefig=sf, **vel_lims)
    #     thin_channel_images_rb('cii', 'hcn', gb, savefig=sf, **vel_lims)
    #     thin_channel_images_rb('cii', 'hcop', gb, savefig=sf, **vel_lims)
