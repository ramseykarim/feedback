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


def compare_carma_to_sofia_crosscut():
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
    """
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    molecule = 'hcn'
    molecule_name = 'HCN'
    mol_fn = [f for f in filepaths if molecule in f].pop()
    mol_conv_fn = [f for f in filepaths_conv if molecule in f].pop()

    reg_filename = catalog.utils.search_for_file("catalogs/p1_threads.reg")
    selected_region = 3 # 0 to 3
    vel_lims = [(24.4, 25.5), (24.8, 25.7), (23.7, 25.8), (25., 27.)][selected_region]
    cc_path = crosscut.coords_from_region(reg_filename, index=selected_region)

    fig = plt.figure(figsize=(17, 7))
    img_ax_f = lambda w: plt.subplot2grid((1, 4), (0, 0), colspan=1, projection=w)
    xcut_ax = plt.subplot2grid((1, 4), (0, 1), colspan=3)
    cco = crosscut.CrossCut(cc_path, vlims=vel_lims)
    cco.setup_figure(fig=fig, xcut_axis=xcut_ax)
    layers = [
        crosscut.DataLayer("[CII]", cube_utils.CubeData("sofia/M16_CII_U.fits"), cube=True, alpha=0.7),
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
    plt.title(f"{molecule_name} (CII beam) {vel_lims}")
    plt.show()


def compare_carma_to_sofia_pv():
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
    """
    # copied from the crosscut version
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcn"), *get_both_mol("hcop"),]
    colors = ['r', 'b', 'cyan', 'violet', 'magenta']
    names = ['CII', 'HCN', 'HCN (CII beam)', 'HCO+', 'HCO+ (CII beam)']

    reg_filename = catalog.utils.search_for_file("catalogs/p1_threads.reg")
    selected_region = 3 # 0 to 3
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_region, width=None) # Try nonzero width later
    vel_limits = np.array([20, 30])*u.km/u.s

    ax_sl, sl_grid_wcs, sl_grid_shape, sl_grid_header = None, None, None, None
    for i, c_fn in enumerate(cube_filenames):
        cube = cube_utils.CubeData(c_fn)
        # should actually use cps2 cutout as wrapper for this and save
        # the one that will make the reference image
        sl = pvextractor.extract_pv_slice(cube.data.spectral_slab(*vel_limits), pv_path)
        if ax_sl is None:
            sl_grid_wcs = WCS(sl.header)
            sl_grid_header = sl.header
            ax_sl = plt.subplot2grid((1, 2), (0, 1), projection=sl_grid_wcs)
            sl_grid_shape = sl.data.shape
            contour_args = (sl.data,)
        else:
            sl.header['CTYPE2'] = 'VRAD' # thanks to the note in m16_pictures.single_parallel_pillar_pvs
            contour_args = (reproject_interp((sl.data, sl.header), sl_grid_header, return_footprint=False),)
        ax_sl.contour(*contour_args, linewidths=1.2, colors=colors[i])

    ax_sl.coords[1].set_format_unit(u.km/u.s)
    ax_sl.coords[0].set_format_unit(u.arcsec)
    ax_sl.coords[0].set_major_formatter('x')
    ax_sl.set_title("PV Diagram")
    ax_sl.set_xlabel("Offset, from E to W (arcseconds)")
    ax_sl.tick_params(axis='x', direction='in')
    ax_sl.tick_params(axis='y', direction='in')

    vel_limits = [(24.4, 25.5), (24.8, 25.7), (23.7, 25.8), (25., 27.)][selected_region]
    # cube = cube_utils.CubeData(cube_filenames[0]) # can also use cps2 cutout function
    # img = cube.data.spectral_slab(*(v*u.km/u.s for v in vel_limits)).moment0().to(u.K * u.km / u.s)

    img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    w = WCS(hdr)
    vlims = dict(vmin=0.1, vmax=0.7)

    ax_img = plt.subplot2grid((1, 2), (0, 0), projection=w)
    ax_img.imshow(img, origin='lower', **vlims)

    handles = [mpatches.Patch(color=c, label=n) for ii, n, c in zip(range(len(cube_filenames)), names, colors)]
    ax_sl.legend(handles=handles, loc='lower right')
    ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color='red', linestyle='-', lw=3, transform=ax_img.get_transform('world'))
    plt.show()


def thin_channel_images_rb(c1_idx, c2_idx, blue_if_true=True, vel_start=24.5, vel_stop=25.5, savefig=True):
    """
    May 27, 2021 (post-MVA appt)
    This will be similar to m16_pictures.make_image_thin_channel_maps
    Following from suggestions from my May 26th M16 meeting, I'll compare thin
    channel maps (like 1 km/s wide) of CO/molecular and CII to see what sorts of
    offsets and gradients I see

    May 28: this function has been really successful and generalized to a few different
    molecular lines and any combination between them. I will extend it to CO 6-5
    and 3-2.
    """
    kms = u.km/u.s
    # copied from the crosscut version
    filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
    filepaths_conv = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.SOFIAbeam.fits"))
    get_mol = lambda mol, fp_list : [f for f in fp_list if mol in f].pop()
    get_both_mol = lambda mol : (get_mol(mol, filepaths), get_mol(mol, filepaths_conv))
    cube_filenames = ["sofia/M16_CII_U.fits", *get_both_mol("hcn"), *get_both_mol("hcop"), "bima/M16_12CO1-0_7x4.fits", "bima/M16_12CO1-0_14x14.fits"]
    colors = ['r', 'b', 'cyan', 'violet', 'magenta', 'LimeGreen', 'g']
    names = ['CII', 'HCN', 'HCN (CII beam)', 'HCO+', 'HCO+ (CII beam)', '12CO(1-0)', '12CO(1-0) (CII beam)']
    short_names = ['cii', 'hcn', 'hcnCONV', 'hcop', 'hcopCONV', 'co10', 'co10CONV']
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
    # blue_if_true = True # green if False
    c1_lo, c1_hi = vlims_r[c1_idx]
    c2_lo, c2_hi = (vlims_b if blue_if_true else vlims_g)[c2_idx]
    cube1 = cube_utils.CubeData(cube_filenames[c1_idx]).data # Cube 1 should have larger footprint (CII)
    cube2 = cube_utils.CubeData(cube_filenames[c2_idx]).data
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
        img1_raw = cube1.spectral_slab(*vel_lims).moment0()
        img2_raw = cube2.spectral_slab(*vel_lims).moment0()
        img2, fp = reproject_interp((img2_raw.to_value(), img2_raw.wcs), img1_raw.wcs, shape_out=img1_raw.shape, return_footprint=True)
        min_cutout_sl = minimum_valid_cutout(fp > 0.5)
        img1 = img1_raw.to_value()[min_cutout_sl]
        img2 = img2[min_cutout_sl]
        nanmask = np.isnan(img1) | np.isnan(img2)
        stretch = np.arcsinh # adjust & apply this later
        img1[nanmask] = np.min(img1)
        img1 -= np.min(img1)
        img2[nanmask] = np.min(img2)
        img2 -= np.min(img2)
        img1, img2 = stretch(img1), stretch(img2)
        # img1 = mpl_colors.Normalize(*misc_utils.flquantiles(img1.flatten(), 1000))(img1)
        img1 = mpl_colors.Normalize(np.median(img1)+c1_lo*np.std(img1), np.max(img1)+c1_hi*np.std(img1))(img1)
        # img2 = mpl_colors.Normalize()(img1)
        # img2 = mpl_colors.Normalize(*misc_utils.flquantiles(img2.flatten(), 20))(img2)
        img2 = mpl_colors.Normalize(np.median(img2)+c2_lo*np.std(img2), np.max(img2)+c2_hi*np.std(img2))(img2)
        # img2 = mpl_colors.Normalize()(img2)
        # Stack in RGB order for matplotlib imshow
        maps = [img1, img2, np.zeros_like(img1)] # just 2 valid layers (R G B order)
        if blue_if_true:
            maps = [maps[0]] + maps[1:][::-1]
        maps = np.stack(maps, axis=-1)
        fig.clear()
        ax = plt.subplot(111)
        ax.imshow(maps, origin='lower')
        vel_str = f"{current_velocity.to_value():.1f}$-${(current_velocity + channel_width).to_value():.1f} {current_velocity.unit}"
        ax.text(0.05, 0.90, f"R: {names[c1_idx]}", transform=ax.transAxes, c='r')
        ax.text(0.05, 0.85, f"{('B' if blue_if_true else 'G')}: {names[c2_idx]}", transform=ax.transAxes, c=('b' if blue_if_true else 'g'))
        ax.set_xlabel("RA"), ax.set_ylabel("Dec")
        ax.set_title(f"R-{('B' if blue_if_true else 'G')} image with channel maps at {vel_str}")
        if savefig:
            try:
                fig.savefig(f"/home/ramsey/Pictures/2021-05-27-work/rgb/{unique_label.replace('CONV', '')}/r{('b' if blue_if_true else 'g')}_{unique_label}_{current_velocity.to_value():04.1f}.png")
            except:
                fig.savefig(f"/home/ramsey/Pictures/2021-05-27-work/rgb/r{('b' if blue_if_true else 'g')}_{unique_label}_{current_velocity.to_value():04.1f}.png")
        current_velocity += channel_width
    if not savefig:
        plt.show()



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

    fig = plt.figure(figsize=(18, 10))

    ax_img = plt.subplot2grid((1, 3), (0, 0))
    ref_img, ref_hdr = fits.getdata(reference_filename, header=True)
    ref_wcs = WCS(ref_hdr)
    im = ax_img.imshow(ref_img, origin='lower', cmap='Greys_r', vmin=0.1, vmax=0.6)
    fig.colorbar(im, ax=ax_img)
    ax_img.set_title(reference_name)


    # reg_idx = 0
    reg = reg_list[reg_idx]

    ax_spec = plt.subplot2grid((1, 3), (0, 1), colspan=2, rowspan=1)
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
    fig.savefig(f"/home/rkarim/Pictures/2021-06-01-work/pillar_samples_{reg_idx}.png")




if __name__ == "__main__":
    reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/pillar_samples.reg"))
    for i in range(3, len(reg_list)):
        pillar_sample_spectra(i)




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
