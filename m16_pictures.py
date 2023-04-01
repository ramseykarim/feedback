"""
A designated place to plot nice M16 images for presentation

Created: October 29, 2020
    Preparing for UJC (and pre-presentation to committee)
First actual use: November 11, 2020
Second use: channel maps for thesis proposal, December 18, 2020
Not sure how much other stuff I've added but adding another figure for the M16
paper, January 13, 2022
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 13}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import sys
import os
import datetime

from math import ceil

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5

from reproject import mosaicking

import pandas as pd

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
# nice blue 50a8ec

kms = u.km/u.s


def marcs_rgb_in_cii():
    """
    See Marc's 1998 paper Fig 3
    Moment 0 images, RGB
    B: 18-20
    G: 20-27
    R: 27-31
    not a PV diagram, but related!!! (must have come from pvdiagrams_2?)
    """
    filenames = ["bima/M16_12CO1-0_7x4.fits", None]
    filenames=[None,]
    fig = plt.figure(figsize=(15, 15)) # (20, 9.5 for CII/CO), ()
    subplot = 111
    axes = []
    vel_limits = [(27, 31), (20, 27), (18, 20),]
    for filename in filenames:
        # length_scale_mult=1.3 for the BIMA/SOFIA RGB, None for the SOFIA solo rgb
        subcube = cps2.cutout_subcube(length_scale_mult=None, reg_filename=catalog.utils.search_for_file("catalogs/across_all_pillars.reg"), reg_index=1, data_filename=filename)
        if filename is None:
            # SOFIA
            # v_limits = [(15, 35), (20, 170), (5, 20),]  # just the pillars
            v_limits = [(15, 120), (15, 180), (5, 30),]  # entire image, use lognorm
        elif 'bima' in filename:
            # BIMA!!!
            v_limits = [(10, 30), (10, 100), (4, 10),]
        v_limits = [dict(vmin=a, vmax=b) for a, b in v_limits]
        norms = [mpl_colors.Normalize(**vl) for vl in v_limits]
        moment0s = [subcube.spectral_slab(*(v*u.km/u.s for v in vl)).moment0() for vl in vel_limits]
        nanmask = np.isnan(moment0s[0])

        ax = plt.subplot(subplot, projection=moment0s[0].wcs)
        subplot += 1
        rgb = [n(m.to_value()) for n, m in zip(norms, moment0s)]
        ### use the following for CII solo rgb
        rgb.append(1 - 0.6*mpl_colors.LogNorm(vmin=20, vmax=8000)(np.sum([m.to_value() for m in moment0s], axis=0)))
        # rgb.append(np.ones_like(rgb[0])*0.9)
        for img in rgb[:2]:
            img[nanmask] = 0
        rgb[3][nanmask] = 1
        rgb = np.stack(rgb, axis=-1)

        ### these are just here for reference I think
        # alpha_floor = 0.1
        # rgb.append(1 - ((1-alpha_floor)*np.sum(rgb, axis=0)/3 + alpha_floor))

        ax.imshow(rgb, origin='lower')
        # ax.set_facecolor('grey')
        # ax.set_title(f"{vel_limits[i]}")

        patch = subcube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.1]), misc_utils.get_pixel_scale(moment0s[0].wcs))
        patch.set_alpha(0.9)
        patch.set_facecolor('white')
        patch.set_edgecolor('grey')
        ax.add_artist(patch)
        axes.append(ax)
    # axes[0].set_title("$^{12}$CO (1-0), from Pound (1998)")
    # axes[1].set_title("[CII], same velocities")
    axes[0].legend(handles=[mpatches.Patch(color=c, label=f"V = {vl[0]:.0f} to {vl[1]:.0f} km/s") for c, vl in zip(('r', 'g', 'b'), vel_limits)], loc='upper left')
    axes[0].set_xlabel("RA"), axes[0].set_ylabel("Dec")

    for ax in axes:
        for ax_name in ('x', 'y'):
            ax.tick_params(axis=ax_name, direction='in', labelleft=False, labelbottom=False)
    axes[0].set_ylabel(" ")
    axes[0].set_xlabel(" ")
    # axes[0].tick_params(axis='y', labelleft=False)
    plt.tight_layout(pad=2, rect=[0.04, 0.02, 1, 0.98], h_pad=0.5)
    plt.show()
    # fig.savefig("/home/ramsey/Pictures/12-21-20-work/cii_rgb_large_AAS1.png", facecolor='k', edgecolor=None)


def m16_channel_maps():
    """
    CII: vlims: 2, 150 (arcsinh)
        label text position: 0.62, 0.05
        ticks: 5, 25, 125
        velocity limits: 15, 35
        grid shape: 5, 4
        figure size: 16, 20 or 20,28 if large?
        text color: black
    CO10: vlims: 4, 125 (arcsinh)
        label text position: 0.70, 0.68
        ticks: 5, 25, 125
        velocity limits: 15, 31
        grid shape: 4, 4
        figure size: 16, 16
        text color: white
    HCO+ & HCN & CS: vlims: 0, 15 (arcsinh)
        label text position: 0.70, 0.68
        ticks: 1, 3, 10
        velocity limits: 18, 30
        grid shape: 4, 3
        figure size: 12, 16
        text color: white
    N2HP: vlims: 0, 7
        ticks: 1, 3, 7
        velocity limits: 18, 27 (another multi-line thing)
        all else same as above
    12CO32: vlims: 0, 75
        ticks: 0, 25, 75
        velocity lims: 15, 35
        grid shape: 5, 4
        figsize: 24, 30
    13CO32: vlims: 0, 25
        ticks: 1, 5, 25
        all else same as 12CO32
    """
    # fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits")
    # fn = catalog.utils.search_for_file("bima/M16_12CO1-0_7x4.fits")
    # fn = catalog.utils.search_for_file("carma/M16.ALL.hcop.sdi.cm.subpv.fits")
    # fn = catalog.utils.search_for_file("carma/M16.ALL.hcn.sdi.cm.subpv.fits")
    # fn = catalog.utils.search_for_file("carma/M16.ALL.cs.sdi.cm.subpv.fits")
    fn = catalog.utils.search_for_file("carma/M16.ALL.n2hp.sdi.cm.subpv.fits")
    # fn = catalog.utils.search_for_file("apex/M16_13CO3-2_truncated.fits")
    cube = cube_utils.CubeData(fn).convert_to_K().data
    # cube = cps2.cutout_subcube(length_scale_mult=12., reg_index=2, data_filename=fn)
    # cube._unit = u.K
    # cube.plot_channel_maps(2, 2, [50, 60, 70, 80], cmap='jet')
    # moments = cube_utils.make_moment_series(cube, (5*kms, 40*kms), 1*kms)
    moments = cube_utils.make_moment_series(cube, (18*kms, 27*kms), 1*kms)
    # assert len(moments) == 20
    grid_shape = (3, 3)
    fig = plt.figure(figsize=(12, 12))
    stretch = np.arcsinh
    ax, im = None, None

    img_to_contour, img_to_contour_hdr = fits.getdata(catalog.utils.search_for_file("sofia/M16_CII_U.mom0.18-28.fits"), header=True)
    contour_args = None

    for i in range(len(moments)):
        v_left, v_right, mom0 = moments[i]
        ax = plt.subplot2grid(grid_shape, (i//grid_shape[1], i%grid_shape[1]), projection=mom0.wcs)
        im = ax.imshow(stretch(mom0.to_value()), vmin=stretch(0), vmax=stretch(7), cmap='inferno')
        for axis_name in ('x', 'y'):
            ax.tick_params(axis=axis_name, direction='in')
            ax.tick_params(axis=axis_name, labelbottom=False, labelleft=False)
        ax.text(0.07, 0.95, f"{v_left.to_value():.1f}$-${v_right.to_value():.1f}\n{v_left.unit}", transform=ax.transAxes, fontsize=16, va='top', color='w')
        if contour_args is None:
            contour_args = (reproject_interp((img_to_contour, img_to_contour_hdr), mom0.wcs, shape_out=mom0.shape, return_footprint=False),)
        ax.contour(*contour_args, colors=marcs_colors[6], linewidths=1, levels=np.linspace(30, 300, 10))
    plt.tight_layout(h_pad=0, w_pad=0, pad=1.01)
    # dx, dy = 0.01, 0.015
    # plt.subplots_adjust(wspace=0, hspace=0, left=dx, right=1-dx, top=1-dy, bottom=dy)

        # if not (i//grid_shape[1] == grid_shape[0]-1 and i%grid_shape[1] == 0):
        #     # for every plot except the bottom left corner
        #     ax.set_xticks([]), ax.set_yticks([])
        # else:
        #     ax.set_ylabel("Dec")
        #     ax.set_xlabel("RA")
    insetcax = inset_axes(ax, width="5%", height="60%", loc='lower right', bbox_to_anchor=(0, 0.01, 0.97, 1), bbox_transform=ax.transAxes)

    ticks = [1, 3, 7]
    cbar = fig.colorbar(im, cax=insetcax, orientation='vertical',
                        ticks=stretch(ticks))
    insetcax.set_yticklabels([f"{x:d}" for x in ticks], fontsize=14)
    insetcax.tick_params(axis='y', colors='w')
    insetcax.yaxis.set_ticks_position('left')
    ax.text(0.70, 0.68, "T (K km/s)", color='w', transform=ax.transAxes, fontsize=14, zorder=10)

    # plt.savefig("/home/ramsey/Pictures/2021-08-31-work/m16_cii_channel_maps_1kms.png")
    fig.savefig("/home/ramsey/Pictures/2022-01-24-work/m16_n2hp_channel_maps_1kms.png",
        metadata=catalog.utils.create_png_metadata(title='channel maps n2hp with cii contours 18-28 from 30 Kkms +30',
        file=__file__, func='m16_channel_maps'))
    # plt.show()


def m16_individual_channel_maps():
    fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits")
    cube = SpectralCube.read(fn)
    cube._unit = u.K
    kms = u.km/u.s
    crop = cps2.cutout_subcube(length_scale_mult=7, return_cutout=True)
    # cube.plot_channel_maps(2, 2, [50, 60, 70, 80], cmap='jet')
    moments = cube_utils.make_moment_series(cube, (11*kms, 35*kms), 1*kms)
    # assert len(moments) == 20
    grid_shape = (3, 5)
    stretch = np.arcsinh
    ax, im = None, None
    for i in range(len(moments)):
        v_left, v_right, mom0 = moments[i]
        fig = plt.figure(figsize=(9.6*1.5, 3.6*1.5))
        ax = plt.subplot(121, projection=mom0.wcs)
        ax_cutout = plt.subplot(122, projection=crop.wcs)
        im = ax.imshow(stretch(mom0.to_value()), vmin=stretch(2), vmax=stretch(150), cmap='inferno')
        im_cutout = ax_cutout.imshow(mom0.to_value()[crop.slices_original], vmin=0, vmax=70, cmap='inferno')
        for axis_name in ('x', 'y'):
            ax.tick_params(axis=axis_name, direction='in')
            # ax.tick_params(axis=axis_name, labelbottom=False, labelleft=False)
            ax_cutout.tick_params(axis=axis_name, labelbottom=False, labelleft=False)
            ax_cutout.tick_params(axis=axis_name, direction='in')
        ax.text(0.1, 0.9, f"{v_left.to_value():.1f}$-${v_right.to_value():.1f} {v_left.unit}", transform=ax.transAxes, fontsize=14)
        ax_cutout.text(0.1, 0.9, f"{v_left.to_value():.1f}$-${v_right.to_value():.1f} {v_left.unit}", transform=ax_cutout.transAxes, fontsize=14, color='white')

        x_box = [crop.xmin_original, crop.xmax_original]
        x_box = x_box + x_box[::-1] + [x_box[0]]
        y_box = [crop.ymin_original, crop.ymax_original]
        y_box = [y_box[0], y_box[0], y_box[1], y_box[1], y_box[0]]
        ax.plot(x_box, y_box, '-', color='white', linewidth=0.5, alpha=0.8)

        insetcax = inset_axes(ax, width="5%", height="60%", loc='lower right', bbox_to_anchor=(0, 0.01, 0.97, 1), bbox_transform=ax.transAxes)

        ticks = [5, 25, 125]
        cbar = fig.colorbar(im, cax=insetcax, orientation='vertical',
                            ticks=stretch(ticks))
        insetcax.set_yticklabels([f"{x:d}" for x in ticks], fontsize=12)
        insetcax.yaxis.set_ticks_position('left')
        cbar_cutout = fig.colorbar(im_cutout, ax=ax_cutout, ticks=np.arange(0, 71, 10), label="T (K km/s)")
        ax.text(0.62, 0.05, "T (K km/s)", transform=ax.transAxes, fontsize=12, zorder=10)
        ax.set_xlabel("RA"), ax.set_ylabel("Dec")
        plt.tight_layout(pad=1, rect=[0.01, 0.05, 1, 1])
        plt.show()
        print(i)
        # plt.savefig(f"/home/ramsey/Pictures/12-21-20-work/channelmaps/map{i:02d}.png")



def residuals_and_wings():
    """
    I need the original cube as arg, and then I can load the fitted stuff
    Revamped based on a pretty sweet DS9 viz I did (Oct 22, 2020)

    Copied from cube_pixel_spectra_2 and reworked to be a standalone function
    """
    cube = cps2.cutout_subcube(length_scale_mult=4)
    kms = u.km/u.s
    vel_bounds = 20*kms, 30*kms
    smooth_beam = cube_utils.Beam(18*u.arcsec)
    cps2.ImgContourPair.smooth_spatial_ = lambda img, w: cps2.smooth_spatial(img, w, cube, smooth_beam)

    pillar_1_highlight = cube.spectral_slab(25*kms, 27*kms).moment0()
    pillar_1_highlight = cps2.ImgContourPair(pillar_1_highlight, "P1", levels=[20, 30, 40, 50, 60], color='k')

    background_35_highlight = cube.spectral_slab(32*kms, 36*kms).moment0()
    background_35_highlight = cps2.ImgContourPair(background_35_highlight, "BG35", levels=[11, 20, 35], color='r')

    background_30_highlight = cube.spectral_slab(29*kms, 30*kms).moment0()
    background_30_highlight = cps2.ImgContourPair(background_30_highlight, "BG30", levels=[6,11], color='orange')

    highlight_27 = cube.spectral_slab(27*kms, 30*kms).moment0()
    highlight_27 = cps2.ImgContourPair(highlight_27, "27", levels=[10, 20, 30], color='violet')

    # original_mom0 = cube.spectral_slab(*vel_bounds).moment(order=0).to(u.K*u.km/u.s).to_value()

    filename_stub = "models/gauss_fit_above_1G_v4_smooth"
    param_fn = cube_utils.os.path.join(cps2.cube_info['dir'], filename_stub+".param.fits")
    resid_fn = cube_utils.os.path.join(cps2.cube_info['dir'], filename_stub+".resid.fits")
    model_fn = cube_utils.os.path.join(cps2.cube_info['dir'], filename_stub+".model.fits")
    resid_cube = cube_utils.SpectralCube.read(resid_fn)
    red_wing_highlight = resid_cube.spectral_slab(27*kms, 30*kms).moment0().to(u.K*u.km/u.s)
    red_wing_highlight = cps2.ImgContourPair(red_wing_highlight, "Redshifted Wing", levels=[6, 9], color='w')
    # resid_mom0 = resid_cube.spectral_slab(*vel_bounds).moment(order=0).to(u.K*u.km/u.s).to_value()
    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(131, projection=cube.wcs, slices=('x', 'y', 0))
    ax2 = plt.subplot(132, projection=resid_cube.wcs, slices=('x', 'y', 0))
    ax3 = plt.subplot(133, projection=cube.wcs, slices=('x', 'y', 0))

    with fits.open(param_fn) as hdul:
        std_fitted = hdul['stddev_FIT'].data

    # metric = (-resid_cube.spectral_slab(*vel_bounds).unmasked_data[:]*np.sign(cube.spectral_slab(*vel_bounds).unmasked_data[:])).to_value()
    # metric[metric < 0] = 0
    # metric = metric.sum(axis=0)
    # ############### add this in

    def plot_axes(img1, v1, img2, v2, *additional_contours):
        im = ax1.imshow(img1.img(), origin='lower', vmin=v1[0], vmax=v1[1], cmap='viridis')
        # im = ax1.imshow(std_fitted, origin='lower', vmin=1.2, vmax=3, cmap='seismic')
        # ax1.set_title("1")
        # ax1.contour(*contour_args, **contour_kwargs)
        ax1.contour(*img1.carg(), **img1.ckwarg())
        fig.colorbar(im, ax=ax1)

        im = ax2.imshow(img2.img(), origin='lower', vmin=v2[0], vmax=v2[1], cmap='viridis')
        ax2.set_title("2")
        for c_img in additional_contours:
            ax2.contour(*c_img.carg(), **c_img.ckwarg())
        ax2.contour(*img2.carg(), **img2.ckwarg())
        fig.colorbar(im, ax=ax2)

    plot_axes(pillar_1_highlight, (5, 80), red_wing_highlight, (0, 30), pillar_1_highlight, background_35_highlight)
    im = ax3.imshow(red_wing_highlight.img()/highlight_27.img(), origin='lower', vmin=0, vmax=1, cmap='viridis')
    ax3.contour(*background_35_highlight.carg(), **background_35_highlight.ckwarg())
    ax3.contour(*red_wing_highlight.carg(), **red_wing_highlight.ckwarg())
    ax3.contour(red_wing_highlight.img()/highlight_27.img(), levels=np.arange(0, 1.01, 0.1), colors='violet', linewidths=0.3, alpha=0.7)
    ax3.set_title('3')
    fig.colorbar(im, ax=ax3)

    plt.show()

def single_parallel_pillar_pvs():
    """
    copied from pvdiagrams_2 on Dec 30, 2020
    The goal now is to have a single parallel cut down each pillar like Marc's
    1998 figures. This function is a good enough template

    I made a new set of pillar cuts: parallelpillars_single
    I should use APEX res versions of all cuts?
        No I should start with CII and CO1-0 so I can use 14x14 for both
        I will leave out CO3-2 until we sort out the spatial offset issue

    Older notes (pre dec 30 2020):
    For this figure I'm thinking of some stuff Lee told me about backing away
    from the Gaussian fits for a bit and going back to spectra/PVs
    I want to make PVs across the pillar, and also compare parallel PVs
    along the pillar (maybe one on each side and one down the center),
    and I'll have to take care to make the beginning/end points comparable

    Also, as of right now, I can't compare M16 CO to CII because there may be
    a pointing error in the APEX data

    Reg files with vectors:
    m16_lines_of_interest.reg is the bunch of labelled lines that mostly
        run perpendicular to the pillars. There's also a good one for the background
        cloud.
    eagpvcuts is Marc's region list, mostly parallel to pillars
    across_all_pillars is just one long region that looks like I made it for
        try_reproject_pv. I added some other long regions to it, it's decent
    parallelpillars.reg is 3 sets of cuts for each of 3 main pillars (9 total)
        One right up the pillar, base to top, and then a "South" to the SW and a "North" to NE
    parallelpillars_2.reg is just 2 cuts per pillar, one on each side, and only
        for the 3 "main" pillars
    parallelpillars_single.reg is just 3 cuts, one for each pillar. They're
        selected to match up to both the CO and CII as best as possible. The
        cut for pillar 1 goes between the horns, doesn't go over either

    Updated September 23, 2021: regridding to CO10 instead of CII to take advantage
    of CO spectral resolution
    """
    colors = marcs_colors[:2][::-1] # ['r', 'k']
    reg_filename = catalog.utils.search_for_file("catalogs/parallelpillars_single.reg") # 3 regions in this file
    pvpath_width = pvdiagrams.m16_allpillars_series_kwargs['pvpath_width']
    pvpath_width = None
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=pvpath_width)
    print("WIDTH", 'interpolate' if pvpath_width is None else pvpath_width.to(u.arcsec))
    chosen_cmap = 'Greys' # 'cool'

    fig = plt.figure(figsize=(20, 14))
    pillar_names = ['Pillar 1', 'Pillar 2', 'Pillar 3']
    marker_styles = ['-', '--', ':']
    paths = []
    # axes_sl = []

    subcube_kwargs = dict(reg_index=1, length_scale_mult=2, reg_filename=reg_filename)
    # CII, native resolution
    subcube_cii = cps2.cutout_subcube(**subcube_kwargs).with_spectral_unit(u.km/u.s) # original
    # subcube_cii = cps2.cutout_subcube(data_filename="sofia/M16_CII_pillar1_BGsubtracted.fits", **subcube_kwargs).with_spectral_unit(u.km/u.s) # bgsub
    # CO 1-0 BIMA, 14x14

    def process_co(fn):
        cube_co = cube_utils.CubeData(fn)
        cutout_co = cps2.cutout_subcube(data_filename=fn, **subcube_kwargs)
        cube_co.data = cutout_co
        cube_co.refresh_wcs()
        cube_co.convert_to_K()
        cube_co.data = cube_co.data.with_spectral_unit(kms)
        return cube_co.data

    subcube_co = process_co("bima/M16_12CO1-0_14x14.fits")
    subcube_co_highres = process_co("bima/M16_12CO1-0_7x4.fits")

    # return subcube_cii, subcube_co

    vel_lims = (18*u.km/u.s, 30*u.km/u.s)
    vel_str = f"[{vel_lims[0].to_value():.0f}, {vel_lims[1].to_value():.0f}] km/s"
    for idx, path in enumerate(path_list):
        sl = pvextractor.extract_pv_slice(subcube_co.spectral_slab(*vel_lims), path)
        sl.header['CTYPE2'] = 'VRAD'
        sl_wcs = WCS(sl.header)
        ax_sl = plt.subplot2grid((3, 6), (1, idx*2), colspan=2, rowspan=2, projection=sl_wcs)
        # axes_sl.append(ax_sl)
        ######### IMAGES
        ### Several possibilities for the images
        #########
        # im = ax_sl.imshow(np.zeros_like(sl.data), origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap=chosen_cmap, vmin=0, vmax=1)
        # im = ax_sl.imshow(sl.data, origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap=chosen_cmap)
        sl_highres = pvextractor.extract_pv_slice(subcube_co_highres.spectral_slab(*vel_lims), path)
        ax_sl.imshow(sl_highres.data, origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap=chosen_cmap)
        ######### END images
        ax_sl.coords[1].set_format_unit(u.km/u.s)
        ax_sl.coords[0].set_format_unit(u.arcsec)
        ax_sl.coords[0].set_major_formatter('x')
        ax_sl.invert_yaxis()
        if idx == 0:
            ax_sl.tick_params(axis='x', direction='in')
            ax_sl.tick_params(axis='y', direction='in')
            ax_sl.set_xlabel("Offset, from SE to NW (arcseconds)")
            ax_sl.set_ylabel("Velocity (km/s)")
        else:
            ax_sl.tick_params(axis='x', direction='in')
            ax_sl.tick_params(axis='y', direction='in', labelleft=False)
            ax_sl.set_xlabel(" ")
        ax_sl.set_title(f"{pillar_names[idx]} PV diagram")
        contour_args = (sl.data,)
        if idx == 2:
            levels = np.arange(8, 61, 4)
        else:
            levels = np.arange(10, 71, 10)
        contour_kwargs = dict(linewidths=1.2, colors=colors[1], alpha=1, levels=levels)
        c = ax_sl.contour(*contour_args, **contour_kwargs, zorder=10)
        try:
            ax_sl.clabel(c, levels, inline=True, fontsize=10, fmt='%.0f')
        except:
            pass
        if idx == 0:
            handles = []
            handles.append(mpatches.Patch(color=colors[1], label="CO (1$-$0)"))

        ##########
        ### Now Cii
        ##########
        sl_cii = pvextractor.extract_pv_slice(subcube_cii.spectral_slab(*vel_lims), path)
        ####################################################
        # sl_co.header['CTYPE2'] = 'VRAD' # this is super important and solved a lot of my problems!!!!1
        ####################################################
        contour_args = (reproject_interp((sl_cii.data, sl_cii.header), sl_wcs, shape_out=sl.data.shape, return_footprint=False),)
        # im = ax_sl.imshow(contour_args[0], origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap=chosen_cmap)
        contour_kwargs['colors'] = colors[0]
        contour_kwargs['alpha'] = 1
        c = ax_sl.contour(*contour_args, **contour_kwargs, zorder=9)
        try:
            ax_sl.clabel(c, levels, inline=True, fontsize=10, fmt='%.0f')
        except:
            pass
        if idx == 0:
            handles.append(mpatches.Patch(color=colors[0], label="[CII]"))
            ax_sl.legend(handles=handles, loc='lower right')





    def plot_ellipse_patch(ax, wcs_obj, subcube):
        patch = subcube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.1]), misc_utils.get_pixel_scale(wcs_obj))
        patch.set_alpha(0.5)
        patch.set_facecolor('grey')
        patch.set_edgecolor('k')
        ax.add_artist(patch)


    vel_lims = (20*u.km/u.s, 26*u.km/u.s)
    vel_str = f"[{vel_lims[0].to_value():.0f}, {vel_lims[1].to_value():.0f}] km/s"
    img = subcube_cii.spectral_slab(*vel_lims).moment0().to(u.K * u.km / u.s)
    w = img.wcs
    img = img.to_value()
    stretch = np.arcsinh
    stretch_vlims = lambda a, b: dict(vmin=stretch(a), vmax=stretch(b))
    vlims = stretch_vlims(10, 200)
    ax_img = plt.subplot2grid((3, 6), (0, 0), colspan=3, projection=w)
    im = ax_img.imshow(stretch(img), origin='lower', **vlims, cmap=chosen_cmap)
    handles = []
    for idx, p in enumerate(path_list):
        l = ax_img.plot([c.ra.deg for c in p._coords], [c.dec.deg for c in p._coords], color='red', linestyle=marker_styles[idx], lw=3, transform=ax_img.get_transform('world'), label=pillar_names[idx])
    ax_img.set_title(f"[CII] integrated {vel_str} with paths overlaid")
    ax_img.legend()
    ax_img.set_xlabel("RA")
    ax_img.set_ylabel("Dec")
    ax_img.tick_params(axis='x', direction='in')
    ax_img.tick_params(axis='y', direction='in')
    ticks = [10, 20, 50, 100, 200]
    cbar = fig.colorbar(im, ax=ax_img, ticks=stretch(ticks))
    cbar.ax.set_yticklabels([f"{x:d}" for x in ticks])
    cbar.ax.set_ylabel("Integrated intensity (K km/s)")
    plot_ellipse_patch(ax_img, w, subcube_cii)

    img = subcube_co_highres.spectral_slab(*vel_lims).moment0().to(u.K * u.km / u.s)
    w = img.wcs
    img = img.to_value()
    ax_img = plt.subplot2grid((3, 6), (0, 3), colspan=3, projection=w)
    im = ax_img.imshow(stretch(img), origin='lower', **vlims, cmap=chosen_cmap)
    for idx, p in enumerate(path_list):
        ax_img.plot([c.ra.deg for c in p._coords], [c.dec.deg for c in p._coords], color='red', linestyle=marker_styles[idx], lw=3, transform=ax_img.get_transform('world'), label=pillar_names[idx])
    ax_img.set_title(f"CO (1$-$0) integrated {vel_str} with paths overlaid")
    ax_img.set_xlabel(" ")
    ax_img.set_ylabel(" ")
    ax_img.tick_params(axis='y', direction='in')
    ax_img.tick_params(axis='x', direction='in')
    ticks = [10, 20, 50, 100, 200]
    cbar = fig.colorbar(im, ax=ax_img, ticks=stretch(ticks))
    cbar.ax.set_yticklabels([f"{x:d}" for x in ticks])
    cbar.ax.set_ylabel("Integrated intensity (K km/s)")
    plot_ellipse_patch(ax_img, w, subcube_co_highres)




    plt.tight_layout(h_pad=0, w_pad=0, pad=5)
    # 2021-11-04 (jupiter),
    plt.savefig("/home/ramsey/Pictures/2022-02-22/pv_along.png",
        metadata=catalog.utils.create_png_metadata(title='pv along',
        file=__file__, func='single_parallel_pillar_pvs'))
    # plt.show()

"""
I used carma_pvdiagrams.across_pillars_carma to make the across-pillar cuts

There's also a big crosscut in crosscut_2.cut_across_m16_pillars_again
"""

def cii_systematic_emission():
    """
    April 7, 2021
    I just copied and pasted the stuff below from another function to get started
    it still needs to be heavily edited to do anything useful

    I want to make a moment 0 plot to highlight the systematic emission around 25 km/s
    which is close to the mean velocity of pillar 1
    I also want a moment 1 plot to show how the line center shifts around in this systematic emission
    A few sample spectra could help as well, taken as average spectra from hand-picked regions

    I made a copy of this function in m16_investigation.py on May 9, 2021
    and have been working on an improved version there. That function is
    called cii_systematic_emission_2.
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
    reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/systematicvelocity_samples.reg"))

    fig = plt.figure(figsize=(18, 10))

    ax_img = plt.subplot2grid((2, 3), (0, 0))
    mom0 = cube.spectral_slab(*sysvel_limits).moment0()
    im = ax_img.imshow(np.arcsinh(mom0.to_value()), origin='lower', vmin=1.5, vmax=5, cmap='Greys')
    fig.colorbar(im, ax=ax_img)
    ax_img.set_title(f"{line_name} moment 0 {sysvel_stub}")

    ax_img2 = plt.subplot2grid((2, 3), (1, 0))
    mom1_limits = (10*kms, 40*kms)
    mom1 = cube.spectral_slab(*mom1_limits).moment1()
    im = ax_img2.imshow(mom1.to_value(), origin='lower', cmap='Greys', vmin=18, vmax=30)
    fig.colorbar(im, ax=ax_img2)
    ax_img2.set_title(f"{line_name} moment 1 {make_vel_stub(mom1_limits)}")

    ax = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
    for reg in reg_list:
        subcube = cube.subcube_from_regions([reg])
        spectrum = subcube.mean(axis=(1, 2))
        p = ax.plot(cube.spectral_axis.to_value(), spectrum.to_value())
        pixreg = reg.to_pixel(mom0.wcs)
        for a in (ax_img, ax_img2):
            pixreg.plot(ax=a, color=p[0].get_c())
    ax.set_xlabel("velocity (km/s)")
    ax.set_ylabel(f"{line_name} intensity (K)")
    ax.set_title(f"{line_name} spectra averaged over selected positions")

    ax.axvspan(*(svl.to_value() for svl in sysvel_limits), color='k', alpha=0.1)
    # plt.show()
    fig.savefig(f"/home/ramsey/Pictures/2021-05-03-work/selected_spectra_{line_name}.png")


def save_fits_thin_channel_maps():
    """
    Created: May 3, 2021
    The plan here is to just save 3 channel maps (1 km/s wide or so) as FITS
    files so I can go into DS9 and try them as 3-color images and see what I
    can find

    Right now, I'm thinking 24-25, 25-26, 26-27

    Feb 22, 2022
    Repurposing for Arrowhead conference figure, to highlight threads
    and blue cap over HST (too big to reproject in Python)

    June 28, 2022
    Repurposing to highlight the "blue" and "red" features around the pillars
    for the paper / my meeting tomorrow. On Marc's recommendation.
    """
    # fn = catalog.utils.search_for_file("apex/M16_12CO3-2_truncated.fits")
    # fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits")
    cube = cps2.cutout_subcube(length_scale_mult=None)
    # cube = cps2.cutout_subcube(data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.fits",
    #     length_scale_mult=None)
    # cube = SpectralCube.read(fn)
    # cube._unit = u.K
    # cube = cube.with_spectral_unit(kms)
    # vel_start, channel_width = 24.*kms, 1*kms
    # vel_lims_list = [(22, 23.5), (24, 24.5), (25.5, 26)] # same as m16_threads.highlight_threads_moment0
    # vel_lims_list = [(19, 22.5), (23, 27.5)] # to highlight the "2 groups" of structures
    vel_lims_list = [(19, 21.5), (22, 23.5), (24, 27.5)] # to highlight the velocity shifts between the groups
    for i in range(3):
        # vel_limits = (vel_start + i*channel_width, vel_start + (i+1)*channel_width)
        vel_limits = tuple(v*kms for v in vel_lims_list[i])
        mom0 = cube.spectral_slab(*vel_limits).moment0()
        hdr = mom0.wcs.to_header()
        # hdr['DATE'] = "May 3, 2021", "Feb 22, 2022", "June 28, 2022"
        hdr['DATE'] = "June 29, 2022"
        hdr['CREATOR'] = "Ramsey Karim via m16_pictures.save_fits_thin_channel_maps"
        hdr['OBJECT'] = "M16"
        hdr['COMMENT'] = f"CII moment 0 image {make_vel_stub(vel_limits)}"
        hdu = fits.PrimaryHDU(data=mom0.data, header=hdr)
        hdu.writeto(f"{catalog.utils.m16_data_path}sofia/integrated3_{i}.fits")
    print("done")


def make_image_thin_channel_maps():
    """
    Created: May 3, 2021
    Use the images created in save_fits_thin_channel_maps to make images!
    I have played around in DS9 (quicker turnaround) to find good parameters

    For the 24, 25, 26 1km/s channel maps, use 2-60 limits for all channels
        with an arcsinh stretch

    Updated: May 17, 2021
    Per Marc/Xander, do this same thing but shift through channels over whole
    velocity range (maybe 10-40 right now)
    Do overlap, so only step 1 km/s even though image spans 3 km/s
    Replicate the save_fits_thin_channel_maps functionality without saving image

    Start blue, drop the bluest layer and shift everything blue, and add red
    This way, we can go in increasing (red) velocity

    Right now I'm optimizing this for channel_width == step_width
    If I want to change it later, I'll have to rewrite some things
    """
    fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits")
    kms = u.km/u.s
    cube = SpectralCube.read(fn)
    cube._unit = u.K
    cube = cube.with_spectral_unit(kms)
    vel_start, channel_width = 40.*kms, 1*kms
    current_velocity = vel_start
    vel_stop = 45.*kms # vel_start + 3 will run once

    bgr_layers = [] # blue, green, red
    # Get the first two (bluest) layers started
    for i in range(2):
        vel_limits = (current_velocity, current_velocity + channel_width)
        mom0 = cube.spectral_slab(*vel_limits).moment0()
        current_velocity += channel_width
        bgr_layers.append(mom0)
    # Get the bgr_layers list in the right shape with a dummy blue layer
    bgr_layers.insert(0, None)
    # Get the WCS object
    w = mom0.wcs

    fig = plt.figure(figsize=(12, 12))

    # Work blue-to-red, and keep bgr_layers in blue-to-red order, but keep
    # in mind that the image array must be red-to-blue
    while current_velocity < vel_stop:
        # create the new red layer and slide the list towards red
        vel_limits = (current_velocity, current_velocity + channel_width)
        mom0 = cube.spectral_slab(*vel_limits).moment0()
        bgr_layers.append(mom0)
        del bgr_layers[0]
        # stack the arrays (red-to-blue)
        maps = np.stack(bgr_layers[::-1], axis=-1)
        # Set up stretch with the parameters from the initial image (sqrt, 2-80)
        stretch = np.sqrt
        v_limits = dict(vmin=stretch(2), vmax=stretch(80))
        maps = stretch(maps)
        norm = mpl_colors.Normalize(**v_limits)
        nanmask = np.isnan(maps[:, :, 0])
        maps = norm(maps)
        maps[nanmask] = 0
        fig.clear()
        ax = plt.subplot(111, projection=w)
        ax.imshow(maps, origin='lower')
        ax.text(0.05, 0.90, f"R: {(current_velocity - 0*channel_width).to_value():.0f} - {(current_velocity - -1*channel_width).to_value():.0f} {current_velocity.unit}", transform=ax.transAxes, c='r')
        ax.text(0.05, 0.85, f"G: {(current_velocity - 1*channel_width).to_value():.0f} - {(current_velocity - 0*channel_width).to_value():.0f} {current_velocity.unit}", transform=ax.transAxes, c='g')
        ax.text(0.05, 0.80, f"B: {(current_velocity - 2*channel_width).to_value():.0f} - {(current_velocity - 1*channel_width).to_value():.0f} {current_velocity.unit}", transform=ax.transAxes, c='b')
        ax.set_xlabel("RA"), ax.set_ylabel("Dec")
        fig.savefig(f"/home/ramsey/Pictures/2021-05-17-work/rgb/rgb_1_g{(current_velocity - 1*channel_width).to_value():02.0f}.png")
        current_velocity += channel_width
        print("saved", current_velocity)
    return


    # from 0 to 2 in blue-to-red order, so reverse them so they're RGB
    maps = np.stack([fits.getdata(f"{catalog.utils.m16_data_path}sofia/thin_channel_{i}.fits") for i in range(3)][::-1], axis=-1)
    w = WCS(fits.getdata(f"{catalog.utils.m16_data_path}sofia/thin_channel_0.fits", header=True)[1])
    kms = u.km/u.s
    vel_start, channel_width = 24.*kms, 1*kms
    stretch = np.sqrt
    v_limits = dict(vmin=stretch(2), vmax=stretch(80))
    maps = stretch(maps)
    norm = mpl_colors.Normalize(**v_limits)
    nanmask = np.isnan(maps[:, :, 0])
    maps = norm(maps)
    maps[nanmask] = 0
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, projection=w)
    ax.imshow(maps, origin='lower')

    # optional: add CO contours
    fn = catalog.utils.search_for_file("apex/M16_12CO3-2_truncated.fits")
    cube = SpectralCube.read(fn)
    colors = ('b', 'LimeGreen', 'r')
    for i in range(3):
        vel_limits = (vel_start + i*channel_width, vel_start + (i+1)*channel_width)
        mom0 = cube.spectral_slab(*vel_limits).moment0()
        mom0_reproj = reproject_interp((mom0.to_value(), mom0.wcs), w, maps.shape[:2], return_footprint=False)
        # mom0_reproj[nanmask] = np.nan
        # [1.5, 3., 4.5] for original, [1.5, 4.5] for 2, [3] for 3
        ax.contour(mom0_reproj, levels=[3], linewidths=0.7, colors=colors[i])

    # plt.show()
    fig.savefig("/home/ramsey/Pictures/2021-05-03-work/rgb_channel_COoverlay3.png")


def quick_make_CO_mom0():
    """
    Created: May 4, 2021
    CO3-2 moment 0 between 20-27 km/s to compare with CO1-0 mom0 that Marc made
    so that I can compare to HST and CO1-0 to prove pointing offset in ds9

    May 9, 2021
    Also used this to make a 25-26 km/s moment 0 and for 19-21
    May 12, 2021
    Also used this to make 20-27 km/s moment 0 for CO 6-5
    """
    fn = catalog.utils.search_for_file("apex/M16_CO6-5.fits")
    cube = SpectralCube.read(fn)
    kms = u.km/u.s
    cube._unit = u.K
    cube = cube.with_spectral_unit(kms)
    vel_limits = (20*kms, 27*kms)
    mom0 = cube.spectral_slab(*vel_limits).moment0()
    hdr = mom0.wcs.to_header()
    hdr['CREATOR'] = "Ramsey Karim via m16_pictures.quick_make_CO_mom0"
    hdr['DATE'] = "May 12, 2021"
    hdr['OBJECT'] = "M16"
    hdr['BUNIT'] = "K km s-1"
    hdr['COMMENT'] = "12CO(6-5) moment 0 between 20-27 km/s"
    hdu = fits.PrimaryHDU(data=mom0.to_value(), header=hdr)
    hdu.writeto(catalog.utils.m16_data_path+"apex/M16_12CO6-5_mom0.fits")


def quick_make_BIMA_HST_APEX_overlay():
    """
    Created: May 5-6, 2021
    two images
    both with HST img base
    each with contours, one from APEX, one from BIMA
    use 13CO for both
    """
    img_hst, hdr_hst = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    img_apex = reproject_interp(catalog.utils.search_for_file("apex/M16_12CO3-2_mom0.fits"), hdr_hst)
    plt.subplot(121)
    plt.imshow(img_hst, origin='lower', vmin=0., vmax=0.6, cmap='Greys_r')
    plt.subplot(122)
    plt.imshow(img_apex, origin='lower')
    # this doesn't work :(
    # reproject_interp is too memory intensive for the HST grid
    plt.show()


def compare_32_65_10():
    co32_cube = SpectralCube.read(catalog.utils.search_for_file("apex/M16_12CO3-2_truncated.fits"))
    co10_cube = SpectralCube.read(catalog.utils.search_for_file("bima/M16_12CO1-0_APEXbeam.fits"))
    co65_cube = SpectralCube.read(catalog.utils.search_for_file("apex/M16_CO6-5_APEXbeam.fits"))
    vel_lims = (20*kms, 27*kms)
    if False:
        # this is how we convolved the 6-5 data
        # Update 2022-08-18, I am convolving 6-5 to CII in m16_investigation.convolve_carma_to_sofia
        co65_cube._unit = u.K
        co65_cube = co65_cube.convolve_to(co32_cube.beam)
        co65_cube.write(catalog.utils.m16_data_path+"apex/M16_CO6-5_APEXbeam.fits", format='fits')
    select = 1
    mom0_co65 = co65_cube.spectral_slab(*vel_lims).moment0()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection=mom0_co65.wcs)
    if select < 0:
        ax.imshow(mom0_co65.to_value(), origin='lower', cmap='Greys')
    else:
        ax.imshow(np.zeros_like(mom0_co65.to_value()), origin='lower', cmap='Greys', vmin=0, vmax=1)
    ax.contour(mom0_co65.to_value(), colors='k')
    handles = [] # FIXME do this legend stuff
    handles.append(mpatches.Patch(color='k', label="APEX 12CO(6-5)"))
    overlay_stub = ""
    if select % 2 == 0:
        co32_mom0 = co32_cube.spectral_slab(*vel_lims).moment0()
        co32_img = reproject_interp((co32_mom0.to_value(), co32_mom0.wcs), mom0_co65.wcs, shape_out=mom0_co65.shape, return_footprint=False)
        overlay_img = co32_img
        overlay_stub = "APEX 12CO(3-2)"
        overlay_color = 'red'
        ax.contour(overlay_img, colors=overlay_color, linewidths=1)
        handles.append(mpatches.Patch(color=overlay_color, label=overlay_stub))
    if select > 0:
        co10_mom0 = co10_cube.spectral_slab(*vel_lims).moment0()
        co10_img = reproject_interp((co10_mom0.to_value(), co10_mom0.wcs), mom0_co65.wcs, shape_out=mom0_co65.shape, return_footprint=False)
        overlay_img = co10_img
        co10_stub = "BIMA 12CO(1-0)"
        if overlay_stub:
            overlay_stub = overlay_stub + " and " + co10_stub
        else:
            overlay_stub = co10_stub
        overlay_color = 'blue'
        ax.contour(overlay_img, colors=overlay_color, linewidths=1)
        handles.append(mpatches.Patch(color=overlay_color, label=co10_stub))
    if select < 0:
        ax.set_title(f"APEX 12CO(6-5)", fontsize=10)
    else:
        ax.set_title(f"APEX 12CO(6-5) compared to {overlay_stub}", fontsize=10)
    ax.set_xlabel("RA"); ax.set_ylabel("Dec")
    ax.legend(handles=handles)
    # plt.show()
    # 2021-05-12,
    plt.savefig("/home/ramsey/Pictures/2022-08-10/CO_65_10.png",
        metadata=catalog.utils.create_png_metadata(title='co lines comparison at APEX beam',
            file=__file__, func='compare_32_65_10'))


def justify_background_figure():
    """
    January 13, 2022
    A figure to justify the use of background subtraction from the pillars
    """
    cube = cps2.cutout_subcube(length_scale_mult=15)
    mom0 = cube.moment0().to_value()

    fig = plt.figure(figsize=(13, 5))
    ax_img = plt.subplot2grid((1, 3), (0, 0))
    ax_spec = plt.subplot2grid((1, 3), (0, 1), colspan=2)

    ax_img.imshow(mom0, origin='lower')
    spec1 = cube.mean(axis=(1, 2)).to_value()
    pillar_moment = cube.spectral_slab(19*kms, 27*kms).moment0().to_value()
    pillar_mask = pillar_moment < 70
    masked_cube = cube.with_mask(pillar_mask)
    spec2 = masked_cube.mean(axis=(1, 2)).to_value()

    ax_img.contour(pillar_moment, levels=[70], colors='w')
    ax_img.set_title("CII integrated line intensity")
    ax_img.xaxis.set_ticks([])
    ax_img.yaxis.set_ticks([])
    ax_spec.plot(cube.spectral_axis.to_value(), spec1, label='Full field')
    ax_spec.plot(cube.spectral_axis.to_value(), spec2, label='Outside pillars')
    ax_spec.axvspan(25, 26, color='k', alpha=0.1)
    ax_spec.axvline(19, linestyle='--', color='k', alpha=0.1)
    ax_spec.axvline(27, linestyle='--', color='k', alpha=0.1)
    ax_spec.set_title("Average CII spectrum")
    ax_spec.set_xlabel("Velocity (km/s)")
    ax_spec.set_ylabel("CII line intensity (K)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/ramsey/Pictures/2022-01-14-work/cii_background_justify.png",
        metadata=catalog.utils.create_png_metadata(title="Moment 0 at full range, contour using cutoff 70 Kkm/s between 19-27km/s",
            file=__file__, func='justify_background_figure'))


def background_samples_figure():
    """
    January 14, 2022
    A follow-up to the above figure, justify_background_figure(), where I show
    the spectra from catalogs/pillar_background_sample_multiple_4.reg (which
    is what I use in cube_pixel_spectra_2.get_cii_background)
    I'm thinking to show the individual spectra from each region in different
    colors (first 4 of marc's colors) and then a thick black line showing the
    average
    """
    cii_cube = cps2.cutout_subcube(length_scale_mult=7)
    bg_reg_filename_short = "catalogs/pillar_background_sample_multiple_5.reg" # this used to be "5_v2" but I renamed "5_v2" to "5", so "5" is the most updated as of Dec 6, 2022
    bg_reg = regions.read_ds9(catalog.utils.search_for_file(bg_reg_filename_short))

    #### Average north or all?
    average_north_or_all = 'north'
    if average_north_or_all == 'north':
        avg_stub = average_north_or_all
        bg_reg_subset = bg_reg[:-1]
    elif average_north_or_all == 'all':
        avg_stub = average_north_or_all
        bg_reg_subset = bg_reg

    default_label_text_size = 12

    cii_bg_spectrum = cii_cube.subcube_from_regions(bg_reg_subset).mean(axis=(1, 2))
    kwargs = {'fill': False}
    fig = plt.figure(figsize=(17.5, 6))
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.99, top=0.97, bottom=0.1, wspace=0.3)

    img, img_hdr = fits.getdata(catalog.utils.search_for_file("misc_regrids/cii_regrid_0.fits"), header=True)
    img_wcs = WCS(img_hdr)
    print(img_wcs)
    ax_img = fig.add_subplot(gs[0, 0], projection=img_wcs)
    im = ax_img.imshow(img, origin='lower', cmap='plasma', vmin=0, vmax=230)
    # Format axis labels
    ax_img.coords[0].set_ticklabel(rotation=0, rotation_mode='anchor', pad=None, fontsize=default_label_text_size, ha='right', va='top')
    ax_img.coords[1].set_ticklabel(fontsize=default_label_text_size)
    ax_img.set_xlabel('Right Ascension')
    ax_img.set_ylabel('Declination', labelpad=0)
    ax_img.tick_params(axis='both', direction='in')
    # Format colorbar
    ax_cbar = ax_img.inset_axes([1, 0, 0.05, 1])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(f"Integrated intensity ({(u.K * kms).to_string('latex_inline')})", size=default_label_text_size)

    ax_spec = fig.add_subplot(gs[0, 1:])
    ax_spec.plot(cii_cube.spectral_axis.to_value(), cii_bg_spectrum.to_value(), color='k', lw=4, label=f"Average {'northern' if avg_stub=='north' else 'all'} background", alpha=0.6)
    for idx, reg in enumerate(bg_reg):
        if idx == len(bg_reg)-1:
            color = 'k'
            ls = ':'
            lw = 3
            label = 'Southern background'
            short_label = "Southern"
            line_kwargs = dict(ls=ls, lw=lw, label=label, color=color)
        else:
            short_label = str(idx+1)
            # color = marcs_colors[idx]
            color = None
            line_kwargs = dict(label=f'{idx+1}')
        # Get and plot spectrum
        spectrum = cii_cube.subcube_from_regions([reg]).mean(axis=(1, 2)).to_value()
        p = ax_spec.plot(cii_cube.spectral_axis.to_value(), spectrum, **line_kwargs)
        # Plot region on reference image
        reg_pixel = reg.to_pixel(img_wcs)
        if color is None:
            color = p[0].get_c()
        artist = reg_pixel.as_artist(**kwargs, color=color)
        ax_img.add_artist(artist)
        x, y = reg_pixel.center.xy
        ax_img.text(x, y, short_label, color=color, ha='center', va='center', fontsize=default_label_text_size)

    beam_patch_coords = [0.9, 0.1] # used to be [0.06, 0.94] # Axes coords
    beam_patch = cii_cube.beam.ellipse_to_plot(*(ax_img.transAxes + ax_img.transData.inverted()).transform(beam_patch_coords), misc_utils.get_pixel_scale(img_wcs))
    beam_patch_kwargs = dict(alpha=0.9, hatch='////', facecolor='white', edgecolor='grey')
    beam_patch.set(**beam_patch_kwargs)
    ax_img.add_artist(beam_patch)
    ax_img.text(beam_patch_coords[0]-0.06, beam_patch_coords[1], cube_utils.cubenames['cii']+'\nbeam', fontsize=default_label_text_size, color='Gainsboro', transform=ax_img.transAxes, va='center', ha='right')

    # ax_img.set_title(f"Integrated CII intensity, {make_vel_stub(vel_lims)}", fontsize=12)
    # for coord in ax_img.coords:
    #     coord.set_ticks_visible(False)
    #     coord.set_ticklabel_visible(False)
    #     coord.set_axislabel('')
    ax_spec.set_xlabel(f"Velocity ({kms.to_string('latex_inline')})")
    ax_spec.set_ylabel(f"{cube_utils.cubenames['cii']} line intensity ({u.K.to_string('latex_inline')})", labelpad=0)
    ax_spec.axhline(0, color='k', alpha=0.1)
    for v in range(20, 29):
        # Velocity gridlines around important velocities
        ax_spec.axvline(v, color='k', alpha=0.07)
    ax_spec.legend()
    # 2022-01-14, 2022-11-29, 2023-02-22, 03-31
    fig.savefig(f"/home/ramsey/Pictures/2023-03-31/cii_background_spectra_avg{avg_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=f'bg regions from {bg_reg_filename_short}, avg {avg_stub}',
            file=__file__, func='background_samples_figure'))


def background_samples_figure_molecular():
    """
    January 21, 2022
    Liz gave me a great idea to check the background of tracers of warm molecular
    gas, like CO(1-0) or CO(3-2) (which I won't publish, but whatever, educational)
    The blueshifted wing in CO1-0 rattled me, and I can see a background in
    CO3-2 in the channel maps. So I want to compare the CII background to
    these CO backgrounds to see if any of the same components are present
    If I find something, then I can talk about it in the background subtraction
    section in the paper
    """
    # cii_cube = cps2.cutout_subcube(length_scale_mult=7)
    cube = cps2.cutout_subcube(data_filename="bima/M16_12CO1-0_7x4.fits", length_scale_mult=None)
    # cube = cps2.cutout_subcube(data_filename="apex/M16_12CO3-2_truncated.fits", length_scale_mult=9)

    line_name = "$^{12}$CO(1$-$0)"
    line_stub = "12co10"

    # line_name = "$^{12}$CO(3$-$2)"
    # line_stub = "12co32"

    bg_reg_filename_short = "catalogs/pillar_background_sample_multiple_4.reg"
    bg_reg = regions.read_ds9(catalog.utils.search_for_file(bg_reg_filename_short))

    # cii_bg_spectrum = cii_cube.subcube_from_regions(bg_reg).mean(axis=(1, 2))

    kwargs = {'fill': False}
    fig = plt.figure(figsize=(14, 6))
    ax_img = plt.subplot2grid((1, 3), (0, 0), projection=cube[0, :, :].wcs)
    ax_spec = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    vel_lims = (19*kms, 27*kms)
    ax_img.imshow(cube.spectral_slab(*vel_lims).moment0().to_value(), origin='lower')

    avg_bg_spectrum = cube.subcube_from_regions(bg_reg).mean(axis=(1, 2)).to_value()
    ax_spec.plot(cube.spectral_axis.to_value(), avg_bg_spectrum, color='k', lw=4, label='Average background', alpha=0.6)

    for idx, reg in enumerate(bg_reg):
        reg_pixel = reg.to_pixel(cube[0, :, :].wcs)
        artist = reg_pixel.as_artist(**kwargs, color=marcs_colors[idx])
        ax_img.add_artist(artist)
        x, y = reg_pixel.center.xy
        ax_img.text(x, y, str(idx+1), color=marcs_colors[idx], ha='center', va='center', fontsize=12)
        spectrum = cube.subcube_from_regions([reg]).mean(axis=(1, 2)).to_value()
        ax_spec.plot(cube.spectral_axis.to_value(), spectrum, color=marcs_colors[idx], label=f'{idx+1}')

    beam_patch_coords = [0.06, 0.94] # Axes coords
    beam_patch = cube.beam.ellipse_to_plot(*(ax_img.transAxes + ax_img.transData.inverted()).transform(beam_patch_coords), misc_utils.get_pixel_scale(cube[0, :, :].wcs))
    beam_patch.set_alpha(0.9)
    beam_patch.set_facecolor('w')
    beam_patch.set_edgecolor('w')
    ax_img.add_artist(beam_patch)
    ax_img.text(beam_patch_coords[0]+0.06, beam_patch_coords[1], f'{line_name}\nbeam', fontsize=9, color='w', alpha=0.9, transform=ax_img.transAxes, va='center', ha='left')
    ax_img.set_title(f"Integrated {line_name} intensity, {make_vel_stub(vel_lims)}", fontsize=12)
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    ax_spec.set_xlabel("Velocity (km/s)")
    ax_spec.set_ylabel(f"{line_name} line intensity (K)")
    ax_spec.legend()
    plt.tight_layout()
    # Previously saved on 2022-01-14
    fig.savefig(f"/home/ramsey/Pictures/2022-01-21-work/{line_stub}_background_spectra.png",
        metadata=catalog.utils.create_png_metadata(title=f'bg regions from {bg_reg_filename_short}',
            file=__file__, func='background_samples_figure'))


def irac8um_to_cii_figure():
    """
    January 17, 2021 (MLK birthday)
    Xander asked for an image of 8micron converted to CII using Cornelia's
    nonlinear relation.
    Xander's message:
        GUSTO is a balloon project to observe [CII].
        It has a beam of 45” which makes it well suited to survey large fields.
        Could you make a one square degree mock-up map of your respective
        regions in the IRAC 8um map at this spatial resolution and then
        translate this to the expected [CII] intensity using Cornelia’s
        non-linear relation ?
    The relation from Pabst+2017 (the Horsehead paper) appears to be:
        I [C II] [erg s^−1 cm^−2 sr^−1] = 2.2 x 10^−2 (I [8um] [ erg s^−1 cm^−2 sr^−1 ])^0.79
    I'll load the IRAC 8um data and I'll have to remind myself of the units
        Units are MJy/sr
        I need to get rid of a Hz-1 unit, so I need to integrate over a bandpass
        or just  multiply by an effective width. The IRAC documentation
        gives an effective width at IRAC channel 4 of
        3.94e12 Hz (3.94 THz, so the spectrometry R~10 since 8um~37Thz)
        I will simply multiply the MJy/sr values by the effective width and use
        that as I_8um
    I found the effective width in Table 4.3 of this doc:
    https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/15/

    It would also be nice to get the figure rotated to align with WCS.
    """
    already_aligned = True
    intensity_unit_cgs = u.Unit('erg s-1 cm-2 sr-1')
    if not already_aligned:
        irac4_fn = catalog.utils.search_for_file("spitzer/SPITZER_I4_mosaic.fits")
        irac4_out_fn = irac4_fn.replace('.fits', '_ALIGNED.fits')
        hdul = fits.open(irac4_fn)
        hdr = hdul[0].header
        keys_to_save = ['ORIGIN', 'INSTRUME', 'CHNLNUM',
            'AOT_TYPE', 'AORLABEL', 'OBJECT', 'BUNIT']
        creator = hdr['CREATOR']
        original_src_file = creator.split(', ')[1] + 'y' # the y got cutoff
        original_src_file = "/".join(original_src_file.split('/')[-2:]) + '::mosaic_irac()'
        new_src_file = "/".join(__file__.split('/')[-2:]) + '::irac8um_to_cii_figure()'

        # Note to future self: this is how I am aligning it with WCS
        # If you just feed find_optimal_celestial_wcs the unaligned WCS, it aligns it!
        w_original = WCS(hdul[0].header)
        w_aligned, shape_aligned = mosaicking.find_optimal_celestial_wcs(
            [(hdul[0].data, hdul[0].header)],
        )
        data_aligned = reproject_interp(hdul[0], w_aligned, shape_out=shape_aligned, return_footprint=False)

        fig = plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(121, projection=w_original)
        ax2 = plt.subplot(122, projection=w_aligned)
        ax1.imshow(hdul[0].data, origin='lower')
        ax2.imshow(data_aligned, origin='lower')
        fig.savefig('/home/ramsey/Pictures/2021-01-17-work/irac4_regrid.png',
            metadata=catalog.utils.create_png_metadata(title='IRAC 4 aligned with RA-DEC',
                file=__file__, func='irac8um_to_cii_figure'))
        hdu_aligned = fits.PrimaryHDU(data=data_aligned, header=w_aligned.to_header())
        hdu_aligned.header['AUTHOR'] = "Ramsey Karim"
        hdu_aligned.header['CREATOR'] = new_src_file
        hdu_aligned.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        # Copy some keys from the previous header
        for k in keys_to_save:
            hdu_aligned.header[k] = hdul[0].header[k]
        hdul.close()
        hdu_aligned.header['HISTORY'] = "Archival files GLM_01650+0075 and 01750_0075 were mosaicked"
        hdu_aligned.header['HISTORY'] = "The result was rotated to align with RA-DEC"
        hdu_aligned.header['HISTORY'] = 'Functions used:'
        hdu_aligned.header['HISTORY'] = original_src_file
        hdu_aligned.header['HISTORY'] = new_src_file
        hdu_aligned.writeto(irac4_out_fn)
        print("DONE")
        return
    already_converted = True
    if not already_converted:
        # Already aligned with RA-DEC, so I can just load that file now
        irac4_fn = catalog.utils.search_for_file("spitzer/SPITZER_I4_mosaic_ALIGNED.fits")
        hdul = fits.open(irac4_fn)
        mjysr = u.Unit(hdul[0].header['BUNIT'])
        irac4_effective_width = 3.94e12 * u.Hz
        desired_unit = intensity_unit_cgs # erg s-1 cm-2 sr-1
        I_8um = (hdul[0].data * mjysr * irac4_effective_width).to(desired_unit)
        """
        Cornelia's equation:
        I [C II] [erg s^−1 cm^−2 sr^−1] = 2.2 x 10^−2 (I [8um] [ erg s^−1 cm^−2 sr^−1 ])^0.79
        """
        I_CII = 2.2e-2 * (I_8um/desired_unit).decompose()**0.79 * desired_unit
        hdul[0].header['BUNIT'] = str(desired_unit)
        hdul[0].header['COMMENT'] = 'This is the estimated integrated CII line intensity from observed 8um'
        hdul[0].header['COMMENT'] = 'Estimated using the relation from Pabst et al. 2017'
        hdul[0].header['COMMENT'] = 'With both I [C II] and I [8um] in units of BUNIT'
        hdul[0].header['COMMENT'] = "I [C II] = 2.2 x 10^-2 (I [8um])^0.79"
        hdul[0].header['COMMENT'] = "8um converted from MJy/sr to BUNIT using effective width"
        hdul[0].header['COMMENT'] = '3.94 THz from Table 4.3 of IRAC Instrument Handbook'
        hdul[0].data = (I_CII/desired_unit).decompose().to_value()
        hdul.writeto(os.path.join(os.path.dirname(irac4_fn), 'estimated_cii_intensity.fits'), overwrite=True)
        hdul.close()
    """
    Convert the CII to intensity to compare to the 8um estimate
    Referencing the similar conversions made in m16_deepdive.py::prepare_pdrt_tables()
    and cii_pacs_contrib.py
    """
    cii_cube = cps2.cutout_subcube(length_scale_mult=None)
    new_hdr = cii_cube[0, :, :].wcs.to_header()
    del new_hdr['SPECSYS']
    new_hdr['AUTHOR'] = "Ramsey Karim"
    new_hdr['CREATOR'] = "/".join(__file__.split('/')[-2:]) + '::irac8um_to_cii_figure()'
    new_hdr['BUNIT'] = str(intensity_unit_cgs)
    new_hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    new_hdr['COMMENT'] = 'Created from the SOFIA CII cube M16_CII_U.fits'
    new_hdr['COMMENT'] = 'Integrated across full range'
    new_hdr['COMMENT'] = 'Converted to CGS using channel width as the spectral term'
    new_hdr['COMMENT'] = 'mom0[K km/s] / [km/s] -> [Jy/sr], *= [km/s]->[Hz], -> CGS'
    cii_fn_out = os.path.join(cps2.cube_info['dir'], 'cii_integrated_intensity_cgs.fits')
    line_center = cii_cube.header['RESTFREQ']*u.Hz
    velocity_axis = cii_cube.spectral_axis[::-1].to(u.Hz, equivalencies=u.doppler_radio(line_center))
    dv_channel = np.mean(np.diff(velocity_axis))
    print(dv_channel.to(u.MHz))
    cii_mom0 = cii_cube.moment0().to(u.K*kms)
    cii_mom0_cgs = (cii_mom0/kms).decompose().to(u.Jy/u.sr, equivalencies=u.brightness_temperature(line_center)) * dv_channel
    cii_mom0_cgs = cii_mom0_cgs.to(intensity_unit_cgs)
    hdu = fits.PrimaryHDU(data=(cii_mom0_cgs/intensity_unit_cgs).decompose().to_value(),
        header=new_hdr)
    hdu.writeto(cii_fn_out)
    print()


def simple_mom0_carma_molecules(line_name, pacs70_reproj=None):
    """
    April 21, 2022
    Lee wanted to see integrated intensity maps of M16
    """
    if line_name == 'cii':
        cube = cps2.cutout_subcube(length_scale_mult=10)
    else:
        cube = cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line_name], length_scale_mult=None)
    mom0 = cube.moment0().to(u.K*kms)
    nchannels = cube.shape[0]
    cube_dv = np.abs(np.diff(cube.spectral_axis[:2])[0].to(kms).to_value())
    channel_noise = cube_utils.onesigmas[line_name]
    moment_noise = cube_dv * channel_noise * np.sqrt(nchannels)
    if pacs70_reproj is None:
        pacs70_fn = catalog.utils.search_for_file("herschel/anonymous1603389167/1342218995/level2_5/HPPJSMAPB/hpacs_25HPPJSMAPB_blue_1822_m1337_00_v1.0_1471714532334.fits.gz")
        pacs70_img, pacs70_hdr = fits.getdata(pacs70_fn, header=True)
        pacs70_reproj = reproject_interp((pacs70_img, pacs70_hdr), mom0.wcs, shape_out=mom0.shape, return_footprint=False)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection=mom0.wcs)
    im = ax.imshow(np.arcsinh(pacs70_reproj), origin='lower', cmap='Greys', vmin=0.6, vmax=1.6)
    fig.colorbar(im, ax=ax, label='PACS 70 micron (arcsinh Jy/pixel)')
    if line_name in ['hcop', 'hcn', 'cs', 'n2hp']:
        levels = [(i**2)*moment_noise for i in range(1, 9, 1)] # carma
    elif line_name in ['13co10', 'c18o10']:
        levels = [(i)*moment_noise for i in range(1, 100, 2)] # 13 and 18 co
    elif line_name == '12co10':
        levels = [(i)*moment_noise for i in range(1, 100, 5)] # 12 co
    elif line_name == 'cii':
        levels = [i*moment_noise for i in range(10, 200, 6)]
    else:
        raise RuntimeError("Haven't hardcoded " + line_name + " yet")
    linestyles = ['--'] + ['-']*(len(levels) - 1)
    ax.contour(mom0.to_value(), colors=marcs_colors[0], linewidths=1, linestyles=linestyles, levels=levels)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title("Integrated " + cube_utils.cubenames[line_name] + " contours on PACS70", fontsize=10)


    patch = cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.1]), misc_utils.get_pixel_scale(mom0.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('white')
    patch.set_edgecolor('grey')
    ax.add_artist(patch)

    plt.savefig(f"/home/ramsey/Pictures/2022-04-25/{line_name}-moment0.pdf")


def overlay_every_spectrum():
    """
    April 26, 2022
    Quick look at overlaying every spectrum (which I am using above) just to see
    where all the emission is. The spectra will be averaged across the field of
    the pillars (length_scale_mult=10 or so)
    """
    line_names_list = ['hcop', 'hcn', 'cs', 'n2hp', '12co10', 'cii']
    fig = plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)
    for line_name in line_names_list:
        if line_name == 'cii':
            cube = cps2.cutout_subcube(length_scale_mult=10)
        else:
            cube = cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line_name], length_scale_mult=None) #.spectral_slab(18*kms, 27*kms)
        cube_x = cube.spectral_axis.to_value()
        mom0 = cube.moment0()
        noise = np.sqrt(cube.shape[0]) * cube_utils.onesigmas[line_name] * np.abs(cube_x[1] - cube_x[0])
        mask = mom0 > 3*noise*u.K*kms
        spectrum = cube.with_mask(mask).mean(axis=(1, 2)).to_value()
        # spectrum /= spectrum.max()
        ax.plot(cube_x, spectrum, label=line_name)
    ax.legend()
    plt.show()



def advanced_carma_molecules():
    """
    April 26, 2022
    Following simple_mom0_carma_molecules, Lee wanted to look deeper into the
    linewidth of N2H+ and CS compared to HCN and HCO+
    I will put spectra on top of each other and then make moment 0 images in the
    range of the thinnest line, and then the complement.
    Lee said it's fine to make moment images which only use the emission part of
    the spectrum.
    """
    if False:
        # these are through the brightest regions
        reg_filename_short = "catalogs/pillar1_emissionpeaks.moreprecise.reg"
    else:
        # these are through a variety of regions and are definitely spaced apart by > 1 SOFIA beam
        reg_filename_short = "catalogs/pillar1_pointsofinterest_v3.reg"
    sky_regions = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    selected_region = sky_regions[3]
    line_names_list = ['hcop', '13co10', 'cs']
    vel_lims = (18*kms, 27*kms)
    fig = plt.figure(figsize=(8, 10))
    grid_shape = (len(line_names_list), 2)
    axes_spec = [plt.subplot2grid(grid_shape, (i, 0)) for i in range(len(line_names_list))]
    axes_img = []
    for i, line_name in enumerate(line_names_list):
        original_cube = cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line_name], length_scale_mult=None)
        cube = original_cube.spectral_slab(*vel_lims)
        cube_x = cube.spectral_axis.to_value()
        channel_noise = cube_utils.onesigmas[line_name]
        mask = (cube > channel_noise*u.K) | (cube < -channel_noise*u.K)
        mom0 = cube.with_mask(mask).moment0()
        noise = np.sqrt(cube.shape[0]) * channel_noise * np.abs(cube_x[1] - cube_x[0])
        ax_img = plt.subplot2grid(grid_shape, (i, 1), projection=mom0.wcs)
        axes_img.append(ax_img)
        ax_img.imshow(mom0.to_value(), origin='lower')
        ax_img.contour(mom0.to_value(), levels=[noise*k for k in (5,)], linewidths=0.7, colors='r')
        ax_img.set_title(line_name)
        ### get spectrum from point
        pix_coords = tuple(round(x) for x in selected_region.to_pixel(mom0.wcs).center.xy[::-1])
        spectrum = original_cube[(slice(None), *pix_coords)].to_value()
        ax_img.plot([pix_coords[1]], [pix_coords[0]], 'x', color='k')
        axes_spec[i].plot(original_cube.spectral_axis.to_value(), spectrum)

        for v in range(21, 29):
            axes_spec[i].axvline(v, color='k', linewidth=0.7)
        axes_spec[i].set_xlim((17, 30))

    plt.show()


def advanced_mom0_carma_molecules():
    """
    April 26, 2022
    Follow up to the stuff above, actually doing the moment 0s now.
    """
    if False:
        # these are through the brightest regions
        reg_filename_short = "catalogs/pillar1_emissionpeaks.moreprecise.reg"
    else:
        # these are through a variety of regions and are definitely spaced apart by > 1 SOFIA beam
        reg_filename_short = "catalogs/pillar1_pointsofinterest_v3.reg"
    sky_regions = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    selected_region = sky_regions[3]
    line_names_list = ['hcn', 'cs']
    fig = plt.figure(figsize=(12, 10))

    if line_names_list[1] == 'cs':
        vel_lims_inner = (23.1*kms, 26.7*kms)
    elif line_names_list[1] == 'n2hp':
        vel_lims_inner = (22.5*kms, 25.7*kms)
    vel_lims_outer = (22.5*kms, 26.7*kms)

    grid_shape = (2, 3)
    ax_spec = plt.subplot2grid(grid_shape, (0, 0), colspan=3)
    axes_img = [plt.subplot2grid(grid_shape, (1, k)) for k in range(3)]

    spectra = []

    zoom_reg, zoom_reg_idx = "catalogs/p1_IDgradients_thru_head.reg", 2
    lsm = 1
    cube_1, cube_2 = tuple(cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line_name], length_scale_mult=lsm, reg_filename=zoom_reg, reg_index=zoom_reg_idx) for line_name in line_names_list)
    subcube_1, subcube_2 = tuple(cube.spectral_slab(*vel_lims_inner) for cube in (cube_1, cube_2))

    channel_noise_1, channel_noise_2 = tuple(cube_utils.onesigmas[line_name] for line_name in line_names_list)
    mask_1 = (subcube_1 > channel_noise_1*u.K) | (subcube_1 < -channel_noise_1*u.K)
    mask_2 = (subcube_2 > channel_noise_2*u.K) | (subcube_2 < -channel_noise_2*u.K)
    mom0_1 = subcube_1.with_mask(mask_1).moment0()
    mom0_2 = subcube_2.with_mask(mask_2).moment0()
    mom0_list = [mom0_1, mom0_2]

    cube_x_1 = cube_1.spectral_axis.to_value()
    noise_1 = np.sqrt(cube_1.shape[0]) * channel_noise_1 * np.abs(cube_x_1[1] - cube_x_1[0])

    axes_img[0].imshow(mom0_1.to_value(), origin='lower')
    axes_img[1].imshow(mom0_2.to_value(), origin='lower')

    noise_cutoff_coeff = 5
    noise_cutoff = noise_cutoff_coeff*noise_1

    bright_mask = mom0_1.to_value() > noise_cutoff

    axes_img[0].set_title(f"{line_names_list[0]} Inner (blue range)")
    axes_img[1].set_title(f"{line_names_list[1]} Inner (blue range)")
    axes_img[2].set_title(f"{line_names_list[0]} Outer (red range)")

    for i, cube in enumerate((cube_1, cube_2)):
        ### get spectrum from point
        spectrum = cube.with_mask(bright_mask).mean(axis=(1, 2)).to_value()
        ax_spec.plot(cube.spectral_axis.to_value(), spectrum/spectrum.max(), label=line_names_list[i])


    outer_lim_lo = (vel_lims_outer[0], vel_lims_inner[0])
    outer_lim_hi = (vel_lims_inner[1], vel_lims_outer[1])

    for v in range(21, 29):
        ax_spec.axvline(v, color='k', linewidth=0.7, alpha=0.4)
    ax_spec.axvspan(vel_lims_inner[0].to_value(), vel_lims_inner[1].to_value(), alpha=0.2, color='b', label=f"Inner: {make_vel_stub(vel_lims_inner)}")
    ax_spec.set_xlim((17, 30))


    outer_mom0_result = mom0_1 * 0
    if outer_lim_lo[0] < outer_lim_lo[1]:
        print("doing lower outer")
        ax_spec.axvspan(outer_lim_lo[0].to_value(), outer_lim_lo[1].to_value(), alpha=0.2, color='r', label=f"Outer: {make_vel_stub(outer_lim_lo)}")
        outer_subcube_lo = cube_1.spectral_slab(*outer_lim_lo)
        outer_mask_lo = (outer_subcube_lo > channel_noise_1*u.K) | (outer_subcube_lo < -channel_noise_1*u.K)
        outer_mom0_lo = outer_subcube_lo.with_mask(outer_mask_lo).moment0()
        outer_mom0_result += outer_mom0_lo
    if outer_lim_hi[0] < outer_lim_hi[1]:
        print("doing upper outer")
        ax_spec.axvspan(outer_lim_hi[0].to_value(), outer_lim_hi[1].to_value(), alpha=0.2, color='r', label=f"Outer: {make_vel_stub(outer_lim_hi)}")
        outer_subcube_hi = cube_1.spectral_slab(*outer_lim_hi)
        outer_mask_hi = (outer_subcube_hi > channel_noise_1*u.K) | (outer_subcube_hi < -channel_noise_1*u.K)
        outer_mom0_hi = outer_subcube_hi.with_mask(outer_mask_hi).moment0()
        outer_mom0_result += outer_mom0_hi

    axes_img[2].imshow(outer_mom0_result.to_value(), origin='lower')

    ax_spec.legend()
    ax_spec.set_title(f"Spectra from within red contours ({cube_utils.cubenames[line_names_list[0]]} inner range moment0 > {noise_cutoff_coeff} $\sigma$)")

    for idx in (0, 1):
        axes_img[idx].contour(mom0_1.to_value(), levels=[noise_cutoff,], linewidths=0.7, colors='r')

    plt.savefig(f"/home/ramsey/Pictures/2022-04-26/mom0_{line_names_list[0]}-{line_names_list[1]}.png",
        metadata=catalog.utils.create_png_metadata(title=f"test image, zoom is lsm={lsm} {zoom_reg} #{zoom_reg_idx}",
            file=__file__, func="advanced_mom0_carma_molecules"))
    # plt.show()


def hh216_co32():
    """
    June 14, 2022
    Follow up John Bally's interest in HH216 in the FEEDBACK meeting this morning
    I couldn't find a direct obvious counterpart to the HH in any CARMA map,
    and it's outside the FOV of the BIMA data, but I did notice a feature at
    around 34 km/s in APEX 12CO 3-2.
    Marc suggested I make a channel map from 32 to 36 or so to see where that
    feature goes
    I think my plan here is to make a channel map in that interval, make another
    one between 17-30 (pillars) and contour that one over the image of 32-36.
    I have regions highlighting HH 216 in catalogs/hh216.reg (a line and a circle)
    """
    cube = SpectralCube.read(catalog.utils.search_for_file("apex/M16_12CO3-2_truncated.fits"))
    # cube = cube[:, 50:100, 70:120]
    pillar_vlims = (17*kms, 30*kms)
    mom0_pillars = cube.spectral_slab(*pillar_vlims).moment0().to(u.K*kms)
    feature_vlims = (32*kms, 36*kms)
    mom0_feature = cube.spectral_slab(*feature_vlims).moment0().to(u.K*kms)

    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, projection=mom0_feature.wcs)
    im = ax.imshow(mom0_feature.to_value())
    fig.colorbar(im, ax=ax, label="$^{12}$CO (3$-$2) integrated intensity (K km/s)")
    ax.contour(mom0_pillars.to_value(), levels=np.arange(15, 245, 45), linewidths=1, colors='r', alpha=0.5)
    ax.set_title("$^{12}$CO (3$-$2) moment 0 highlighting high velocity feature towards HH 216")
    ax.text(0.05, 0.95, "Image: "+make_vel_stub(feature_vlims), transform=ax.transAxes, ha='left', va='top', color='white')
    ax.text(0.05, 0.9, "Contours: "+make_vel_stub(pillar_vlims), transform=ax.transAxes, ha='left', va='top', color='white')

    reg_list = regions.Regions.read(catalog.utils.search_for_file('catalogs/hh216.reg'))
    for reg in reg_list:
        reg.visual['width'] = 1
        reg.to_pixel(mom0_feature.wcs).plot(ax=ax, color='k', alpha=0.8)

    fig.savefig("/home/ramsey/Pictures/2022-06-14/hh216_co32_channel_large.png",
        metadata=catalog.utils.create_png_metadata(title='17,30 pillars 32,36 feature, contours range 15,245,45',
            file=__file__, func="hh216_co32"), dpi=150)


def save_smaller_CO32_maps():
    """
    August 10, 2022
    Repeating the "_truncated" process for the newer CO32 maps that Rolf gave us
    in September 2021 (I have not looked at them until now)
    I couldn't find where I trimmed the older CO32 maps.
    """
    raise RuntimeError("Already ran this on August 10, 2022")
    tw_or_th = "13"
    original_fn = catalog.utils.search_for_file(f"apex/M16_{tw_or_th}CO3-2_ref.fits")
    save_fn = original_fn.replace('_ref.fits', '_truncated.fits') # so that we don't have to rename all the usages
    cube = cube_utils.CubeData(original_fn).data
    subcube = cube.spectral_slab(-10*kms, 60*kms).with_spectral_unit(kms)
    subcube.meta['HISTORY'] = "rkarim trimmed spectra to -10,60 km/s on 2022-08-10"
    subcube.write(save_fn)


def try_component_velocity_figure():
    """
    August 22, 2022
    Trying some version of a component velocity figure to improve on a table.
    This will be sort of a chart, not an astronomical image
    August 26, 2022 Trying again with more data
    """
    csv_fn_template = lambda param : f"/home/ramsey/Pictures/2022-09-12/line_info_2_{param}.csv"
    params = ['mean', 'linewidth', 'amplitude']
    dfs = {param: pd.read_csv(csv_fn_template(param)) for param in params}
    for p in dfs:
        df = dfs[p]
        df.set_index("Region Name", inplace=True)
    line_columns = [y for y in dfs['mean'].columns if y.strip() not in ['reg_name']]
    unique_lines = ['cii', '12co10', '13co10', '12co32', 'co65', 'hcop', 'hcn', 'cs', 'n2hp']
    try:
        assert frozenset(unique_lines) == frozenset(x[:-2] for x in line_columns)
    except AssertionError:
        print("I wrote down: ", unique_lines)
        print("The files contain: ", frozenset(x[:-2] for x in line_columns))
        return
    lines_dict = {x: [] for x in unique_lines}
    for component in line_columns:
        # put the numbered components in a dictionary under the line name key
        lines_dict[component[:-2]].append(component)

    line_offsets = {line: x for line, x in zip(unique_lines, np.linspace(-0.22, 0.28, len(unique_lines)))}

    # Figure out normalization factors for amplitudes
    amplitude_graph_max = 0.15 # Given that the regions are spaced by 1.0, maximum allowable amplitude spread
    line_amp_norms = {}
    for line in unique_lines:
        line_amplitudes = []
        for c in lines_dict[line]:
            line_amplitudes.extend(list(dfs['amplitude'][c].values))
        line_amp_norms[line] = amplitude_graph_max / np.nanmax(line_amplitudes)

    x_keys = ['SE Thread', 'NE Thread', 'SE Head', 'NE Head', 'SW Thread', 'NW Thread', 'SW Head', 'NW Head',]
    x_values = list(range(1, 9))

    markers = ["$c$", 'v', 'D', 'x', '^', '<', '>', 'o', 's']
    assigned_markers = {l: m for l, m in zip(unique_lines, markers)}
    colors = ['b', 'g', 'r', 'k']

    fig = plt.figure(figsize=(13, 10.5))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)


    for i, line in enumerate(unique_lines):
        components = lines_dict[line]
        for c in components:
            xarr1, yarr1 = [], []
            xarr2, yarr2 = [], []
            lwarr1, amparr1 = [], []
            lwarr2, amparr2 = [], []
            c_index = int(c[-1]) - 1
            for xk, xv in zip(x_keys, x_values):
                yv = dfs['mean'].loc[xk, c]
                lw = dfs['linewidth'].loc[xk, c]
                amp = dfs['amplitude'].loc[xk, c]
                if np.isfinite(yv):
                    (xarr1 if xv<5 else xarr2).append(xv)
                    (yarr1 if xv<5 else yarr2).append(yv)
                    (lwarr1 if xv<5 else lwarr2).append(lw)
                    (amparr1 if xv<5 else amparr2).append(amp)
            if xarr1:
                ax1.errorbar(np.array(xarr1)+line_offsets[line], yarr1, xerr=np.array(amparr1)*line_amp_norms[line], yerr=np.array(lwarr1)*2.355/2, color=colors[c_index], marker=assigned_markers[line], alpha=0.7, label=(cube_utils.cubenames[line] if c_index == 1 else None), linestyle='none')
            if xarr2:
                ax2.errorbar(np.array(xarr2)+line_offsets[line], yarr2, xerr=np.array(amparr2)*line_amp_norms[line], yerr=np.array(lwarr2)*2.355/2, color=colors[c_index], marker=assigned_markers[line], alpha=0.7, label=(cube_utils.cubenames[line] if c_index == 1 else None), linestyle='none')
    for i, ax in enumerate((ax1, ax2)):
        ax.set_ylim([20.5, 29.5])
        ax.set_xlim([4*i + 0.5, 4*(i+1) + 0.6])
        ax.set_xticks(x_values[4*i:4*(i+1)])
        ax.set_xticklabels(x_keys[4*i:4*(i+1)], rotation=0, fontsize=13)
        ax.set_ylabel("$V_{\\rm LSR}$ (km/s)")
    ax2.legend(loc='lower center', ncol=len(unique_lines), handletextpad=0.05, handlelength=0.8)
    plt.subplots_adjust(bottom=0.05, hspace=0.2, top=0.95, left=0.05, right=0.97)
    # 2022-08-23,31, 09-09,12
    savename = "/home/ramsey/Pictures/2022-09-12/model_fit_table_viz"
    save_as_png = True
    if save_as_png:
        fig.savefig(f"{savename}.png",
            metadata=catalog.utils.create_png_metadata(title='bunch of lines, unfinished list though',
                file=__file__, func='try_component_velocity_figure'))
    else:
        fig.savefig(f"{savename}.pdf")


def column_density_figure():
    """
    November 22, 2022
    Side-by-side of the column density figures
    I will try interpolating-nearest to keep the pixelization of the lower-resolution maps
    """
    filenames_dict = {'cii': "sofia/Cp_coldens_and_mass_lsm8_ff1.0_with_uncertainty_v2.fits",
        'co': "bima/13co10_column_density_and_more_with_uncertainty_v2.fits",
        'dust': "herschel/coldens_70-160_sampled_1000.fits",
    }
    column_extnames = {'cii': 'Hcoldens', 'co': 'H2coldens_all', 'dust': 'Hcoldens_best'}

    # divide by something (dust: N(H) -> N(H2))
    column_factors = {'dust': 2}
    h_label, h2_label = "{\\rm H}", "{\\rm H}_2"
    column_labels = {'cii': h_label, 'co': h2_label, 'dust': h2_label}
    panel_labels = {'cii': cube_utils.cubenames['cii'], 'co': 'CO (1$-$0)', 'dust': 'FIR dust'}
    vmaxes = {'cii': 3e22, 'co': 1.5e23, 'dust': 7e22}

    # JWST footprint for plotting
    # jwst_footprint_fn = catalog.utils.search_for_file("catalogs/jwst_reproj_footprint.reg")

    # The map whose WCS we will use for the image
    reference_name = 'co'
    ref_hdr = fits.getheader(catalog.utils.search_for_file(filenames_dict[reference_name]), extname=column_extnames[reference_name])
    ref_wcs = WCS(ref_hdr)

    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, 4, left=0.05, right=0.99, top=0.98, bottom=0.05, wspace=0.25)
    axes = []
    tick_labelsize = 9

    """ Cycle thru the column densities """
    for i, line_name in enumerate(['cii', 'co', 'dust']):
        ax = fig.add_subplot(gs[0, i], projection=ref_wcs)
        axes.append(ax)

        img_raw, hdr_raw = fits.getdata(catalog.utils.search_for_file(filenames_dict[line_name]), extname=column_extnames[line_name], header=True)
        if line_name == reference_name:
            img = img_raw
        else:
            img = reproject_interp((img_raw, hdr_raw), ref_hdr, order='nearest-neighbor', return_footprint=False)
        if line_name in column_factors:
            img = img/column_factors[line_name]
        im = ax.imshow(img, origin='lower', vmin=0, vmax=vmaxes[line_name], cmap='plasma')
        # Colorbar
        cax = ax.inset_axes([1, 0, 0.05, 1])
        cbar = fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=tick_labelsize)
        cbar_tick_labelsize = tick_labelsize + 1
        cbar.set_label("$N("+column_labels[line_name]+")$ ("+(u.cm**-2).to_string('latex_inline')+")", size=cbar_tick_labelsize)
        cax.yaxis.offsetText.set(size=cbar_tick_labelsize)
        # Label
        text_x = 0.1 # 0.025
        ax.text(text_x, 0.94, panel_labels[line_name], transform=ax.transAxes, fontsize=tick_labelsize+2, color=marcs_colors[1], weight='bold', ha='left', va='center')


    """ JWST reference image """
    img_fn = catalog.utils.search_for_file("jwst/MAST_2022-10-26T1800/JWST/jw02739-o001_t001_nircam_clear-f335m/f335m_rotated.fits")
    img, hdr = fits.getdata(img_fn, header=True)
    img = reproject_interp((img, hdr), ref_hdr, order='bilinear', return_footprint=False)

    ax = fig.add_subplot(gs[0, 3], projection=ref_wcs)
    stretch = np.arcsinh
    img_vlims = (1, 100)
    ax.imshow(stretch(img), origin='lower', vmin=stretch(img_vlims[0]), vmax=stretch(img_vlims[1]), cmap='Greys')

    """ Regions: points """
    reg_list = regions.Regions.read(catalog.utils.search_for_file("catalogs/pillar123_pointsofinterest_v2.reg"))
    xs, ys = [], []
    for reg in reg_list:
        x, y = reg.to_pixel(ref_wcs).center.xy
        xs.append(x)
        ys.append(y)
    ax.plot(xs, ys, linestyle='None', marker='s', markersize=8, mfc=marcs_colors[1], mec='k')
    axes.append(ax)

    """ Regions: boxes """
    chosen_colors = [marcs_colors[x] for x in [0, 7]]
    for i, box_reg_filenames in enumerate(["catalogs/mass_boxes_v2.reg", "catalogs/p123_boxes_head_body_withlabels_v3.reg"]):
        reg_list = regions.Regions.read(catalog.utils.search_for_file(box_reg_filenames))
        for j, reg in enumerate(reg_list):
            if 'noise' in reg.meta['label']:
                continue
            box_artist = reg.to_pixel(ref_wcs).as_artist(ec=chosen_colors[i], fill=False)
            ax.add_artist(box_artist)

    """ Ticks, labels, etc """
    for ax in axes:
        ax.coords[0].set_ticklabel(rotation=30, rotation_mode='anchor', pad=13, fontsize=tick_labelsize, ha='right', va='top')
        ax.coords[1].set_ticklabel(fontsize=tick_labelsize)
        ss = ax.get_subplotspec()
        ax.tick_params(axis='both', direction='in')
        ax.coords.grid(color='grey', alpha=0.5, linestyle='solid')
        ss = ax.get_subplotspec()
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        if not ss.is_first_col():
            # hide ticklabels all but left column
            ax.tick_params(axis='y', labelleft=False)

    # Titles
    fig.supxlabel("Right Ascension")
    fig.supylabel("Declination")
    # 2023-03-21,22
    fig.savefig("/home/ramsey/Pictures/2023-03-22/column_density_figure.png",
        metadata=catalog.utils.create_png_metadata(title='column figure',
                file=__file__, func='column_density_figure'))


def multi_panel_moment_images():
    """
    January 5, 2023 (First work of 2023)
    Create a multi-panel figure showing moment images (over same interval) and
    photometry, all on the same grid and cutout, but all at native resolution
    The key here will be picking a grid and regridding everything to it.

    Probably the 12co10 grid. 12co10 pixels are 0.5'' while IRAC is at 0.6''.
    The CARMA grids (e.g. hcop) are also at 0.5'' and have a larger footprint.

    The only finer grids are optical/NIR, but I'll present those somewhere else
    The data that go here are:
    cii
    oi (maybe cii as overlay?)
    12co10 (c18o10 overlay?)
    13co10 (should this also be an overlay?)
    12co32 (13co32 overlay)
    12co65
    hcop
    hcn (n2hp overlay?)
    cs
    8um
    70um (160um overlay)

    n2hp, c18o10, and any other limited-emission lines can go in a contour overlay somewhere (maybe on this plot?)
    """
    selected_data = [
        'cii', 'oi', '8um', '70um',
        '12co10', '13co10', '12co32', 'co65',
        'hcop', 'hcn', 'cs', 'n2hp',
    ]
    overlays = { # overlay: image on which to overlay
        'c18o10': '12co10', '13co32': '12co32', '160um': '70um'
    }
    grid_shape = (3, 4) # production
    figsize=(19.5, 15.5)
    plots_adjust_kwargs = dict(left=0.05, right=0.96, top=0.95, bottom=0.07, wspace=0.16, hspace=0.08)

    contour_levels_base = np.arange(2, 111, 10)
    contour_levels_coeff = {
        '13co10': 2., 'c18o10': 0.25, '160um': 90000, # 15000 originally
        '13co32': 2., 'n2hp': 1,
    }
    special_contour_levels = {'160um': np.array([1, 2, 3, 5,])*contour_levels_coeff['160um']}
    img_limits = {
        '12co10': (0, 300), '13co10': (0, 55), 'cii': (0, 230), '8um': (None, 1200), 'oi': (0, 40),
        'co65': (0, 60), '12co32': (0, 130),
    }
    img_stretches = {'8um': np.arcsinh, '70um': np.arcsinh, '160um': np.arcsinh} # if not here, linear
    stretch_inverses = {np.arcsinh: np.sinh}
    def apply_stretch(value_or_image, label, invert=False):
        """ Apply stretch. If key not in img_stretches, it's linear. invert=True does inverse stretch """
        if value_or_image is None:
            # Short circuit for vmin/vmaxes left unset
            return None
        if label in img_stretches:
            stretch_fn = img_stretches[label]
            if invert:
                return stretch_inverses[stretch_fn](value_or_image)
            else:
                return stretch_fn(value_or_image)
        else:
            return value_or_image
    def get_vlims(label):
        return dict(vmin=apply_stretch(img_limits[label][0], label), vmax=apply_stretch(img_limits[label][1], label)) if label in img_limits else {} # empty dict if not there
    # img_ticks = {'160um': ()}

    ################################ for the draft version
    ################################ for the draft version
    ################################ for the draft version
    ################################ for the draft version
    # selected_data = ['12co10', 'cii', '70um']
    # overlays = {'c18o10': '12co10', '160um': '70um'}
    # grid_shape = (1, 3) # testing
    ################################ for the draft version
    ################################ for the draft version
    ################################ for the draft version

    # Velocity limits
    vel_lims = (18, 27)
    # Add units to velocity limits and make a string description
    vel_lims = tuple(x*kms for x in vel_lims)
    vel_lims_stub = make_vel_stub(vel_lims)

    # Lookup table for 2D data filenames and info (beams!)
    herschel_dir_stub = "/herschel/anonymous1603389167/1342218995/level2_5/"
    # Beam is tuple (major, minor, PA) where major,minor are in arcseconds and PA in degrees
    photometry_lookup = {
        # (filename, needs_unit_conversion,)
        '8um': ("spitzer/SPITZER_I4_mosaic.fits", (1.98, 1.98, 0)), '12co10': ("bima/M16_12CO1-0_7x4_mom0.fits", None), '13co10': ("bima/M16.BIMA.13co.mom0.fits", None), 'c18o10': ("bima/M16.BIMA.c18o.masked_mom0.fits", None),
        '70um': (herschel_dir_stub+"HPPJSMAPB/hpacs_25HPPJSMAPB_blue_1822_m1337_00_v1.0_1471714532334.fits.gz", (9.0, 5.75, 62)), '160um': (herschel_dir_stub+"HPPJSMAPR/hpacs_25HPPJSMAPR_1822_m1337_00_v1.0_1471714553094.fits.gz", (13.32, 11.31, 40.9)),
        'hcop': ("carma/M16.ALL.hcop.sdi.mom0pv.fits", None), 'cs': ("carma/M16.ALL.cs.sdi.mom0pv.fits", None), 'hcn': ("carma/M16.ALL.hcn.sdi.mom0pv.fits", None), 'n2hp': ("carma/M16.ALL.n2hp.sdi.mom0pv.fits", None),
    }

    def print_stats(stub, img):
        """
        Print out the min, max, mean, median, stddev values for the image
        """
        print(f"---{stub}---")
        for f in (np.min, np.max, np.mean, np.median, np.std):
            print(f"{f.__name__}: {f(img[np.isfinite(img)])}")
        print('-'*(6 + len(stub)))


    def load_helper(cube_or_image_stub):
        # Load a cube or image
        # Perform a moment 0 integration if it's a cube
        # Return the image and WCS
        # TODO: Units of images. Cubes will be in K km/s
        """
        For the record, this procedure is very similar to DataLayer, which I made for CrossCut.
        But it's pretty simple and DataLayer does a lot of other stuff that I don't need to do here,
        so I'm going to rewrite the wheel. It's just a circle.
        """
        if cube_or_image_stub in photometry_lookup:
            filename, beam_info = photometry_lookup[cube_or_image_stub]
            img, header = fits.getdata(catalog.utils.search_for_file(filename), header=True)
            wcs_obj = WCS(header, naxis=2)
            img = np.squeeze(img)
            if beam_info is None:
                beam = cube_utils.Beam.from_fits_header(header)
            else:
                major, minor, pa = beam_info
                beam = cube_utils.Beam(major=major*u.arcsec, minor=minor*u.arcsec, pa=pa*u.deg)
            if 'RESTFREQ' in header: # Catches all the line cubes, nothing else
                # This routine I think is pretty specific to MIRIAD, and Marc did both the bima and carma mom0s in MIRIAD
                restfrq = header['RESTFREQ'] * u.Hz
                data_units = tuple(u.Unit(y.lower().replace('jy', 'Jy')) for y in header['BUNIT'].split('.'))
                print(header['BUNIT'])
                print(data_units)
                img = (img*data_units[0]).to(u.K, equivalencies=u.brightness_temperature(restfrq, beam.sr)) * data_units[1]
                print(img.unit)
            else:
                try:
                    img = img * u.Unit(header['BUNIT'])
                    print(f"successfully obtained unit {img.unit} for data {cube_or_image_stub}")
                except Exception as e:
                    print(e)
                    img_unit = header['BUNIT']
                    print(f"Failed to obtain unit {img_unit} for data {cube_or_image_stub}")
            if img.unit == u.Jy/u.pix:
                mjysr = u.MJy/u.sr
                print(f"Converting {img.unit} to {mjysr}")
                # Note that u.pix is unfortunately locked in as a distance unit rather than area, so while this conversion looks clean, it is a little sketchy but will work
                img = (img * (u.pix / misc_utils.get_pixel_scale(ref_wcs)**2)).to(mjysr)

        else:
            cube_obj = cube_utils.CubeData(cube_or_image_stub)
            cube_obj.convert_to_K()
            mom0 = cube_obj.data.spectral_slab(*vel_lims).moment0().to(u.K*kms)
            img = mom0.quantity
            wcs_obj = mom0.wcs
            beam = cube_obj.data.beam
        print_stats(cube_or_image_stub, img)
        return img, wcs_obj, beam

    def make_regrid_stub(data_stub, order):
        order_stub = f"_{order:d}" if (order is not None) else ""
        return f"misc_regrids/{data_stub}_regrid{order_stub}.fits"

    def load_helper_memoize(cube_or_image_stub, order=None):
        """
        Check if there's a saved regridded 2D version of the cube/image and load that instead
        :param order: order of interpolation. 0 is nearest neighbor, good for showing pixellation in images
            1 is bilinear, good for contours. None is no interpolation, good for the reference image.
            order is saved to the filename so that we don't misuse different interpolations if we change the data presentation
        """
        try:
            fn = catalog.utils.search_for_file(make_regrid_stub(cube_or_image_stub, order))
            print(f"Found memoized data for {cube_or_image_stub} order={order}")
        except FileNotFoundError:
            print(f"No memoized data found for {cube_or_image_stub} order={order}")
            return False
        data, header = fits.getdata(fn, header=True)
        wcs_obj = WCS(header)
        beam = cube_utils.Beam.from_fits_header(header)
        if header['BUNIT'] != "GAIN":
            data = data*u.Unit(header['BUNIT'])
        return data, wcs_obj, beam

    def save_helper_memoize(cube_or_image_stub, img_reproj, wcs_obj, beam, unit, order=None):
        """
        Save a regridded version with the beam info in the header
        """
        fn = f"{catalog.utils.m16_data_path}{make_regrid_stub(cube_or_image_stub, order)}"
        header = wcs_obj.to_header()
        header['BUNIT'] = str(unit)
        header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        header['AUTHOR'] = "Ramsey Karim"
        header['CREATOR'] = f"rkarim, via {__file__}.multi_panel_moment_images"
        header.update(beam.to_header_keywords())
        header['COMMENT'] = f"reprojected to {moment_helper_make_nice_wcs(get='filename')} grid"
        hdu = fits.PrimaryHDU(data=img_reproj, header=header)
        print(f"Memoizing data for {cube_or_image_stub} order={order}")
        hdu.writeto(fn)

    """ Plotting config and defaults """
    fig = plt.figure(figsize=figsize)
    plot_kwargs = dict(origin='lower', cmap='plasma')
    contour_kwargs = dict(colors='cyan', linewidths=1)
    default_label_text_size = 16
    cbar_orientation = 'vertical'
    def config_cbar_labels(cbar_ax, cbar_obj, im_obj, cbar_label_unit, data_stub):
        if cbar_orientation == 'horizontal':
            cbar_ax.xaxis.set_ticks_position('top')
            labelpad = -53
        else:
            cbar_ax.yaxis.set_ticks_position('right')
            labelpad = None
        cbar_obj.set_label(cbar_label_unit.to_string('latex_inline'), labelpad=labelpad, size=default_label_text_size)
        if True and data_stub in img_stretches: # only do this if there is a nonlinear stretch
            # flip True to False if it breaks
            """ This might break for other photometric images, but it works ok now! """

            print(cbar_obj.get_ticks())
            default_tick_list = np.array([x for x in cbar_obj.get_ticks() if (im_obj.norm.vmin <= x <= im_obj.norm.vmax)])
            if np.any(default_tick_list <= 0):
                print("below zero ticks: ", default_tick_list)
            lowest_tick_power = int(np.round(np.log10(np.min(apply_stretch(default_tick_list[default_tick_list > 0], data_stub, invert=True)))))
            print("Lowest tick power ", lowest_tick_power)
            best_tick_list_unstretched = [np.round(apply_stretch(x, data_stub, invert=True), decimals=-lowest_tick_power) for x in default_tick_list]
            if lowest_tick_power > 2:
                denom = 10**lowest_tick_power
                cbar_obj.set_label(cbar_label_unit.to_string('latex_inline'))
                cbar_obj.ax.text(1, 1, f"1e{lowest_tick_power:d}", transform=ax.transAxes, va='bottom', ha='center', fontsize=default_label_text_size)
            else:
                denom = 1
            cbar_obj.set_ticks(apply_stretch(best_tick_list_unstretched, data_stub), labels=[f"{int(x/denom)}" for x in best_tick_list_unstretched])
        cbar_ax.tick_params(labelsize=default_label_text_size)
    default_text_kwargs = dict(fontsize=16, color=marcs_colors[1], ha='left', va='center', weight='bold')
    text_x = 0.02
    beam_patch_kwargs = dict(alpha=0.9, hatch='////', facecolor='white', edgecolor='grey')

    # Grab the reference HEADER text file (no longer using an image)
    # This header, catalogs/inclusive_wcs_and_footprint.txt, uses a footprint inclusive of both JWST and CO10 and the pixel size of 12CO10
    reference_header = moment_helper_make_nice_wcs()
    ref_shape = (reference_header['NAXIS2'], reference_header['NAXIS1']) # reverse XY order to ij
    ref_wcs = WCS(reference_header)

    # Memoize axes, and use the ref_wcs to make the Axes objects
    gs = fig.add_gridspec(*grid_shape, **plots_adjust_kwargs)
    axes = {}
    def get_axis(index):
        # Index is 1D index of stub in selected_data list
        if index not in axes:
            # ax = plt.subplot2grid(grid_shape, np.unravel_index(index, grid_shape), projection=ref_wcs)
            ax = fig.add_subplot(gs[np.unravel_index(index, grid_shape)], projection=ref_wcs)
            axes[index] = ax
        return axes[index]
    """ end plot config and defaults """

    """ General loop over the (non-reference) images """
    for i, data_stub in enumerate(selected_data):
        ax = get_axis(i)
        # Stuff to do to ALL images
        ax.tick_params(axis='both', direction='in')
        ax.coords.grid(color='Gainsboro', alpha=0.5, linestyle='solid', linewidth=1)
        # ax.coords[1].set_format_unit(u.deg)
        # ax.coords[0].set_format_unit(u.deg)
        # ax.coords[0].set_major_formatter('hh:mm:ss')
        # Colorbar
        if cbar_orientation == 'horizontal':
            ax_cbar = ax.inset_axes([0, 1, 1, 0.05])
        else:
            ax_cbar = ax.inset_axes([1, 0, 0.05, 1])


        ss = ax.get_subplotspec()
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        if not ss.is_last_row():
            # hide ticklabels all but bottom row
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.coords[0].set_ticklabel(rotation=25, rotation_mode='anchor', pad=17, fontsize=default_label_text_size, ha='right', va='top')
        if not ss.is_first_col():
            # hide ticklabels all but left column
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.coords[1].set_ticklabel(fontsize=default_label_text_size)

        # Stuff to do to all images (I'm using a reference header only, so nothing has been plotted before this loop)
        interp_order = 0
        memoized_data = load_helper_memoize(data_stub, order=interp_order)
        if memoized_data:
            img_reproj, _, beam = memoized_data
            del memoized_data
            unit = img_reproj.unit
            img_reproj = img_reproj.to_value()
        else:
            raw_img, raw_wcs, beam = load_helper(data_stub)
            # Regrid to reference wcs. Use nearest-neighbor (order=0) to preserve pixellation
            img_reproj = reproject_interp((raw_img.to_value(), raw_wcs), ref_wcs, shape_out=ref_shape, order=interp_order, return_footprint=False)
            unit = raw_img.unit
            save_helper_memoize(data_stub, img_reproj, ref_wcs, beam, raw_img.unit, order=interp_order)
        im = ax.imshow(apply_stretch(img_reproj, data_stub), **plot_kwargs, **get_vlims(data_stub))


        """ Gain Curve """
        # Check for memoized gain curve; let gain always have bilinear interp since it's a contour
        memoized_gain_data = load_helper_memoize(data_stub+'gain', order=1)
        if memoized_gain_data:
            # Get memoized gain map
            gain_map, _, _ = memoized_gain_data
            del memoized_gain_data
        else:
            gain_map, gain_wcs = cube_utils.get_gain_map(data_stub)
            if gain_map is not None:
                # Reproject gain map
                gain_map_reproj = reproject_interp((gain_map, gain_wcs), ref_wcs, shape_out=ref_shape, order=1, return_footprint=False)
                # Memoize gain map
                save_helper_memoize(data_stub+'gain', gain_map_reproj, ref_wcs, beam, 'GAIN', order=1)
                # Reassign gain_map to the reprojected version for simplicity
                gain_map = gain_map_reproj
            else:
                # There is no gain map, no action necessary
                pass
        if gain_map is not None:
            ax.contour(gain_map, levels=[0.5], linewidths=1, colors='SlateGray', alpha=0.8)


        cbar = fig.colorbar(im, cax=ax_cbar, orientation=cbar_orientation)
        config_cbar_labels(ax_cbar, cbar, im, unit, data_stub)
        # Beam
        patch = beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.1]), misc_utils.get_pixel_scale(ref_wcs))
        patch.set(**beam_patch_kwargs)
        ax.add_artist(patch)
        ax.text(text_x, 0.94, cube_utils.cubenames[data_stub], transform=ax.transAxes, **default_text_kwargs)

        # Plot scale bar on top-left panel
        if i == 0:
            # Scale bar 1 pc should be 237 pixels at 0.5'' pixels
            # Good location is XY: ([463, 700], 882) on the March 30, 2023 grid
            ax.plot([463-15, 700-15], [882, 882], color=default_text_kwargs['color'], linewidth=2)
            # Text location approximately XY: 661, 846 (DS9 estimate)
            text_kwargs = dict(**default_text_kwargs)
            # text_kwargs['color'] = 'LimeGreen'
            text_kwargs['ha'] = 'right'
            ax.text(661, 846, "1 pc", **text_kwargs)
            del text_kwargs


    """ Loop over contour data """
    for data_stub in overlays:
        # the key, data_stub, is the overlay data name`
        # the value is the image to be overlayed on. find its index in selected data
        i = selected_data.index(overlays[data_stub])
        ax = get_axis(i)

        # raw_img, raw_wcs, beam = load_helper(data_stub)
        # img_reproj = reproject_interp((raw_img.to_value(), raw_wcs), ref_wcs, shape_out=ref_shape, order=1, return_footprint=False)
        ##### keep this around until ur sure u did it right

        interp_order = 1
        memoized_data = load_helper_memoize(data_stub, order=interp_order)
        if memoized_data:
            img_reproj, _, beam = memoized_data
            del memoized_data
            unit = img_reproj.unit
            img_reproj = img_reproj.to_value()
        else:
            raw_img, raw_wcs, beam = load_helper(data_stub)
            # Use bilinear (order=1) for contours, they don't need to be pixellated like the images should be
            img_reproj = reproject_interp((raw_img.to_value(), raw_wcs), ref_wcs, shape_out=ref_shape, order=interp_order, return_footprint=False)
            unit = raw_img.unit
            save_helper_memoize(data_stub, img_reproj, ref_wcs, beam, raw_img.unit, order=interp_order)

        if data_stub in special_contour_levels:
            levels = special_contour_levels[data_stub]
            print(data_stub, levels)
        else:
            # Normal case
            levels = contour_levels_base*contour_levels_coeff[data_stub]
        ax.contour(img_reproj, **contour_kwargs, levels=levels)
        # Beam
        patch = beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.8, 0.1]), misc_utils.get_pixel_scale(ref_wcs))
        patch.set(**beam_patch_kwargs)
        ax.add_artist(patch)
        ax.text(text_x, 0.86, cube_utils.cubenames[data_stub] + " (c)", transform=ax.transAxes, **default_text_kwargs)

    fig.supxlabel("Right Ascension", fontsize=default_text_kwargs['fontsize']+1)
    fig.supylabel("Declination", fontsize=default_text_kwargs['fontsize']+1)

    # 2023-03-09,10,13,20,22,30,31
    plt.savefig("/home/ramsey/Pictures/2023-03-31/moment_panel.png",
        metadata=catalog.utils.create_png_metadata(title="moment panel img for paper",
            file=__file__, func='multi_panel_moment_images'))


def moment_helper_make_nice_wcs(get='header'):
    """
    March 30, 2023
    Load in 12co10, get flat wcs, copy out the header

    the box I want is:
    CENTER_VAL: 18:18:52.7904 -13:51:10.914 (but let WCS find this based on CRPIX)
    CENTER_PIX (12co10): 262.29711 (X) 170.83123 (Y) (reverse for IJ)
    reset CRPIX to be at the center of the image with corresponding CRVAL

    WIDTH (arcsec): 359.427 (X) 458.330 (Y) (taller in Dec than in RA)
    WIDTH (pix): 718.85425 (X) 916.66025 (Y)
    no rotation, let it be exactly aligned with RA-Dec
    """
    short_filepath = "catalogs/inclusive_wcs_and_footprint.txt"
    filepath = catalog.utils.m16_data_path + short_filepath
    if get == 'header':
        return fits.Header.fromtextfile(filepath)
    elif get == 'filename':
        return short_filepath
    elif get == 'rewrite':
        raise RuntimeError("Already ran this on March 30 2023!")
        cube_obj = cube_utils.CubeData('12co10')
        wcs_flat = cube_obj.wcs_flat
        new_wcs_header = wcs_flat.to_header()
        for k in ['RESTFRQ', 'SPECSYS']:
            del new_wcs_header[k]
        crpix = (718, 916)
        center_xy_in_old_pixels = ([262], [171])
        center_sky = tuple(x[0] for x in wcs_flat.pixel_to_world_values(*center_xy_in_old_pixels))
        new_keys = {
            'NAXIS': 2, 'NAXIS1': crpix[0], 'NAXIS2': crpix[1],
            'CRVAL1': center_sky[0], 'CRVAL2': center_sky[1],
            'CRPIX1': crpix[0]//2, 'CRPIX2': crpix[1]//2
        }
        new_wcs_header.update(new_keys)
        new_wcs_header.totextfile(filepath, overwrite=False) # set overwrite=True if you really mean it, just a safeguard for me


def pv_vertical_series_thru_pillars(pillar_name, line_stub):
    save_dir = "/home/ramsey/Pictures/2023-02-23/pv_anim"
    if pillar_name == 'p2':
        path_info = pvdiagrams.linear_series_from_ds9(catalog.utils.search_for_file("catalogs/pillar_series_p2.reg"), pvpath_width=None, n_steps=30)
        save_dir = os.path.join(save_dir, "p2")
        pv_hi = {'hcop': 12, 'cii': 27, '12co10': 12, 'n2hp': 4, 'cs': 6}[line_stub]
        vel_lims = (18, 26)
    else:
        print("no")
        return
    cube = cube_utils.CubeData(line_stub)
    cube.convert_to_K()
    pvdiagrams.run_plot_and_save_series(cube, vel_lims, *path_info,
        os.path.join(save_dir, f"img_{cube.filename_stub()}.png"),
        pv_lims=(0, cube_utils.onesigmas[line_stub]*pv_hi), contours=cube_utils.onesigmas[line_stub]*1)



def paper_pv_diagrams(choose_file=0, molecular_line_stub='12co10'):
    """
    March 7, 2023
    The PV diagrams that I will put into the paper
    Based off m16_pictures.single_parallel_pillar_pvs, but for a wider variety of
    pv slices
    Keep the "high res background", "CII resolution contours" scheme and colors
    But put the finder image as a 4th panel in line (or in 2x2 grid)


    molecular_line_stub = "12co10" # whatever it is must have a valid "CONV" counterpart
    """


    """ Setup paths """
    colors = marcs_colors[:2][::-1] # ['r', 'k']
    reg_filename_dict = {
        'large-along': "parallelpillars_single.reg",
        'threads': "pillar1_threads_pv_v6_withboxes.reg",
        'cap': "p1_IDgradients_thru_head.reg",
        'p2': "pillar2_across.reg",
        'misc': "misc_pillar_pv_cuts.reg",
        'paper': "paper_pv_cuts.reg",
    }
    cii_contour_levels_options = [np.arange(10, 131, 10), np.arange(8, 81, 4), np.arange(5, 101, 5)]
    if molecular_line_stub == '12co10':
        molecular_contour_levels_options = cii_contour_levels_options
    else:
        molecular_contour_levels_options = [np.arange(1, 26, 2)]*3

    # I don't think I have python 3.10 so I don't have switch or case or whatever it's called
    # So instead, I have this!
    if choose_file == 0:
        reg_key = 'large-along'
        reg_slice = slice(0, 3)
        pv_vel_lims = (18, 30)
    elif choose_file in [1, 2]:
        reg_key = 'threads'
        if choose_file == 1:
            reg_slice = slice(0, 3)
        else:
            reg_slice = slice(3, 6)
        pv_vel_lims = (20, 28)
    elif choose_file == 3:
        reg_key = 'cap'
        reg_slice = slice(0, 3)
        pv_vel_lims = (20, 28)
    elif choose_file == 4:
        reg_key = 'p2'
        reg_slice = slice(0, 3)
        pv_vel_lims = (18, 26)
    elif choose_file in [5, 6]:
        reg_key = 'misc'
        pv_vel_lims = (18, 30)
        if choose_file == 5:
            reg_slice = slice(0, 3)
        elif choose_file == 6:
            reg_slice = slice(3, 6)
    elif choose_file >= 7:
        reg_key = 'paper'
        if choose_file == 7:
            pv_vel_lims = (18, 30)
            reg_slice = slice(0, 3)
        elif choose_file == 8:
            pv_vel_lims = (18, 28)
            reg_slice = slice(3, 6)
        else:
            raise NotImplementedError("havent set this up yet")

    # path_names = ['Pillar 1', 'Pillar 2', 'Pillar 3']
    # path_names = ['1', '2', '3']
    # marker_styles = ['-', '--', ':']

    pv_vel_lims = tuple(x*kms for x in pv_vel_lims)


    reg_filename_short = reg_filename_dict[reg_key]
    reg_filename = catalog.utils.search_for_file("catalogs/"+reg_filename_short) # 3 regions in this file
    pvpath_width = pvdiagrams.m16_allpillars_series_kwargs['pvpath_width']
    pvpath_width = None
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=pvpath_width)
    if isinstance(reg_slice, slice):
        # Slice object
        path_list = path_list[reg_slice]
    else:
        # Tuple of indices (out of order in the reg file)
        path_list = [path_list[i] for i in reg_slice]


    """ Setup colors """
    chosen_cmap = 'Greys' # 'cool'
    line_color = marcs_colors[1]

    # paths = [] # delete if it doesnt crash

    """ Load cubes """
    default_region_filename = catalog.utils.search_for_file("catalogs/parallelpillars_single.reg") # good for the cutout
    subcube_kwargs = dict(reg_index=1, length_scale_mult=2, reg_filename=default_region_filename)
    # CII, native resolution
    subcube_cii = cps2.cutout_subcube(**subcube_kwargs) # TODO: move away from cutout_subcube here and just load the whole cube

    def process_mol(line_stub):
        cube_obj = cube_utils.CubeData(line_stub)
        cube_obj.convert_to_K()
        return cube_obj
    # Get both native and convolved resolution cubes
    cube_obj_mol_natres = process_mol(molecular_line_stub)
    cube_obj_mol_ciires = process_mol(molecular_line_stub+"CONV")

    def make_make_pv_slice(path):
        """
        I'm just doing this for fun, gotta use 61A for something
        """
        def make_pv_slice(cube_obj, just_cube=False):
            if just_cube:
                cube = cube_obj
            else:
                cube = cube_obj.data
            return pvextractor.extract_pv_slice(cube.with_spectral_unit(kms).spectral_slab(*pv_vel_lims), path)
        return make_pv_slice

    """ Grid stuff """
    fig = plt.figure(figsize=(21, 7))
    gs_whole = gridspec.GridSpec(1, 4, wspace=0.15, left=0.05, right=0.99, top=0.98, bottom=0.05)

    # Start reference image stuff
    # I need the WCS of the image
    reference_img_instead_of_cube = True
    if reference_img_instead_of_cube:
        img, hdr = fits.getdata(catalog.utils.search_for_file("jwst/MAST_2022-10-26T1800/JWST/jw02739-o001_t001_nircam_clear-f335m/f335m_rotated.fits"), header=True)
        ref_wcs = WCS(hdr)
        stretch = np.arcsinh
        vlims = (1, 100)
        img_ticks = [1, 2, 5, 10, 25, 50, 100]
    else:
        img_vel_lims = (20*u.km/u.s, 26*u.km/u.s)
        img_vel_str = make_vel_stub(img_vel_lims)
        img = subcube_cii.spectral_slab(*img_vel_lims).moment0().to(u.K * u.km / u.s)
        ref_wcs = img.wcs
        img = img.to_value()
        stretch = np.arcsinh
        vlims = (10, 200)
        img_ticks = [10, 20, 50, 100, 200]


    # Make image Axes
    gs_img = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_whole[0, :1], width_ratios=(3, 1), wspace=0) # leave whitespace around the plot to make room for axes labels and colorbar
    ax_img = fig.add_subplot(gs_img[0, 0], projection=ref_wcs)
    ax_cbar = ax_img.inset_axes([1, 0, 0.05, 1])
    # ax_cbar = fig.add_subplot(gs_img[0, 1])

    # Briefly finish reference image stuff because I don't want the image hanging out in memory forever
    stretch_vlims = lambda a, b: dict(vmin=stretch(a), vmax=stretch(b))
    im = ax_img.imshow(stretch(img), origin='lower', **stretch_vlims(*vlims), cmap=chosen_cmap)
    del img # save memory i guess?

    gs_sl = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_whole[0, 1:], wspace=0.01)

    def make_sl_ax(flat_index, wcs_obj):
        """
        Given flat index 0-2 of path, assign grid location within grid_shape
        """
        return fig.add_subplot(gs_sl[0, flat_index], projection=wcs_obj)

    """ end grid stuff """


    """ Loop through paths """
    for idx, path in enumerate(path_list):
        make_pv_slice = make_make_pv_slice(path)

        sl_mol_ciires = make_pv_slice(cube_obj_mol_ciires)
        sl_mol_natres = make_pv_slice(cube_obj_mol_natres)
        # WCSs are equivalent, resolution doesn't matter

        sl_mol_ciires.header['CTYPE2'] = 'VRAD'
        sl_wcs = WCS(sl_mol_ciires.header)

        ax_sl = make_sl_ax(idx, sl_wcs)
        ax_sl.imshow(sl_mol_natres.data, origin='lower', aspect=(sl_mol_natres.data.shape[1]/sl_mol_natres.data.shape[0]), cmap=chosen_cmap, vmin=0)
        ax_sl.coords[1].set_format_unit(u.km/u.s)
        ax_sl.coords[0].set_format_unit(u.arcsec)
        ax_sl.coords[0].set_major_formatter('x')
        cube_obj_mol_ciires.help_plot_pv(ax_sl)

        if idx == 0:
            ax_sl.tick_params(axis='x', direction='in')
            ax_sl.tick_params(axis='y', direction='in')
            ax_sl.set_xlabel("Offset (arcseconds)")
            ax_sl.set_ylabel("Velocity (km s$^{-1}$)")
        else:
            ax_sl.tick_params(axis='x', direction='in')
            ax_sl.tick_params(axis='y', direction='in', labelleft=False)
            ax_sl.set_xlabel(" ")
        # ax_sl.set_title(f"{path_names[idx]} PV diagram")

        """ Contours """
        contour_args = (sl_mol_ciires.data,)
        if (choose_file == 0 and idx == 2):
            contour_levels_selection = 1
        elif choose_file in [1, 4, 5]:
            contour_levels_selection = 2
        else:
            contour_levels_selection = 0
        contour_kwargs = dict(linewidths=1.2, colors=colors[1], alpha=1)
        c = ax_sl.contour(*contour_args, **contour_kwargs, zorder=10, levels=molecular_contour_levels_options[contour_levels_selection])
        try:
            ax_sl.clabel(c, molecular_contour_levels_options[contour_levels_selection], inline=True, fontsize=10, fmt='%.0f')
        except Exception as e:
            print(e)
        if idx == 0:
            handles = []
            handles.append(mpatches.Patch(color=colors[1], label=cube_utils.cubenames[molecular_line_stub]))

        sl_cii = make_pv_slice(subcube_cii, just_cube=True)
        contour_args = (reproject_interp((sl_cii.data, sl_cii.header), sl_wcs, shape_out=sl_mol_ciires.data.shape, return_footprint=False),)
        contour_kwargs['colors'] = colors[0]
        contour_kwargs['alpha'] = 1
        c = ax_sl.contour(*contour_args, **contour_kwargs, zorder=9, levels=cii_contour_levels_options[contour_levels_selection])
        try:
            ax_sl.clabel(c, cii_contour_levels_options[contour_levels_selection], inline=True, fontsize=10, fmt='%.0f')
        except Exception as e:
            print(e)
        if idx == 0:
            handles.append(mpatches.Patch(color=colors[0], label=cube_utils.cubenames['cii']))
            ax_sl.legend(handles=handles, loc='lower right')
        """ end contours """

    def plot_ellipse_patch(ax, wcs_obj, subcube):
        patch = subcube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.1]), misc_utils.get_pixel_scale(wcs_obj))
        patch.set_alpha(0.5)
        patch.set_facecolor('grey')
        patch.set_edgecolor('k')
        ax.add_artist(patch)

    """ Finish reference image stuff """
    # handles = []
    for idx, p in enumerate(path_list):
        l = ax_img.plot([c.ra.deg for c in p._coords], [c.dec.deg for c in p._coords], color=line_color, linestyle='-', lw=3, transform=ax_img.get_transform('world'))#, label=path_names[idx])
        ax_img.text(p._coords[0].ra.deg + 8*u.arcsec.to(u.deg), p._coords[0].dec.deg - 2*u.arcsec.to(u.deg), f"{idx +1}", color=line_color, fontsize=16, va='top', ha='center', transform=ax_img.get_transform('world'))

    # ax_img.set_title(f"[CII] integrated {img_vel_str} with paths overlaid")
    # ax_img.legend()
    ax_img.set_xlabel("RA")
    ax_img.set_ylabel("Dec")
    ax_img.tick_params(axis='x', direction='in')
    ax_img.tick_params(axis='y', direction='in')
    cbar = fig.colorbar(im, cax=ax_cbar, ticks=stretch(img_ticks))
    cbar.ax.set_yticklabels([(f"{x:d}" if isinstance(x, int) else f"{x:.1f}") for x in img_ticks])
    if reference_img_instead_of_cube:
        cbar.ax.set_ylabel("3.3$\mu$m flux density (MJy sr$^{-1}$)")
    else:
        cbar.ax.set_ylabel("Integrated intensity (K km s$^{-1}$)")
    if not reference_img_instead_of_cube:
        plot_ellipse_patch(ax_img, ref_wcs, subcube_cii)
    # 2023-03-08,09,20
    plt.savefig(f"/home/ramsey/Pictures/2023-03-20/pv_along_draft_cii_{molecular_line_stub}_{choose_file}.png",
        metadata=catalog.utils.create_png_metadata(title=f'pv_along {reg_filename_short}',
            file=__file__, func="paper_pv_diagrams"))


def paper_channel_maps():
    """
    March 15, 2023
    Channel maps for paper. Would like an overlay here but not sure what yet.
    Reference: the channel maps at the top of this file, m16_pictures, as well as
    m16_threads.channel_maps_again, where I do overlay channel maps.

    I will need to bin all cubes to 1km/s for this, and I have started with CII.
    I should do the same to CO or HCO+, which may be a little more complex (but I can just use a gaussian, see spectral_cube documentation for this)

    It's only fair to give the spectral_cube method plot_channel_maps a shot, but I think it falls short for several reasons.
    First, even though they try to expose some of the plot options via kwargs sent to imshow, I think you still lose a ton of control.
    Second, that simple command below has a really strange aspect ratio and spacings between plots.
    Third, it forces nx * ny to be equal to the number of channels. Fourth, you specify channels by index (I guess that's fine but I don't like it personally.)
    Fifth, it's not clear that it returns the Figure or list of Axes for further editing.
    I think it's good for quick-look type plots, but to be honest, I don't see a benefit over DS9 for quick looks.
    I think a publication quality plot must be done by hand for this.
    # cii_cube.plot_channel_maps(5, 3, list(range(3, 18)))
    # plt.show()
    """
    # Load spectrally rebinned cube
    cii_fn_stub = "sofia/M16_CII_U_1kms_jwstfootprint.fits"
    cii_cube = cube_utils.SpectralCube.read(catalog.utils.search_for_file(cii_fn_stub))
    wcs_flat = cii_cube[0,:,:].wcs
    print("First and last available channels: ", cii_cube.spectral_axis[0], cii_cube.spectral_axis[-1])
    # Get channels by velociy (can expose these as arguments later)
    first_channel, last_channel = 17*kms, 32*kms
    first_channel_idx, last_channel_idx = (cii_cube.closest_spectral_channel(x) for x in (first_channel, last_channel))
    # Make subcube (this is how spectral_slab works under the hood, but I want to be explicit about it)
    # Announce how many channels need to be plotted (then I can make the gridspec)
    print(f"Will plot {last_channel_idx+1-first_channel_idx} channels from {first_channel} to {last_channel}")
    print(f"That's indices from {first_channel_idx} to {last_channel_idx} (inclusive, so +1)")
    # Figure and Gridspec
    grid_shape = (3, 6)
    fig = plt.figure(figsize=(14, 8))
    # Messed up gridspec setup so I can get a big colorbar on the side
    mega_gridspec = fig.add_gridspec(right=0.85, left=0.1, top=0.98, bottom=0.1)
    mega_axis = mega_gridspec.subplots()
    mega_axis.set_axis_off() # Hide this axis but use it as a frame for the colorbar
    gs = mega_gridspec[0,0].subgridspec(*grid_shape, hspace=0, wspace=0)
    # Memoize axes
    axes = {}
    def get_axis(index):
        # Index is 1D index of channel, first_channel_idx -> 0
        if index not in axes:
            # I prefer this to GridSpec.subplots() because I may have blank axes so lazy creation is easier
            axes[index] = fig.add_subplot(gs[np.unravel_index(index-first_channel_idx, grid_shape)], projection=wcs_flat)
        return axes[index]

    # Text
    text_x, text_y = 0.06, 0.94
    default_text_kwargs = dict(fontsize=12, color='k', ha='left', va='center')
    tick_labelsize = 9
    tick_labelrotation = 45
    # Colors
    cmap = 'Greys'

    for channel_idx in range(first_channel_idx, last_channel_idx+1):
        print(channel_idx - first_channel_idx)
        velocity = cii_cube.spectral_axis[channel_idx]
        channel_data = cii_cube[channel_idx].to_value()
        # Setup axis
        ax = get_axis(channel_idx)
        ss = ax.get_subplotspec()
        # Formatting
        ax.tick_params(axis='both', direction='in')
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        if ss.is_last_row() and ss.is_first_col():
            ax.coords[0].set_ticklabel(rotation=30, rotation_mode='anchor', pad=13, fontsize=tick_labelsize, ha='right', va='top')
            ax.coords[1].set_ticklabel(fontsize=tick_labelsize)
        else:
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelleft=False)

        # Check data limits
        print([f(channel_data) for f in (np.min, np.mean, np.median, np.max)])
        im = ax.imshow(channel_data, origin='lower', cmap=cmap, vmin=1, vmax=50)
        ax.text(text_x, text_y, f"{velocity.to_value():.0f} {velocity.unit.to_string('latex_inline')}", transform=ax.transAxes, **default_text_kwargs)
        # Beam (on every panel)
        patch = cii_cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.1]), misc_utils.get_pixel_scale(wcs_flat))
        patch.set(alpha=0.9, facecolor='white', edgecolor='grey')
        ax.add_artist(patch)


    # Colorbar
    cbar_ax = mega_axis.inset_axes([1.03, 0, 0.03, 1])
    cbar = fig.colorbar(im, cax=cbar_ax, label='T$_{\\rm MB}$ (K)')
    cbar.set_ticks([1, 10, 20, 30, 40, 50])
    # Titles
    fig.supxlabel("Right Ascension")
    fig.supylabel("Declination")

    # 2023-03-15,20
    fig.savefig("/home/ramsey/Pictures/2023-03-20/cii_channel_maps.png",
        metadata=catalog.utils.create_png_metadata(title=f'cii {cii_fn_stub}',
        file=__file__, func="paper_channel_maps"))

def paper_spectra():
    """
    March 28, 2023
    Show spectra through selected points. Use the same points as the tables.
    Very similar code to m16_investigation.pillar_sample_spectra
    For now, use CII resolutions for all lines, but consider using native resolution
    """
    reg_filename_short = "catalogs/pillar123_pointsofinterest_v2.reg"
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    multiplier = {'cii': 1,
        '12co10CONV': 0.5, '13co10CONV': 1, 'c18o10CONV': 15,
        'co65CONV': 2, '12co32': 1, '13co32': 1,
        'hcopCONV': 2, 'hcnCONV': 1, 'csCONV': 4, 'n2hpCONV': 4,
        'oiCONV': 1, 'ciCONV': 2}

    def get_multiplier(stub):
        if stub in multiplier:
            return multiplier[stub]
        elif stub+"CONV" in multiplier:
            return multiplier[stub+"CONV"]
        else:
            # intentionally throw error
            return multiplier[stub]


    """ Make 2 sets of spectra """
    # short_names = ['cii', '12co10', '12co32', 'co65', 'hcop', 'cs']; set_stub = ""; set_number = 1
    short_names = ['oi', '13co10', '13co32', 'hcn', 'n2hp']; set_stub = "set2"; set_number = 2

    using_conv = True # set to True to use CONV versions if applicable
    if using_conv:
        short_names = [(x if x in ("cii", "12co32", "13co32") else x+"CONV") for x in short_names]

    def check_if_region_is_southern(reg_name):
        """
        Based on what I said in the table in my paper, check if I should use northern or southern background.
        'south' if southern, 'north' if northern. Also 'north' for other strings, so be careful.
        """
        # Southern regions are Horns, Shared Base, and Shelf. All others are northern
        if reg_name[-4:] == "Horn":
            return 'south'
        elif reg_name[:2] == 'Sh':
            # Shared base and Shelf
            return 'south'
        else:
            return 'north'

    # Create Figure and Axes
    fig = plt.figure(figsize=(18, 18))
    grid_shape = (4, 3)
    gs = fig.add_gridspec(*grid_shape, hspace=0.05, wspace=0.05, left=0.05, right=0.95, bottom=0.05, top=0.95)
    axes = [] # hold the axes in same order as reg_list
    # Iterate thru reg_list and make an Axes for each
    for reg_idx in range(len(reg_list)):
        ax = fig.add_subplot(gs[np.unravel_index(reg_idx, grid_shape)])
        axes.append(ax)

    # Iterate thru cubes, and then thru regs inside of those (load 1 cube at a time, regs are much quicker already loaded)
    for line_idx, line_stub in enumerate(short_names):
        cube = cube_utils.CubeData(line_stub)
        cube.convert_to_K()
        cube.data = cube.data.with_spectral_unit(kms)
        if line_stub == 'cii':
            bgs = {ns: cps2.get_cii_background(cii_cube=cube.data, select=ns) for ns in ('north', 'south')}
        for reg_idx, reg in enumerate(reg_list):
            pixreg = reg.to_pixel(cube.wcs_flat)
            j, i = [int(round(c)) for c in pixreg.center.xy]
            try:
                spectrum = cube.data[:, i, j]
            except IndexError:
                spectrum = np.full(cube.data.shape[0], np.nan) * cube.data.unit
            if line_stub == 'cii':
                # Subtract CII BG spectrum and save the unsubtracted spectrum
                unsub_spectrum = spectrum
                spectrum = spectrum - bgs[check_if_region_is_southern(reg.meta['label'])]
            multiplier_stub = '' if get_multiplier(line_stub) == 1 else f' $\\times${get_multiplier(line_stub)}'
            line_name = cube_utils.cubenames[line_stub.replace("CONV", "")]
            p = axes[reg_idx].plot(cube.data.spectral_axis.to_value(), spectrum.to_value()*get_multiplier(line_stub), label=f"{line_name}{multiplier_stub}")
            if line_stub == 'cii':
                # Plot unsubtracted CII spectrum in dotted line with same color as CII
                axes[reg_idx].plot(cube.data.spectral_axis.to_value(), unsub_spectrum.to_value()*get_multiplier(line_stub), linestyle=':', color=p[0].get_c(), alpha=0.6)
        if line_stub == 'cii':
            del bgs

    # Iterate thru regs/Axes again and dress up the plots
    for reg_idx in range(len(reg_list)):
        ax = axes[reg_idx]
        ss = ax.get_subplotspec()
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
        if not ss.is_last_row():
            # hide ticklabels except bottom row
            ax.tick_params(axis='x', labelbottom=False)
        if not ss.is_first_col():
            # hide ticklabels except left column
            ax.tick_params(axis='y', labelleft=False)
        if reg_list[reg_idx].meta['label'].lower()[:2] == "p3":
            ax.legend(loc='upper right', fontsize=13)
        ax.set_xlim((15, 35))
        if set_number == 1:
            ax.set_ylim((-5, 45))
        elif set_number == 2:
            ax.set_ylim((-3, 23))

        ax.text(0.06, 0.94, reg_list[reg_idx].meta['label'], transform=ax.transAxes, fontsize=15, color='k', ha='left', va='center')
        for v in range(20, 29):
            # Some light velocity gridlines around the important velocities
            ax.axvline(v, color='k', alpha=0.07)
        ax.axhline(0, color='k', alpha=0.1)
        # Use this line to verify background subtractions done correctly
        # print(f"{reg_list[reg_idx].meta['label']} {check_if_region_is_southern(reg_list[reg_idx].meta['label'])}")

    fig.supxlabel(f"Velocity ({kms.to_string('latex_inline')})")
    fig.supylabel(f"Line intensity ({u.K.to_string('latex_inline')})")

    conv_text = "using " + ("conv" if using_conv else "native") + " resolutions"
    conv_stub = "" if using_conv else "_unconv"
    # 2023-03-28,31
    fig.savefig(f"/home/ramsey/Pictures/2023-03-31/sample_spectra{conv_stub}{set_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=f"{conv_text}, points: {reg_filename_short}",
        file=__file__, func="paper_spectra"))


if __name__ == "__main__":
    # single_parallel_pillar_pvs() # most recently uncommented (april 21, 2022)

    # simple_mom0_carma_molecules('cii')
    # advanced_mom0_carma_molecules()

    # background_samples_figure()

    # pv_vertical_series_thru_pillars('p2', 'cs')

    # try_component_velocity_figure()
    # column_density_figure()

    # multi_panel_moment_images()
    paper_spectra()
