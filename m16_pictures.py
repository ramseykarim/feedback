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
import sys
import os
import datetime

from math import ceil

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5

from reproject import mosaicking

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
    of CII spectral resolution
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
        ### Now CO
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
    bg_reg_filename_short = "catalogs/pillar_background_sample_multiple_4.reg"
    bg_reg = regions.read_ds9(catalog.utils.search_for_file(bg_reg_filename_short))
    cii_bg_spectrum = cii_cube.subcube_from_regions(bg_reg).mean(axis=(1, 2))
    kwargs = {'fill': False}
    fig = plt.figure(figsize=(14, 6))
    ax_img = plt.subplot2grid((1, 3), (0, 0), projection=cii_cube[0, :, :].wcs)
    ax_spec = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    vel_lims = (19*kms, 27*kms)
    ax_img.imshow(cii_cube.spectral_slab(*vel_lims).moment0().to_value(), origin='lower')
    ax_spec.plot(cii_cube.spectral_axis.to_value(), cii_bg_spectrum.to_value(), color='k', lw=4, label='Average background', alpha=0.6)
    for idx, reg in enumerate(bg_reg):
        reg_pixel = reg.to_pixel(cii_cube[0, :, :].wcs)
        artist = reg_pixel.as_artist(**kwargs, color=marcs_colors[idx])
        ax_img.add_artist(artist)
        x, y = reg_pixel.center.xy
        ax_img.text(x, y, str(idx+1), color=marcs_colors[idx], ha='center', va='center', fontsize=12)
        spectrum = cii_cube.subcube_from_regions([reg]).mean(axis=(1, 2)).to_value()
        ax_spec.plot(cii_cube.spectral_axis.to_value(), spectrum, color=marcs_colors[idx], label=f'{idx+1}')
    beam_patch_coords = [0.06, 0.94] # Axes coords
    beam_patch = cii_cube.beam.ellipse_to_plot(*(ax_img.transAxes + ax_img.transData.inverted()).transform(beam_patch_coords), misc_utils.get_pixel_scale(cii_cube[0, :, :].wcs))
    beam_patch.set_alpha(0.9)
    beam_patch.set_facecolor('w')
    beam_patch.set_edgecolor('w')
    ax_img.add_artist(beam_patch)
    ax_img.text(beam_patch_coords[0]+0.06, beam_patch_coords[1], '[CII]\nbeam', fontsize=9, color='w', alpha=0.9, transform=ax_img.transAxes, va='center', ha='left')
    ax_img.set_title(f"Integrated CII intensity, {make_vel_stub(vel_lims)}", fontsize=12)
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    ax_spec.set_xlabel("Velocity (km/s)")
    ax_spec.set_ylabel("CII line intensity (K)")
    ax_spec.legend()
    plt.tight_layout()
    fig.savefig("/home/ramsey/Pictures/2022-01-14-work/cii_background_spectra.png",
        metadata=catalog.utils.create_png_metadata(title=f'bg regions from {bg_reg_filename_short}',
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


if __name__ == "__main__":
    # m16_channel_maps()
    # save_fits_thin_channel_maps()
    # single_parallel_pillar_pvs() # most recently uncommented (april 21, 2022)

    # simple_mom0_carma_molecules('cii')
    # advanced_mom0_carma_molecules()

    compare_32_65_10()
