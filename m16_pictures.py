"""
A designated place to plot nice M16 images for presentation

Created: October 29, 2020
    Preparing for UJC (and pre-presentation to committee)
First actual use: November 11, 2020
Second use: channel maps for thesis proposal, December 18, 2020
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

from math import ceil

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5

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


def marcs_rgb_in_cii():
    """
    See Marc's 1998 paper Fig 3
    Moment 0 images, RGB
    B: 18-20
    G: 20-27
    R: 27-31
    not a PV diagram, but related!!!
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
            ax.tick_params(axis=ax_name, direction='in', labelled=False, labelbottom=False)
    axes[0].set_ylabel(" ")
    axes[0].set_xlabel(" ")
    # axes[0].tick_params(axis='y', labelleft=False)
    plt.tight_layout(pad=2, rect=[0.04, 0.02, 1, 0.98], h_pad=0.5)
    plt.show()
    # fig.savefig("/home/ramsey/Pictures/12-21-20-work/cii_rgb_large_AAS1.png", facecolor='k', edgecolor=None)


def m16_channel_maps():
    fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits")
    cube = SpectralCube.read(fn)
    cube._unit = u.K
    kms = u.km/u.s
    # cube.plot_channel_maps(2, 2, [50, 60, 70, 80], cmap='jet')
    moments = make_moment_series(cube, (2.5*kms, 40*kms), 2.5*kms)
    # assert len(moments) == 20
    grid_shape = (3, 5)
    fig = plt.figure(figsize=(20, 13))
    stretch = np.arcsinh
    ax, im = None, None
    for i in range(len(moments)):
        v_left, v_right, mom0 = moments[i]
        ax = plt.subplot2grid(grid_shape, (i//grid_shape[1], i%grid_shape[1]), projection=mom0.wcs)
        im = ax.imshow(stretch(mom0.to_value()), vmin=stretch(2), vmax=stretch(150), cmap='inferno')
        for axis_name in ('x', 'y'):
            ax.tick_params(axis=axis_name, direction='in')
            ax.tick_params(axis=axis_name, labelbottom=False, labelleft=False)
        ax.text(0.1, 0.9, f"{v_left.to_value():.1f}$-${v_right.to_value():.1f} {v_left.unit}", transform=ax.transAxes, fontsize=16)
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

    ticks = [5, 25, 125]
    cbar = fig.colorbar(im, cax=insetcax, orientation='vertical',
                        ticks=stretch(ticks))
    insetcax.set_yticklabels([f"{x:d}" for x in ticks], fontsize=14)
    insetcax.yaxis.set_ticks_position('left')
    ax.text(0.62, 0.05, "T (K km/s)", transform=ax.transAxes, fontsize=14, zorder=10)

    # plt.savefig("/home/ramsey/Pictures/12-21-20-work/m16_channel_maps.png")
    plt.show()


def m16_individual_channel_maps():
    fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits")
    cube = SpectralCube.read(fn)
    cube._unit = u.K
    kms = u.km/u.s
    crop = cps2.cutout_subcube(length_scale_mult=7, return_cutout=True)
    # cube.plot_channel_maps(2, 2, [50, 60, 70, 80], cmap='jet')
    moments = make_moment_series(cube, (11*kms, 35*kms), 1*kms)
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



def make_moment_series(cube, velocity_range, velocity_spacing):
    """
    :param cube: SpectralCube with at least good velocity units
    :param velocity_range: two-element sequence of (low, high) velocity
        limits. The high limit is not inclusive, the low limit is.
        The limits should be Quantities.
        These are the first two arguments for a "range" function
    :param velocity_spacing: Spacing for the channel maps. Should be a Quantity.
        This is the third argument to a "range" function.
    """
    v_unit = velocity_spacing.unit
    # This is the "left edge" of each channel map. The right edge will be this
    # plus velocity_spacing
    v0_range = np.arange(*(v.to(v_unit).to_value() for v in velocity_range), velocity_spacing.to_value()) * v_unit
    # Gather moments into list
    moments = []
    for v0 in v0_range:
        v1 = v0 + velocity_spacing
        moments.append((v0, v1, cube.spectral_slab(v0, v1).moment0().to(u.K*v_unit)))
    return moments


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
    """
    colors = ['r', 'k']
    reg_filename = catalog.utils.search_for_file("catalogs/parallelpillars_single.reg") # 3 regions in this file
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=pvdiagrams.m16_allpillars_series_kwargs['pvpath_width'])
    chosen_cmap = 'Greys'

    fig = plt.figure(figsize=(20, 14))
    pillar_names = ['Pillar 1', 'Pillar 2', 'Pillar 3']
    marker_styles = ['-', '--', ':']
    paths = []
    # axes_sl = []

    subcube_kwargs = dict(reg_index=1, length_scale_mult=2, reg_filename=reg_filename)
    # CII, native resolution
    subcube_cii = cps2.cutout_subcube(**subcube_kwargs).with_spectral_unit(u.km/u.s)
    # CO 1-0 BIMA, 14x14
    fn_co = "bima/M16_12CO1-0_14x14.fits"
    cube_co = cube_utils.CubeData(fn_co)
    subcube_co = cps2.cutout_subcube(data_filename=fn_co, **subcube_kwargs)
    cube_co.data = subcube_co
    cube_co.refresh_wcs()
    cube_co.convert_to_K()
    cube_co.data = cube_co.data.with_spectral_unit(u.km/u.s)
    subcube_co = cube_co.data

    # return subcube_cii, subcube_co

    vel_lims = (18*u.km/u.s, 30*u.km/u.s)
    vel_str = f"[{vel_lims[0].to_value():.0f}, {vel_lims[1].to_value():.0f}] km/s"
    for idx, path in enumerate(path_list):
        sl = pvextractor.extract_pv_slice(subcube_cii.spectral_slab(*vel_lims), path)
        sl_wcs = WCS(sl.header)
        ax_sl = plt.subplot2grid((3, 6), (1, idx*2), colspan=2, rowspan=2, projection=sl_wcs)
        # axes_sl.append(ax_sl)
        im = ax_sl.imshow(np.zeros_like(sl.data), origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap=chosen_cmap, vmin=0, vmax=1)
        ax_sl.coords[1].set_format_unit(u.km/u.s)
        ax_sl.coords[0].set_format_unit(u.arcsec)
        ax_sl.coords[0].set_major_formatter('x')
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
            levels = np.arange(10, 61, 10)
        contour_kwargs = dict(linewidths=1.2, colors=colors[0], alpha=1, levels=levels)
        c = ax_sl.contour(*contour_args, **contour_kwargs, zorder=10)
        ax_sl.clabel(c, levels, inline=True, fontsize=10, fmt='%.0f')
        if idx == 0:
            handles = []
            handles.append(mpatches.Patch(color=colors[0], label="[CII]"))

        sl_co = pvextractor.extract_pv_slice(subcube_co.spectral_slab(*vel_lims), path)
        ####################################################
        sl_co.header['CTYPE2'] = 'VRAD' # this is super important and solved a lot of my problems!!!!1
        ####################################################
        contour_args = (reproject_interp((sl_co.data, sl_co.header), sl_wcs, shape_out=sl.data.shape, return_footprint=False),)
        # contour_args = (sl_co.data,)
        contour_kwargs['colors'] = colors[1]
        contour_kwargs['alpha'] = 1
        c = ax_sl.contour(*contour_args, **contour_kwargs, zorder=9)
        ax_sl.clabel(c, levels, inline=True, fontsize=10, fmt='%.0f')
        if idx == 0:
            handles.append(mpatches.Patch(color=colors[1], label="CO (1$-$0)"))
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

    img = subcube_co.spectral_slab(*vel_lims).moment0().to(u.K * u.km / u.s)
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
    plot_ellipse_patch(ax_img, w, subcube_co)




    plt.tight_layout(h_pad=0, w_pad=0, pad=5)
    # plt.savefig("/home/ramsey/Pictures/12-29-20-iposter/pv_along.png")
    plt.show()

"""
I used carma_pvdiagrams.across_pillars_carma to make the across-pillar cuts

There's also a big crosscut in crosscut_2.cut_across_m16_pillars_again
"""


if __name__ == "__main__":
    args = single_parallel_pillar_pvs()
