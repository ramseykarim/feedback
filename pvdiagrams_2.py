"""
Same spirit as crosscut_2, just too much stuff in pvdiagrams for how many times
I import that as a module for other things.
Created: November 13, 2020

I moved some functions from pvdiagrams to here, since they probably shouldn't be
there.
"""

import sys
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 12}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib import patches

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D

from . import crosscut
pvdiagrams = crosscut.pvdiagrams
misc_utils = pvdiagrams.misc_utils
catalog = pvdiagrams.catalog
cube_utils = pvdiagrams.cube_utils
reproject_interp = pvdiagrams.reproject_interp
pvextractor = pvdiagrams.pvextractor

# watch out for circular dependencies
# https://stackabuse.com/python-circular-imports/
from . import cube_pixel_spectra as cps1
from . import cube_pixel_spectra_2 as cps2

mpl_cm = pvdiagrams.mpl_cm
mpl_colors = pvdiagrams.mpl_colors
mpl_transforms = pvdiagrams.mpl_transforms
mpatches = pvdiagrams.mpatches

if __name__ != "__main__":
    # Just a reminder
    raise RuntimeError("Where the hell are you importing this to, dependencies are already on thin ice")

def main():
    # m16_pv_again2()
    try_reproject_pv()


def pv_thru_p3_shelves():
    """
    Pretty much just for Marc's suggestion of doing P3 "tail" in BIMA CO 1-0
    CII will be too low-res, even at 14x14
    """
    colors, path_list, path_name, vlims, grid_shape, region_name = crosscut.setup_paths(1)
    axes_sl = []

    subcube = cps2.cutout_subcube(reg_filename=reg_filename, reg_index=reg_index, length_scale_mult=None, data_filename="bima/M16_12CO1-0_7x4.fits",)
    # subcube = cps2.smooth(subcube)
    vel_lims = (18*u.km/u.s, 36*u.km/u.s)
    for idx in range(len(path_list)):
        path = path_list[idx]
        sl = pvextractor.extract_pv_slice(subcube.spectral_slab(*vel_lims), path)
        if idx == 0:
            sl_wcs = WCS(sl.header)
            ax_sl = plt.subplot2grid(grid_shape, (0, 1), rowspan=2, projection=sl_wcs)
            axes_sl.append(ax_sl)
            im = ax_sl.imshow(sl.data, origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap='Greys')
            ax_sl.coords[1].set_format_unit(u.km/u.s)
            ax_sl.coords[1].set_major_formatter('{x:4.2f}') # # TODO: FIGURE THIS OUT
            ax_sl.coords[0].set_format_unit(u.arcsec)
            ax_sl.coords[0].set_major_formatter('x.xx')
            ax_sl.set_title(f"{path_name[idx]} PV slice")
        contour_args = (sl.data,)
        contour_kwargs = dict(linewidths=0.7, colors=[colors[idx]], alpha=1, levels=levels)
        ax_sl.contour(*contour_args, **contour_kwargs)
        handles.append(mpatches.Patch(color=colors[idx], label=path_name[idx]))
    ax_sl.legend(handles=handles)

    img_select = 'hst'
    if img_select == 'sofia':
        img = subcube.moment0().to(u.K * u.km / u.s)
        w = img.wcs
        img = img.to_value()
        vlims = dict(vmin=45, vmax=200)
    elif img_select == 'hst':
        img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
        w = WCS(hdr)
        vlims = dict(vmin=0.1, vmax=0.7)
    else:
        raise NotImplementedError
    ax_img = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection=w)
    im = ax_img.imshow(img, origin='lower', **vlims, cmap='Greys')
    for idx, p in enumerate(path_list):
        ax_img.plot([c.ra.deg for c in p._coords], [c.dec.deg for c in p._coords], color=colors[idx], transform=ax_img.get_transform('world'), label=path_name[idx])
    ax_img.set_title(f"Paths across {pillar_names[selected_pillar]}")
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
    ax_sl.plot([x_offset, x_offset + beam_size_mean], [0.9, 0.9], transform=beamtransform, color='k', marker='|', alpha=0.5)
    # fig.savefig(f"/home/ramsey/Pictures/11-11-20-work/pillar{selected_pillar+1}_PVs_{img_select}.png")
    plt.show()



"""
From pvdiagrams.py, moved here


Adding some stuff, Oct 8, 2020, more M16 PV diagrams
This time my big idea is to emphasize use of contours, to help the eye see
the intensity of the lines, and to reproject PV diagrams onto other grids.
"""


def try_reproject_pv():
    """
    Modeling this function on the plot_pv function from compressed_CO_pv.py
    Created: Oct 8, 2020?
    Edited: Jan 25-26, 2021 for thesis proposal, since it shows the differences in emission across the filaments
        Also using APEXbeam resolution for all 4 lines
        Jan 29 update: how about that GME stock, huh? replicating this same cut/image as
        a crosscut, bc Marc thinks the point would come across more clearly that way

        July 12, 2021: prepping for the Future of Airborne astro workshop and
        thinking to use a similar image here, maybe just one pillar at a time?
        I think I would also cap resolution at SOFIA and get rid of APEX CO3-2
        I won't do this yet, I might stick to one line at a time in which case
        I'll use a different function. And I'll push this code before I mess
        with anything
    """
    colors = ['r', 'b', 'DarkOrchid', 'DarkGreen']
    linestyles = ['-', '--', ':', '--']
    line_names = ['[CII]', '$^{12}$CO(3$-$2)', '$^{13}$CO(3$-$2)', '$^{12}$CO(1$-$0)']
    reg_filename = catalog.utils.search_for_file("catalogs/across_all_pillars.reg") # 5 regions in this now
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=pvdiagrams.m16_allpillars_series_kwargs['pvpath_width'])
    # subcube = cps2.cutout_subcube(data_filename=filename, reg_filename=reg_filename)
    fn_co10 = "bima/M16_12CO1-0_APEXbeam.fits"

    reg_index = 2 # for "across_all_pillars": 0 is "longest", 2 is "thru peaks"
    cube_co10 = cube_utils.CubeData(fn_co10)
    subcubes = list(c for c in cps2.get_all_subcubes(length_scale_mult=2, data_filename="sofia/M16_CII_U_APEXbeam.fits", extra_filename=fn_co10, reg_filename=reg_filename, reg_index=reg_index))
    for i in range(len(subcubes)):
        s = subcubes[i]
        if s.header['CTYPE3'] != "VRAD":
            s.header['CTYPE3'] = "VRAD"

            cube_co10.data = s
            cube_co10.refresh_wcs()
            cube_co10.header = s.header
            cube_co10.convert_to_K()
            cube_co10.data = cube_co10.data.with_spectral_unit(u.km/u.s)
            subcubes[i] = cube_co10.data

    path = path_list[reg_index]
    sl_list = [None]*len(subcubes)
    for idx, subcube in enumerate(subcubes):
        sl = pvextractor.extract_pv_slice(subcube.spectral_slab(20*u.km/u.s, 30*u.km/u.s), path)
        sl_list[idx] = sl
        if sl.header['CTYPE2'] != 'VRAD':
            sl.header['CTYPE2'] = 'VRAD'

    selected_index = 0
    sl_cii_wcs = WCS(sl_list[selected_index].header)
    fig = plt.figure(figsize=(18, 6))
    ax_sl = plt.subplot2grid((1, 7), (0, 2), colspan=5, projection=sl_cii_wcs)
    im = ax_sl.imshow(np.zeros_like(sl_list[selected_index].data), origin='lower', aspect=(0.45*sl_list[selected_index].data.shape[1]/sl_list[selected_index].data.shape[0]), cmap='Greys', vmin=0, vmax=1)

    contour_args_list = [None]*len(subcubes)
    contour_kwargs_list = [None]*len(subcubes)
    for idx, sl in enumerate(sl_list):
        if idx == selected_index and False:
            continue
        else:
            sl_reproj = reproject_interp((sl.data, sl.header), sl_cii_wcs, shape_out=sl_list[selected_index].data.shape, return_footprint=False)
            contour_args_list[idx] = (sl_reproj,)
            ########### LEFT OFF HERE Jan 25, 2021
            levels = [10, 25]
            if idx == 3:
                levels = [l*2 for l in levels]
            elif idx == 2:
                levels = [l/2 for l in levels]
            contour_kwargs_list[idx] = dict(linewidths=1, colors=colors[idx], alpha=1, levels=levels, linestyles=linestyles[idx])

            contour_obj = ax_sl.contour(*contour_args_list[idx], **contour_kwargs_list[idx])
            # ax_sl.clabel(contour_obj, levels, inline=True, fontsize=10, fmt='%.1f')

    ax_sl.legend(handles=[mpatches.Patch(color=c, label=l) for c, l, _ in zip(colors, line_names, subcubes)])
    ax_sl.coords[1].set_format_unit(u.km/u.s)
    ax_sl.coords[1].set_major_formatter('x.xx')
    ax_sl.coords[0].set_format_unit(u.arcsec)
    ax_sl.coords[0].set_major_formatter('x.xx')
    ax_sl.set_title(f"PV diagrams")
    ax_sl.set_xlabel("Displacement (from top-left to bottom-right; arcsec)")
    ax_sl.set_ylabel("Velocity (km/s)")

    # contour label positions?
    # ax_sl.plot([6.95, 8.3, 28.5, 41], [4.2, 6.25, 1.5, 2.8], 'x', color=colors[0])
    # ax_sl.plot([7, 9.7], [12.6, 10.7], 'x', color=colors[1])


    del sl_list, sl, subcubes
    # img, hdr = fits.getdata(catalog.utils.search_for_file("spitzer/SPITZER_I4_mosaic.fits"), header=True)
    img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
    ax_img = plt.subplot2grid((1, 7), (0, 0), colspan=2, projection=WCS(hdr))
    # ax_img.imshow(img, origin='lower', vmin=100, vmax=600)
    ax_img.imshow(img, origin='lower', vmin=0.1, vmax=0.7, cmap='Greys')
    ax_img.plot([c.ra.deg for c in path._coords], [c.dec.deg for c in path._coords], color='LimeGreen', transform=ax_img.get_transform('world'))
    ax_img.set_title("HST F657N image")
    ax_img.set_xlabel(" ")
    ax_img.set_ylabel(" ")
    plt.subplots_adjust(left=0.05, right=0.99)
    # fig.savefig("/home/ramsey/Pictures/2021-01-26-imgs/across_all_2.png")
    plt.show()


"""
Figure for the RCW 49 paper (November 9, 2020 or so)
"""

def rcw49_shell_pv_figure():
    """
    Create a specific figure for the RCW 49 paper, a PV diagram (like Marc's)
    that traces around the visible shell
    Created: November 8, 2020
    Nov 12, 2020: Maitraiyee says 12CO is fine, and I'll put this figure right
    into Appendix C
    """
    reg_filename = catalog.utils.search_for_file("catalogs/ellipse_mask.reg")
    # Only one valid pvextractor region (segment)
    shell_path, = pvextractor.paths_from_regfile(reg_filename)
    shell_path.width = (1*u.pc * u.radian/(4.16*u.kpc)).to(u.arcsec)
    vel_bounds = (-30*u.km/u.s, 30*u.km/u.s)
    vel_str = f"[{vel_bounds[0]:.0f}, {vel_bounds[1]:.0f}]"
    fig = plt.figure(figsize=(15.5, 7))

    cube_filenames = {'sofia': "sofia/rcw49-cii.fits", 'apex12': "apex/apexCO/RCW49_12CO.fits", 'apex13': "apex/apexCO/RCW49_13CO.fits"}
    sl_positions = {'sofia': 1, 'apex12': 2}

    stub = 'sofia'
    # Load the cube (as my CubeData wrapper)
    cube = cube_utils.CubeData(cube_filenames[stub])
    # Get the slice
    sl = pvextractor.extract_pv_slice(cube.data.spectral_slab(*vel_bounds), shell_path)
    # Save the slice header for later, reprojecting
    cii_sl_header, cii_sl_shape = sl.header, sl.data.shape
    ax_sl = plt.subplot2grid((1, 7), (0, 3), colspan=4, projection=WCS(sl.header))
    im = ax_sl.imshow(sl.data, origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), vmin=0, vmax=30, cmap='Greys')
    cbar = fig.colorbar(im, ax=ax_sl)
    cbar.ax.set_ylabel("[CII] Intensity (K)")
    ax_sl.coords[1].set_format_unit(u.km/u.s)
    cube.help_plot_pv(ax_sl)
    ax_sl.set_ylabel("Velocity (km s$^{-1}$)")
    ax_sl.coords[0].set_format_unit(u.arcmin)
    ax_sl.coords[0].set_major_formatter('x')
    ax_sl.set_xlabel("Displacement (\')")
    ax_sl.set_title("Position-velocity diagram along RCW 49 shell")

    img_vel_bounds = (-10*u.km/u.s, 0*u.km/u.s)
    kms_str = " km s$^{-1}$"
    img_vel_str = f"[{img_vel_bounds[0].to_value():.0f} {kms_str}, {img_vel_bounds[1].to_value():.0f} {kms_str}]"
    mom0 = cube.data.spectral_slab(*img_vel_bounds).moment0().to(u.K * u.km / u.s)
    # Save the CII mom0 WCS for later
    cii_img_wcs, cii_img_shape = mom0.wcs, mom0.shape
    ax_img = plt.subplot2grid((1, 7), (0, 0), colspan=3, projection=mom0.wcs)
    img_stretch = np.log
    im = ax_img.imshow(img_stretch(mom0.to_value()), origin='lower', vmin=img_stretch(18), vmax=img_stretch(175), cmap='Greys')
    ticks = [20, 40, 80, 160]
    cbar = fig.colorbar(im, ax=ax_img, ticks=img_stretch(np.array(ticks)))
    cbar.ax.set_yticklabels(ticks)
    cbar.ax.set_ylabel("Integrated [CII] intensity (K km s$^{-1}$)")
    ax_img.set_title(f"[CII] Integrated Intensity {img_vel_str}")
    ax_img.set_xlabel("Right Ascension")
    ax_img.set_ylabel("Declination")

    # Get the 12CO cube
    cube = cube_utils.CubeData(cube_filenames['apex12'])
    # Get the slice
    sl = pvextractor.extract_pv_slice(cube.data.spectral_slab(*vel_bounds), shell_path)
    # The limits are similar; create moment image and PV slice contours
    # Slice contours (limits 0 to 30)
    # Need to reproject these things before we use them
    pv_contour_args = (np.log(reproject_interp((sl.data, sl.header), cii_sl_header, shape_out=cii_sl_shape, return_footprint=False)),)
    co_mom0 = cube.data.spectral_slab(*img_vel_bounds).moment0().to(u.K * u.km / u.s)
    img_contour_args = (np.log(reproject_interp((co_mom0.to_value(), co_mom0.wcs), cii_img_wcs, shape_out=cii_img_shape, return_footprint=False)),)
    contour_kwargs = dict(linewidths=0.8, alpha=0.7)

    sl_levels = np.array([4, 8, 16])
    # sl_levels = np.array([2, 4, 6])
    ax_sl.contour(*pv_contour_args, **contour_kwargs, levels=np.log(sl_levels), colors=['k', 'b', 'r'])
    img_levels = np.array([20, 50, 125])
    # img_levels = np.array([4, 6, 20])
    ax_img.contour(*img_contour_args, **contour_kwargs, levels=np.log(img_levels), colors=['k', 'b', 'r'])


    line_ra, line_dec = [], []
    for c in shell_path._coords:
        line_ra.append(c.ra.deg)
        line_dec.append(c.dec.deg)
    ax_img.plot(line_ra, line_dec, color='r', linewidth=0.8, transform=ax_img.get_transform('world'))
    # ax_img.plot(line_ra[0], line_dec[0], marker='x', color='b', linewidth=0.8, transform=ax_img.get_transform('world'))
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.02, right=0.99, wspace=0.2)
    fig.savefig("/home/ramsey/Pictures/11-12-20-work/shell_PV_cii_12co.png")
    # plt.show()

"""
Back to M16, primarily for the Nov 13 presentation
November 10, 2020
"""
def m16_pv_again():
    """
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
    """
    colors = ['Orchid', 'DarkGreen', 'DarkOrange']
    reg_filename = catalog.utils.search_for_file("catalogs/parallelpillars_2.reg") # 5 regions in this now
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=pvdiagrams.m16_allpillars_series_kwargs['pvpath_width'])

    fig = plt.figure(figsize=(8, 7))
    pillar_names = ['Pillar 1', 'Pillar 2', 'Pillar 3']
    path_name = ['North', 'South']
    selected_pillar = 0
    paths = []
    axes_sl = []
    handles = []


    subcube = cps2.cutout_subcube(reg_filename=reg_filename, reg_index=selected_pillar*3 + 1, length_scale_mult=2)
    vel_lims = (18*u.km/u.s, 36*u.km/u.s)
    for idx in range(2):
        path = path_list[selected_pillar*2 + idx]
        sl = pvextractor.extract_pv_slice(subcube.spectral_slab(*vel_lims), path)
        if idx == 0:
            sl_wcs = WCS(sl.header)
            ax_sl = plt.subplot2grid((2, 2), (0, 1), rowspan=2, projection=sl_wcs)
            axes_sl.append(ax_sl)
            im = ax_sl.imshow(sl.data, origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]))
            ax_sl.coords[1].set_format_unit(u.km/u.s)
            ax_sl.coords[0].set_format_unit(u.arcsec)
            ax_sl.coords[0].set_major_formatter('x')
        contour_args = (sl.data,)
        contour_kwargs = dict(linewidths=0.7, colors=colors[idx], alpha=1, levels=[10, 20])
        ax_sl.contour(*contour_args, **contour_kwargs)
        handles.append(mpatches.Patch(color=colors[idx], label=path_name[idx]))
    ax_sl.legend(handles=handles)

    # del sl, subcube

    img_select = 'sofia'
    if img_select == 'sofia':
        subcube = cps2.cutout_subcube(reg_filename=reg_filename, reg_index=selected_pillar*2 + 1, length_scale_mult=2)
        img = subcube.spectral_slab(*vel_lims).moment0().to(u.K * u.km / u.s)
        w = img.wcs
        img = img.to_value()
        vlims = dict(vmin=45, vmax=200)
    elif img_select == 'hst':
        img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
        w = WCS(hdr)
        vlims = dict(vmin=0.1, vmax=0.7)
    else:
        raise NotImplementedError
    ax_img = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection=w)
    im = ax_img.imshow(img, origin='lower', **vlims)
    for idx, p in enumerate(path_list[selected_pillar*2:(selected_pillar+1)*2]):
        ax_img.plot([c.ra.deg for c in p._coords], [c.dec.deg for c in p._coords], color=colors[idx], transform=ax_img.get_transform('world'), label=path_name[idx])
    ax_img.set_title(f"Paths along {pillar_names[selected_pillar]}")
    plt.show()


def m16_pv_again2():
    """
    did not find anything interesting in those parallel cuts. try across??

    July 12, 2021 update: I might edit this (I will push to github first)
    for use in the upcoming Future of Airborne astro conference.
    The only thing is, I don't think I need multuple paths overlaid?
    Can just reference this function and write it into m16_deepdive.easy_pv
    with each pv in a different subplot and all the paths on the HST
    """
    colors = ['MediumOrchid', 'LimeGreen', 'DarkOrange', 'MediumBlue']
    cmap = mpl_cm.get_cmap('autumn')
    colors = [mpl_colors.to_hex(cmap(x)) for x in (0, 0.33, 0.66, 0.99)]
    reg_filename = catalog.utils.search_for_file("catalogs/across_each_pillar.reg") # 5 regions in this now
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=pvdiagrams.m16_allpillars_series_kwargs['pvpath_width'])

    fig = plt.figure(figsize=(9.5, 5))
    pillar_names = ['Pillar 1', 'Pillar 2', 'Pillar 3']
    path_name = ['North', 'Mid-N', 'Mid', 'South']
    selected_pillar = 1
    path_list = path_list[selected_pillar*4:(selected_pillar+1)*4]
    if selected_pillar == 2:
        path_name.pop(1)
        levels = [7, 9]
    else:
        path_name[2] += '-S'
        # levels = [15, 20]
        levels = [20]
        if selected_pillar == 1:
            levels = [10] + levels[:1]
    reg_index = selected_pillar*4 + 1
    paths = []
    axes_sl = []
    handles = []

    # data_filename=f"apex/M16_12CO3-2_truncated.fits",
    subcube = cps2.cutout_subcube(reg_filename=reg_filename, reg_index=reg_index, length_scale_mult=2)
    subcube = cps2.smooth(subcube)
    vel_lims = (18*u.km/u.s, 36*u.km/u.s)
    for idx in range(len(path_list)):
        path = path_list[idx]
        sl = pvextractor.extract_pv_slice(subcube.spectral_slab(*vel_lims), path)
        if idx == 0:
            sl_wcs = WCS(sl.header)
            ax_sl = plt.subplot2grid((2, 2), (0, 1), rowspan=2, projection=sl_wcs)
            axes_sl.append(ax_sl)
            im = ax_sl.imshow(sl.data, origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap='Greys')
            ax_sl.coords[1].set_format_unit(u.km/u.s)
            ax_sl.coords[1].set_major_formatter('x.xx')
            ax_sl.coords[0].set_format_unit(u.arcsec)
            ax_sl.coords[0].set_major_formatter('x.xx')
            ax_sl.set_title(f"{path_name[idx]} PV slice")
        contour_args = (sl.data,)
        contour_kwargs = dict(linewidths=0.7, colors=[colors[idx]], alpha=1, levels=levels)
        ax_sl.contour(*contour_args, **contour_kwargs)
        handles.append(mpatches.Patch(color=colors[idx], label=path_name[idx]))
    ax_sl.legend(handles=handles)

    img_select = 'sofia'
    if img_select == 'sofia':
        img = subcube.moment0().to(u.K * u.km / u.s)
        w = img.wcs
        img = img.to_value()
        vlims = dict(vmin=45, vmax=200)
    elif img_select == 'hst':
        img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
        w = WCS(hdr)
        vlims = dict(vmin=0.1, vmax=0.7)
    else:
        raise NotImplementedError
    ax_img = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection=w)
    im = ax_img.imshow(img, origin='lower', **vlims, cmap='Greys')
    for idx, p in enumerate(path_list):
        ax_img.plot([c.ra.deg for c in p._coords], [c.dec.deg for c in p._coords], color=colors[idx], transform=ax_img.get_transform('world'), label=path_name[idx])
    ax_img.set_title(f"Paths across {pillar_names[selected_pillar]}")
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
    ax_sl.plot([x_offset, x_offset + beam_size_mean], [0.9, 0.9], transform=beamtransform, color='k', marker='|', alpha=0.5)
    # fig.savefig(f"/home/ramsey/Pictures/11-11-20-work/pillar{selected_pillar+1}_PVs_{img_select}.png")
    plt.show()


if __name__ == "__main__":
    main()
