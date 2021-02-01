"""
I want to recreate the pvdiagrams_2 cuts across the pillars with Marc's CARMA
maps
Created: November 16, 2020

The imports are all copied directly from pvdiagrams_2.py
"""

import sys
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 13}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib import patches
import glob
import os

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

filepaths = glob.glob(os.path.join(catalog.utils.m16_data_path, "carma/M16.ALL.*subpv.fits"))
"""
This contains:
CS, N2H+, HCO+, HCN
"""
fn = catalog.utils.search_for_file("sofia/M16_CII_U.fits")
fn = catalog.utils.search_for_file("bima/M16_12CO1-0_14x14.fits")
# fn = filepaths[2]


def main():
    across_pillars_carma()

def across_pillars_carma():
    """
    Copied from pvdiagrams_2, "m16_pv_again2"
    did not find anything interesting in those parallel cuts. try across??

    Updated Dec 30-31 2020 for AAS iPoster
    """
    colors = ['MediumOrchid', 'LimeGreen', 'DarkOrange', 'MediumBlue']
    cmap = mpl_cm.get_cmap('terrain')
    colors = [mpl_colors.to_hex(cmap(x)) for x in (0, 0.1, 0.3, 0.7)]
    reg_filename = catalog.utils.search_for_file("catalogs/across_each_pillar.reg") # 5 regions in this now
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=pvdiagrams.m16_allpillars_series_kwargs['pvpath_width'])
    linestyles = ['--', '-']
    hatchings = ['|||', None]

    cube = cube_utils.CubeData(fn)
    line_name = cube.name().split(' ')[1].replace('+', 'p').replace('12', '')

    fig = plt.figure(figsize=(19, 8))
    pillar_names = ['Pillar 1', 'Pillar 2', 'Pillar 3']
    path_name = ['North', 'Mid-N', 'Mid', 'South']
    selected_pillar = 1
    path_list = path_list[selected_pillar*4:(selected_pillar+1)*4]
    reg_index = selected_pillar*4 + 1
    lsm = 1.4
    if selected_pillar == 2:
        path_name.pop(1)
        levels = [7, 9]
    else:
        path_name[2] += '-S'
        levels = [2]
        if line_name == 'HCOp':
            levels = [2, 4, 10]
        if line_name == 'HCN':
            levels = [4]
        if selected_pillar == 0:
            reg_index += 1
            lsm = 1.8
        #     levels = [10] + levels[:1]
    paths = []
    axes_sl = []
    handles = []
    levels = []

    vel_lims = (18*u.km/u.s, 30*u.km/u.s) # used to be 36
    # data_filename=f"apex/M16_12CO3-2_truncated.fits",
    subcube = cps2.cutout_subcube(data_filename=fn, reg_filename=reg_filename, reg_index=reg_index, length_scale_mult=lsm)
    cube.data = subcube
    cube.refresh_wcs()
    cube.convert_to_K()
    cube.data = cube.data.with_spectral_unit(u.km/u.s)
    cube.data = cube.data.spectral_slab(*vel_lims)
    subcube = cube.data
    # subcube = cps2.smooth(subcube)
    for idx in range(len(path_list)):
        # ######
        # if idx > 0:
        #     continue
        # ###
        path = path_list[idx]
        sl = pvextractor.extract_pv_slice(subcube, path)
        if idx == 0:
            sl_wcs = WCS(sl.header)
            ax_sl = plt.subplot2grid((2, 2), (0, 1), rowspan=2, projection=sl_wcs)
            axes_sl.append(ax_sl)
            im = ax_sl.imshow(np.zeros_like(sl.data), origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap='Greys', vmin=0, vmax=1)
            ax_sl.coords[1].set_format_unit(u.km/u.s)
            ax_sl.coords[1].set_major_formatter('x.xx')
            ax_sl.coords[0].set_format_unit(u.arcsec)
            ax_sl.coords[0].set_major_formatter('x.xx')
            ax_sl.set_title(f"PV diagrams")
            ax_sl.set_ylabel("Velocity (km/s)")
            ax_sl.set_xlabel("Offset, from NE to SW (arcseconds)")
            cube.help_plot_pv(ax_sl)
        contour_args = (sl.data,)

        levels = [np.max(sl.data)*2/3]
        if cube.telescope.lower() == 'sofia':
            if selected_pillar == 1:
                levels = [10, 15]
            if selected_pillar == 0:
                levels = [20, 25]
        if cube.telescope.lower() == 'bima':
            if selected_pillar == 1:
                levels = [20, 35]
            if selected_pillar == 0:
                if np.max(sl.data)*2/3 > 30:
                    levels = [30, 45]
                else:
                    levels = [15, 30]
        # print("LEVEL ", levels[0])

        contour_kwargs = dict(linewidths=2, colors=[colors[idx]], alpha=1, levels=levels, linestyles=linestyles[idx%2])
        c = ax_sl.contour(*contour_args, **contour_kwargs)
        ax_sl.clabel(c, levels, inline=True, fontsize=10, fmt='%.1f')
        handles.append(mpatches.Patch(color=colors[idx], label=path_name[idx], hatch=hatchings[idx%2], fill=bool(idx%2)))
    ax_sl.legend(handles=handles)

    if selected_pillar == 1:
        vel_lims = (20*u.km/u.s, 24*u.km/u.s)
    elif selected_pillar == 0:
        # ax_sl.axhline(y=25, color='k', alpha=0.5, linestyle=':', transform=ax_sl.get_transform('world')) # need to fix transform thing
        vel_lims = (23*u.km/u.s, 27*u.km/u.s)
    else:
        vel_lims = (20*u.km/u.s, 24*u.km/u.s)
    vel_str = f"[{vel_lims[0].to_value():.0f}, {vel_lims[1].to_value():.0f}] km/s"
    img_select = cube.telescope.lower()
    if img_select != 'hst':
        img = subcube.spectral_slab(*vel_lims).moment0().to(u.K * u.km / u.s)
        w = img.wcs
        img = img.to_value()
        if img_select == 'carma':
            vlims = dict(vmin=0, vmax=20)
        elif img_select == 'sofia':
            if selected_pillar == 1:
                vlims = dict(vmin=0, vmax=75)
            elif selected_pillar == 0:
                vlims = dict(vmin=0, vmax=150)
            else:
                vlims = dict(vmin=0, vmax=75)
        elif img_select == 'bima':
            if selected_pillar == 1:
                vlims = dict(vmin=0, vmax=125)
            elif selected_pillar == 0:
                vlims = dict(vmin=0, vmax=175)
            else:
                vlims = dict(vmin=0, vmax=75)
    elif img_select == 'hst':
        img, hdr = fits.getdata(catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits"), header=True)
        w = WCS(hdr)
        vlims = dict(vmin=0.1, vmax=0.7)
    else:
        raise NotImplementedError
    ax_img = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection=w)
    im = ax_img.imshow(img, origin='lower', **vlims, cmap='Greys')
    for idx, p in enumerate(path_list):
        ax_img.plot([c.ra.deg for c in p._coords], [c.dec.deg for c in p._coords], color=colors[idx], transform=ax_img.get_transform('world'), label=path_name[idx], lw=2, linestyle=linestyles[idx%2])
    ax_img.set_title(f"{line_name} integrated {vel_str} with paths across {pillar_names[selected_pillar]} overlaid")
    ax_img.set_xlabel("RA")
    ax_img.set_ylabel("Dec")
    # Plot the beam on the image
    patch = subcube.beam.ellipse_to_plot(*(ax_img.transAxes + ax_img.transData.inverted()).transform([0.9, 0.1]), misc_utils.get_pixel_scale(w))
    patch.set_alpha(0.5)
    patch.set_facecolor('GhostWhite')
    patch.set_edgecolor('k')
    ax_img.add_artist(patch)
    cbar = fig.colorbar(im, ax=ax_img)
    cbar.ax.set_ylabel("Integrated intensity (K km/s)")
    # Plot the beam as a line in the PV slice
    beam_size_mean = np.sqrt(subcube.beam.major*subcube.beam.minor).to(u.deg).to_value()
    beamtransform = mpl_transforms.blended_transform_factory(ax_sl.get_transform('world'), ax_sl.transAxes)
    x_offset = 5*u.arcsec.to(u.deg)
    # Plot the beam in degrees in the x coord and axes in the y coord
    ax_sl.plot([x_offset, x_offset + beam_size_mean], [0.9, 0.9], transform=beamtransform, color='k', marker='|', alpha=0.5)

    plt.tight_layout(h_pad=0, w_pad=0, pad=5)
    # fig.savefig(f"/home/ramsey/Pictures/12-29-20-iposter/pillar{selected_pillar+1}_PVs_{cube.telescope.lower()}_{line_name.replace('[', '').replace(']', '')}.png")
    plt.show()


def along_pillars_carma():
    """
    Copied from m16_pv_again (the first one)
    which I just edited to be 2 parallel cuts, not 3
    """
    colors = ['MediumBlue', 'DarkOrange', 'LimeGreen',]
    # cmap = mpl_cm.get_cmap('autumn')
    # colors = [mpl_colors.to_hex(cmap(x)) for x in (0, 0.8)]
    reg_filename = catalog.utils.search_for_file("catalogs/parallelpillars_2.reg") # 5 regions in this now
    path_list = pvdiagrams.path_from_ds9(reg_filename, None, width=pvdiagrams.m16_allpillars_series_kwargs['pvpath_width'])

    cube = cube_utils.CubeData(fn)
    line_name = cube.name().split(' ')[1].replace('+', 'p').replace('[', '').replace(']', '')

    fig = plt.figure(figsize=(9.5, 5))
    pillar_names = ['Pillar 1', 'Pillar 2', 'Pillar 3']
    path_name = ['North', 'South']
    selected_pillar = 0
    path_list = path_list[selected_pillar*2:(selected_pillar+1)*2]
    reg_index = selected_pillar*2
    if selected_pillar == 2:
        levels = [7, 9]
    else:
        levels = [2]
        if line_name == 'HCOp':
            levels = [2, 4, 10]
        if line_name == 'HCN':
            levels = [4]
    paths = []
    axes_sl = []
    handles = []

    vel_lims = (18*u.km/u.s, 36*u.km/u.s)
    # data_filename=f"apex/M16_12CO3-2_truncated.fits",
    subcube = cps2.cutout_subcube(data_filename=fn, reg_filename=reg_filename, reg_index=reg_index, length_scale_mult=1.4)
    cube.data = subcube
    cube.refresh_wcs()
    cube.convert_to_K()
    cube.data = cube.data.with_spectral_unit(u.km/u.s)
    cube.data = cube.data.spectral_slab(*vel_lims)
    subcube = cube.data
    # subcube = cps2.smooth(subcube)
    for idx in range(len(path_list)):
        path = path_list[idx]
        sl = pvextractor.extract_pv_slice(subcube, path)
        if idx == 0:
            sl_wcs = WCS(sl.header)
            ax_sl = plt.subplot2grid((2, 2), (0, 1), rowspan=2, projection=sl_wcs)
            axes_sl.append(ax_sl)
            im = ax_sl.imshow(sl.data, origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), cmap='Greys')
            ax_sl.coords[1].set_format_unit(u.km/u.s)
            ax_sl.coords[1].set_major_formatter('x.xx')
            ax_sl.coords[0].set_format_unit(u.arcsec)
            ax_sl.coords[0].set_major_formatter('x.xx')
            ax_sl.set_title(f"{path_name[idx]} PV slice, {cube.telescope.upper()} {line_name}")
            cube.help_plot_pv(ax_sl)
        contour_args = (sl.data,)
        nanmax = np.nanmax(sl.data)
        levels = [nanmax/3, nanmax*2/3]
        # print("LEVEL ", levels[0])
        contour_kwargs = dict(linewidths=0.7, colors=[colors[idx]], alpha=1, levels=levels)
        c = ax_sl.contour(*contour_args, **contour_kwargs)
        ax_sl.clabel(c, levels, inline=True, fontsize=6, fmt='%.1f')
        handles.append(mpatches.Patch(color=colors[idx], label=path_name[idx]))
    ax_sl.legend(handles=handles)

    # del sl, subcube

    img_select = cube.telescope.lower()
    if img_select != 'hst':
        img = subcube.moment0().to(u.K * u.km / u.s)
        w = img.wcs
        img = img.to_value()
        if img_select == 'carma':
            vlims = dict(vmin=0, vmax=20)
        elif img_select == 'bima' or img_select == 'sofia':
            vlims = dict(vmin=0, vmax=200)
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
    ax_img.set_title(f"Paths across {pillar_names[selected_pillar]} ({img_select} image)")
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
    # fig.savefig(f"/home/ramsey/Pictures/12-29-20-work/pillar{selected_pillar+1}_PVs_{cube.telescope.lower()}_{line_name}.png")
    plt.show()



if __name__ == "__main__":
    main()
