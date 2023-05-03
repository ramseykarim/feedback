"""
A designated place for M16 entire region / bubble related procedures and images
Descended from m16_investigation and m16_pictures and m16_deepdive


Created: April 25, 2023
Just circulated the M16 pillars paper and jumping into the next project before
my keyboard gets cold.
Starting with some moment 0 and PV diagrams of the entire region in CII and
CO(3-2). Maybe even the CO(1-0) that I found online
"""
__author__ = "Ramsey Karim"

# All imports dumped from the last file, m16_deepdive
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 13}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import sys
import os
import glob
import datetime
import time

# from math import ceil
# from scipy import signal
# from scipy.interpolate import UnivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
# from astropy.table import Table, QTable
from astropy import constants as const

import pandas as pd
# from io import StringIO

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

from .mantipython.physics import greybody, dust, instrument


make_vel_stub = lambda x : f"[{x[0].to_value():.1f}, {x[1].to_value():.1f}] {x[0].unit}"
kms = u.km/u.s
marcs_colors = ['#377eb8', '#ff7f00','#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

large_map_filenames = {
        'cii': "sofia/m16_CII_final_15_5_0p5_clean.fits",
        '12co10': "purplemountain/G17co12.fits", '13co10': "purplemountain/G17co13.fits",
        'c18o10': "purplemountain/G17c18o.fits",
}

def get_map_filename(line_stub, beam=None):
    """
    May 2, 2023
    Shortcut function to get filenames and stuff
    """
    if line_stub in large_map_filenames:
        fn = large_map_filenames[line_stub]
        if beam == 'APEX':
            return fn.replace('.fits', '.APEXbeam.fits')
        else:
            return fn
    else:
        # Use the cube_utils default
        return line_stub

def manual_pv_slice_series():
    """
    April 25, 2023
    I wanna see if I can just plot array slices for PV diagrams since pvextractor takes a while on large/many slices
    Ok yes this works, WCS likes to throw lots of warnings about it but it is ultimately fine

    I can expand this out to PV movies along RA and Dec (non-aligned paths will need to be done with pvextractor still)
    but this will take more time than I have tonight before the meeting tomorrow

    April 26, 2023: trying to expand this into movies
    """

    """
    PV cut orientation, vertical or horizontal
    Vertical means slice at a single RA and plot velocity vs Dec
    Horizontal means slice at a single Dec and plot velocity vs RA
    """
    orientation = 'horizontal'
    start_idx, step_idx = 25, 50

    # Load cube
    line_stub = 'cii'
    if line_stub in large_map_filenames:
        # Use the custom filename rather than the default
        filename = large_map_filenames[line_stub]
    else:
        # Use default filename from cube_utils (many of these are centered around Pillars)
        filename = line_stub
    cube_obj = cube_utils.CubeData(filename).convert_to_K().convert_to_kms()
    dimension_size = (cube_obj.data.shape[2] if orientation=='vertical' else cube_obj.data.shape[1])

    # Make image
    ref_vel_lims = (10*kms, 35*kms)
    ref_mom0 = cube_obj.data.spectral_slab(*ref_vel_lims).moment0()
    ref_img = ref_mom0.to_value()

    # Set colors
    pv_cmap = 'plasma'
    img_cmap = 'Greys_r'
    line_color = marcs_colors[1]

    # Loop thru slice index
    for slice_idx in range(start_idx, dimension_size, step_idx):

        if orientation == 'vertical':
            # Cube index order is V,Y,X = Velocity,Dec,RA = V,I,J
            cube_slices = (slice(None), slice(None), slice_idx)
        else:
            cube_slices = (slice(None), slice_idx, slice(None))

        pv_slice = cube_obj.data[cube_slices]

        # First try to remake fig/axes each time. Try persistent if slow
        fig = plt.figure(figsize=(8, 10))
        gs = fig.add_gridspec(2, 1)
        ax_img = fig.add_subplot(gs[0,0], projection=cube_obj.wcs_flat)
        ax_pv = fig.add_subplot(gs[1,0], projection=pv_slice.wcs)

        im = ax_img.imshow(ref_img, origin='lower', vmin=0, cmap=img_cmap)
        fig.colorbar(im, ax=ax_img, label=ref_mom0.unit.to_string('latex_inline'))

        im = ax_pv.imshow(pv_slice.to_value(), origin='lower', vmin=0, cmap=pv_cmap)
        fig.colorbar(im, ax=ax_pv, label=pv_slice.unit.to_string('latex_inline'), orientation='horizontal')

        # Plot line
        if orientation == 'vertical':
            plot_line = ax_img.axvline
        else:
            plot_line = ax_img.axhline
        plot_line(slice_idx, color=line_color, linewidth=2)
        # Reference image velocity interval stamp
        ax_img.text(0.1, 0.9, make_vel_stub(ref_vel_lims), color=line_color, ha='left', va='bottom')

        # Clean up axes labels
        # ax_img.set_xlabel("RA")
        # ax_img.set_ylabel("Dec")
        ax_pv.coords[1].set_format_unit(kms)

        savename = f"/home/ramsey/Pictures/2023-04-26/m16_pv_{orientation}_{slice_idx:03d}.png"
        fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=f'{line_stub}, using stub/file {filename}', file=__file__, func='manual_pv_slice_series'))


def m16_large_moment0():
    """
    August 25, 2023
    Make some big moment0 maps of CII and CO3-2 to showcase the interesting
    features and places for further study
    """
    # Use the big CII map. Needs some cleaning but generally much larger than the original I was using before.

    # line_stub = 'cii'
    # line_stub = '12co32'
    line_stub = 'c18o10'

    if line_stub in large_map_filenames:
        # Use the custom filename rather than the default
        filename = large_map_filenames[line_stub]
    else:
        # Use default filename from cube_utils (many of these are centered around Pillars)
        filename = line_stub

    velocity_intervals = [ # what I have for now.. will add later
        (10, 17), (17, 21), (23.5, 27), (27, 33), (30, 35),
    ]

    cube_obj = cube_utils.CubeData(filename).convert_to_K().convert_to_kms()

    # Can always remove this loop
    for i in range(len(velocity_intervals)):

        vel_lims = tuple(x*kms for x in velocity_intervals[i])
        mom0 = cube_obj.data.spectral_slab(*vel_lims).moment0()

        fig = plt.figure()
        ax = plt.subplot(111, projection=cube_obj.wcs_flat)
        im = ax.imshow(mom0.to_value(), origin='lower', vmin=0, cmap='plasma')
        fig.colorbar(im, ax=ax, label=f"{cube_utils.cubenames[line_stub]} {mom0.unit.to_string('latex_inline')}")
        vel_stub_simple = ".".join(f"{x.to_value():.1f}" for x in vel_lims)
        savename = f"/home/ramsey/Pictures/2023-04-25/mom0_{line_stub}_{vel_stub_simple}.png"
        if not os.path.exists(os.path.dirname(savename)):
            os.makedirs(os.path.dirname(savename))
            print("Created", os.path.dirname(savename))
        fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=make_vel_stub(vel_lims),
            file=__file__, func='m16_large_moment0'))


def real_easy_pv(reg_idx=0):
    """
    May 1, 2023
    Trying to stay true to the name this time. Put in a file, and index, and
    make the PV. No funny business. I'll try to dress it up nicely.
    I think m16_deepdive.easy_pv_2 is a good model.
    """
    # Leave None to use the moment image, or set to an image filename
    reference_filename_short = None

    # line_stub = 'cii'; vmax = None # 20
    line_stub = '12co32'; vmax = None # 30
    if line_stub in large_map_filenames:
        cube_filename_short = large_map_filenames[line_stub]
    else:
        cube_filename_short = line_stub

    # reg_filename_short = "catalogs/N19_pv_paths.reg"
    reg_filename_short = "catalogs/m16_small_bubble_pvs.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    # reg_idx = 1

    vel_lims = (9*kms, 41*kms)

    cube = cube_utils.CubeData(cube_filename_short).convert_to_K().convert_to_kms().data
    pv_path = pvdiagrams.path_from_ds9(reg_filename, index=reg_idx)
    sl = pvextractor.extract_pv_slice(cube.spectral_slab(*vel_lims), pv_path)

    # Reload the region using regions to get its name
    pv_name = regions.Regions.read(reg_filename)[reg_idx].meta['text']

    if reference_filename_short is None:
        mom0 = cube.spectral_slab(*vel_lims).moment0()
        ref_img_wcs = mom0.wcs
        ref_img = mom0.to_value()
        ref_colorbar_label = mom0.unit.to_string('latex_inline')
    else:
        raise NotImplementedError

    pv_cmap = 'plasma'
    img_cmap = 'Greys_r'
    line_color = marcs_colors[1]

    fig = plt.figure(figsize=(8,12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
    ax_img = fig.add_subplot(gs[0,0], projection=ref_img_wcs)
    ax_pv = fig.add_subplot(gs[1,0], projection=WCS(sl.header))

    im = ax_img.imshow(ref_img, origin='lower', vmin=0, cmap=img_cmap)
    fig.colorbar(im, ax=ax_img, label=ref_colorbar_label)

    im = ax_pv.imshow(sl.data, origin='lower', vmin=0, vmax=vmax, cmap=pv_cmap, aspect=(sl.data.shape[1]/(1.5*sl.data.shape[0])))
    fig.colorbar(im, ax=ax_pv, label=cube.unit.to_string('latex_inline'), orientation='horizontal')

    ax_pv.coords[1].set_format_unit(kms)
    ax_img.text(0.1, 0.9, make_vel_stub(vel_lims), color=line_color, ha='left', va='bottom')

    ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color=line_color, linestyle='-', lw=3, transform=ax_img.get_transform('world'))
    ax_img.text(pv_path._coords[0].ra.deg, pv_path._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color=line_color, fontsize=10, va='bottom', ha='center', transform=ax_img.get_transform('world'))

    # 2023-05-01,02
    savename = f"/home/ramsey/Pictures/2023-05-02/m16_pv_{pv_name}_{line_stub}.png"
    fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=f'{line_stub}, using stub/file {cube_filename_short}', file=__file__, func="real_easy_pv"))

def real_easy_spectra(reg_idx=0):
    """
    May 1, 2023
    Quick look spectra for CO3-2 and CII. Keeping it real simple
    """
    # Leave None to use the moment image, or set to an image filename
    reference_filename_short = "spitzer/SPITZER_I4_mosaic_ALIGNED.fits"

    line_stub_list = ['cii', '12co32']

    reg_filename_short = "catalogs/N19_points.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    reg = regions.Regions.read(reg_filename)[reg_idx]
    reg_name = reg.meta['text']

    ref_img, ref_hdr = fits.getdata(catalog.utils.search_for_file(reference_filename_short), header=True)
    ref_wcs = WCS(ref_hdr)

    fig = plt.figure(figsize=(8,12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
    ax_img = fig.add_subplot(gs[0,0], projection=ref_wcs)
    ax_spec = fig.add_subplot(gs[1,0])

    ax_img.imshow(ref_img, origin='lower', vmin=45, vmax=290)

    for i, line_stub in enumerate(line_stub_list):
        # Load cube
        if line_stub in large_map_filenames:
            cube_filename_short = large_map_filenames[line_stub]
        else:
            cube_filename_short = line_stub
        cube = cube_utils.CubeData(cube_filename_short).convert_to_K().convert_to_kms()
        pixreg = reg.to_pixel(cube.wcs_flat)
        j, i = [int(round(c)) for c in pixreg.center.xy]
        spectrum = cube.data[:, i, j]

        ax_spec.plot(cube.data.spectral_axis.to_value(), spectrum.to_value(), label=f"{cube_utils.cubenames[line_stub]}")

        pixreg = reg.to_pixel(ref_wcs)
        x, y = pixreg.center.xy
        ax_img.plot([x], [y], 'o', mfc=marcs_colors[1], mec='k')

    ax_spec.legend()

    # 2023-05-01
    savename = f"/home/ramsey/Pictures/2023-05-01/spectrum_{reg_name}.png"
    fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=f'{line_stub}, using stub/file {cube_filename_short}', file=__file__, func="real_easy_spectra"))

def convolve_cii_to_co32():
    """
    May 2, 2023
    Quick convolve CII to CO(3-2)
    """
    reference_cube_fn = '12co32'
    reference_cube = cube_utils.CubeData(reference_cube_fn).convert_to_K().data

    target_cube_fn = large_map_filenames['cii']
    target_cube = cube_utils.CubeData(target_cube_fn).convert_to_K().data

    savename = catalog.utils.search_for_file(target_cube_fn).replace('.fits', '.APEXbeam.fits')
    print("writing to ", savename)
    print("Convolving...")

    convolved_cube = target_cube.convolve_to(reference_cube.beam)
    convolved_cube.write(savename, format='fits')
    print("done")


def real_medium_spectra(velocity_index=0):
    """
    May 2, 2023
    Follow-up to m16_bubble.real_easy_spectra, a slightly more fine-tuned figure
    with moment images and velocity intervals and stuff like that.
    Didn't want to work it into real_easy_spectra because that should be kept
    simple and reusable.

    Going to go full on with this image. I want 3 rows: moments/images up top,
    PV in the middle, and spectra on the bottom.
    Three images up top: Spitzer for reference, then CII and CO.
    Middle, just one long PV with bars showing velocity intervals.
    Bottom, two to three spectra taken along the PV diagram. Maybe mark their
    approximate locations along the PV with vertical bars (have to approx this in DS9).
    """
    # Leave None to use the moment image, or set to an image filename
    reference_filename_short = "spitzer/SPITZER_I4_mosaic_ALIGNED.fits"

    line_stub_list = ['12co32', 'cii', '13co32']

    # global vmaxes
    pv_vmaxes = {'cii': 20, '12co32': 30}
    img_vmaxes = {'cii': [10, 14, 19, 30, 30, 25, 12],
        '12co32': [6, 21, 25, 25, 25, 19, 3]}

    # each (i, i+1) is an interval
    velocity_intervals = (10, 18, 21.5, 23.5, 26, 28, 31, 34)
    # velocity_index = 0 # current interval
    velocity_interval = tuple(velocity_intervals[velocity_index+delta_i]*kms for delta_i in (0, 1))
    full_velocity_limits = (8*kms, 35*kms)

    """ Hillenbrand Stars """
    star_df = pd.read_csv(catalog.utils.search_for_file("catalogs/hillenbrand_stars_1.csv"), index_col='ID')
    star_ra = star_df['RA'].values
    star_dec = star_df['DE'].values
    del star_df

    """
    Let the region file have 1 vector and 3 points.
    Regions will not read the vector, and pvextractor will not read the points. Perfect!
    """
    # reg_filename_short = "catalogs/N19_points_along_path_1.reg"
    # reg_filename_short = "catalogs/m16_up_points_along_path.reg"
    # reg_filename_short = "catalogs/m16_across_pillars_points_along_path.reg"
    # reg_filename_short = "catalogs/spire_up_points_along_path.reg"
    reg_filename_short = "catalogs/m16_across_points_along_path.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    point_reg_list = regions.Regions.read(reg_filename)
    pv_path = pvdiagrams.path_from_ds9(reg_filename, index=0)

    ref_img, ref_hdr = fits.getdata(catalog.utils.search_for_file(reference_filename_short), header=True)
    ref_wcs = WCS(ref_hdr)

    fig = plt.figure(figsize=(20,16))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])


    """ Image row: Spitzer, (line_1), (line_2) """
    ax_ref_img = fig.add_subplot(gs[0,0], projection=ref_wcs)
    axes_moment_imgs = {line_stub: (0, i+1) for i, line_stub in enumerate(line_stub_list[:2])}

    # ax_pv will use gs[1, :] but needs its WCS first
    ax_pv = (1, slice(None))
    pv_ref_hdr = None
    pv_ref_shape = None
    pv_cbar = None

    """ Make room for 3 spectrum locations """
    axes_spec = [fig.add_subplot(gs[2,i]) for i in range(3)]

    # Reference
    ax_ref_img.imshow(ref_img, origin='lower', vmin=45, vmax=290)

    for line_idx, line_stub in enumerate(line_stub_list):
        # Load cube
        cube_filename_short = get_map_filename(line_stub)
        cube = cube_utils.CubeData(cube_filename_short).convert_to_K()

        # Spectrum for each location
        for reg_idx, reg in enumerate(point_reg_list):
            pixreg = reg.to_pixel(cube.wcs_flat)
            j, i = [int(round(c)) for c in pixreg.center.xy]
            spectrum = cube.data[:, i, j]
            # Only label first spectrum plot
            spectrum_kwargs = (dict(label=f"{cube_utils.cubenames[line_stub]}") if reg_idx == 0 else {})
            axes_spec[reg_idx].plot(cube.data.spectral_axis.to(kms).to_value(), spectrum.to_value(), **spectrum_kwargs)

            # Do a few things only once
            if line_idx == 0:
                # Plot the PV path on the Spitzer image
                ax_ref_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color=marcs_colors[1], linestyle='-', lw=1, transform=ax_ref_img.get_transform('world'))
                ax_ref_img.text(pv_path._coords[0].ra.deg, pv_path._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color=marcs_colors[1], fontsize=10, va='center', ha='right', transform=ax_ref_img.get_transform('world'))

                # Plot stars
                ref_xlim = ax_ref_img.get_xlim()
                ref_ylim = ax_ref_img.get_ylim()
                ax_ref_img.scatter(star_ra, star_dec, color=marcs_colors[3], marker='*', transform=ax_ref_img.get_transform('world'))
                ax_ref_img.set_xlim(ref_xlim)
                ax_ref_img.set_ylim(ref_ylim)

                # Plot the points on the Spitzer image
                pixreg = reg.to_pixel(ref_wcs)
                x, y = pixreg.center.xy
                ax_ref_img.plot([x], [y], 'o', mfc=marcs_colors[1], mec='k')

                # Plot velocity interval bars on the spectra
                for v in velocity_intervals:
                    axes_spec[reg_idx].axvline(v, color='grey', alpha=0.7, linestyle='--')
                # Highlight current interval
                for vkms in velocity_interval:
                    axes_spec[reg_idx].axvline(vkms.to_value(), color='k', alpha=0.9, linestyle='-', linewidth=2)
                axes_spec[reg_idx].axvspan(*(vkms.to_value() for vkms in velocity_interval), color='grey', alpha=0.25)
                # Plot 0
                axes_spec[reg_idx].axhline(0, color='grey', alpha=0.7, linestyle='--')
                # Set xlims
                axes_spec[reg_idx].set_xlim([vkms.to_value() for vkms in full_velocity_limits])



        # Stop this loop for 13co since we only want spectra
        if line_stub == '13co32':
            # No PV, no moment, all done
            continue
        # Keep going for cii and 12co

        """ Moment 0 """
        mom0 = cube.data.spectral_slab(*velocity_interval).moment0().to(u.K*kms)
        avg_intensity = (mom0 / (velocity_interval[1] - velocity_interval[0])).decompose()
        ax_i, ax_j = axes_moment_imgs[line_stub]
        ax_mom = fig.add_subplot(gs[ax_i, ax_j], projection=mom0.wcs)
        axes_moment_imgs[line_stub] = ax_mom
        im = ax_mom.imshow(avg_intensity.to_value(), origin='lower', vmin=0, vmax=img_vmaxes[line_stub][velocity_index])
        fig.colorbar(im, ax=ax_mom, label="Avg Tmb ("+avg_intensity.unit.to_string('latex_inline')+")")
        ax_mom.text(0.05, 0.9, cube_utils.cubenames[line_stub], fontsize=15, color='k', transform=ax_mom.transAxes)
        # Plot the PV path on the Spitzer image
        ax_mom.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color=marcs_colors[1], linestyle='-', lw=1, transform=ax_mom.get_transform('world'))
        ax_mom.text(pv_path._coords[0].ra.deg, pv_path._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color=marcs_colors[1], fontsize=10, va='center', ha='right', transform=ax_mom.get_transform('world'))
        for reg in point_reg_list:
            # Plot the points on the Spitzer image
            pixreg = reg.to_pixel(mom0.wcs)
            x, y = pixreg.center.xy
            ax_mom.plot([x], [y], 'o', mfc=marcs_colors[1], mec='k')

        xlim = ax_mom.get_xlim()
        ylim = ax_mom.get_ylim()
        ax_mom.scatter(star_ra, star_dec, ec=marcs_colors[3], fc='None', alpha=0.2, marker='o', transform=ax_mom.get_transform('world'))
        ax_mom.set_xlim(xlim)
        ax_mom.set_ylim(ylim)


        """ PV """
        sl_raw = pvextractor.extract_pv_slice(cube.data.spectral_slab(*full_velocity_limits), pv_path)
        if pv_ref_hdr is None:
            # Plot this PV as the image and save its header
            pv_ref_hdr = sl_raw.header
            pv_ref_shape = sl_raw.data.shape
            pv_ref_hdr['CTYPE2'] = 'VRAD'
            ax_pv = fig.add_subplot(gs[ax_pv], projection=WCS(pv_ref_hdr))
            im = ax_pv.imshow(sl_raw.data, origin='lower', cmap='plasma', vmin=0, vmax=pv_vmaxes[line_stub], aspect=(sl_raw.data.shape[1]/(3*sl_raw.data.shape[0])))
            cs = ax_pv.contour(sl_raw.data, colors='k', linewidths=1, levels=np.arange(5, 41, 5))
            pv_cbar = fig.colorbar(im, ax=ax_pv, label=cube.data.unit.to_string('latex_inline'))
            for l in cs.levels:
                pv_cbar.ax.axhline(l, color='grey')
            ax_pv.text(0.05, 0.95, cube_utils.cubenames[line_stub], fontsize=13, color=marcs_colors[1], va='top', ha='left', transform=ax_pv.transAxes)

            ax_pv.coords[1].set_format_unit(u.km/u.s)
            ax_pv.coords[1].set_major_formatter('x.xx')
            ax_pv.coords[0].set_format_unit(u.deg)
            ax_pv.coords[0].set_major_formatter('x.xx')

        else:
            """ Stupid bug fix! I don't think this step should be necessary but it appears to be necessary """
            sl_raw.header['RESTFRQ'] = pv_ref_hdr['RESTFRQ']
            sl_reproj = reproject_interp((sl_raw.data, sl_raw.header), WCS(pv_ref_hdr), shape_out=pv_ref_shape, return_footprint=False)
            cs = ax_pv.contour(sl_reproj, cmap='viridis_r', linewidths=1.5, levels=np.arange(3, 37, 4), vmax=26)
            for l in cs.levels:
                pv_cbar.ax.axhline(l, color='white', linewidth=2)
            ax_pv.text(0.05, 0.90, cube_utils.cubenames[line_stub] + ' (c)', fontsize=10, color='white', va='top', ha='left', transform=ax_pv.transAxes)

            # Plot velocity intervals
            x_length = pv_path._coords[0].separation(pv_path._coords[1]).deg
            for v in velocity_intervals:
                ax_pv.plot([0, x_length], [v*1e3]*2, color='grey', alpha=0.7, linestyle='--', transform=ax_pv.get_transform('world'))
            for vkms in velocity_interval:
                ax_pv.plot([0, x_length], [vkms.to_value()*1e3]*2, color='k', alpha=0.9, linestyle='-', linewidth=3, transform=ax_pv.get_transform('world'))

            # Plot region positions
            # Assume their position in arcseconds is in the reg.meta['text'] after '/'
            for reg in point_reg_list:
                offset = float(reg.meta['text'].split('/')[1]) * u.arcsec
                ax_pv.plot([offset.to(u.deg).to_value()]*2, [v.to_value()*1e3 for v in full_velocity_limits], color='k', alpha=1, linestyle=':', linewidth=4, transform=ax_pv.get_transform('world'))


    axes_spec[0].legend()
    plt.tight_layout()

    # 2023-05-02
    savename = f"/home/ramsey/Pictures/2023-05-02/mega_spectrum_{os.path.basename(reg_filename_short).replace('.reg', '')}_vidx{velocity_index:02d}.png"
    print(savename)
    fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=f'{line_stub}, using stub/file {cube_filename_short}', file=__file__, func="real_medium_spectra"))


def pv_regrid_debug():
    """
    May 2, 2023
    There is some issue with PV diagram reprojecting, I need to isolate it and sort it out
    """

    reg_filename_short = "catalogs/N19_pv_paths.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    reg_idx = 1

    vel_lims = (9*kms, 41*kms)

    pv_path = pvdiagrams.path_from_ds9(reg_filename, index=reg_idx)

    # Reload the region using regions to get its name
    pv_name = regions.Regions.read(reg_filename)[reg_idx].meta['text']

    fig = plt.figure(figsize=(8,12))

    line_stub = 'cii'; vmax = 20
    cube_filename_short = line_stub#get_map_filename(line_stub)
    cube = cube_utils.CubeData(cube_filename_short).convert_to_K().convert_to_kms().data
    sl = pvextractor.extract_pv_slice(cube.spectral_slab(*vel_lims), pv_path)

    ax_pv = fig.add_subplot(211, projection=WCS(sl.header))
    im = ax_pv.imshow(sl.data, origin='lower', vmin=0, vmax=vmax, cmap='plasma', aspect=(sl.data.shape[1]/(1.5*sl.data.shape[0])))
    fig.colorbar(im, ax=ax_pv, label=cube.unit.to_string('latex_inline'), orientation='horizontal')



    line_stub = '12co32'; vmax = 30
    cube_filename_short = line_stub #get_map_filename(line_stub)
    cube = cube_utils.CubeData(cube_filename_short).convert_to_K().convert_to_kms().data
    sl2 = pvextractor.extract_pv_slice(cube.spectral_slab(*vel_lims), pv_path)
    sl2_header = sl2.header
    """ THIS IS THE FIX!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! this is new... and not good. """
    sl2_header['RESTFRQ'] = sl.header['RESTFRQ']

    sl2_regrid = reproject_interp((sl2.data, sl2.header), WCS(sl.header), shape_out=sl.data.shape, return_footprint=False)


    ax_pv2 = fig.add_subplot(212, projection=WCS(sl.header))
    im = ax_pv2.imshow(sl2_regrid, origin='lower', vmin=0, vmax=vmax, cmap='plasma', aspect=(sl.data.shape[1]/(1.5*sl.data.shape[0])))
    fig.colorbar(im, ax=ax_pv2, label=cube.unit.to_string('latex_inline'), orientation='horizontal')


    # ax_pv.coords[1].set_format_unit(kms)

    savename = f"/home/ramsey/Pictures/2023-05-02/pv_debug.png"
    plt.show()
    # fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title='debug', file=__file__, func="pv_regrid_debug"))


if __name__ == "__main__":
    real_medium_spectra(6)
    # for i in range(1,7):
