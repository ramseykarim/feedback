"""
A designated place for even MORE nice M16 images
Descended from m16_investigation and m16_pictures

Created: July 9, 2021
Starting with investigating the "light feature" over P2 and repeating some
transverse PV diagrams
I hope the paper is within reach. Maybe august?? Rise and grind
(Feb 2023: lol, lmao)

m16_investigation got too long so I'm starting this one. I estimate that I'll
need maybe just one more to wrap up M16. After that, I should see if I can
consolidate any of this into reusable packages. Though tbh I should wait for
one more region to make sure everything truly is reusable. Tho I do have RCW49..
"""
__author__ = "Ramsey Karim"

# All imports dumped from the last file, m16_investigation
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

from math import ceil
from scipy import signal
from scipy.interpolate import UnivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.table import Table, QTable
from astropy import constants as const

import pandas as pd
from io import StringIO

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

from .mantipython.physics import greybody, dust, instrument


make_vel_stub = lambda x : f"[{x[0].to_value():.1f}, {x[1].to_value():.1f}] {x[0].unit}"
kms = u.km/u.s
marcs_colors = ['#377eb8', '#ff7f00','#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

ratio_12co_to_H2 = 8.5e-5
Cp_H_ratio = 1.6e-4 # Sofia et al 2004, what Tiwari et al 2021 used in the N(C+)->N(H) section

ratio_12co_to_13co = 44.65 # call it 45 in paper, the difference will be miniscule

los_distance_M16 = 1740 * u.pc # Kuhn et al 2019
err_los_distance_M16 = 130 * u.pc # Kuhn et al 2019; it was +130, -120, so for a symmetric error bar I'll take +/- 130

H_mass_amu = 1.00794
Hmass = const.u * H_mass_amu
# H2_mass is 1/1e4 larger than 2x Hmass. All of these ignore He
# no longer using this: # H2_mass_permole = 2.016 * u.g / u.mol
# no longer using this: H2_mass =  (H2_mass_permole / const.N_A).decompose()
# Fix all instances of H2_mass to be 2*Hmass (easier to find them thru NameErrors)

# Adopting Y = 0.25, Z=0
mean_molecular_weight_neutral = 1.33


# Let us begin with a rewrite of m16_investigation.compare_carma_to_sofia_pv
def easy_pv():
    """
    July 9, 2021
    Despite the name of this function, this may not be so easy
    I just want a somewhat general function to make PVs from lines or vector
    regions and from any cube. Let's see how I do!

    The first use case will be on Pillar 2 to examine the transverse gradient
    right where the light feature is
    But I suppose I can test it on whatever if it's supposed to be general
    And remember the VOPT VRAD thing..

    August 2, 2021: tried this with filament_p23south.reg and it doesn't look
    that great. So OK to overwrite all this

    November 10, 2021: trying this with p1_IDgradients_thru_head.reg to
    estimate the velocity of different components as they pass through the
    middle of the head
    """
    # This PV will be in color and perhaps also in contour
    main_pv_fn = "carma/M16.ALL.hcop.sdi.cm.subpv.SMOOTH.fits"
    cube = cube_utils.CubeData(main_pv_fn).data
    # Is that necessary? Can it just wait for the loop?
    reg_filename = catalog.utils.search_for_file("catalogs/p1_IDgradients_thru_head.reg")

    selected_path = 2

    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_path, width=5*u.arcsec)
    sl = pvextractor.extract_pv_slice(cube.spectral_slab(20*kms, 28*kms), pv_path)
    sl_wcs = WCS(sl.header)
    ax_sl = plt.subplot(111, projection=sl_wcs)
    ax_sl.imshow(sl.data, origin='lower', aspect=1.4)
    ax_sl.coords[1].set_format_unit(u.km/u.s)
    ax_sl.coords[1].set_major_formatter('x.xx')
    ax_sl.coords[0].set_format_unit(u.arcsec)
    ax_sl.coords[0].set_major_formatter('x.xx')
    ax_sl.contour(sl.data, colors='k')

    arcsec = u.arcsec.to(u.deg)
    if 'sofia' in main_pv_fn:
        if selected_path == 0:
            x_lo, x_hi = 23, 61
            y_lo, y_hi = 25.45, 24.62
            x0, x1 = 0, 74
        elif selected_path == 1:
            x_lo, x_hi = 23, 78
            y_lo, y_hi = 25.35, 24.95
            x0, x1 = 0, 98
        elif selected_path == 2:
            x_lo, x_hi = 8, 59
            y_lo, y_hi = 23, 26
            x0, x1 = 0, 63


    elif 'hcop' in main_pv_fn:
        if selected_path == 0:
            x_lo, x_hi = 26, 46
            y_lo, y_hi = 25.73, 25.4
            x0, x1 = 0, 74
        elif selected_path == 1:
            x_lo, x_hi = 38, 51
            y_lo, y_hi = 25.08, 24.89
            x0, x1 = 0, 98
        elif selected_path == 2:
            x_lo, x_hi = 20, 56
            y_lo, y_hi = 23.47, 24.93
            x0, x1 = 0, 63


    m = (y_hi - y_lo)/(x_hi - x_lo)
    b = y_lo - x_lo*m
    ymxb = lambda x: x*m + b
    y0, y1 = ymxb(x0), ymxb(x1)
    print(x0, y0)
    print(x1, y1)
    ax_sl.plot([x_lo*arcsec, x_hi*arcsec], [y_lo*1e3, y_hi*1e3], '--', color='red', transform=ax_sl.get_transform('world'))
    ax_sl.plot([x0*arcsec, x1*arcsec], [y0*1e3, y1*1e3], '-.', color='orange', transform=ax_sl.get_transform('world'))
    ax_sl.set_title(f"slope = {m*u.arcmin.to(u.arcsec):.2f} km/s / arcmin")
    # plt.savefig(f"/home/rkarim/Pictures/2021-11-10-work/hcop_pillar1_gradient_{selected_path}.png")
    plt.savefig(f"/home/ramsey/Pictures/2022-01-24-work/hcop_pillar1_gradient_{selected_path}.SMOOTH.png")
    # plt.show()

def easy_pv_2():
    """
    January 24, 2022
    I want almost the same thing as easy_pv but with a couple tweaks, but I wanna
    leave the original function intact
    Created to make a nice PV across the blue cap
    """
    main_pv_fn = "carma/M16.ALL.hcop.sdi.cm.subpv.fits"
    cube = cps2.cutout_subcube(data_filename=main_pv_fn, length_scale_mult=5)
    reg_filename_short = "catalogs/p1_IDgradients_thru_head.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    selected_path = 2 # 0,1 are threads, 2 is blue cap
    vel_lims = (22.5*kms, 24.0*kms) # to highlight the cap
    mom0 = cube.spectral_slab(*vel_lims).moment0()
    pv_path = pvdiagrams.path_from_ds9(reg_filename, selected_path, width=None)
    pv_path_length = pv_path._coords[0].separation(pv_path._coords[1]).to(u.arcsec)
    sl = pvextractor.extract_pv_slice(cube.spectral_slab(20*kms, 28*kms), pv_path)

    fig = plt.figure(figsize=(15, 6))
    ax_img = plt.subplot2grid((1, 2), (0, 1), projection=mom0.wcs)
    im = ax_img.imshow(mom0.to_value(), origin='lower', cmap='Greys_r')
    fig.colorbar(im, ax=ax_img, label='K km/s')
    ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color='red', linestyle='-', lw=3, transform=ax_img.get_transform('world'))
    ax_img.text(pv_path._coords[0].ra.deg, pv_path._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color='red', fontsize=10, va='bottom', ha='center', transform=ax_img.get_transform('world'))
    ax_img.text(pv_path._coords[1].ra.deg, pv_path._coords[1].dec.deg - 4*u.arcsec.to(u.deg), f'Offset = {pv_path_length.to_value():.1f}\"', color='red', fontsize=12, va='top', ha='center', transform=ax_img.get_transform('world'))
    ax_sl = plt.subplot2grid((1, 2), (0, 0), projection=WCS(sl.header))
    im = ax_sl.imshow(sl.data, origin='lower', aspect=1.4)
    fig.colorbar(im, ax=ax_sl, label='K')
    ax_sl.coords[1].set_format_unit(u.km/u.s)
    ax_sl.coords[1].set_major_formatter('x.xx')
    ax_sl.coords[0].set_format_unit(u.arcsec)
    ax_sl.coords[0].set_major_formatter('x.xx')
    ax_sl.set_xlabel("Offset (arcseconds)")
    ax_sl.set_ylabel("Velocity (km/s)")
    ax_sl.set_title("HCO+ PV diagram")
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    ax_img.set_title(f"Integrated HCO+ line intensity {make_vel_stub(vel_lims)}")
    hcop_noise = 0.3 # I did this one by hand in DS9 on 2022-01-24, I think the sample that yielded 0.546 was in the corner where noise gets worse
    ax_sl.contour(sl.data, colors='k', levels=np.arange(hcop_noise*5, np.max(sl.data), hcop_noise*5), lw=2, alpha=0.7)
    plt.tight_layout()
    plt.subplots_adjust(left=0.07, bottom=0.1, wspace=0.05)
    fig.savefig("/home/ramsey/Pictures/2022-01-24-work/blue-cap_hcop_pv.png",
        metadata=catalog.utils.create_png_metadata(title=f"reg i={selected_path} from {reg_filename_short}",
        file=__file__, func="easy_pv_2"))


def oi_image():
    """
    July 20, 2021
    Make OI integrated intensity image from Nicola's cube
    """
    vel_lims = (20*kms, 27*kms)
    cii_levels = [60, 90, 120, 150, 180, 210]
    cii_cube = cps2.cutout_subcube(length_scale_mult=6)
    cii_mom0 = cii_cube.spectral_slab(*vel_lims).moment0()
    # plt.imshow(cii_mom0.to_value(), origin='lower')
    # plt.contour(cii_mom0.to_value(), levels=cii_levels, colors='k')
    # plt.show()
    # return

    fn = catalog.utils.search_for_file("sofia/m16_OI_63.fits")
    cube = cube_utils.CubeData(fn)
    cube = cube.data.with_spectral_unit(kms)
    mom0 = cube.spectral_slab(*vel_lims).moment0()

    cii_mom0_reproj = reproject_interp((cii_mom0.to_value(), cii_mom0.wcs), mom0.wcs, shape_out=mom0.shape, return_footprint=False)
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, projection=mom0.wcs)
    im = ax.imshow(mom0.to_value(), origin='lower', vmin=0, vmax=50)
    fig.colorbar(im, ax=ax, label='[OI] integrated intensity (K km/s)')
    ax.contour(cii_mom0_reproj, levels=cii_levels, colors='k')
    for coord in ax.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    ax.set_title("[OI] 63 $\mu$m with [CII] 158 $\mu$m in contours")
    ax.text(0.03, 0.9, "Integrated intensity between\n20-27 km/s", transform=ax.transAxes, fontsize=13)
    ax.text(0.03, 0.7, "[CII] integrated intensity\ncontours at 60, 90, 120,\n150, 180, 210 K km/s", transform=ax.transAxes, fontsize=13)
    # OI beam
    patch = cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.06]), misc_utils.get_pixel_scale(mom0.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('grey')
    ax.add_artist(patch)
    ax.text(0.9, 0.06, "OI", ha='center', va='center', transform=ax.transAxes, fontsize=12)
    # CII beam
    patch = cii_cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.8, 0.06]), misc_utils.get_pixel_scale(mom0.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('grey')
    ax.add_artist(patch)
    ax.text(0.8, 0.06, "CII", ha='center', va='center', transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig("/home/ramsey/Pictures/2021-07-20-work/oi.png")
    # plt.show()


def simple_mom0(selected_region=0, selected_line=0):
    """
    July 20, 2021
    Moment 0 images for the presentation at the Stuttgart conference
    HCO+, CII, CO1-0

    Edited and reused: August 2, 2021
    Edited again: August 16, 2021
        catalogs/shelf.reg , catalogs/filament_p23south.reg
        set up to highlight the four "features" that I'm discussing around the
        pillars
    """
    # blue streamer, P4, shared base, shelf
    # selected_region = 0
    region_name = ['bluestreamer', 'p4', 'sharedbase', 'shelf'][selected_region]

    # select region file
    if selected_region < 2:
        reg_filename = "catalogs/filament_p23south.reg"
        length_scale_mult = 2
    else:
        reg_filename = "catalogs/shelf.reg"
        length_scale_mult = 2.5

    # select line; cii, hcop, 12co10
    # selected_line = 1

    if selected_line == 0:
        # set up velocity / color limits for CII
        if selected_region == 0:
            # blue streamer
            vel_lims = (19, 22)
            vlims = dict(vmin=0, vmax=70)
            levels = np.sinh(np.linspace(np.arcsinh(20), np.arcsinh(80), 8)) # np.linspace(20, 80, 8)
            levels_stub = "arcsinh"
        elif selected_region == 1:
            # P4
            vel_lims = (20, 24)
            vlims = dict(vmin=0, vmax=140)
            levels = np.linspace(30, 150, 8)
            levels_stub = "linear"
        elif selected_region == 2:
            # shared base
            vel_lims = (20, 24)
            vlims = dict(vmin=0, vmax=140)
            levels = np.linspace(30, 150, 8)
            levels_stub = "linear"
        elif selected_region == 3:
            # shelf
            vel_lims = (24, 27)
            vlims = dict(vmin=0, vmax=140)
            levels = np.linspace(30, 150, 8)
            levels_stub = "linear"
        else:
            raise NotImplementedError("I only have 4 regions in m16_deepdive.simple_mom0 right now")
        cube = cps2.cutout_subcube(length_scale_mult=length_scale_mult, reg_index=0, reg_filename=reg_filename)
        line_title = "upGREAT [CII] 158$\mu$m"
        line_stub = 'cii'

    elif selected_line == 1:
        vlims = dict(vmin=0, vmax=40)
        levels = list(np.linspace(4, 42, 5))
        # set up velocity / color limits for CII
        if selected_region == 0:
            # blue streamer
            vel_lims = (19, 22)
            vlims = dict(vmin=0, vmax=10)
            levels = np.linspace(1.2, 10, 6)
            levels_stub = "linear"
        elif selected_region == 1:
            # P4
            vel_lims = (20, 24)
            vlims = dict(vmin=0, vmax=20)
            levels = (np.linspace(np.sqrt(1.5), np.sqrt(20), 6))**2 #np.linspace(1.5, 20, 5)
            levels_stub = "sqrt"
        elif selected_region == 2:
            # shared base
            vel_lims = (20, 24)
            vlims = dict(vmin=0, vmax=20)
            levels = (np.linspace(np.sqrt(1.5), np.sqrt(20), 6))**2 #np.linspace(1.5, 20, 5)
            levels_stub = "sqrt"
        elif selected_region == 3:
            # shelf
            vel_lims = (24, 27)
            vlims = dict(vmin=0, vmax=40)
            levels = (np.linspace(np.sqrt(1.2), np.sqrt(37), 8))**2 #np.linspace(1.2, 38, 8)
            levels_stub = "sqrt"
        else:
            raise NotImplementedError("I only have 4 regions in m16_deepdive.simple_mom0 right now")

        cube = cube_utils.CubeData("carma/M16.ALL.hcop.sdi.cm.subpv.fits")
        cube.convert_to_K()
        cube = cube.data
        line_title = "HCO+"
        line_stub = 'hcop'

    elif selected_line == 2:
        vlims = dict(vmin=0, vmax=300)
        levels = list(np.linspace(80, 300, 5))

        cube = cube_utils.CubeData("bima/M16_12CO1-0_7x4.fits")
        cube.convert_to_K()
        cube = cube.data
        line_title = "$^{12}$CO (J=1$-$0)"
        line_stub = "12co10"

    cube = cube.with_spectral_unit(kms).spectral_slab(*(v*kms for v in vel_lims))
    mom0 = cube.moment0()
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, projection=mom0.wcs)
    im = ax.imshow(mom0.data, origin='lower', cmap='cividis', **vlims)
    fig.colorbar(im, ax=ax, label='Integrated intensity (K km/s)')
    ax.contour(mom0.data, levels=levels, colors='k')
    for coord in ax.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    ax.set_title(f"{line_title}, integrated between {vel_lims[0]}$-${vel_lims[1]} km/s")
    # ax.set_title(", integrated between 20$-$27 km/s")
    patch = cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.06]), misc_utils.get_pixel_scale(mom0.wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('grey')
    ax.add_artist(patch)
    # plt.show(); return
    plt.tight_layout()
    plt.savefig(f"/home/ramsey/Pictures/2021-08-16-work/{line_stub}_{region_name}.png", metadata={"Comment": f"{levels_stub} spaced contours; m16_deepdive.simple_mom0"})


def prepare_pdrt_tables(line1, line2=None, reg_filename=None, regions_to_do=None, convert_units=False):
    """
    Created: Aug 16, 2021
    Creating the ASCII tables for PDRT
    going to start with just CII and CII/CO10
    Then can branch out to CO 32 and 65 (pending alignment correction)
    And can search for FIR (Nicola should know where it is)


    Todo:
    get cii, integrate between 20-24
    remap co10 (14x14) moment 0 to the same grid
    use catalogs/pdrt_test_pillar2.reg to grab the pixels at that location
    save cii and cii/co10 to tables (and figure out how tables work)
    tables: https://docs.astropy.org/en/stable/io/unified.html#ascii-formats and https://docs.astropy.org/en/stable/table/io.html#getting-started

    2022-01-17 comment: I'm not sure if the moment integration limits are the
    important spectral quantity here. I think K km/s is kinda just saying
    K * Hz but with km/s instead, so the relevant quantity would be the channel
    width integrated over in the moment calculation... I should revisit this!
    TODO: revisit K km/s -> CGI unit conversion, remake pdrt tables
    (done as of Sept 2022)

    Written under an unfinished (unstarted) prepare_pdrt_tables_3 (and there's a 2!)
        Created August 15, 2022
        This is going to be the best version yet
        I want to be able to make a quick table (or Measurement, idk) at a given
        location for all lines (in a specified list) at a fixed resolution
        (largest of the samples).
        These will be used in spaghetti plots.
        I want to include a diagnostic image for each sample, which includes the
        location on a moment 0 image within the velocity limits I am integrating
        over, and the full spectrum at that location with lines showing the velocity
        limits. I want to write this diagnostic image out for each sample and each
        location so that I can show that each location makes sense.

    Updated September 14, 2022
    Going to standardize this for any region, any line.
    Part of this is detecting where the line is and only integrating over it.
    I should save a diagnostic figure for that, for each line/region
    (I hadn't seen prepare_pdrt_tables_3 yet when I wrote this, but sounds like
    I have a consistent plan!)

    Updated September 30, 2022
    I will remove my own conversion from K km/s to cgs (but I will keep those
    old diagnostic images) so I can let pdrtpy do the conversion (it knows how!)
    I will have to store the line frequency in the table in this case.
    """
    region_filenames = ["catalogs/pillar1_pointsofinterest_v3.reg", # The 8 classic P1a regions
        "catalogs/pillar123_pointsofinterest_v1.reg", # Several more around P1b, 2, 3, and Shelf
        ]
    if reg_filename is None:
        reg_filename = 0
    # reg_filename can either be a filename or an index into the list above
    if isinstance(reg_filename, int):
        reg_filename = region_filenames[reg_filename]
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename))

    if regions_to_do is not None:
        if isinstance(regions_to_do, str):
            regions_to_do = [regions_to_do]
        else:
            assert isinstance(regions_to_do, list)
    #### testing
    # reg_list = [reg_list[0]]
    ####

    # utility functions for later
    slice_from = lambda s: slice(s[0], s[1]+1)
    thin_slice_from = lambda s: slice(s, s+1)

    # reg = reg_list[0]
    # x = reg.meta['text']
    # print(x)
    # return

    line_stub_list = [line1]
    if line2:
        line_stub_list.append(line2)
    line_name_list = [cube_utils.cubenames[l] for l in line_stub_list]

    # Describe the arguments to this call in one short string
    # Used for saving the table at the end
    version_stub = '' if convert_units else '_v2'
    unique_run_identifier = "__".join(line_stub_list) + version_stub + "__" + os.path.basename(reg_filename).replace('.reg', '')
    print(unique_run_identifier)

    reference_filename = catalog.utils.search_for_file("hst/hlsp_heritage_hst_wfc3-uvis_m16_f657n_v1_drz.fits")
    reference_name = "HST F657N"
    ref_img, ref_hdr = fits.getdata(reference_filename, header=True)
    ref_wcs = WCS(ref_hdr)
    plot_ref_img = lambda ax : ax.imshow(ref_img, origin='lower', cmap='Greys_r', vmin=0.1, vmax=0.6)

    preset_search_bounds = {
        # More broad bounds just for a spectral_slab within which a search for the line is made, not  meant to be the final line integration bounds (like preset_bounds is)
        'oiCONV': (15, 35), 'ciCONV': (15, 35),
    }

    preset_bounds = {
        # All co10 entries are in reverse order because the spectral axis is reversed (high-to-low)
        '12co10CONV-broad-line': (None, 21), '12co10CONV-E-peak': (27.5, None), '12co10CONV-NE-thread': (None, 21), '12co10CONV-NW-thread': (27.5, None),
        '12co10CONV-SE-thread': (27.5, 23), '12co10CONV-S-peak': (27.5, None), '12co10CONV-SW-thread': (27, None), '12co10CONV-W-peak': (27.5, 21),
        '13co10CONV-SW-thread': (24.5, 27), '12co32-E-peak': (None, 27.5), '12co32-W-peak': (None, 27.5), '12co32-SW-thread': (None, 27.5), '12co32-S-peak': (None, 27.5), '12co32-E-peak': (None, 27.5), '12co32-SE-thread': (None, 27.5), '12co32-NW-thread': (None, 27.5), '12co32-E-peak': (None, 27.5),
        '12co10CONV-Western-Horn': (27, None), '12co10CONV-Eastern-Horn': (27, None), '12co10CONV-Inter-horn': (27, None), '12co10CONV-Shared-Base-Mid': (None, 20), '12co10CONV-Shared-Base-W': (26, 18),
        '12co32-Eastern-Horn': (23, 27),

        # positions for paper
        # Horns
        '12co10CONV-E-Horn': (27, None), '12co10CONV-W-Horn': (27, None),
        '12co32-E-Horn': (23, 27), # 12co32 W-Horn is ok on its own
        # P1a head
        '12co10CONV-P1a-center': (27, None), '12co10CONV-P1a-edge': (27.5, 20),
        '12co32-P1a-center': (None, 27.5), '12co32-P1a-edge': (None, 27.5),
        # Threads
        '12co32-P1a-E-thread': (None, 27.5), '12co32-P1a-W-thread': (None, 27.5),
        '12co10CONV-P1a-E-thread': (27.5, 22.5), '12co10CONV-P1a-W-thread': (27, None)

    }

    def check_if_region_is_southern(reg_name):
        """
        (coped from m16_pictures.paper_spectra, 2023-04-07)
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

    result_list = []
    for line_stub in line_stub_list:
        cube_obj = cube_utils.CubeData(line_stub).convert_to_K()
        cube = cube_obj.data
        if line_stub in preset_search_bounds:
            cube = cube.spectral_slab(preset_search_bounds[line_stub][0]*kms, preset_search_bounds[line_stub][1]*kms)
        # # TODO: subtract CII background
        if 'cii' in line_stub:
            print(f"subtracting CII background ({line_stub})")
            bgs = {ns: cps2.get_cii_background(cii_cube=cube, select=ns) for ns in ('north', 'south')}
        channel_noise = cube_utils.onesigmas[line_stub]
        reg_coord_list = [tuple(round(x) for x in reg.to_pixel(cube[0, :, :].wcs).center.xy[::-1]) for reg in reg_list]
        for pix_coords, reg in zip(reg_coord_list, reg_list):
            # Get some basic information
            reg_name = reg.meta['text'].replace(" ", "-")
            if (regions_to_do is not None) and (reg_name not in regions_to_do):
                # If we are specifying only SOME regions, and this is NOT one of them, skip!
                continue
            try:
                spectrum = cube[(slice(None), *pix_coords)]
            except IndexError:
                print("Shape", cube.shape[1:], "Pixel coord", pix_coords, "; pixel out of bounds")
                spectrum = np.full(cube.shape[0], np.nan) * u.K
            if line_stub == 'cii':
                spectrum = spectrum - bgs[check_if_region_is_southern(reg_name)]
            rest_freq = cube.header['RESTFRQ'] * u.Hz

            # Check if pixel is valid (whether it's in the observed area or not)
            if not np.all(np.isnan(spectrum)): # if spectrum has SOME valid values
                # pixel is valid, not all NaNs
                # Make the 1sigma mask for line finding
                mask_1sigma = spectrum.to_value() > channel_noise
                line_bounding_indices = list(misc_utils.identify_longest_run(mask_1sigma))
                preset_key = f"{line_stub}-{reg_name}"
                if preset_key in preset_bounds:
                    for idx, vel in enumerate(preset_bounds[preset_key]):
                        if vel:
                            line_bounding_indices[idx] = cube.closest_spectral_channel(vel*kms)
                line_bounding_indices = tuple(line_bounding_indices)
                # This indexing trick cube[:, x:x+1, y:y+1].moment0()[0,0] lets us use the moment function on a single 1D spectrum
                integrated_intensity = cube[slice_from(line_bounding_indices), thin_slice_from(pix_coords[0]), thin_slice_from(pix_coords[1])].moment0()[0, 0]

                # Uncertainty calculation
                cube_dv = np.abs(np.diff(cube.spectral_axis[:2]))[0].to(kms)
                n_channels = line_bounding_indices[1]+1 - line_bounding_indices[0]
                err_integrated_intensity = channel_noise*u.K * cube_dv * np.sqrt(n_channels)
            else:
                # pixel is invalid, spectrum all NaNs. because regions falls outside of observed area in this line
                integrated_intensity = np.nan * u.K * kms
                err_integrated_intensity = np.nan * u.K * kms

            # Add in the PDRT version of the line name
            line_ID_pdrt = cube_utils.cubeIDs_pdrt[line_stub.replace('CONV', '')]

            if convert_units:
                # Unit conversion
                # Get the conversion between velocity and frequency
                vel_to_freq_f = lambda v: v.to(u.Hz, equivalencies=u.doppler_optical(rest_freq))
                # Frequency interval of 1 km/s (center it around 25 km/s just cause)
                freq_interval = vel_to_freq_f(24*kms) - vel_to_freq_f(25*kms)
                # Divide the integrated intensity by 1 km/s and convert to cgs and multiply by the frequency interval
                def convert_Kkms_to_cgs(int_intens):
                    """
                    int(S)df = S{int(T)dV / Delta_V} * Delta_f
                    and then make sure final units are good
                    """
                    return (freq_interval * (int_intens / (1*kms)).to(u.Jy/u.sr, equivalencies=u.brightness_temperature(rest_freq))).to(u.erg / (u.s * u.cm**2 * u.sr))

                integrated_intensity_cgs = convert_Kkms_to_cgs(integrated_intensity)
                err_integrated_intensity_cgs = convert_Kkms_to_cgs(err_integrated_intensity)
                # Append everything to the results list
                result_list.append((integrated_intensity_cgs, err_integrated_intensity_cgs, line_ID_pdrt, reg_name))
            else:
                # Append everything to the results list, leave it in K km/s
                result_list.append((integrated_intensity, err_integrated_intensity, line_ID_pdrt, reg_name, rest_freq.to(u.Hz)))

            # Now do all the plotting
            fig = plt.figure(num=1, figsize=(15,6), clear=True)
            ax_spec = plt.subplot2grid((1, 4), (0, 0), colspan=3)
            ax_img = plt.subplot2grid((1, 4), (0, 3), projection=ref_wcs)
            # Plot HST image
            plot_ref_img(ax_img)
            # Pixel coordinate of the region in the reference HST image
            ref_pix_coords = reg.to_pixel(ref_wcs).center.xy
            ax_img.plot(*ref_pix_coords, color='r', marker='+')

            ax_spec.plot(cube.spectral_axis.to_value(), spectrum, color=marcs_colors[0])
            for sign in (-1, 1):
                ax_spec.axhline(channel_noise*sign, color='grey', linestyle='--', alpha=0.4)

            text_kwargs = dict(transform=ax_spec.transAxes, fontsize=10)

            # Some things to only plot if it's not a NaN spectrum
            if np.isfinite(integrated_intensity_cgs.to_value()):
                # Mark integration range
                for bounding_index in line_bounding_indices:
                    ax_spec.axvline(cube.spectral_axis.to_value()[bounding_index], color='k', alpha=0.6)
                # Mark where data exceeds noise level
                ax_spec.fill_between(cube.spectral_axis.to_value(), mask_1sigma.astype(float)*channel_noise*2, color='pink', alpha=0.3)

                ax_spec.set_title(f"{line_stub} {rest_freq:.2E} ({line_bounding_indices[0]}, {line_bounding_indices[1]})")
                ax_spec.text(0.05, 0.95, f"({cube.spectral_axis[line_bounding_indices[0]]:.2f}, {cube.spectral_axis[line_bounding_indices[1]]:.2f})", **text_kwargs)

            ax_img.set_title(f"{reg_name}")
            ax_spec.text(0.05, 0.85, f"{integrated_intensity:.4f}", **text_kwargs)
            if convert_units:
                ax_spec.text(0.05, 0.75, f"{integrated_intensity_cgs:.4E}", **text_kwargs)
            ax_spec.set_xlabel("Velocity (km/s)")
            ax_spec.set_ylabel("T (K)")

            plt.tight_layout()
            print(f"saving <{line_stub}_{reg_name}>")
            fig.savefig(f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt/diagnostic_figs/diagnostic{version_stub}_{line_stub}_{reg_name}.png",
                metadata=catalog.utils.create_png_metadata(title="make pdrt tables diagnostic image",
                    file=__file__, func="prepare_pdrt_tables"))
            fig.clear()

    # Save data into a QTable (so don't need to specify my own units)
    names = ['data', 'uncertainty', 'identifier', 'region']
    if not convert_units:
        names.append('rest_freq')
    tab = QTable(list(zip(*result_list)), names=names)
    tab.write(f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt/{unique_run_identifier}.txt", format='ipac',
        overwrite=True)


def prepare_pdrt_tables_fir(reg_filename=None):
    """
    Created: September 21, 2022
    Same as above but for FIR, which is just an image
    """

    region_filenames = ["catalogs/pillar1_pointsofinterest_v3.reg", # The 8 classic P1a regions
        "catalogs/pillar123_pointsofinterest_v1.reg", # Several more around P1b, 2, 3, and Shelf
        ]
    if reg_filename is None:
        reg_filename = 0
    # reg_filename can either be a filename or an index into the list above
    if isinstance(reg_filename, int):
        reg_filename = region_filenames[reg_filename]
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename))

    # FIR filename, old one is "herschel/m16-I_FIR.fits" from 2021, new one was made Oct 2022
    # fir_fn_short = "herschel/M16_I_FIR_from70-160.fits" # better resolution, like 13'' instead of 18''
    fir_fn_short = "herschel/M16_I_FIR_from70-160_fluxbgsub.fits" # same as above but background already removed
    fir_img, fir_hdr = fits.getdata(catalog.utils.search_for_file(fir_fn_short), header=True)
    fir_wcs = WCS(fir_hdr)
    reg_coord_list = [tuple(round(x) for x in reg.to_pixel(fir_wcs).center.xy[::-1]) for reg in reg_list]

    value_list = [fir_img[i, j] for (i, j) in reg_coord_list]
    uncertainty_list = [10]*len(value_list)
    id_list = ['FIR']*len(value_list)
    reg_name_list = [reg.meta['text'].replace(' ', '-') for reg in reg_list]


    fig = plt.figure(figsize=(15, 15))
    ax = plt.subplot(111, projection=fir_wcs)
    im = ax.imshow(fir_img, origin='lower', cmap='Greys_r')
    ax.set_title(f"FIR intensity, herschel/m16-I_FIR.fits")
    fig.colorbar(im, ax=ax, label=fir_hdr['BUNIT'])
    for reg_coord in reg_coord_list:
        ax.plot([reg_coord[1]], [reg_coord[0]], '+', color='r')

    version_stub = ""
    fig.savefig(f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt/diagnostic_figs/diagnostic_FIR{version_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=f"from {fir_fn_short}",
            file=__file__, func="prepare_pdrt_tables_fir"))

    unique_run_identifier = "FIR__" + os.path.basename(reg_filename).replace('.reg', '')

    # Save data into a Table (put units in as dictionary)
    tab = Table([value_list, uncertainty_list, id_list, reg_name_list], names=('data', 'uncertainty', 'identifier', 'region'), units={'data': fir_hdr['BUNIT'], 'uncertainty': '%'})
    tab.write(f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt/{unique_run_identifier}{version_stub}.txt", format='ipac')


def prepare_pdrt_tables_g0(reg_filename=None):
    """
    Created: September 21, 2022
    Make a table of G0 values at given locations
    Get G0 from both the Herschel UV (Nicola made this) and the stars (I made that)
    """
    region_filenames = ["catalogs/pillar1_pointsofinterest_v3.reg", # The 8 classic P1a regions
        "catalogs/pillar123_pointsofinterest_v1.reg", # Several more around P1b, 2, 3, and Shelf
        ]
    if reg_filename is None:
        reg_filename = 0
    # reg_filename can either be a filename or an index into the list above
    if isinstance(reg_filename, int):
        reg_filename = region_filenames[reg_filename]
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename))
    reg_name_list = [reg.meta['text'].replace(' ', '-') for reg in reg_list]

    data_dir = "/home/ramsey/Documents/Research/Feedback/m16_data"
    herschel_uv_fn = "herschel/uv_m16_repro_CII.fits"
    stars_g0_fn = "catalogs/g0_hillenbrand_stars_fuvgt4.5_ltxarcmin.fits"

    results = [[], []]
    names = []
    for idx, fn in enumerate([herschel_uv_fn, stars_g0_fn]):
        data, hdr = fits.getdata(os.path.join(data_dir, fn), header=True)
        data = np.squeeze(data)
        wcs_obj = WCS(hdr, naxis=2)

        fig = plt.figure(figsize=(15, 15))
        ax = plt.subplot(111, projection=wcs_obj)
        if idx == 0:
            im = ax.imshow(data, origin='lower', cmap='Greys_r')
            ax.set_title(f"Herschel uv, {os.path.basename(fn)}")
            fig.colorbar(im, ax=ax, label='G0 (Habing)')
        else:
            im = ax.imshow(np.log10(data), origin='lower', cmap='Greys_r')
            ax.set_title(f"Stars g0, {os.path.basename(fn)}")
            fig.colorbar(im, ax=ax, label='G0 (Habing)')

        for reg in reg_list:
            pixel_coords_ij = tuple(round(x) for x in reg.to_pixel(wcs_obj).center.xy[::-1])
            value = data[pixel_coords_ij]
            results[idx].append(value)
            ax.plot([pixel_coords_ij[1]], [pixel_coords_ij[0]], '+', color='r')

        unique_run_identifier = "-".join(os.path.basename(fn).split('.')[:-1]) + "__" + os.path.basename(reg_filename).replace('.reg', '')
        names.append(unique_run_identifier)
        fig.savefig(f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt/diagnostic_figs/diagnostic_g0_{unique_run_identifier}.png",
            metadata=catalog.utils.create_png_metadata(title="make pdrt tables diagnostic image",
                file=__file__, func="prepare_pdrt_tables_g0"))


    uncertainty_list = [10]*len(reg_list)
    ids = ['Herschel_G0', 'Stars_G0']
    for i, name in enumerate(names):
        id_list = [ids[i]]*len(reg_list)
        tab = Table([results[i], uncertainty_list, id_list, reg_name_list], names=('data', 'uncertainty', 'identifier', 'region'), units={'data': 'Habing unit', 'uncertainty': '%'})
        tab.write(f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt/{name}.txt", format='ipac')




def prepare_pdrt_tables_2():
    """
    Created: September 8, 2021
    sequel to the first one, but this time using Measurement properly
    """
    pillar = 1 # 1 or 2
    select = "fir"

    if pillar == 1:
        vel_lims = (20*kms, 28*kms) # P1 regions
    elif pillar == 2:
        vel_lims = (20*kms, 24*kms) # P2 regions

    cii_cube = cps2.cutout_subcube(length_scale_mult=4, reg_index=2)
    cii_restfreq = cii_cube.header['RESTFRQ'] * u.Hz
    cii_mom0 = cii_cube.spectral_slab(*vel_lims).moment0()
    del cii_cube
    reproj_wcs = cii_mom0.wcs
    reproj_shape = cii_mom0.shape
    reproj_hdr = cii_mom0.wcs.to_header()
    del reproj_hdr['RESTFRQ'], reproj_hdr['SPECSYS']

    if select == "ciico":

        cii_vel_to_freq_equiv = u.doppler_optical(cii_restfreq)
        cii_vel_to_freq_f = lambda v: v.to(u.Hz, equivalencies=cii_vel_to_freq_equiv)
        cii_dv = (cii_vel_to_freq_f(vel_lims[0]) - cii_vel_to_freq_f(vel_lims[1]))
        ### that's a conversion factor from km/s (over the moment0 integration limits) to Hz

        co10_cube = cube_utils.CubeData("bima/M16_12CO1-0_14x14.fits").convert_to_K().data
        co10_restfreq = co10_cube.header['RESTFRQ'] * u.Hz
        co10_mom0 = co10_cube.spectral_slab(*vel_lims).moment0().to(u.K*kms)
        del co10_cube
        co10_reproj = reproject_interp((co10_mom0.to_value(), co10_mom0.wcs), reproj_wcs, shape_out=reproj_shape, return_footprint=False)
        co10_hdr = co10_mom0.header
        co10_hdr.update(reproj_hdr)
    elif select == "fir":
        fir_img, fir_hdr = fits.getdata(catalog.utils.search_for_file("herschel/results/m16-I_FIR.fits"), header=True)
        fir_img = fir_img[cps2.cube_info['cutout'].slices_original]

    reg_list = regions.read_ds9(catalog.utils.search_for_file(f"catalogs/pdrt/pdrt_test_pillar{pillar}.reg"))
    tmp_path = '/home/rkarim/Downloads/tmp'
    if not os.path.exists(tmp_path):
        print("creating directory ", tmp_path)
        os.mkdir(tmp_path)
    else:
        print("tmp directory already exists")
    for i, reg in enumerate(reg_list):
        pixreg = reg.to_pixel(reproj_wcs)
        reg_mask = pixreg.to_mask().to_image(reproj_shape)
        tmp_filename = os.path.join(tmp_path, f'reg_p{pillar}_cii_{i}.fits')
        # cii_img_copy = cii_mom0.copy()
        # cii_img_copy[~(reg_mask==1)] = np.nan
        # cii_img_copy.write(tmp_filename)

        # tmp_filename = os.path.join(tmp_path, f'reg_p{pillar}_co10_{i}.fits')
        # co10_img_copy = co10_reproj.copy()
        # co10_img_copy[~(reg_mask==1)] = np.nan
        # fits.PrimaryHDU(data=co10_img_copy, header=co10_hdr).writeto(tmp_filename)

        tmp_filename = os.path.join(tmp_path, f'reg_p{pillar}_fir_{i}.fits')
        fir_img_copy = fir_img.copy()
        fir_img_copy[~(reg_mask==1)] = np.nan
        fits.PrimaryHDU(data=fir_img_copy, header=fir_hdr).writeto(tmp_filename)

    print("done")


def fit_molecular_components_with_gaussians(region_name, cii=False, regrid=False):
    """
    Created October 22, 2021
    Try my hand at fitting with Gaussians again
    This time it's the CO (1-0) data (maybe....13...?)
    Try to find distinct components and see if they can be responsible for the CII profile without
    major velocity shifts

    Dec 7, 2021: now using this for the regridded HCO+ to check if the single-
    component linewidth changed (it should've by a little bit due to rebin)
    """
    # cube_co = cube_utils.CubeData("bima/M16_12CO1-0_7x4.fits").convert_to_K().data.with_spectral_unit(kms)
    fn = f"carma/M16.ALL.hcop.sdi.cm.subpv{'.SOFIAbeam.regrid' if regrid else ''}.fits"
    if not cii:
        cube = cube_utils.CubeData(fn).convert_to_K().data.with_spectral_unit(kms)
    else:
        cube = cube_utils.CubeData("sofia/M16_CII_pillar1_BGsubtracted.fits").data
    # try with HCO+ smoothed first, then see if it holds up for unsmoothed
    # good_pixel = (446, 304)
    # di, dj = 20, 30

    img_idx = 4

    hcop_regions = ('western horn', 'bluest component', 'bluest component 2', # 0, 1, 2
        'western thread N', 'just off peak', # 3, 4
        'eastern thread N', 'eastern thread S', 'main red', # 5, 6, 7
        'east of peak', # 8
        )
    # region_name = 'peak W'
    print(region_name)
    good_pixel, (di, dj), g = cps2.select_pixels_and_models(('hcop' + ('-cii' if regrid else '')) if not cii else 'cii', region_name, var_mean=1, var_std=1)
    # # mean = 24.91
    # ## this is the western thread, it seems to always be around 25 km/s
    # good_pixel = (540, 430) # mean = 24.87
    # good_pixel = (527, 410) # 24.96
    # good_pixel = (512, 401) # 25.07
    # good_pixel = (487, 373) # 25.21
    # ## good example of probable multiple components



    # Identify noise level
    if regrid:
        ## for the CII-grid HCO+
        noise_pixel = (cube.shape[1]-6, 5) # top left area
    else:
        ## for the original HCO+
        noise_pixel = (644, 318) # noise for 2x2 pixels is ~0.5 K

    vel_lims = (23, 26)

    i_lims = i_lo, i_hi = tuple(good_pixel[0] + sign*di + offset for offset, sign in enumerate((-1, 1)))
    j_lims = j_lo, j_hi = tuple(good_pixel[1] + sign*dj + offset for offset, sign in enumerate((-1, 1)))

    noise_i = slice(*(noise_pixel[0] + sign*di + offset for offset, sign in enumerate((-1, 1))))
    noise_j = slice(*(noise_pixel[1] + sign*dj + offset for offset, sign in enumerate((-1, 1))))
    print(noise_i, noise_j)

    # vel_lims = (23, 24)
    mom0 = cube.spectral_slab(*(v*kms for v in vel_lims)).moment0()
    fig = plt.figure(figsize=(15, 7))


    ax_img = plt.subplot(121)
    if regrid:
        ## for the CII-grid HCO+ cube
        show_box_i_lims = show_box_i_lo, show_box_i_hi = 0, mom0.shape[0]
        show_box_j_lims = show_box_j_lo, show_box_j_hi = 0, mom0.shape[1]
    else:
        ## for the original HCO+ cube
        show_box_i_lims = show_box_i_lo, show_box_i_hi = 370, 665 # 0, mom0.shape[0]
        show_box_j_lims = show_box_j_lo, show_box_j_hi = 277, 592 # 0, mom0.shape[1]
    show_box_lo_lims = (show_box_i_lo, show_box_j_lo)
    show_box = cps2.make_show_box(show_box_i_lims, show_box_j_lims)
    ax_img.imshow(mom0.to_value()[show_box], origin='lower')

    noise = np.std(cube[:, noise_i, noise_j].mean(axis=(1, 2)).to_value())
    print("NOISE", noise)
    # if regrid:
    #     cube = cube.with_mask(cube > -1*noise*u.K)

    x_axis = cube.spectral_axis.to_value()
    spectrum = cube[:, i_lo:i_hi, j_lo:j_hi].mean(axis=(1, 2))

    ax_spec = plt.subplot(122)
    ax_spec.set_ylabel(f"Noise: {noise:.3f}")

    # mark some things on each plot
    cps2.plot_noise_and_vlims(ax_spec, noise, vel_lims)
    cps2.plot_box(ax_img, i_lims, j_lims, show_box_lo_lims)
    cps2.plot_noise_img(ax_img, noise_pixel, show_box_lo_lims)

    mask = spectrum.to_value() > -100

    fitter = cps2.fitting.LevMarLSQFitter(calc_uncertainties=True) # SLSQPLSQFitter() is old method, LevMarLSQFitter lets us do uncertainties
    g_fit = fitter(g, x_axis[mask], spectrum.to_value()[mask])
    print(g_fit)
    cps2.plot_everything_about_models(ax_spec, x_axis, spectrum.to_value(), g_fit)

    ax_spec.set_title(f"{'HCO+' if not cii else 'CII'} spectrum from within box (see left), with fit")
    ax_img.set_title(f"{'HCO+' if not cii else 'CII'} moment 0 between {vel_lims[0]}, {vel_lims[1]} km/s")
    ax_spec.set_xlim(17, 30)
    reg_stub = region_name.replace(' ', '-')
    # plt.savefig(f'/home/ramsey/Pictures/2021-11-11-work/fit_molecular_components_3G_fixedstd_{reg_stub}.png')
    plt.show()
    return


def test_fitting_2_gaussians_with_1(j, x, label):
    """
    Created October 26, 2021
    I want to see the pattern in the residuals made by fitting 2 gaussians with 1
    """
    # Set up the model
    A, mean, stddev = 10, 25, 1
    dv = 1
    c_A = 1
    c_std = x
    g1 = cps2.models.Gaussian1D(amplitude=A*c_A, mean=mean - (dv/2), stddev=stddev)
    g2 = cps2.models.Gaussian1D(amplitude=A, mean=mean + (dv/2), stddev=stddev*c_std)
    g_all = g1 + g2
    # Set up "observations"
    x_axis = np.arange(18, 33, 0.25)
    y_obs = g_all(x_axis)
    high_res_x = np.linspace(18, 33, 100) # for plotting the model
    # Set up a guess
    g_guess = cps2.models.Gaussian1D(amplitude=A, mean=mean, stddev=stddev,
        bounds={"amplitude": (0, 200), "mean": (20, 30), "stddev": (0.2, 5)})
    fitter = cps2.fitting.SLSQPLSQFitter()
    # Fit
    g_fit = fitter(g_guess, x_axis, y_obs, verblevel=1)
    print(g_fit)
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.title("2 Gaussians fit with 1 Gaussian")
    plt.plot(high_res_x, g_all(high_res_x), color='r', label='underlying truth')
    for i, g in enumerate(g_all):
        plt.plot(high_res_x, g(high_res_x), color='r', linestyle='--', label=f'underlying component {i}')
    for sign in (-1, 1):
        plt.axvline(mean + sign*dv/2, color='r', linestyle='--', alpha=0.3)
    plt.plot(x_axis, y_obs, '|', color='k', label='observations')
    plt.plot(high_res_x, g_fit(high_res_x), color='b', linestyle=':', label='Single component fit')
    plt.axvline(g_fit.mean, color='b', linestyle=':', alpha=0.3)
    residuals = y_obs - g_fit(x_axis)
    plt.plot(x_axis, residuals, color='grey', linestyle=':', label='Residuals')
    plt.legend()

    plt.subplot(122)
    plt.title("Residuals")
    plt.plot(x_axis, residuals, color='grey', linestyle=':', label='Residuals')
    # Plot the models and fit normalized to the residuals
    plt.plot(high_res_x, g_all(high_res_x)*np.max(residuals)/np.max(g_all(high_res_x)), color='r', alpha=0.5)
    for sign in (-1, 1):
        plt.axvline(mean + sign*dv/2, color='r', linestyle='--', alpha=0.3)
    plt.plot(high_res_x, g_fit(high_res_x)*np.max(residuals)/np.max(g_fit(high_res_x)), color='b', linestyle=':', alpha=0.5)
    plt.axvline(g_fit.mean, color='b', linestyle=':', alpha=0.3)

    plt.legend()
    # plt.xlabel(f"Component 0's A is {x} x Component 1's")
    # plt.xlabel(f"Components shifted {x} km/s apart")
    plt.xlabel(f"Component 1's $\sigma$ is {x} x Component 0's")
    plt.savefig(f'/home/rkarim/Pictures/2021-10-26-work/doubleGaussian_{label}_{j:02}.png')
    plt.clf()
    # plt.show()


from numpy.random import default_rng

def test_fitting_2_gaussians_with_2(dv=1, c_A=1, c_std=1, snr=20, numbers_only=False, seed=None):
    """
    Created Nov 10, 2021
    On Lee's advice, I want to see how well I can fit 2 gaussians with noise
    """
    A, mean, stddev = 10, 25, 1
    g1 = cps2.models.Gaussian1D(amplitude=A*c_A, mean=mean - (dv/2), stddev=stddev)
    g2 = cps2.models.Gaussian1D(amplitude=A, mean=mean + (dv/2), stddev=stddev*c_std)
    g_all = g1 + g2
    # Set up "observations"
    x_axis = np.arange(18, 33, 0.25)
    y_obs = g_all(x_axis)
    # ADD NOISE
    noise_rms = A/snr
    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(seed)
    y_obs += rng.normal(scale=noise_rms, size=y_obs.shape)
    high_res_x = np.linspace(18, 33, 100) # for plotting the model
    # Set up a guess
    g_guess_1 = cps2.models.Gaussian1D(amplitude=A, mean=mean, stddev=stddev,
        bounds={"amplitude": (0, 200), "mean": (20, 30), "stddev": (0.2, 5)})
    g_guess = g_guess_1 + g_guess_1.copy()
    g_guess.mean_1 = g_guess.mean_0 + 0.1
    g_guess.stddev_1.tied = lambda m: m.stddev_0
    fitter = cps2.fitting.SLSQPLSQFitter()
    # Fit
    g_fit = fitter(g_guess, x_axis, y_obs, verblevel=1)
    if numbers_only:
        return g_fit.parameters
    print(g_fit)
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    plt.title("2 Gaussians fit with 2 Gaussian")
    cps2.plot_everything_about_models(ax, x_axis, y_obs, g_all, m_color='k', text_x=0.8, text_y=0.3, dy=0.05)
    # # TRUTH
    # plt.plot(high_res_x, g_all(high_res_x), color='k', label='underlying truth')
    # # TRUTH COMPONENTS
    # for i, g in enumerate(g_all):
    #     plt.plot(high_res_x, g(high_res_x), color='k', linestyle='--', label=f'underlying component {i}')
    # # TRUTH COMPONENTS LINE CENTERS
    # for sign in (-1, 1):
    #     plt.axvline(mean + sign*dv/2, color='k', linestyle='--', alpha=0.3)
    # OBSERVATIONS
    plt.plot(x_axis, y_obs, 'x', color='k', label='observations')

    # fit
    cps2.plot_everything_about_models(ax, high_res_x, None, g_fit)

    # plt.plot(high_res_x, g_fit(high_res_x), color='b', linestyle=':', label='Single component fit')
    # plt.axvline(g_fit.mean, color='b', linestyle=':', alpha=0.3)
    residuals = y_obs - g_fit(x_axis)
    # plt.plot(x_axis, residuals, color='grey', linestyle=':', label='Residuals')
    plt.legend()
    # plt.show()
    plt.savefig("/home/ramsey/Pictures/2021-11-11-work/fitting_2G_with_2G_2.png")


def test_fitting_2G_with_2G_wrapper():
    """
    November 10, 2021
    Loop over the previous function and see how things change as parameters and noise changes
    """
    txt_file = "/home/ramsey/Desktop/gauss_test_dv_snr25.txt"
    n_trials = 10
    trials = np.zeros((n_trials, 2+6)) # SNR, dV, 6 parameters
    trials[:, 0] = 25
    trials[:, 1] = np.linspace(0.2, 2, n_trials)
    for i in range(n_trials):
        fitted_params = test_fitting_2_gaussians_with_2(dv=trials[i, 1], snr=trials[i, 0], numbers_only=True)
        trials[i, 2:] = fitted_params
    np.savetxt(txt_file, trials)


def save_bgsub_cii():
    """
    November 4, 2021
    Save a background-subtracted version of the CII cube (small cutout)
    Use all 4 regions from catalogs/pillar_background_sample_multiple_4.reg
    """
    raise RuntimeError("Already ran this on Nov 4, 2021")
    cii_bg_spectrum = cps2.get_cii_background()
    cii_cube = cps2.cutout_subcube(length_scale_mult=4)
    cii_cube = cii_cube - cii_bg_spectrum[:, np.newaxis, np.newaxis]
    # cii_cube.write(catalog.utils.m16_data_path + "sofia/M16_CII_pillar1_BGsubtracted.fits")


def moment2_cii_and_hcop():
    """
    November 29, 2021 (Cyber monday)
    Based on Marc's suggestion from the last meeting, compare the moment 2
    images of CII (bg subtracted) and HCO+
    I don't recall exactly what we will gain from this, since the CII are almost
    definitely wider profiles than the HCO+
    """
    # Load cubes
    cii_cube = cube_utils.CubeData("sofia/M16_CII_pillar1_BGsubtracted.fits").data
    hcop_cube = cps2.cutout_subcube(data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.fits", length_scale_mult=4)
    # Make subcubes
    vel_lims = (19*kms, 27*kms)
    cii_cube = cii_cube.spectral_slab(*vel_lims)
    hcop_cube = hcop_cube.spectral_slab(*vel_lims)
    # Noise cutoffs
    cii_noise = 1.*u.K
    hcop_noise = 0.5*u.K
    # Setup axes
    fig = plt.figure(figsize=(18, 8))
    ax_cii = plt.subplot(121, projection=cii_cube[0, :, :].wcs)
    ax_hcop = plt.subplot(122, projection=hcop_cube[0, :, :].wcs)
    # Make moment 0 images
    cii_mom0 = cii_cube.with_mask(cii_cube > cii_noise).moment0().to_value()
    hcop_mom0 = hcop_cube.with_mask(hcop_cube > hcop_noise).moment0().to_value()
    # Make moment 2 images
    cii_mom2 = cii_cube.with_mask(cii_cube > 2*cii_noise).moment(order=2)
    mom2_unit = cii_mom2.unit
    cii_mom2 = cii_mom2.to_value()
    hcop_mom2 = hcop_cube.with_mask(hcop_cube > 2*hcop_noise).moment(order=2).to_value()
    # Plot those images
    vlims = {'vmin': 0, 'vmax': 3}
    # vlims = {'vmin': 23, 'vmax': 26}
    cii_mom2[cii_mom0 < 30] = np.nan
    hcop_mom2[hcop_mom0 < 2.5] = np.nan
    im_cii = ax_cii.imshow(cii_mom2, origin='lower', **vlims)
    im_hcop = ax_hcop.imshow(hcop_mom2, origin='lower', **vlims)
    # Colorbars
    fig.colorbar(im_cii, ax=ax_cii, label=str(mom2_unit))
    fig.colorbar(im_hcop, ax=ax_hcop, label=str(mom2_unit))
    for ax in (ax_cii, ax_hcop):
        for coord in ax.coords:
            # coord.set_ticks_visible(False)
            coord.set_ticklabel_visible(False)
            coord.set_axislabel('')
    # Contours of moment 0
    ax_cii.contour(cii_mom0, colors='k', levels=np.arange(30, 211, 10))
    ax_hcop.contour(hcop_mom0, colors='k', levels=np.arange(2.5, 46, 5))
    # Labels
    ax_cii.set_title("CII Moment 2 (Moment 0 in contours) [20, 30] km/s")
    ax_hcop.set_title("same for HCO+")
    # Save figure
    # plt.tight_layout()
    # plt.show()
    # fig.savefig("/home/ramsey/Pictures/2021-12-06-work/moment2_cii_and_hcop.png")


def make_quick_image():
    cube = cps2.cutout_subcube(data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.fits", length_scale_mult=4.)
    mom0 = cube.spectral_slab(20*kms, 27*kms).moment0()
    plt.imshow(mom0.to_value(), origin='lower', cmap='nipy_spectral')
    plt.colorbar()
    plt.title("HCO+, moment 0 between 20-27 km/s")
    plt.show()


def setup_fitting_defaults(pillar):
    """
    August 22, 2022
    Return a dictionary of things that I will reuse in multiple fitting functions
    This would honestly make more sense as a Class, just structurally, but I'm
    already in too deep and it doesn't matter
    """
    utils_dict = {}

    if pillar == 1:
        reg_filename_short = "catalogs/pillar1_pointsofinterest_v3.reg"
    elif pillar == 2:
        reg_filename_short = "catalogs/pillar2_pointsofinterest_v2.reg"
    elif pillar == 3:
        reg_filename_short = "catalogs/pillar3_pointsofinterest.reg"
    elif pillar == '1b':
        reg_filename_short = "catalogs/pillar123_pointsofinterest_v1.reg"

    utils_dict['reg_filename_short'] = reg_filename_short

    utils_dict['spectrum_ylims'] = {
        'hcopCONV': (-1, 15), '13co10CONV': (-2, 17),
        'cii': (-3, 40), 'ciiAPEX': (-3, 40),
        'hcn': (-2, 15), 'hcnCONV': (-1, 15),
        'n2hpCONV': (-1, 4),
        'csCONV': (-0.5, 8),
        '12co10': (-10, 130), 'hcop': (-2, 15),
        '13co10': (-4, 20), 'cs': (-2, 8),
        '12co10CONV': (-5, 120), '12co10APEX': (-5, 120),
        'co65CONV': (-2, 25), '12co32': (-3, 45), '13co32': (-1, 23),
        'c18o10CONV': (-1, 4),
    }

    if pillar == 1:
        vel_lims = (24, 27)
    elif pillar == 2:
        vel_lims = (21, 24)
    elif pillar == 3:
        vel_lims = (21, 24)
    elif pillar == '1b':
        vel_lims = (20, 27)
    utils_dict['vel_lims'] = vel_lims

    img_vmin = { # these are all for 1 km/s moment, so multiply by km/s width of moment
        'hcopCONV': 0, '13co10CONV': 0,
        'cii': 10. if pillar!=3 else 5., 'ciiAPEX': 10. if pillar!=3 else 5.,
        'hcn': 1./3, 'hcnCONV': 0,
        'n2hpCONV': None,
        'csCONV': 0,
        '12co10': 10, 'hcop': 1./3,
        '13co10': 1, 'cs': 0,
        '12co10CONV': 10., '12co10APEX': 10.,
        'c18o10CONV': 0,
        'co65CONV': 1, '12co32': 0, '13co32': 0,
    }
    utils_dict['img_vmin'] = img_vmin

    img_vmax = { # same deal as above. Only used for pillar 3!!!!!!!!
        'hcopCONV': 2, '13co10CONV': None,
        'hcnCONV': 2, '12co10CONV': 75./3, '12co10APEX': 75./3,
        'cii':55./3, 'ciiAPEX':55./3,
    }
    utils_dict['img_vmax'] = img_vmax

    if pillar == 1:
        cutout_args = dict(length_scale_mult=4)
    elif pillar == 2:
        cutout_args = dict(length_scale_mult=3, reg_filename='catalogs/pillar2_across.reg', reg_index=2)
    elif pillar == 3:
        cutout_args = dict(length_scale_mult=1.3, reg_filename='catalogs/parallelpillars_2.reg', reg_index=5)
    elif pillar == '1b':
        cutout_args = dict(length_scale_mult=1.3, reg_filename='catalogs/across_all_pillars.reg', reg_index=4) # cut across shared base
    utils_dict['cutout_args'] = cutout_args

    def trim_cube(line_name, cube):
        if line_name.replace('CONV', '') in ('hcn', '13co10'):
            # Get rid of satellite lines and negatives
            # Satellite line correction depends on the pillar, because it's a constant velocity offset from the main line
            full_cube = cube
            if line_name.replace('CONV', '') == 'hcn':
                lo_lim = [None, 20, 17, 17][pillar if isinstance(pillar, int) else 1]
                hi_lim = [None, 27.5, 25.5, 25.5][pillar if isinstance(pillar, int) else 1]
            else:
                lo_lim = 17
                hi_lim = 27.5
            cube = cube.spectral_slab(lo_lim*kms, hi_lim*kms)
        elif line_name.replace('CONV', '') in ('12co10', '12co32'):
            # Get rid of the redshifted feature
            # This velocity does not depend on which pillar, because the feature is at a mostly fixed velocity
            full_cube = cube
            cube = cube.spectral_slab(17*kms, 27*kms)
        else:
            full_cube = None
        return cube, full_cube
    utils_dict['trim_cube'] = trim_cube

    def choose_vmin_vmax(line_name):
        vmin = None if img_vmin[line_name] is None else img_vmin[line_name]*(vel_lims[1] - vel_lims[0])
        if pillar == 3:
            vmax = None if ((line_name not in img_vmax) or (img_vmax[line_name] is None)) else img_vmax[line_name]*(vel_lims[1]-vel_lims[0])
        else:
            vmax = None
        return vmin, vmax
    utils_dict['choose_vmin_vmax'] = choose_vmin_vmax

    if pillar == 1:
        velocity_gridline_range = (22, 28)
    elif pillar == 2:
        velocity_gridline_range = (18, 24)
    elif pillar == 3:
        velocity_gridline_range = (18, 24)
    elif pillar == '1b':
        velocity_gridline_range = (20, 27)
    utils_dict['velocity_gridline_range'] = velocity_gridline_range

    return utils_dict


def fit_molecular_and_cii_with_gaussians(n_components=1, lines=None, pillar=1, select=0):
    """
    Created October 27 2021, 25 minutes before my meeting
    Let's do this
    Fit the HCO+ peak with 2 Gaussians, check it on the 13CO(1-0) peak (optional)
    and then check it on the CII peak

    Dec 7, 2021 the night before a meeting
    I deleted everything and am starting over
    But the goal is the same
    I have regridded the HCO+ to the CII grid now so I can do more direct
    comparisons
    I will use regions to mark the points so that I can also use this on the
    high-res HCO+
    Starting this rewrite while making the rice + tuna dish that gave that
    person on tik tok mercury poisoning because they ate it every day for 2 months
    I do not plan to eat it that often
    Dec 9, 2021 still working :/
    Use the pillar1_emissionpeaks.hcopregrid.moreprecise.reg and p1_threads_pathsandpoints.reg regions for fitting
    Jan 20, 2022 about to try to use this as production quality figure
    The coding I did to smash two regions into this plot is some of my worst work yet

    April 21, 2022: I want to rewrite this entire thing, it looks so bad
    I want to use it for comparing 13co10 to hcop and cii all at once,
    so I should convert the entire thing into a loop over some lines.
    I will drop the constraint that the spectral axes have to match; not great,
    but I'm not regridding 13co10, that's too much work for 1 or 2 plots.
    I will use Point regions to select spectra.
    I should just loop through each line and fit the template model to the
    spectrum. I can make a few versions for different template models and stitch
    them together in Google slides if I really want to.
    I probably only need 1 comp. free, 2 comp. free, and 3 comp. fixed/tied std.
    Also, should standardize the way I put in regions (commented filenames
    are messy)
    April 22, 2022: This is so much better!

    May 19, 2022 (astro commencement is today I think)
    I am going to try to make this do Pillar 2 as well.
    I have regions in catalogs/pillar2_pointsofinterest.reg (point regions)

    June 3, 2022 (Guangwei's defense yesterday)
    I will update for P3 now too. I made 4 points in
    catalogs/pillar3_pointsofinterest.reg (point regions)
    """
    utils_dict = setup_fitting_defaults(pillar) # contains useful things
    # reg_filename_short = "catalogs/pillar1_emissionpeaks.hcopregrid.moreprecise.reg" # order appears to be [HCO+, CII]
    # reg_filename_short = "catalogs/p1_threads_pathsandpoints.reg" # order appears to be North-E, North-W, South-E, South-W
    # reg_filename_short = "catalogs/pillar1_pointsofinterest.reg" # order is wide profile, blue tail, north of west thread, bluest component, top of western thread (where IRAC4 peaks), above western thread (where IRAC4 is dark)
    # reg_filename_short = "catalogs/pillar1_peak_degeneracyboundary.reg" # order is 3 component peak, 2 component peak
    reg_filename_short = utils_dict['reg_filename_short']
    sky_regions = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    # print("There are ", len(sky_regions), " regions")
    # now supporting multiple selected_regions
    if pillar == 1:
        # selected_region_list = [sky_regions[0]]
        if select == 0:
            selected_region_list = [sky_regions[x-1] for x in [3, 6, 4, 7]] # threads (3, 4 works good, 6, 7 are noisier)
        elif select == 1:
            selected_region_list = [sky_regions[x-1] for x in [1, 2, 8, 5]] # head
        elif select == 2:
            selected_region_list = [sky_regions[x-1] for x in [4, 8, 5]] # Wthread
        # Select 3, 4, 5, 6 are the latest idea for the paper (2022-08-20)
        elif select == 3:
            selected_region_list = [sky_regions[x-1] for x in [5, 4, 7]] # single component fits (W thread and NW head)
        elif select == 4:
            selected_region_list = [sky_regions[x-1] for x in [1, 2, 8]] # 2 component fits (rest of head)
        elif select == 5:
            selected_region_list = [sky_regions[x-1] for x in [1, 3, 6]] # E thread + NE head
        elif select == 6:
            selected_region_list = [sky_regions[1 - 1]] # position 1 is the 3-component position
    elif pillar == 2:
        """
        Order is
        1: N, 2: E-mid-N, 3: W-mid-N
        4: E-mid-S, 5: W-mid-S
        6: E-W, 7: W-S, 8: S
        9: Head, 10: Mid-pillar Feature
        """
        if select == 0:
            selected_region_list = [sky_regions[x-1] for x in [1, 2, 3]]
        elif select == 1:
            selected_region_list = [sky_regions[x-1] for x in [4, 5, 6, 7]]
        elif select == 2:
            selected_region_list = [sky_regions[x-1] for x in [8, 9, 10]]
        elif select == 'head':
            selected_region_list = [sky_regions[x-1] for x in [1, 9, 2, 3]]
    elif pillar == 3:
        """
        Order is N, S, E, Center
        I probably want N Center E S
        """
        selected_region_list = [sky_regions[x-1] for x in [1, 4, 3, 2]]

    elif pillar == '1b':
        """
        Order is:
        E Horn, W Horn, Inter Horn (0, 1, 2)
        Shared Base E, Mid, W (3, 4, 5)
        P2, P3, Shelf (not using these much)
        """
        if select == 0: # Horns
            selected_region_list = [sky_regions[x] for x in [0, 1, 2]]
        elif select == 1: # Shared Base
            selected_region_list = [sky_regions[x] for x in [3, 4, 5]]
        elif select == 2: # Shelf
            selected_region_list = [sky_regions[8]]
    if len(selected_region_list) > 1:
        pixel_name = "-and-".join([reg.meta['text'].replace(" ", '-') for reg in selected_region_list])
    else:
        pixel_name = selected_region_list[0].meta['text'].replace(" ", '-')

    spectrum_ylims = utils_dict['spectrum_ylims']
    vel_lims = utils_dict['vel_lims']
    img_vmin = utils_dict['img_vmin']
    img_vmax = utils_dict['img_vmax']

    # Decide which things are fixed in the models
    fixedstd = False
    tiestd = True
    untieciistd = False
    fixed_cii_std = False
    fixedmean = False


    # Process list of line names
    if lines is None:
        line_names_list = ['12co10', 'cs', 'cii',]
        print("Setting line_names_list to default of ", line_names_list)
    else:
        line_names_list = lines

    if 'cii' in line_names_list:
        cii_index = line_names_list.index('cii')
        assert cii_index == len(line_names_list)-1 # always last, so i have the template fit ready first
        cii_background_spectrum = cps2.get_cii_background().to_value()
        template_line = 'hcopCONV' # The line to use to fix CII line ceners
        # template_line = 'cs' # The line to use to fix CII line ceners
        try:
            template_index = line_names_list.index(template_line)
        except ValueError:
            assert not fixedmean
            template_index = None
    else:
        cii_index = None
        template_index = None


    # Set up all Axes
    fig = plt.figure(figsize=(18, 10))
    n_lines = len(line_names_list)
    n_regions = len(selected_region_list)
    grid_shape = (n_lines, n_regions + 1)
    axes_spec = [[plt.subplot2grid(grid_shape, (i, j)) for j in range(n_regions)] for i in range(n_lines)]
    axes_img = [plt.subplot2grid(grid_shape, (i, grid_shape[1]-1)) for i in range(n_lines)]

    # Set up initial model for fitting
    if pillar == 1:
        default_mean = 25.
        mean_bounds = (20, 30)
    elif pillar == 2:
        default_mean = 22.
        mean_bounds = (18, 24)
    elif pillar == 3:
        default_mean = 22.
        mean_bounds = (18, 24)
    elif pillar == '1b':
        if select == 0 or select == 2:
            default_mean = 25.
        else:
            default_mean = 23.5
        mean_bounds = (21, 26)

    stddev_hcop = 0.47
    g0 = cps2.models.Gaussian1D(amplitude=7, mean=default_mean, stddev=stddev_hcop,
        bounds={'amplitude': (0, None), 'mean': mean_bounds})
    if n_components > 1:
        g1 = g0.copy()
        if n_components > 2:
            g2 = g0.copy()
            g0.mean = default_mean - 1.5
            g1.mean = default_mean - 0.5
            g2.mean = default_mean + 0.5
            g = g0 + g1 + g2
        else:
            g0.mean = default_mean - 1
            g1.mean = default_mean
            g = g0 + g1
    else:
        g = g0
    # Apply the constraints
    fitter = cps2.fitting.LevMarLSQFitter(calc_uncertainties=True)
    if tiestd:
        cps2.tie_std_models(g)
    if fixedstd:
        cps2.fix_std(g)
    # Number of dimensions of least constrained fit
    ndim = len(get_fittable_param_names(g))

    """
    model_list will have len==len(line_names_list), and each element will
    have len==len(selected_region_list)
    """
    model_list = []
    cutout_args = utils_dict['cutout_args']

    for line_idx, line_name in enumerate(line_names_list):
        # Extract spectra at each region
        cube = cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line_name], **cutout_args)
        cube, full_cube = utils_dict['trim_cube'](line_name, cube) # get rid of satellite lines & other features in certain lines
        cube_x = cube.spectral_axis.to_value()
        # PLOT IMAGE!!!
        vmin, vmax = utils_dict['choose_vmin_vmax'](line_name)
        im = axes_img[line_idx].imshow(cube.spectral_slab(*(v*kms for v in vel_lims)).moment0().to_value(), origin='lower', cmap='Blues', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes_img[line_idx])
        # PLOT BEAM!!!
        patch = cube.beam.ellipse_to_plot(*(axes_img[line_idx].transAxes + axes_img[line_idx].transData.inverted()).transform([0.9, 0.06]), misc_utils.get_pixel_scale(cube[0, :, :].wcs))
        patch.set_alpha(0.9)
        patch.set_facecolor('grey')
        patch.set_edgecolor('grey')
        axes_img[line_idx].add_artist(patch)

        channel_noise = cube_utils.onesigmas[line_name]

        models_for_this_line = []

        for reg_idx, reg in enumerate(selected_region_list):
            # Get spectra
            pix_coords = tuple(round(x) for x in reg.to_pixel(cube[0, :, :].wcs).center.xy[::-1])
            spectrum = cube[(slice(None), *pix_coords)].to_value()
            if line_idx == cii_index:
                spectrum = spectrum - cii_background_spectrum
            # If there is a "full" spectrum (if I trimmed off some velocity range), get it so I can plot it
            if full_cube is not None:
                full_spectrum = full_cube[(slice(None), *pix_coords)].to_value()
                if line_idx == cii_index:
                    full_spectrum -= cii_background_spectrum
            else:
                full_spectrum = None

            # Plot markers
            #### Don't plot a mark, just use the number as the marker
            if len(selected_region_list) > 1:
                # Label region, if more than one
                pad = 1.5
                # dx = pad if reg_idx else -pad
                # dy = pad
                dx, dy = 0, 0
                axes_img[line_idx].text(pix_coords[1]+dx, pix_coords[0]+dy, str(reg_idx + 1), color='r', fontsize=12, ha='center', va='center')
            else:
                axes_img[line_idx].plot([pix_coords[1]], [pix_coords[0]], 'o', markersize=5, color='r')


            # Plot noise stuff on spectrum Axes
            cps2.plot_noise_and_vlims(axes_spec[line_idx][reg_idx], channel_noise, vel_lims)

            # Fit!
            # First, do the special stuff for CII
            if line_idx == cii_index:
                if fixedmean:
                    g_init = model_list[template_index][reg_idx].copy()
                    cps2.fix_mean(g_init)
                else:
                    g_init = g.copy()
                cps2.tie_std_models(g_init, untie=untieciistd)
                if not fixed_cii_std:
                    cps2.unfix_std(g_init)
                else:
                    for m in cps2.iter_models(g_init):
                        m.stddev = 1.
                    cps2.fix_std(g_init)
                ndim = len(get_fittable_param_names(g_init))
            else:
                g_init = g.copy()
            # Now actually fit!
            g_fit = fitter(g_init, cube_x, spectrum, weights=np.full(spectrum.size, 1.0/channel_noise))

            # Plot fit
            if full_spectrum is not None:
                axes_spec[line_idx][reg_idx].plot(full_cube.spectral_axis.to_value(), full_spectrum, color='grey', alpha=0.5)
            cps2.plot_everything_about_models(axes_spec[line_idx][reg_idx], cube_x, spectrum, g_fit, noise=channel_noise, dof=(cube_x.size - ndim))

            models_for_this_line.append(g_fit)
        model_list.append(models_for_this_line)

    # Adjust axes things
    for ax in sum(axes_spec, []) + axes_img:
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
    for ax in sum(axes_spec, []):
        ax.set_xlim([17, 30])
        velocity_gridline_range = utils_dict['velocity_gridline_range']
        for v in range(*velocity_gridline_range):
            ax.axvline(v, color='gray', alpha=0.2)
    for line_name, ax in zip(line_names_list, axes_img):
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        ax.text(0.95, 0.9, cube_utils.cubenames[line_name], fontsize=15, ha='right', va='center', transform=ax.transAxes)
    for ax in sum(axes_spec[::-1][1:], []):
        # Get rid of x ticks above the bottom row
        ax.xaxis.set_ticklabels([])
    for ax in sum(tuple(zip(*axes_spec))[1:], ()):
        # Get rid of y ticks to the right of the leftmost column
        ax.yaxis.set_ticklabels([])
    for line_name, axes in zip(line_names_list, axes_spec):
        for reg_idx, ax in enumerate(axes):
            # Only label region number if there are multiple regions
            index_stub = f"\n{reg_idx+1}" if len(selected_region_list) > 1 else ""
            # Label each spectrum plot
            ax.text(0.8, 0.9, f'{cube_utils.cubenames[line_name].replace(" (CII beam)", "")}{index_stub}', fontsize=15, ha='center', va='center', transform=ax.transAxes)
            ax.set_ylim(spectrum_ylims[line_name])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)

    pillar_stub = "" if pillar == 1 else f"p{pillar}_"

    fixedstd_stub = f"_fixedstd{stddev_hcop:04.2f}" if fixedstd else ''
    tiestd_stub = f"_hcopUNtiedstd" if not tiestd else ''

    untieciistd_stub = f"_ciiUNtiedstd" if untieciistd else ''
    fixedciistd_stub = "_ciifixedstd" if fixed_cii_std else ''
    if cii_index is not None:
        fixedmean_stub = f"_fixedciimean" if fixedmean else ''
    else:
        fixedmean_stub = ''
    lines_stub = "-".join(line_names_list)
    # plt.savefig(f'/home/ramsey/Pictures/2021-12-21-work/fit_{g.n_submodels}molecular_components_and_CII_{pixel_name}_{fixedstd_stub}{tiestd_stub}{untieciistd_stub}{fixedmean_stub}.png')
    # 2022-01-20, 2022-04-22, 2022-04-25, 2022-04-26, 2022-04-28, 2022-05-03, 2022-05-19,
    # 2022-06-03, 2022-06-07, 2022-08-18,19,20,24, 2022-09-08,12, 2022-10-29,31, 2022-11-01,07
    savename = f"/home/ramsey/Pictures/2022-11-07/fit_{pillar_stub}{g.n_submodels}_{lines_stub}_{pixel_name}{fixedstd_stub}{tiestd_stub}{untieciistd_stub}{fixedciistd_stub}{fixedmean_stub}"
    ###########################
    save_as_png = True
    ###########################
    if save_as_png:
        fig.savefig(f"{savename}.png",
            metadata=catalog.utils.create_png_metadata(title=f'regions from {reg_filename_short}',
                file=__file__, func='fit_molecular_and_cii_with_gaussians'))
    else:
        fig.savefig(f"{savename}.pdf")
    # plt.show()


def fit_spectrum_detailed(line_stub, n_components=1, pillar=1, reg_number=0):
    """
    August 22, 2022
    Fit one line spectrum in one region. Very similar to fit_molecular_and_cii_with_gaussians()
    The main difference is that I'll keep that function nice and general, and this
    one can have a bunch of detailed constraints to make the fits look nice.
    The goal here is to characterize the CO line asymmetry by fitting one or two
    components, and since it often fits poorly I'll have to do a little more
    work with initial conditions and constraints to make the fits come out better
    :param reg_number: 1-indexed!!!!!!!!!!!
    """
    utils_dict = setup_fitting_defaults(pillar)
    reg_filename_short = utils_dict['reg_filename_short']
    # I have been 1-indexing the regions so I'll keep doing that
    reg = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))[reg_number - 1]
    pixel_name = reg.meta['text'].replace(' ', '-')

    spectrum_ylims = utils_dict['spectrum_ylims']
    vel_lims = utils_dict['vel_lims']
    img_vmin = utils_dict['img_vmin']
    img_vmax = utils_dict['img_vmax']

    # Decide which things are fixed in the models
    fixedstd = False
    tiestd = True
    fixedmean = False

    fig = plt.figure(figsize=(15, 6))
    grid_len = 8
    grid_spec_frac = 6
    axes_spec = plt.subplot2grid((1, grid_len), (0, 0), colspan=grid_spec_frac)
    axes_img = plt.subplot2grid((1, grid_len), (0, grid_spec_frac), colspan=(grid_len - grid_spec_frac))

    dictionary_of_fitting_parameters_that_work = {
        # put things in here by "{line_stub}-{pixel_name}-{n_components}"
        "hcopCONV-NW-thread-1": {'m1': 24.9, 'm1f': True, 's1': 0.3, 'a1': 3.5},
        "hcopCONV-NW-thread-2": {'m1': 24.9, 'm1f': True, 's1': 0.3, 'a1': 3.5, 'm2': 25.5, 's2': 0.3, 'a2': 1.26},
        'co65CONV-NW-thread-2': {'m1': 24.9, 'm1f': False, 's1': 0.3, 'a1': 3.5, 'm2': 25.5, 's2': 0.3, 'a2': 1.26},
        'co65CONV-SW-thread-2': {'m1': 24.9, 'm1f': False, 's1': 0.3, 'a1': 3.5, 'm2': 25.5, 's2': 0.3, 'a2': 1.26},
        '12co32-NW-thread-2': {'m1': 25.31, 'm1f': False, 's1': 0.7, 's1b': (0.3, 0.9), 'm2': 26.1, 'm2b': (26, 30)},
        '12co32-SW-thread-2': {'m1': 25.42, 'm1f': True, 's1': 0.7, 'm2': 26.1, 'm2b': (26, 30)},
        '12co32-NW-thread-3': {'m1': 25.31, 'm1f': True, 's1': 0.5, 's1b': (0.3, 0.5), 'm2': 26, 'm3b': (24, 24.75), 'm3': 24},
        # '12co32-SW-thread-3': {'m1': 25.42, 'm1f': True, 's1': 0.7, 's1b': (0.3, 0.7), 'm2': 26, 'm3b': (24, 24.75), 'm3': 24},

        '13co10CONV-NW-thread-2': {'m1': 25.1, 's1': 0.3, 's1b': (0.1, 0.5), 'm2': 26},

        'csCONV-SE-thread-1': {'m1': 26},
        'hcnCONV-NW-thread-2': {'m1': 24.9, 'm2': 26},
        'hcnCONV-SW-thread-2': {'m1': 25.2, 'm2': 26},
        'hcnCONV-NE-thread-2': {'m1': 23., 'm2': 25.6},
        'hcnCONV-SE-thread-2': {'m1': 25., 'm2': 25.9},

        '12co10CONV-E-peak-2': {'m1': 23.5, 's1': 0.85, 'a1': 32, 'm2': 25.4, 's2': 0.85, 'a2': 66.4},
        '12co10CONV-E-peak-3': {'m1': 23.5, 's1': 0.85, 'a1': 32, 'm2': 25.4, 's2': 0.85, 'a2': 66.4, 's1b': (0.2, 0.8)},
        '12co10CONV-E-peak-4': {'m1': 23.5, 's1': 0.85, 'a1': 32, 'm2': 25.4, 's2': 0.85, 'a2': 66.4, 's1b': (0.2, 0.65), 's4t': False, 's4': 80, 's4f': True, 'm4': 18, 'm4f': True, 'a4': 3.5, 'a4f': False},
        '12co10CONV-S-peak-4': {'m1': 23.5, 's1': 0.85, 'a1': 32, 'm2': 25.4, 's2': 0.85, 'a2': 66.4, 's1b': (0.2, 0.7), 's4t': False, 's4': 80, 's4f': True, 'm4': 18, 'm4f': True, 'a4': 3.5, 'a4f': False},
        # '12co10CONV-broad-line-2': {'m1': 23.1, 'm1b': (22, 24), 's1': 1, 'a1': 10, 'm2': 24.7, 's2': 1, 'a2': 50},
        # '13co10CONV-broad-line-3': {'m1': 22.3, 's1': 0.3, 'm2': 23.7, 'm3': 25.1, 's1b': (0.1, 0.3)},


        '12co10CONV-broad-line-2': {'m1': 24.7, 's1': 1, 'a1': 50, 's2t': False, 's2f': True, 'a2': 3.5, 'a2f': False, 'm2': 18, 'm2f': True, 's2': 80},
        '12co10CONV-broad-line-3': {'m1': 23.1, 'm1b': (22, 24), 's1': 1, 'a1': 10, 'm2': 24.7, 's2': 1, 'a2': 50, 's3t': False, 's3': 80, 's3f': True, 'm3': 18, 'm3f': True, 'a3': 3.5, 'a3f': False},
        '12co10CONV-broad-line-4': {'m1': 23.1, 'm1b': (22, 24), 's1b': (0.2, 0.6), 'a1': 10, 'm2': 24.7, 'a2': 50, 's4t': False, 's4': 80, 's4f': True, 'm4': 18, 'm4f': True, 'a4': 3.5, 'a4f': False},

        '12co10CONV-W-peak-2': {'m1': 25.7, 's1': 1, 'a1': 60, 's2t': False, 's2': 80, 's2f': True, 'm2': 19, 'm2f': True, 'a2': 4.7, 'a2f': False},
        '12co10CONV-W-peak-3': {'m1': 23, 'a1': 20, 's1': 0.7, 's1b': (0.1, 0.95), 'm2': 25.7, 'a2': 60, 's3t': False, 's3': 80, 's3f': True, 'm3': 19, 'm3f': True, 'a3': 4.7, 'a3f': False},
        '12co10CONV-W-peak-4': {'m1': 23, 'a1': 32, 'm2': 24.4, 'm3': 25.4, 'a2': 66.4, 's1b': (0.2, 0.65), 's4t': False, 's4': 80, 's4f': True, 'm4': 18, 'm4f': True, 'a4': 3.5, 'a4f': False},

        '12co10CONV-NW-thread-2': {'m1': 24.9, 'm1f': True, 's1': 0.5, 'a1': 30, 'm2': 26, 'm2f': False, 'a2': 10},

        '12co32-E-peak-3': {'m1': 23.5, 'm2': 24.7, 'm3': 25.7, 's1b': (0.2, 0.7)},

        '12co32-S-peak-2': {'m1': 23.5, 'm2': 25.3, 's1b': (0.2, 0.9)},
        '12co32-S-peak-3': {'m1': 23.5, 'm2': 24.7, 'm3': 25.7, 's1b': (0.2, 0.8)},
        '12co32-W-peak-3': {'m1': 23.5, 'm2': 24.7, 'm3': 25.7, 's1b': (0.2, 0.85)},

        '12co32-broad-line-3': {'m1': 23.5, 'm2': 24.7, 'm3': 25.7, 's1b': (0.2, 0.7)},

        'hcopCONV-E-peak-3': {'m1': 23.6, 'm2': 24.8, 'm3': 25.6, 'a1': 1.5, 'a2': 3.1, 'a3': 9.3, 's1': 0.5},
        'hcopCONV-S-peak-3': {'m1': 23.6, 'm2': 24.8, 'm3': 25.6, 'a1': 1.5, 'a2': 3.1, 'a3': 9.3, 's1': 0.5},
        'hcopCONV-W-peak-3': {'m1': 23.6, 'm2': 24.8, 'm3': 25.6, 'a1': 1.5, 'a2': 3.1, 'a3': 9.3, 's1b': (0.2, 0.45)},

        'csCONV-S-peak-3': {'m1': 24.1, 'm2': 25, 'm3': 25.9, 'a1': 2, 'a2': 8, 'a3': 6, 's1': 0.4},
        'csCONV-S-peak-2': {'m1': 25, 'm2': 25, 'a1': 2, 'a2': 8, 's1': 1.2, 's2': 0.4},
        'csCONV-W-peak-3': {'m1': 24.1, 'm2': 25, 'm3': 25.9, 'a1': 1, 'a2': 3, 'a3': 1, 's1b': (0.1, 0.5)},
        'hcnCONV-E-peak-3': {'m1': 23.6, 'm2': 24.8, 'm3': 25.6, 'a1': 1.5, 'a2': 3.1, 'a3': 9.3, 's1': 0.5},
        'hcnCONV-S-peak-3': {'m1': 23.6, 'm2': 24.8, 'm3': 25.6, 'a1': 1.5, 'a2': 3.1, 'a3': 9.3, 's1': 0.5},
        'hcnCONV-W-peak-2': {'s1b': (0.2, 0.5)},
        'hcnCONV-W-peak-3': {'m1': 23.5, 'm2': 24.1, 'm3': 25, 'a1': 1., 'a2': 9, 'a3': 1, 's1b': (0.2, 0.5)},

        'n2hpCONV-S-peak-2': {'m1': 24.75, 'm2': 25.1, 'a1': 1, 'a2': 1, 's1b': (0.1, 0.3)},
        'n2hpCONV-S-peak-3': {'m1': 24.8, 'm2': 25.2, 'm3': 25.5, 'a1': 1, 'a2': 1, 's1b': (0.1, 0.15)},

        'co65CONV-S-peak-3': {'m1': 23.6, 'm2': 24.8, 'm3': 25.6, 'a1': 1.5, 'a2': 3.1, 'a3': 9.3, 's1': 0.7},
        'co65CONV-W-peak-3': {'m1': 23.6, 'm2': 24.8, 'm3': 25.6, 'a1': 1.5, 'a2': 3.1, 'a3': 9.3, 's1b': (0.2, 0.7)},

        # 'cii-NE-thread-2': {'m1': 23.5, 'm2': 25.6, 'm2f': False, 's1': 0.8},
        'cii-NE-thread-2': {'m1': 23.46, 'm1f': True, 'm2': 25.61, 'm2f': True, 's1': 0.8}, # based on 12CO10 towards the same location
        'cii-SE-thread-3': {'m1': 22.5, 'm1b': (22, 22.7), 'a1': 3, 'a1b': (1, 5), 'm2': 25.6, 'm2b': (25.5, 25.75), 'm3': 28.17, 'm3b': (27.5, 29)},

        # P1b
        '12co32-Shared-Base-Mid-2': {'m1': 22.2, 'm2': 24.6, 'a1': 17.5, 'a2': 17.5, 's1': 1},
        '13co32-Shared-Base-Mid-2': {'m1': 22.2, 'm2': 24.6, 'a1': 17.5, 'a2': 17.5, 's1': 1},
    }
    def parse_and_assign_saved_params(model):
        key = f"{line_stub}-{pixel_name}-{n_components}"
        if key not in dictionary_of_fitting_parameters_that_work:
            return None
        saved_params = dictionary_of_fitting_parameters_that_work[key]
        model_list = list(cps2.iter_models(model))
        for model_index in range(len(model_list)):
            n = model_index + 1
            if f'm{n}' in saved_params:
                model_list[model_index].mean = saved_params[f'm{n}']
            if f'm{n}f' in saved_params:
                model_list[model_index].mean.fixed = saved_params[f'm{n}f']
            if f'm{n}b' in saved_params:
                model_list[model_index].mean.bounds = saved_params[f'm{n}b']

            if f's{n}' in saved_params:
                model_list[model_index].stddev = saved_params[f's{n}']
            if f's{n}t' in saved_params:
                model_list[model_index].stddev.tied = saved_params[f's{n}t']
            if f's{n}f' in saved_params:
                model_list[model_index].stddev.fixed = saved_params[f's{n}f']
            if f's{n}b' in saved_params:
                model_list[model_index].stddev.bounds = saved_params[f's{n}b']

            if f'a{n}' in saved_params:
                model_list[model_index].amplitude = saved_params[f'a{n}']
            if f'a{n}f' in saved_params:
                model_list[model_index].amplitude.fixed = saved_params[f'a{n}f']
            if f'a{n}b' in saved_params:
                model_list[model_index].amplitude.bounds = saved_params[f'a{n}b']
        print(f"Found result for {key}:", saved_params)
        print("Model initialized to")
        print(model)

    default_mean = 25.
    mean_bounds = (20, 30)
    g0 = cps2.models.Gaussian1D(amplitude=7, mean=default_mean, stddev=0.5,
        bounds={'amplitude': (0, None), 'mean': mean_bounds})
    if n_components > 1:
        g1 = g0.copy()
        if n_components > 2:
            g0.mean = default_mean - 2
            g1.mean = default_mean - 0.5
            g2 = g1.copy()
            g2.mean = default_mean + 0.5
            if n_components > 3:
                g3 = g2.copy()
                g3.mean = default_mean + 1
                g = g0 + g1 + g2 + g3
            else:
                g = g0 + g1 + g2
        else:
            g0.mean = default_mean - 1
            g1.mean = default_mean + 0.5
            g = g0 + g1
    else:
        g = g0

    fitter = cps2.fitting.LevMarLSQFitter(calc_uncertainties=True)
    if tiestd:
        cps2.tie_std_models(g)
    if fixedstd:
        cps2.fix_std(g)
    parse_and_assign_saved_params(g)
    ndim = len(get_fittable_param_names(g))

    cutout_args = utils_dict['cutout_args']
    cube = cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line_stub], **cutout_args)
    cube, full_cube = utils_dict['trim_cube'](line_stub, cube)
    cube_x = cube.spectral_axis.to_value()
    # PLOT IMAGE!!!
    vmin, vmax = utils_dict['choose_vmin_vmax'](line_stub)
    im = axes_img.imshow(cube.spectral_slab(*(v*kms for v in vel_lims)).moment0().to_value(), origin='lower', cmap='Blues', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axes_img)
    # PLOT BEAM!!!
    patch = cube.beam.ellipse_to_plot(*(axes_img.transAxes + axes_img.transData.inverted()).transform([0.9, 0.06]), misc_utils.get_pixel_scale(cube[0, :, :].wcs))
    patch.set_alpha(0.9)
    patch.set_facecolor('grey')
    patch.set_edgecolor('grey')
    axes_img.add_artist(patch)

    channel_noise = cube_utils.onesigmas[line_stub]
    pix_coords = tuple(round(x) for x in reg.to_pixel(cube[0, :, :].wcs).center.xy[::-1])
    spectrum = cube[(slice(None), *pix_coords)].to_value()
    is_cii = ('cii' in line_stub)
    if is_cii:
        cii_background_spectrum = cps2.get_cii_background().to_value()
        spectrum = spectrum - cii_background_spectrum
    if full_cube is not None:
        full_spectrum = full_cube[(slice(None), *pix_coords)].to_value()
        if is_cii:
            full_spectrum -= cii_background_spectrum
    else:
        full_spectrum = None
    axes_img.plot([pix_coords[1]], [pix_coords[0]], 'o', markersize=5, color='r')
    cps2.plot_noise_and_vlims(axes_spec, channel_noise, vel_lims)
    g_init = g.copy()
    g_fit = fitter(g_init, cube_x, spectrum, weights=np.full(spectrum.size, 1./channel_noise))
    if full_spectrum is not None:
        axes_spec.plot(full_cube.spectral_axis.to_value(), full_spectrum, color='grey', alpha=0.5)
    cps2.plot_everything_about_models(axes_spec, cube_x, spectrum, g_fit, noise=channel_noise, dof=(cube_x.size - ndim))
    for ax in (axes_spec, axes_img):
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
    axes_spec.set_xlim([17, 30])
    velocity_gridline_range = utils_dict['velocity_gridline_range']
    for v in range(*velocity_gridline_range):
        axes_spec.axvline(v, color='gray', alpha=0.2)
    index_stub = f"\n{reg_number}"
    # Label spectrum plot
    axes_spec.text(0.8, 0.9, f'{cube_utils.cubenames[line_stub].replace(" (CII beam)", "")}{index_stub}', fontsize=15, ha='center', va='center', transform=axes_spec.transAxes)
    axes_spec.set_ylim(spectrum_ylims[line_stub])
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, hspace=0.05, right=0.95)

    pillar_stub = f"p{pillar}_"

    if fixedstd:
        fixedstd_stub = f"_fixedstd{list(cps2.iter_models(g_init))[0].stddev.value:04.2f}"
        tiestd_stub = ''
    else:
        fixedstd_stub = ''
        tiestd_stub = f"_untiedstd" if not tiestd else ''
    fixedmean_stub = f"_fixedmean" if fixedmean else ''
    # 2022-08-22,24, 2022-09-08,09,12,15, 2022-11-01
    savename = f"/home/ramsey/Pictures/2022-11-01/fit_{pillar_stub}{g.n_submodels}_{line_stub}_{pixel_name}{fixedstd_stub}{tiestd_stub}{fixedmean_stub}"
    ###########################
    save_as_png = True
    ###########################
    if save_as_png:
        fig.savefig(f"{savename}.png",
            metadata=catalog.utils.create_png_metadata(title=f'regions from {reg_filename_short}',
                file=__file__, func='fit_spectrum_detailed'))
    else:
        fig.savefig(f"{savename}.pdf")







def get_fittable_param_names(model):
    """
    Find the fittable parameters for this model. Fittable means not fixed
    or tied.
    :param model: template model with all the right things tied and fixed
    :returns: list of fittable param_names valid for this model
    """
    return [ param_name for param_name in model.param_names if not (model.tied[param_name] or model.fixed[param_name]) ]

def get_fittable_parameters(model, fpn):
    """
    Similar to get_fittable_param_names, but this time return the current
    values of those parameters. Length and order of parameters is the exact
    same as in the result of get_fittable_param_names(model).
    There must already be a fittable_param_names array generated by get_fittable_param_names
    :param model: template model that matches the existing fittable_param_names
    array
    :param fpn: fittable_param_names list
    :returns: list of the current param values
    """
    p0 = [] # resulting param array
    for param_name in fpn:
        param = getattr(model, param_name)
        p0.append(param.value)
    return p0

def set_fittable_parameters(p, model, fpn):
    """
    Sets all the fittable parameters of model to the values in the array p
    :param p: array of whatever fittable parameters are in this model.
        should be in order of get_fittable_param_names(model)
    :param model: template model that is OK to edit
    :param fpn: fittable_param_names list
    """
    for i, param_name in enumerate(fpn):
        param = getattr(model, param_name)
        param.value = p[i]

def make_ln_posterior_fn(model, helper_chisq_fn):
    """
    Dec 13, 2021
    make a lnposterior function to pass to emcee.EnsembleSampler
    """
    def lnposterior(p, fpn):
        """
        :param p: parameter array
        :param fpn: fittable_param_names list
        """
        return -1.*helper_chisq_fn(p, model, fpn)
    return lnposterior


def test_fitting_uncertainties_with_emcee(which_line='hcop'):
    """
    December 12, 2021
    Check the range of valid Gaussian fits to a given HCO+/CII spectrum
    Use emcee and corner plots
    :param which_line: cii or hcop
    """
    # Only load in CII if we need it
    if which_line == 'cii':
        cii_cube = cube_utils.CubeData("sofia/M16_CII_pillar1_BGsubtracted.fits").data
    regrid = True # Just in case I want to switch back to regular resolution? doesn't hurt
    fn = f"carma/M16.ALL.hcop.sdi.cm.subpv{'.SOFIAbeam.regrid' if regrid else ''}.fits"
    hcop_cube = cube_utils.CubeData(fn).convert_to_K().data.with_spectral_unit(kms)
    hcop_flat_wcs = hcop_cube[0, :, :].wcs

    # sky_regions = regions.Regions.read(catalog.utils.search_for_file("catalogs/pillar1_emissionpeaks.hcopregrid.moreprecise.reg")) # order appears to be [HCO+, CII]
    # sky_regions = regions.Regions.read(catalog.utils.search_for_file("catalogs/p1_threads_pathsandpoints.reg")) # order appears to be North-E, North-W, South-E, South-W
    sky_regions = regions.Regions.read(catalog.utils.search_for_file("catalogs/pillar1_pointsofinterest.reg")) # order is wide profile, blue tail, north of west thread, bluest component
    pixel_coords = [tuple(round(x) for x in reg.to_pixel(hcop_flat_wcs).center.xy[::-1]) for reg in sky_regions] # converted to (i, j) tuples
    selected_pixel = pixel_coords[0]
    # pixel_name = "bluest-component"
    pixel_name = "wide-profile"

    # Start with one pixel, just fit that first
    assert regrid
    if which_line == 'cii':
        cii_spectrum = cii_cube[(slice(None), *selected_pixel)].to_value()
        cii_x = cii_cube.spectral_axis.to(kms).to_value()
    hcop_spectrum = hcop_cube[(slice(None), *selected_pixel)].to_value()
    hcop_x = hcop_cube.spectral_axis.to(kms).to_value()

    noise_cii = 1 # 1 K has been my estimate for a while
    noise_hcop = 0.12 # estimated from the cube, lower than the original 0.5 due to smoothing to CII beam and rebinning to CII channels
    # cps2.plot_noise_and_vlims(ax_cii_spec, noise_cii, vel_lims)
    # cps2.plot_noise_and_vlims(ax_hcop_spec, noise_hcop, vel_lims)
    # for ax in [ax_cii_spec, ax_hcop_spec]:
    #     ax.set_xlim(20, 30)

    # Options for the models
    fixedstd = True
    tiestd = True
    untieciistd = True
    fixedmean = True
    stddev_hcop = 0.55
    # Setup models
    g0 = cps2.models.Gaussian1D(amplitude=10, mean=24.5, stddev=stddev_hcop,
        bounds={'amplitude': (0, None), 'mean': (20, 30)})
    g1 = g0.copy()
    g1.mean = 25.5
    g2 = g0.copy()
    g2.mean = 26
    g = g0 + g1 + g2

    # The input/output for the fitting
    if which_line ==  'cii':
        x_arr, y_arr = cii_x, cii_spectrum
        e_arr = np.full(cii_spectrum.size, noise_cii)
    elif which_line == 'hcop':
        x_arr, y_arr = hcop_x, hcop_spectrum
        e_arr = np.full(hcop_spectrum.size, noise_hcop)

    def helper_chisq(p, model, fpn):
        """
        :param p: array of whatever parameters are in this model
        :param model: a template of the model to which p are the fittable parameters
            The model need not have its values set to the p values already;
            this function will take care of that
        :param fpn: fittable_param_names list
        :returns: the REDUCED chi squared
            May return positive infinity if the parameter values are illegal
        """
        # Penalize for out-of-bounds parameter values
        for param_val, param_name in zip(p, fpn):
            # Illegal negative values
            if ('amplitude' in param_name or 'stddev' in param_name) and (param_val < 0):
                return np.inf
            # Unreasonably high linewidth
            elif ('stddev' in param_name) and (param_val > 10):
                return np.inf
            # Unreasonably high amplitude
            elif ('amplitude' in param_name) and (param_val > 200):
                return np.inf
            # Unreasonable line center
            elif ('mean' in param_name) and (param_val<15 or param_val>35):
                return np.inf
        set_fittable_parameters(p, model, fpn)
        y_model = model(x_arr)
        chisq = np.sum( (y_arr - y_model)**2. / (e_arr**2) )
        dof = y_arr.size - len(p)
        return chisq / dof

    # Prepare for the fitting, do any necessary pre-fitting
    # HCO+ in either case (need info for CII if doing CII fit)
    fitter = cps2.fitting.LevMarLSQFitter(calc_uncertainties=True)
    if tiestd:
        cps2.tie_std_models(g)
    if fixedstd:
        cps2.fix_std(g)

    import emcee

    # Try fitting either line first and using the result as the
    # initial guess center
    # Simple HCO+ fit just for the info
    g_fit_hcop = fitter(g, hcop_x, hcop_spectrum, weights=np.full(hcop_spectrum.size, 1.0/noise_hcop))
    if which_line == 'cii':
        # Now do the rest of the CII fitting with emcee using info from HCO+ fit
        if fixedmean:
            g = g_fit_hcop.copy()
            print("FIXING MEANS TO HCO+ MEANS:")
            print(g)
            cps2.fix_mean(g)
        if untieciistd:
            cps2.tie_std_models(g, untie=True)
        cps2.unfix_std(g)
        g_fit_cii = fitter(g, cii_x, cii_spectrum, weights=np.full(cii_spectrum.size, 1.0/noise_cii))
        g = g_fit_cii.copy()
    elif which_line == 'hcop':
        g = g_fit_hcop.copy()
    ########################
    # This list won't change within a single run of the outer function
    fittable_param_names = get_fittable_param_names(g)
    # Set up emcee parameters
    niter, nburn = 2000, 1750
    nwalkers, ndim = 15, len(fittable_param_names)
    print(f"N_BURN {nburn}, N_ITER {niter}")
    print(f"N_WALKERS {nwalkers}, N_DIM {ndim}")
    print("FITTABLE PARAMETERS:")
    print(fittable_param_names)
    p0_single = get_fittable_parameters(g, fittable_param_names)
    print("P0 centers:")
    print(p0_single)
    # Get initial spread for each type of parameter
    def param_scale_helper(param_name, initial_value):
        if 'mean' in param_name:
            return 1
        elif 'amplitude' in param_name:
            return max(initial_value, 1)*0.2
        elif 'stddev' in param_name:
            return initial_value*0.1
        else:
            raise RuntimeError(f"Parameter name not recognized: {param_name}")
    p0_param_scale = [param_scale_helper(param_name, init_val) for param_name, init_val in zip(fittable_param_names, p0_single)]
    # Need to make p0 for each walker
    rng = np.random.default_rng()
    p0 = rng.normal(loc=p0_single, scale=p0_param_scale, size=(nwalkers, ndim))
    print("P0 samples for entry into MCMC:")
    print(p0)
    # Run the emcee
    lnposterior = make_ln_posterior_fn(g, helper_chisq)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, args=[fittable_param_names])
    sampler.run_mcmc(p0, niter+nburn)
    # samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    samples = sampler.get_chain(flat=True, discard=nburn)
    # print(samples)
    sample_chisqs = -1*sampler.get_log_prob(flat=True, discard=nburn)
    ########################
    chisq_max = 2.5
    def range_helper(param_name):
        if 'mean' in param_name:
            return (23, 27)
        elif 'amplitude' in param_name:
            return (0, np.max(y_arr))
        elif 'stddev' in param_name:
            return (0, 2.1)
        elif 'chisq' in param_name:
            return (np.min(sample_chisqs), chisq_max)
        else:
            raise RuntimeError("nooooo")

    #### NOW TRY RUNNING THIS!
    # for i, param_name in enumerate(fittable_param_names):
    #     ax = plt.subplot(331+i)
    #range_helper##################
    #     ax.hist(samples[:, i], range=lims)
    #     ax.set_xlabel(param_name)
    # ax = plt.subplot(339)
    # ax.hist(sample_chisqs)
    # plt.show()
    # return
    result = np.concatenate([samples, sample_chisqs[:, np.newaxis]], axis=1)
    print(result.shape)
    result_chisq = result[sample_chisqs < chisq_max]
    print(result_chisq.shape)
    names = fittable_param_names + ['chisq']
    import corner
    fig = plt.figure(figsize=(15, 15))
    corner.corner(result_chisq, labels=names,
        range=[range_helper(param_name) for param_name in names])
    # plt.show()
    # plt.savefig(f'/home/ramsey/Pictures/2021-12-15-work/emcee-3p-{pixel_name}_{which_line}.png')
    # np.savetxt(f"/home/ramsey/Pictures/2021-12-15-work/emcee-3p-{pixel_name}_{which_line}.txt", result_chisq)








def investigate_emcee_result(which_line):
    """
    December 15, 2021
    Check out the emcee results
    """
    # pixel_name = "bluest-component"
    pixel_name = "wide-profile"
    result_chisq = np.loadtxt(f"/home/ramsey/Pictures/2021-12-15-work/emcee-3p-{pixel_name}_{which_line}.txt")
    # Only load in CII if we need it
    if which_line == 'cii':
        cii_cube = cube_utils.CubeData("sofia/M16_CII_pillar1_BGsubtracted.fits").data
    regrid = True # Just in case I want to switch back to regular resolution? doesn't hurt
    fn = f"carma/M16.ALL.hcop.sdi.cm.subpv{'.SOFIAbeam.regrid' if regrid else ''}.fits"
    hcop_cube = cube_utils.CubeData(fn).convert_to_K().data.with_spectral_unit(kms)
    hcop_flat_wcs = hcop_cube[0, :, :].wcs

    # sky_regions = regions.Regions.read(catalog.utils.search_for_file("catalogs/pillar1_emissionpeaks.hcopregrid.moreprecise.reg")) # order appears to be [HCO+, CII]
    # sky_regions = regions.Regions.read(catalog.utils.search_for_file("catalogs/p1_threads_pathsandpoints.reg")) # order appears to be North-E, North-W, South-E, South-W
    sky_regions = regions.Regions.read(catalog.utils.search_for_file("catalogs/pillar1_pointsofinterest.reg")) # order is wide profile, blue tail, north of west thread, bluest component
    pixel_coords = [tuple(round(x) for x in reg.to_pixel(hcop_flat_wcs).center.xy[::-1]) for reg in sky_regions] # converted to (i, j) tuples
    selected_pixel = pixel_coords[0]

    # Start with one pixel, just fit that first
    assert regrid
    if which_line == 'cii':
        cii_spectrum = cii_cube[(slice(None), *selected_pixel)].to_value()
        cii_x = cii_cube.spectral_axis.to(kms).to_value()
    hcop_spectrum = hcop_cube[(slice(None), *selected_pixel)].to_value()
    hcop_x = hcop_cube.spectral_axis.to(kms).to_value()
    # Noise
    noise_cii = 1 # 1 K has been my estimate for a while
    noise_hcop = 0.12 # estimated from the cube, lower than the original 0.5 due to smoothing to CII beam and rebinning to CII channels
    # The input/output for the fitting
    if which_line ==  'cii':
        x_arr, y_arr = cii_x, cii_spectrum
        e_arr = np.full(cii_spectrum.size, noise_cii)
        e_level = noise_cii
    elif which_line == 'hcop':
        x_arr, y_arr = hcop_x, hcop_spectrum
        e_arr = np.full(hcop_spectrum.size, noise_hcop)
        e_level = noise_hcop
    # Options for the models
    fixedstd = True
    tiestd = True
    untieciistd = True
    fixedmean = True
    stddev_hcop = 0.55

    # Setup models
    g0 = cps2.models.Gaussian1D(amplitude=10, mean=24.5, stddev=stddev_hcop,
        bounds={'amplitude': (0, None), 'mean': (20, 30)})
    g1 = g0.copy()
    g1.mean = 25.5
    g2 = g0.copy()
    g2.mean = 26
    g = g0 + g1 + g2
    fitter = cps2.fitting.LevMarLSQFitter(calc_uncertainties=True)
    if tiestd:
        cps2.tie_std_models(g)
    if fixedstd:
        cps2.fix_std(g)
    g_fit_hcop = fitter(g, hcop_x, hcop_spectrum, weights=np.full(hcop_spectrum.size, 1.0/noise_hcop))
    if which_line == 'cii':
        # Now do the rest of the CII fitting with emcee using info from HCO+ fit
        if fixedmean:
            g = g_fit_hcop.copy()
            cps2.fix_mean(g)
        if untieciistd:
            cps2.tie_std_models(g, untie=True)
        cps2.unfix_std(g)
        g_fit_cii = fitter(g, cii_x, cii_spectrum, weights=np.full(cii_spectrum.size, 1.0/noise_cii))
        g = g_fit_cii.copy()
    elif which_line == 'hcop':
        g = g_fit_hcop.copy()
    fittable_param_names = get_fittable_param_names(g)
    # Now grab an emcee sample

    sorted_chisqs = np.argsort(result_chisq[:, 6])
    chosen_chisqs = sorted_chisqs[:25]
    print(chosen_chisqs)
    print(result_chisq[chosen_chisqs, 6])
    # p_sample = np.where(result_chisq[:, 6] < 1.5)
    # print(p_sample)
    # p_sample = result_chisq[p_sample][0]
    # p_sample = result_chisq[current_index]
    for current_chisq_idx in chosen_chisqs:
        p_sample = result_chisq[current_chisq_idx]
        print(p_sample)
        # print(fittable_param_names)
        # print(p_sample)
        set_fittable_parameters(p_sample, g, fittable_param_names)
        # print(g)
        fig = plt.figure(figsize=(15, 9))
        ax = plt.subplot(111)
        cps2.plot_noise_and_vlims(ax, e_level, (23, 26))
        cps2.plot_everything_about_models(ax, x_arr, y_arr, g, dy=-0.08, noise=e_level, dof=(y_arr.size - len(fittable_param_names)))
        ax.set_xlim((20, 30))
        y_model = g(x_arr)
        chisq = np.sum((y_model - y_arr)**2 / (e_arr**2))
        # print(y_model - y_arr)
        # print("000")
        # print(e_arr)
        dof = y_arr.size - len(fittable_param_names)
        print("CHISQ", chisq)
        print("REDUCED CHISQ", chisq/dof)
        # plt.show()
        # plt.savefig(f'/home/ramsey/Pictures/2021-12-15-work/all_cii_emcee_results/emcee-3p-investigation-{pixel_name}_{which_line}_chisqlow_IDX{current_chisq_idx:05d}.png')


def plot_selected_hcop_spectra_fits():
    """
    January 13, 2022
    Intended to be the "final form" figure for a lot of the HCO+ spectra fitting
    done in this file. I selected a handful of points (pillar1_pointsofinterest_v2.reg)
    and will just throw all of those plots into this figure. It'll be messy
    but I think it'll ultimately be a nice figure
    There are 8 points selected so I can maybe do a 3x3 grid with the reference
    at the center
    I'm copying a lot from m16_threads.figure_for_hcop_linewidths() since that's
    nearly the same as what I want to do
    """
    # Get cube
    cube = cps2.cutout_subcube(length_scale_mult=2.5, data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.fits")
    # Get regions and convert to pixel coords
    reg_filename_short = "catalogs/pillar1_pointsofinterest_v2.reg"
    sky_regions = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    pixel_coords = [tuple(round(x) for x in reg.to_pixel(cube[0, :, :].wcs).center.xy[::-1]) for reg in sky_regions]
    assert len(pixel_coords) == 8

    # Set up axes
    fig = plt.figure(figsize=(12, 12))
    ax_spec_list = []
    for i in range(3):
        for j in range(3):
            if (i == 0) and (j == 2):
                # Image Axes
                ax_img = plt.subplot2grid((3, 3), (i, j), projection=cube[0, :, :].wcs)
            else:
                ax_spec_list.append(plt.subplot2grid((3, 3), (i, j)))
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
    g0 = cps2.models.Gaussian1D(amplitude=10, mean=23.5, stddev=0.47,
        bounds={'amplitude': (0, None), 'mean': (22, 30), 'stddev': (0.3, 1.5)})
    g1 = g0.copy()
    g1.mean = 24.5
    g2 = g0.copy()
    g2.mean = 25.5
    g_init = g0 + g1 + g2
    cps2.fix_std(g_init)
    dof = 6
    fit_stub = "3cfixedwidth"
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
        pad = 10
        dj = pad if (idx+1 not in [1, 3, 6]) else -pad
        di = pad if (idx+1 not in [4, 7, 5, 8]) else -pad
        ax_img.text(j+dj, i+di, str(idx+1), color='w', fontsize=12, ha='center', va='center')
        ax_spec_list[idx].text(0.9, 0.9, str(idx+1), color='k', fontsize=14, ha='center', va='center', transform=ax_spec_list[idx].transAxes)
        # Extract, fit, and plot spectrum
        spectrum = cube[:, i, j].to_value()
        g_fit = fitter(g_init, spectral_axis, spectrum, weights=weights)
        cps2.plot_noise_and_vlims(ax_spec_list[idx], noise, None)
        cps2.plot_everything_about_models(ax_spec_list[idx], spectral_axis, spectrum, g_fit, noise=noise, dof=(spectral_axis.size - dof))
        # ax_spec_list[idx].set_xlabel("Velocity (km/s)")
        # ax_spec_list[idx].set_ylabel("HCO+ line intensity (K)")
        ax_spec_list[idx].set_ylim([-2, 22])
        ax_spec_list[idx].xaxis.set_ticks_position('both')
        ax_spec_list[idx].yaxis.set_ticks_position('both')
        ax_spec_list[idx].xaxis.set_tick_params(direction='in', which='both')
        ax_spec_list[idx].yaxis.set_tick_params(direction='in', which='both')
        if idx+1 not in [1, 3, 6]:
            # These are NOT on the left edge, so no y axis labelling
            ax_spec_list[idx].yaxis.set_ticklabels([])
        if idx+1 not in [6, 7, 8]:
            # These are NOT on the bottom, so no x axis labellling
            ax_spec_list[idx].xaxis.set_ticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(f"/home/ramsey/Pictures/2022-01-13-work/hcop_selected_spectra_thru_head_{fit_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=f"points from {reg_filename_short}",
            file=__file__, func='plot_selected_hcop_spectra_fits'))


def plot_selected_CII_spectra_fits():
    """
    January 20, 2022
    Intended to be the "final form" figure the CII fitting
    I don't know how I want to do these comparisons yet.
    Right now I have just copied all of plot_selected_hcop_spectra_fits()
    and will edit that into something.
    """
    # Get cube
    # cube = cps2.cutout_subcube(length_scale_mult=2.5, data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.fits")
    cube = cps2.cutout_subcube(length_scale_mult=2.5)
    cii_background_spectrum = cps2.get_cii_background()
    cube = cube - cii_background_spectrum[:, np.newaxis, np.newaxis]
    # Get regions and convert to pixel coords
    reg_filename_short = "catalogs/pillar1_pointsofinterest_v2.reg"
    sky_regions = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    pixel_coords = [tuple(round(x) for x in reg.to_pixel(cube[0, :, :].wcs).center.xy[::-1]) for reg in sky_regions]
    assert len(pixel_coords) == 8

    # Set up axes
    fig = plt.figure(figsize=(12, 12))
    ax_spec_list = []
    for i in range(3):
        for j in range(3):
            if (i == 0) and (j == 2):
                # Image Axes
                ax_img = plt.subplot2grid((3, 3), (i, j), projection=cube[0, :, :].wcs)
            else:
                ax_spec_list.append(plt.subplot2grid((3, 3), (i, j)))
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
    g0 = cps2.models.Gaussian1D(amplitude=10, mean=23.5, stddev=0.75, # 0.47 for HCO+ native
        bounds={'amplitude': (0, None), 'mean': (22, 30), 'stddev': (0.3, 2)})
    g1 = g0.copy()
    g1.mean = 24.5
    g2 = g0.copy()
    g2.mean = 25.5
    g_init = g0 + g1 + g2
    # cps2.fix_std(g_init)
    dof = 9
    fit_stub = "3cfree75"
    # Fitter
    fitter = cps2.fitting.LevMarLSQFitter(calc_uncertainties=True)
    # Spectral axis
    spectral_axis = cube.spectral_axis.to_value()
    # Noise
    ############
    # noise = 0.546 # HCO+ native resolution
    noise = 1. # CII
    ############
    weights = np.full(spectral_axis.size, 1.0/noise)
    # Loop through the 3 pixels and plot things
    for idx, (i, j) in enumerate(pixel_coords):
        # Label the point on the reference image
        ax_img.plot([j], [i], 'o', markersize=5, color='w')
        ##############
        # pad = 10 # HCO+ native resolution
        pad = 1.5 # CII
        ##############
        dj = pad if (idx+1 not in [1, 3, 6]) else -pad
        di = pad if (idx+1 not in [4, 7, 5, 8]) else -pad
        ax_img.text(j+dj, i+di, str(idx+1), color='w', fontsize=12, ha='center', va='center')
        ax_spec_list[idx].text(0.9, 0.9, str(idx+1), color='k', fontsize=14, ha='center', va='center', transform=ax_spec_list[idx].transAxes)
        # Extract, fit, and plot spectrum
        spectrum = cube[:, i, j].to_value()
        g_fit = fitter(g_init, spectral_axis, spectrum, weights=weights)
        cps2.plot_noise_and_vlims(ax_spec_list[idx], noise, None)
        cps2.plot_everything_about_models(ax_spec_list[idx], spectral_axis, spectrum, g_fit, noise=noise, dof=(spectral_axis.size - dof))
        # ax_spec_list[idx].set_xlabel("Velocity (km/s)")
        # ax_spec_list[idx].set_ylabel("HCO+ line intensity (K)")
        ax_spec_list[idx].set_xlim([15, 35])
        ################
        # -2, 22 is good for HCO+ at native resolution
        # ax_spec_list[idx].set_ylim([-2, 22])
        ax_spec_list[idx].set_ylim([-5, 45])
        ################
        ax_spec_list[idx].xaxis.set_ticks_position('both')
        ax_spec_list[idx].yaxis.set_ticks_position('both')
        ax_spec_list[idx].xaxis.set_tick_params(direction='in', which='both')
        ax_spec_list[idx].yaxis.set_tick_params(direction='in', which='both')
        if idx+1 not in [1, 3, 6]:
            # These are NOT on the left edge, so no y axis labelling
            ax_spec_list[idx].yaxis.set_ticklabels([])
        if idx+1 not in [6, 7, 8]:
            # These are NOT on the bottom, so no x axis labellling
            ax_spec_list[idx].xaxis.set_ticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(f"/home/ramsey/Pictures/2022-01-20-work/cii_selected_spectra_thru_head_{fit_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=f"points from {reg_filename_short}",
            file=__file__, func='plot_selected_CII_spectra_fits'))



def plot_selected_hcop_AND_CII_spectra():
    """
    January 20, 2022
    Final form for comparing CII and HCO+. No fitting right now...
    """
    # Get cube
    hcop_cube = cps2.cutout_subcube(length_scale_mult=None, data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.SOFIAbeam.regrid.fits")
    cii_cube = cps2.cutout_subcube(length_scale_mult=4)
    cii_background_spectrum = cps2.get_cii_background()
    cii_cube = cii_cube - cii_background_spectrum[:, np.newaxis, np.newaxis]
    # Get regions and convert to pixel coords
    reg_filename_short = "catalogs/pillar1_pointsofinterest_v2.reg"
    sky_regions = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    # pixel_coords = [tuple(round(x) for x in reg.to_pixel(cube[0, :, :].wcs).center.xy[::-1]) for reg in sky_regions]
    assert len(sky_regions) == 8

    # Set up axes
    fig = plt.figure(figsize=(12, 12))
    ax_spec_list = []
    for i in range(3):
        for j in range(3):
            if (i == 0) and (j == 2):
                # Image Axes
                ax_img = plt.subplot2grid((3, 3), (i, j), projection=hcop_cube[0, :, :].wcs)
            else:
                ax_spec_list.append(plt.subplot2grid((3, 3), (i, j)))
    # Make moment 0 to plot
    vel_lims = (19*kms, 27*kms)
    mom0 = hcop_cube.spectral_slab(*vel_lims).moment0()
    ax_img.imshow(mom0.to_value(), origin='lower', vmin=0, cmap='plasma')
    for coord in ax_img.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')
    # Loop through the pixels and plot things
    for idx, sky_reg in enumerate(sky_regions):
        # Convert the sky_reg to pixel coords for each cube
        hcop_ij = tuple(round(x) for x in sky_reg.to_pixel(hcop_cube[0, :, :].wcs).center.xy[::-1])
        cii_ij = tuple(round(x) for x in sky_reg.to_pixel(cii_cube[0, :, :].wcs).center.xy[::-1])
        assert cii_ij == hcop_ij
        i, j = hcop_ij
        # Label the point on the reference image
        ax_img.plot([j], [i], 'o', markersize=5, color='w')
        ##############
        # pad = 10 # HCO+ native resolution
        pad = 2 # CII or HCO+ regrid
        ##############
        dj = pad if (idx+1 not in [1, 3, 6]) else -pad
        di = pad if (idx+1 not in [4, 7, 5, 8]) else -pad
        ax_img.text(j+dj, i+di, str(idx+1), color='w', fontsize=12, ha='center', va='center')
        ax_spec_list[idx].text(0.9, 0.9, str(idx+1), color='k', fontsize=14, ha='center', va='center', transform=ax_spec_list[idx].transAxes)
        # Extract, fit, and plot spectrum
        hcop_spectrum = hcop_cube[:, i, j].to_value()
        cii_spectrum = cii_cube[:, i, j].to_value()
        ax_spec_list[idx].plot(hcop_cube.spectral_axis.to_value(), hcop_spectrum/22, color=marcs_colors[0], label='HCO+')
        ax_spec_list[idx].plot(cii_cube.spectral_axis.to_value(), cii_spectrum/40, color=marcs_colors[1], label='CII')
        # ax_spec_list[idx].set_xlabel("Velocity (km/s)")
        # ax_spec_list[idx].set_ylabel("HCO+ line intensity (K)")
        ax_spec_list[idx].set_xlim([15, 35])
        ################
        # -2, 22 is good for HCO+ at native resolution
        # ax_spec_list[idx].set_ylim([-2, 22])
        ax_spec_list[idx].set_ylim([-0.05, 1.05])
        ################
        ax_spec_list[idx].xaxis.set_ticks_position('both')
        ax_spec_list[idx].yaxis.set_ticks_position('both')
        ax_spec_list[idx].xaxis.set_tick_params(direction='in', which='both')
        ax_spec_list[idx].yaxis.set_tick_params(direction='in', which='both')
        if idx+1 == 4:
            ax_spec_list[idx].legend(loc='upper left')
        if idx+1 not in [1, 3, 6]:
            # These are NOT on the left edge, so no y axis labelling
            ax_spec_list[idx].yaxis.set_ticklabels([])
        if idx+1 not in [6, 7, 8]:
            # These are NOT on the bottom, so no x axis labellling
            ax_spec_list[idx].xaxis.set_ticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig("/home/ramsey/Pictures/2022-01-20-work/cii_and_hcop_selected_spectra_thru_head.png",
        metadata=catalog.utils.create_png_metadata(title=f"points from {reg_filename_short}",
            file=__file__, func='plot_selected_hcop_AND_CII_spectra'))



def make_3d_fit_viz_in_2d(n_submodels=3, line='hcop', version=None):
    """
    April 19, 2022
    I want to make a sort of 3d viz but for the paper. So not really 3d.
    Probably just two 2d histograms, one along RA and one along Dec.
    Copying a lot of setup boilerplate from
    cube_pixel_spectra_2.investigate_template_model_fit (written Nov 17, 2021)
    Hopefully this looks okay

    I have an idea for a third panel, where I can just make an RA/Dec map of
    the number of acceptable component fits (so amplitude > some cutoff, etc)
    to show where the model fit "flips" from 3 component to 2 component.
    I could use that in the paper as some evidence for a different type of
    gas interaction (or at least observational obstacle) in that region.

    Updated/edited a little September 1, 2022
    """
    if line[:4] == 'hcop':
        directory = "carma"
        if version is None:
            if line == 'hcop':
                version = 2
            elif line == 'hcop_regrid':
                version = 3
    else:
        directory = 'sofia'
        if version is None:
            version = 1
    filename_stub = f"{directory}/models/gauss_fit_{line}_{n_submodels}G_v{version}"
    param_fn = catalog.utils.search_for_file(filename_stub+".param.fits")
    # resid_fn = catalog.utils.search_for_file(filename_stub+".resid.fits")
    # model_fn = catalog.utils.search_for_file(filename_stub+".model.fits")
    hdul = fits.open(param_fn)
    print(list(hdu.header['EXTNAME'] for hdu in hdul if 'EXTNAME' in hdu.header))
    # resid_cube = cube_utils.SpectralCube.read(resid_fn)
    # model_cube = cube_utils.SpectralCube.read(model_fn)
    means = []
    amplitudes = []
    shape = hdul[1].data.shape
    # ii, jj = tuple(x for x in np.mgrid[0:shape[0], 0:shape[1]])
    ii, jj = np.mgrid[0:shape[0], 0:shape[1]]
    i_axis = np.arange(shape[0])
    j_axis = np.arange(shape[1])

    i_cube = []
    j_cube = []

    if n_submodels > 1:
        for k in range(n_submodels):
            means.append(hdul[f'mean_{k}'].data[:])
            amplitudes.append(hdul[f'amplitude_{k}'].data[:])
            i_cube.append(ii)
            j_cube.append(jj)
        i_cube = np.array(i_cube)
        j_cube = np.array(j_cube)
        means = np.array(means)
        amplitudes = np.array(amplitudes)
    else:
        means = hdul['mean'].data[np.newaxis, :, :]
        amplitudes = hdul['amplitude'].data[np.newaxis, :, :]
        i_cube = ii[np.newaxis, :, :]
        j_cube = jj[np.newaxis, :, :]

    # means = np.array(means)
    # amplitudes = np.array(amplitudes)
    # i_array = np.array(i_array)
    # j_array = np.array(j_array)
    if line == 'hcop':
        amp_cutoff = 2.5
    elif line == 'hcop_regrid':
        amp_cutoff = 0.6
    else:
        amp_cutoff = 5
    amp_mask = amplitudes > amp_cutoff # about 5sigma


    # means = means[amp_mask]
    # amplitudes = amplitudes[amp_mask]
    # i_array = i_array[amp_mask]
    # j_array = j_array[amp_mask]

    # im1 = ax1.hist2d(j_array, means, bins=64)[3]

    if line[:4] == 'hcop':
        n_bins = 128
    else:
        n_bins = 32

    img_ra_vel = np.zeros((n_bins, shape[1]))
    vel_limits = (22, 28)
    for j in j_axis:
        velocities_in_j = means[:, :, j].ravel()
        amplitudes_in_j = amplitudes[:, :, j].ravel()
        vel_hist_in_j, vel_edges = np.histogram(velocities_in_j[amplitudes_in_j > amp_cutoff], bins=n_bins, range=vel_limits)
        img_ra_vel[:, j] = vel_hist_in_j
    vel_centers = (vel_edges[:-1] + vel_edges[1:])/2
    vel_delta = vel_edges[1] - vel_edges[0]


    fig = plt.figure(figsize=(6, 8))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, projection=WCS(hdul[1].header))


    im1 = ax1.imshow(img_ra_vel, origin='lower', aspect=(shape[1]/(vel_limits[1]-vel_limits[0])), extent=[0, shape[1], vel_limits[0], vel_limits[1]])
    fig.colorbar(im1, ax=ax1, label='$N$ valid components')
    # ax1.set_xlabel("RA")
    ax1.set_ylabel(f"Velocity ({kms.to_string('latex_inline')})")
    ax1.xaxis.set_ticks([])
    # ax.invert_xaxis()


    img_ra_dec = np.sum((amplitudes > amp_cutoff).astype(int), axis=0)
    im2 = ax2.imshow(img_ra_dec, origin='lower', vmin=0, vmax=3)
    cbar = fig.colorbar(im2, ax=ax2, ticks=list(range(0, 4)), label='$N$ valid components')
    ax2.set_xlabel("Right Ascension")
    ax2.set_ylabel("Declination")
    ax2.tick_params(axis='x', direction='in')

    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.12, right=0.98, hspace=0.07)


    # im3 = ax3.hist2d(means, i_array, bins=64)[3]
    # fig.colorbar(im3, ax=ax3)
    # ax3.set_xlabel("Velocity (km/s)")
    # ax3.set_ylabel("Dec")

    dpi = 300
    dpi_stub = "" if dpi==100 else f"_dpi{dpi}"

    # plt.show()
    # 2022-09-01,13, 2023-07-25
    savename = os.path.join(catalog.utils.todays_image_folder(), f"p1_3d_viz_in_2d_{line}_{n_submodels}p{dpi_stub}")
    fig.savefig(f"{savename}.png",
        metadata=catalog.utils.create_png_metadata(title="projection of grid fit",
            file=__file__, func="make_3d_fit_viz_in_2d"),
            dpi=dpi)
    # elif True:
    #     from mayavi import mlab
    #     mlab.figure(bgcolor=(0.2, 0.2, 0.2), fgcolor=(0.93, 0.93, 0.93), size=(800, 700))
    #     mlab.axes(ranges=[0, shape[1], 0, shape[0], 20, 30],
    #         xlabel='j (ra)', ylabel='i (dec)', zlabel='velocity (km/s)', nb_labels=10,
    #         line_width=19)
    #     kwargs = dict(mode='cube', colormap='jet',
    #         scale_mode='none', scale_factor=0.7, opacity=0.2)
    #     mlab.points3d(j_array, i_array, -1*means*(30 if line=='hcop' else 4), amplitudes, **kwargs)
    #     mlab.show()


def correlations_between_carma_molecule_intensities(molX, molY, integrated=False):
    """
    April 20, 2022 (at 4:23 PM)
    Correlate the intensities of the carma molecules so that I can say
    stuff in the paper
    """
    fn_template = lambda x : f"carma/M16.ALL.{x}.sdi.cm.subpv.fits"
    cubeX, cubeY = (cps2.cutout_subcube(data_filename=fn_template(x), length_scale_mult=None) for x in (molX, molY))
    arrX, arrY = (cube.unmasked_data[:].to_value().flatten() for cube in (cubeX, cubeY))
    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    ax.hist2d(arrX, arrY, bins=128, norm=LogNorm())
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(xlim, xlim)
    # ax.scatter(arrX, arrY, marker='.', alpha=0.5)
    ax.set_xlabel(f"{molX} intensity (K)")
    ax.set_ylabel(f"{molY} intensity (K)")
    plt.show()


def generate_n2hp_frequency_axis(debug=False):
    """
    Per the text file in Marc's April 28 email, the correlator setup for the
    original N2H+ cube (the 0.08 km/s resolution version)
    Refer to window 1 (first column)

    rest frequency     :  93.173505  93.174000  89.188518  88.631847  87.407000  87.284000
    starting channel   :          1        320        639        958       1277       1596
    number of channels :        319        319        319        319        319        319
    starting frequency :  93.168192  92.679145  89.183598  88.626982  87.402256  87.279268
    frequency interval :  -0.000024  -0.000024  -0.000024  -0.000024  -0.000024  -0.000024
    band width         :  -0.007788  -0.007788  -0.007788  -0.007788  -0.007788  -0.007788
    starting velocity  :     12.511   1587.662     11.953     11.871     11.687     11.668
    ending velocity    :     37.491   1612.643     38.050     38.131     38.315     38.334
    velocity interval  :      0.079      0.079      0.082      0.083      0.084      0.084

    rest frequency     :  97.980968  97.980968  89.188518  88.631847  87.407000  87.284000
    starting channel   :       1915       2234       2553       2872       3191       3510
    number of channels :        319        319        319        319        319        319
    starting frequency :  97.477474  97.966521 101.462068 102.018684 103.243410 103.366398
    frequency interval :   0.000024   0.000024   0.000024   0.000024   0.000024   0.000024
    band width         :   0.007788   0.007788   0.007788   0.007788   0.007788   0.007788
    starting velocity  :   1535.979     39.620 -41260.724 -45285.534 -54321.847 -55243.292
    ending velocity    :   1512.225     15.865 -41286.821 -45311.794 -54348.475 -55269.958
    velocity interval  :     -0.075     -0.075     -0.082     -0.083     -0.084     -0.084
    """
    # Correlator setup (Marc's text file)
    rest_freq_window_1 = 93.173505
    n_channels = 319
    starting_frequency = 93.168192
    frequency_interval = -0.000024
    band_width = -0.007788
    starting_velocity = 12.511
    ending_velocity = 37.491
    velocity_interval = 0.079
    # Derive the axis
    frequency_array = [starting_frequency + i*frequency_interval for i in range(n_channels)] * u.GHz
    if debug:
        velocity_array = frequency_array.to(u.km/u.s, equivalencies=u.doppler_radio(rest_freq_window_1*u.GHz)) # 93.1721 is closest
        print("band_width/frequency_interval = ", band_width/frequency_interval)
        print("calculated starting, ending velocities: ", end='')
        print(velocity_array[0], ', ', velocity_array[-1])
        print("calculated velocity range: ", velocity_array[-1] - velocity_array[0])
        print("velocity range from starting_velocity, ending_velocity: ", ending_velocity-starting_velocity)
        print("calculated velocity interval", np.diff(velocity_array.to_value()).mean())
    else:
        return frequency_array


def generate_n2hp_line_table():
    """
    May 11, 2022
    Get the Aij, weights, frequencies, stuff like that for all these lines
    N2H+ transition data from https://home.strw.leidenuniv.nl/~moldata/datafiles/n2h+_hfs.dat
    The "rest frequency" line for window 1 is the J=1-0, F1=2-1, F=2-1
    The line at -9 km/s from that line is J=1-0, F1=0-1, F=1-2
    """
    # Line data (LAMDA)
    line_columns = "TRANS + U + L + A(s^-1) + FREQ(GHz) + E_u/k(K)".split(" + ")
    raw_line_data = """
     1     4     2  3.896E-05          93.17161030     4.47
     2     5     2  6.040E-06          93.17190510     4.47
     3     5     3  3.293E-05          93.17190510     4.47
     4     6     2  4.652E-06          93.17204230     4.47
     5     6     1  1.983E-05          93.17204240     4.47
     6     6     3  1.448E-05          93.17204240     4.47
     7     7     2  3.293E-05          93.17346770     4.47
     8     7     3  6.040E-06          93.17346770     4.47
     9     8     3  3.896E-05          93.17376420     4.47
    10     9     1  1.219E-05          93.17395870     4.47
    11     9     2  2.525E-05          93.17395870     4.47
    12     9     3  1.533E-06          93.17395870     4.47
    13    10     1  6.944E-06          93.17625430     4.47
    14    10     2  9.068E-06          93.17625430     4.47
    15    10     3  2.296E-05          93.17625430     4.47
    """
    line_table = pd.read_table(StringIO(raw_line_data), sep='\s+', header=None, names=line_columns, index_col=0)
    state_columns = "LEVEL + ENERGIES(cm^-1) + WEIGHT + J_F1_F".split(" + ")
    raw_state_data = """
    1     0.000000000   1.0   0_1_0
    2     0.000000000   3.0   0_1_1
    3     0.000000000   5.0   0_1_2
    4     3.107870389   1.0   1_1_0
    5     3.107880222   5.0   1_1_2
    6     3.107884799   3.0   1_1_1
    7     3.107932345   5.0   1_2_2
    8     3.107942235   7.0   1_2_3
    9     3.107948723   3.0   1_2_1
   10     3.108025296   3.0   1_0_1
    """
    state_table = pd.read_table(StringIO(raw_state_data), sep='\s+', header=None, names=state_columns, index_col=0)
    def jf1f_tuple_to_string(u, l):
        # Convert pair of J_F1_F identifiers to "J=x-y, F1=x-y, F=x-y" form
        quantum_numbers = ('J', 'F1', 'F')
        return ', '.join([f"{x}={ux}-{lx}" for ux, lx, x in zip(u.split('_'), l.split('_'), quantum_numbers)])
    # Make two weight columns using the U and L columns to index the state_table
    line_table['g_u'] = state_table.loc[line_table['U'], 'WEIGHT'].values
    line_table['g_l'] = state_table.loc[line_table['L'], 'WEIGHT'].values
    # Do this again but more complicated, get the J_F1_F entries and combine them into a human-readable string
    get_state_info = lambda line_col_name: state_table.loc[line_table[line_col_name], 'J_F1_F'].values
    line_table['JF1F'] = [jf1f_tuple_to_string(*x) for x in zip(get_state_info('U'), get_state_info('L'))]
    line_table.drop(columns=['U', 'L'], inplace=True)
    del state_table
    return line_table


def plot_n2hp_lines_and_spectrum():
    """
    May 5-12, 2022
    Mostly testing for the fitting
    """
    filename_stub = "carma/n2hp_P1_peak_spectrum.dat"
    filename = catalog.utils.search_for_file(filename_stub)
    data = np.genfromtxt(filename)
    v_axis = data[:, 0] * u.m / u.s
    i_axis = data[:, 1] * u.Jy/u.beam
    freq_axis = generate_n2hp_frequency_axis()

    line_table = generate_n2hp_line_table()
    A_list, freq_list = line_table['A(s^-1)'].values, line_table['FREQ(GHz)'].values
    rest_freq = line_table.loc[7, 'FREQ(GHz)']
    shifted_freq = freq_axis[np.argmax(i_axis)].to_value()
    new_rest_line = line_table.loc[line_table['JF1F'] == 'J=1-0, F1=0-1, F=1-2']
    new_rest_freq = new_rest_line['FREQ(GHz)']
    print(new_rest_line)


    ax1 = plt.subplot(211)
    plt.plot(freq_axis.to(u.GHz), i_axis)
    plt.gca().invert_xaxis()
    print(v_axis.shape, freq_axis.shape)
    ax2 = plt.subplot(212)
    plt.plot(np.array(freq_list) + (shifted_freq - rest_freq), np.array(A_list)*1e5, 'x', markersize=10)
    plt.plot([shifted_freq], [line_table.loc[7, 'A(s^-1)']*1e5], marker='o', mec='k', mfc='b')
    plt.plot([new_rest_freq + shifted_freq - rest_freq], [new_rest_line['A(s^-1)']*1e5], marker='o', mec='k', mfc='orange')
    ax2.set_xlim(ax1.get_xlim())

    ax2.set_xlabel("Frequency, decreasing (GHz)")
    ax2.set_ylabel("$A_{ij}$ ($10^{-5} s^{-1}$)")
    ax1.set_ylabel("Observed brightness (Jy/beam)")

    ax1.xaxis.set_ticklabels([])
    plt.subplots_adjust(hspace=0)
    plt.show()


def fit_n2hp_peak(i=5):
    """
    May 5-12, 2022
    Fit the saved N2H+ peak data. Use the line at like -9 km/s
    """
    filename_stub = f"carma/n2hp_fullres/n2hp_spectrum_pillar1_pointsofinterest_v3_{i-1}.txt.gz"
    filename = catalog.utils.search_for_file(filename_stub)
    data = np.loadtxt(filename)
    v_axis = data[0, :] * u.m / u.s
    i_axis = data[1, :] * u.Jy/u.beam
    freq_axis = generate_n2hp_frequency_axis()

    line_table = generate_n2hp_line_table()
    A_list, freq_list = line_table['A(s^-1)'].values, line_table['FREQ(GHz)'].values
    rest_freq = line_table.loc[7, 'FREQ(GHz)'] * u.GHz
    new_rest_line = line_table.loc[line_table['JF1F'] == 'J=1-0, F1=0-1, F=1-2'].iloc[0]
    new_rest_freq = new_rest_line['FREQ(GHz)'] * u.GHz

    """
    Use the difference in frequency between the two lines to find a velocity offset
    This is an alternative to going from velocity to frequency and back to velocity (more room for error there)
    This way, I can just trust the velocity axis that exists and use only the difference between the known frequencies

    Calculate the difference in velocity using two different rest frequencies:
    Doppler shift (radio) is V(f) = c(f0 - f)/f0
    If V1(f) is the original velocity corresponding to frequency f, given a
    line at rest frequency f1, and V2(f) is that for another line at f2:
    Delta V = V2(f) - V1(f) = c[ (f2-f)/f2 - (f1-f)/f1 ]
    c * Delta V = [ f1(f2 - f) - f2(f1 - f) ] / f1f2
    = [(f1f2 - f1f2) - f1f + f2f]/f1f2
    = f(f2 - f1)/f1f2
    If f1 ~ f2 ~ f, then
    (V2 - V1)/c = (f2 - f1)/f1 = (f2 - f1)/f2
    """
    if False:
        # Some debug to show that the approximation is always good to within 2%
        x = const.c*freq_axis*(new_rest_freq - rest_freq)/(new_rest_freq*rest_freq)
        full_new_vaxis = v_axis + x
        y = const.c*(new_rest_freq - rest_freq)/rest_freq
        print(y)
        approx_new_vaxis = v_axis + y
        diff_axis = ((full_new_vaxis - approx_new_vaxis)/np.abs(v_axis[1] - v_axis[0])).decompose()

    delta_V = const.c * (new_rest_freq - rest_freq)/rest_freq
    new_v_axis = v_axis + delta_V

    mask = new_v_axis < 30*u.km/u.s
    trimmed_v_axis = new_v_axis[mask].to(u.km/u.s).to_value()
    trimmed_i_axis = i_axis[mask].to_value()

    # plt.plot(v_axis, i_axis)
    # plt.show()
    # return

    # plt.plot(new_v_axis, i_axis); plt.title(f"{i}"); plt.show()
    g0 = cps2.models.Gaussian1D(amplitude=3, mean=25., stddev=0.45,
        bounds={'amplitude': (0, None), 'mean': (20, 30)})
    fitter = cps2.fitting.LevMarLSQFitter(calc_uncertainties=True)
    ndim = len(get_fittable_param_names(g0))
    g_fit = fitter(g0, trimmed_v_axis, trimmed_i_axis, weights=np.full(trimmed_i_axis.size, 1.0/cube_utils.onesigmas['n2hp']))

    fig = plt.figure(figsize=(12, 8))
    # ax_img = plt.subplot(121)
    ax_spec = plt.subplot(111)
    cps2.plot_noise_and_vlims(ax_spec, cube_utils.onesigmas['n2hp'], None)
    cps2.plot_everything_about_models(ax_spec, trimmed_v_axis, trimmed_i_axis, g_fit, noise=cube_utils.onesigmas['n2hp'], dof=(trimmed_i_axis.size - ndim))
    plt.show()



def save_n2hp_full_spectra(reg_filename_short=None, index=None):
    """
    May 13, 2022 (Friday the 13th!)
    Save spectra from the large N2H+ cube as .txt.gz file.
    This way I can load and fit them quickly
    """
    if reg_filename_short is None:
        reg_filename_short = "catalogs/pillar1_pointsofinterest_v3.reg"
    sky_regions = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    ### must be run on jupiter or another UMD network computer
    cube_filename = "/n/aurora1/feedback/m16/M16.ALL.n2hp.fullres.sdi.fits"
    cube = cube_utils.SpectralCube.read(cube_filename)
    path_name = catalog.utils.m16_data_path+"carma"
    hdr_savename = os.path.join(path_name, "n2hp_fullres.header")
    cube.header.tofile(hdr_savename, overwrite=True)
    ## save the header using tofile, load using fromfile
    if index is None:
        index_list = list(range(len(sky_regions)))
    else:
        index_list = [index]
    for i in index_list:
        sky_reg = sky_regions[i]
        reg_name = sky_reg.meta['text'].replace(' ', '-')
        pix_coords = tuple(round(x) for x in sky_reg.to_pixel(cube[0, :, :].wcs).center.xy[::-1])
        spectrum = cube[(slice(None), *pix_coords)]
        spectral_axis = cube.spectral_axis
        savename = "n2hp_spectrum_" + reg_filename_short.replace('catalogs/', '').replace('.reg', '') + f"_{i}.txt.gz"
        savename = os.path.join(path_name, savename)
        np.savetxt(savename, np.array([spectral_axis.to_value(), spectrum.to_value()]), header=f"{spectral_axis.unit}, {spectrum.unit}")
        print("saved "+savename)


def save_n2hp_new_vaxis_cube():
    """
    August 24, 2022
    Save a new version of the full resolution N2H+ cube which is has the velocity
    axis correct for the line that's currently at -9 km/s (J=1-0, F1=0-1, F=1-2)
    """
    cube_filename = "/n/aurora1/feedback/m16/M16.ALL.n2hp.fullres.sdi.fits"
    cube = cube_utils.SpectralCube.read(cube_filename)
    path_name = catalog.utils.m16_data_path+"carma"


    v_axis = cube.spectral_axis.to(kms)

    # Following the method in m16_deepdive.fit_n2hp_peak
    line_table = generate_n2hp_line_table()
    rest_freq = line_table.loc[7, 'FREQ(GHz)'] * u.GHz
    new_rest_line = line_table.loc[line_table['JF1F'] == 'J=1-0, F1=0-1, F=1-2'].iloc[0]
    new_rest_freq = new_rest_line['FREQ(GHz)'] * u.GHz

    delta_V = const.c * (new_rest_freq - rest_freq)/rest_freq
    new_v_axis = v_axis + delta_V

    hdr = cube.header
    del hdr['HISTORY']
    hdr['ORIGIN'] = 'rkarim, m16_deepdive.save_n2hp_new_vaxis_cube'
    old_restfrq = hdr['RESTFRQ']
    hdr['HISTORY'] = f"old RESTFRQ value was {old_restfrq} Hz"
    hdr['HISTORY'] = f"Closest line found in LAMDA {line_table.loc[7, 'JF1F']} at {rest_freq}"
    hdr['HISTORY'] = f"New line in LAMDA {new_rest_line['JF1F']} at {new_rest_freq}"
    hdr['HISTORY'] = f"Shifted rest frequency from {rest_freq} to {new_rest_freq}"
    hdr['HISTORY'] = f"by {delta_V}"
    hdr['RESTFRQ'] = (new_rest_freq.to(u.Hz).to_value(), "[Hz] Line rest frequency")
    hdr['RESTFREQ'] = new_rest_freq.to(u.Hz).to_value()
    del hdr['LSTART'], hdr['LSTEP'], hdr['LWIDTH'], hdr['LTYPE']
    hdr['CRVAL3'] = (hdr['CRVAL3'] + delta_V.to_value(), "[m s-1] Coordinate value at reference point")


    print(new_v_axis[88])

    # Mask at < 28.4 km/s
    new_cube = cube_utils.SpectralCube(data=cube.unmasked_data[:], header=hdr, wcs=WCS(hdr))
    new_cube = new_cube[:88, :, :]
    new_cube.write(os.path.join(path_name, "n2hp_fullres_j_10_f1_01_f_12.fits"))


def save_hcop_and_cs_moment_imgs(line='hcop'):
    """
    June 17, 2022
    Save a 32-36 km/s moment img from the fullres hcop and cs maps that Marc uploaded
    Goal here is to look for whatever counterpart to HH 216. Found one in 12CO3-2 and
    I think I see it in HCO+ in ds9. Couldn't ID it in CS in ds9, so I'll see moment0
    Ultimate goal is to overlay these on the HST img in ds9 (reproject too intensive
    in Python) to compare locations
    """
    data_path = "/n/aurora1/feedback/m16"
    filename = f"M16.ALL.{line}.fullres.sdi.fits"
    cube = cube_utils.SpectralCube.read(os.path.join(data_path, filename))
    path_name = catalog.utils.m16_data_path+"carma"
    savename = os.path.join(path_name, f"{line}.mom0.32-36.fits")
    mom0 = cube.spectral_slab(32*kms, 36*kms).moment0()
    mom0.write(savename)


def test_optical_molecular_pillar_2_thing():
    """
    May 24, 2022
    Just a test
    """
    vel_lims_wrongshape = (21, 22)
    vel_lims_rightshape = (22, 23)
    full_vel_lims = (19, 25)
    lines = ['hcop', 'cs', 'n2hp']
    for line in lines:
        cube = cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line], length_scale_mult=None)
        if line == 'n2hp':
            ...
        else:
            for j, vel_lims in enumerate((vel_lims_rightshape, vel_lims_wrongshape)):
                stub = ['shape', 'complement'][j]
                vel_lims_q = (v*kms for v in vel_lims)
                mom0 = cube.spectral_slab(*vel_lims_q).moment0()
                mom0.write(f"/home/ramsey/Downloads/{line}_{stub}.fits")
    print("done")


def test_hcop_hcn_linewidth_magnet_thing():
    """
    Feburary 7, 2023
    The Crutcher 2012 magnetic field molecular cloud review mentions comparing
    linewidths of ions to neutral species. The ions should have thinner lines
    because they are constrained against perpendicular motions to the B field.
    Let me see what a ratio map looks like of their moment2s
    """
    ion_cube = cps2.cutout_subcube(length_scale_mult=None, data_filename='hcop')
    atom_cube = cps2.cutout_subcube(length_scale_mult=None, data_filename='hcn')

    mask1 = (ion_cube > 3*cube_utils.onesigmas['hcop']*u.K)
    mask2 = (atom_cube > 3*cube_utils.onesigmas['hcn']*u.K)
    mask = mask1.include() & mask2.include()

    vel_lims = (22*kms, 27*kms)

    ion_cube_m = ion_cube.with_mask(mask).spectral_slab(*vel_lims)
    atom_cube_m = atom_cube.with_mask(mask).spectral_slab(*vel_lims)

    ion_cube_mom2 = ion_cube_m.linewidth_sigma()
    atom_cube_mom2 = atom_cube_m.linewidth_sigma()

    fig = plt.figure(figsize=(17, 9))
    ax1 = plt.subplot(231)
    ax1.imshow(ion_cube_mom2.to_value(), origin='lower', vmin=0.3, vmax=1)
    ax1.set_title("HCO+")
    ax2 = plt.subplot(233)
    im = ax2.imshow(atom_cube_mom2.to_value(), origin='lower', vmin=0.3, vmax=1)
    ax2.set_title("HCN")
    fig.colorbar(im, ax=ax2, label='1D velocity dispersion (km/s)')

    ax_mid = plt.subplot(232)
    im = ax_mid.imshow(atom_cube_mom2.to_value()-ion_cube_mom2.to_value(), origin='lower', vmin=-0.03, vmax=0.03, cmap='PiYG')
    fig.colorbar(im, ax=ax_mid, label='HCN$-$HCO+ velocity dispersions (km/s)')


    ion_mom0 = ion_cube_m.moment0().to_value()
    atom_mom0 = atom_cube_m.moment0().to_value()

    ax_lo = plt.subplot(235)
    im = ax_lo.imshow(atom_mom0-ion_mom0, origin='lower', vmin=-4, vmax=4, cmap='PiYG')
    fig.colorbar(im, ax=ax_lo, label="HCN$-$HCO+ int intens (K km/s)")


    ax3 = plt.subplot(234)
    ax3.imshow(ion_mom0, origin='lower', vmin=0, vmax=35)
    ax4 = plt.subplot(236)
    im = ax4.imshow(atom_mom0, origin='lower', vmin=0, vmax=35)
    fig.colorbar(im, ax=ax4, label='Integrated intensity (K km/s)')
    # 2023-02-07
    fig.savefig("/home/ramsey/Pictures/2023-02-07/hcop_hcn_velocitydispersion_difference_magnetic_0.png",
        metadata=catalog.utils.create_png_metadata(title=">3sigma in BOTH lines",
            file=__file__, func="test_hcop_hcn_linewidth_magnet_thing"
        ))


def integrate_13_and_18_co_for_column_density():
    """
    May 25, 2022
    Following Mangum and Shirley 2015 (really nice paper)
    Integrate 13co and c18o between 19-27.3 km/s (27.3 to avoid the other cloud)
    and save that for the column density calculation
    """
    lines = ['13co10', 'c18o10']
    vel_lims = [v*kms for v in (19, 27.3)]
    for line in lines:
        cube = cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line], length_scale_mult=None)
        subcube = cube.spectral_slab(*vel_lims)
        mom0 = subcube.moment0().to(u.K*kms)
        # Noise calculation
        channel_noise = cube_utils.onesigmas[line]
        cube_dv = np.abs(np.diff(cube.spectral_axis[:2]))[0].to(kms).to_value()
        nchannels = subcube.shape[0]
        noise = channel_noise * cube_dv * np.sqrt(nchannels)
        # Save and keep noise as comment
        savename = os.path.join(cps2.cube_info['dir'], f"{line}_19-27.3_integrated.fits")
        header = mom0.header.copy()
        start_flag = "<error>"
        end_flag = "</error>"
        header['COMMENT'] = f'{start_flag}{noise:.3f} K km s-1{end_flag}'
        header['HISTORY'] = 'Ramsey wrote this on May 25, 2022 using code in'
        header['HISTORY'] = 'm16_deepdive.integrate_13_and_18_co_for_column_density'
        header['HISTORY'] = 'and updated it April 19, 2023 to use Marcs uncertainties'
        hdu = fits.PrimaryHDU(data=mom0.array, header=header)
        hdu.writeto(savename)


def convert_13co10_integrated_map_to_Kkms():
    """
    April 19, 2023
    please release me
    re-doing the column densities with up-to-date error estimates
    using marc's mom0 map now
    need to convert the map from Jy/beam km/s to K km/s. I have made this conversion before.
        - get frequency and beam
        - use u.brightness_temperature equivalency
        - need to remove velocity unit before and reattach after; the brightness_temperature
        conversion does not like it.
    """
    line_fns = {
        '13co10': "bima/M16.BIMA.13co.mom0.fits",
        'c18o10': "bima/M16.BIMA.c18o.masked_mom0.fits"
    }
    nchannels_dict = {'13co10': 30}
    for line in ['13co10']:
        full_fn = catalog.utils.search_for_file(line_fns[line])
        img, hdr = fits.getdata(full_fn, header=True)
        # noise
        channel_noise = cube_utils.onesigmas[line]
        cube_dv = (2.65635948582E+02 * u.m / u.s).to(kms).to_value()
        nchannels = nchannels_dict[line]
        noise = channel_noise * cube_dv * np.sqrt(nchannels)
        # convert units
        # first wrangle units and remove km/s unit
        data_units = tuple(u.Unit(y.lower().replace('jy', 'Jy')) for y in hdr['BUNIT'].split('.'))
        restfrq = hdr['RESTFREQ'] * u.Hz
        beam = cube_utils.Beam.from_fits_header(hdr)
        img = (img*data_units[0]).to(u.K, equivalencies=u.brightness_temperature(restfrq, beam.sr)) * data_units[1]
        print("Converted", hdr['BUNIT'], "to", img.unit)
        # save
        savename = os.path.join(os.path.dirname(full_fn), f"{line}_19-27.integrated.marcs_version.fits")
        header = WCS(hdr).to_header()
        header.update(beam.to_header_keywords())
        header['BUNIT'] = str(img.unit)
        start_flag = "<error>"
        end_flag = "</error>"
        header['COMMENT'] = f'{start_flag}{noise:.3f} K km s-1{end_flag}'
        header['HISTORY'] = 'Ramsey wrote this on April 19, 2023 using code in'
        header['HISTORY'] = 'm16_deepdive.convert_13co10_integrated_map_to_Kkms'
        header['HISTORY'] = 'it uses Marcs uncertainties'
        hdu = fits.PrimaryHDU(data=img.to_value(), header=header)
        hdu.writeto(savename)


def extract_noise_from_hdr(hdr):
    """
    May 25, 2022
    Following the function above, extract noise as I put it in the header
    :param hdr: FITS Header that I wrote using the function:
        m16_deepdive.integrate_13_and_18_co_for_column_density()
    """
    start_flag = "<error>"
    end_flag = "</error>"
    start_n = str(hdr['COMMENT']).find(start_flag)
    end_n = str(hdr['COMMENT']).find(end_flag)
    noise = str(hdr['COMMENT'])[start_n + len(start_flag):end_n]
    return u.Quantity(noise)


def get_excitation_temperature_12co():
    """
    May 30, 2022
    I think this will be just like the peak temperature function I already wrote at some point.
    then just slap channel errors on it and can propagate them from there
    Yeah so I did peak temperature in m16_investigation.peak_temperature(),
    and for that I just did a max() within a spectral_slab
    """
    line = '12co10'
    vel_lims = [v*kms for v in (19, 27.3)]
    cube = cps2.cutout_subcube(data_filename=cube_utils.cubefilenames[line], length_scale_mult=None)
    subcube = cube.spectral_slab(*vel_lims)
    peak_temperature = subcube.max(axis=0).to(u.K)
    channel_noise = cube_utils.onesigmas[line]
    savename = os.path.join(cps2.cube_info['dir'], f"{line}_19-27.3_peak.fits")
    header = peak_temperature.header.copy()
    start_flag = "<error>"
    end_flag = "</error>"
    header['COMMENT'] = f"{start_flag}{channel_noise} K{end_flag}"
    header['HISTORY'] = "Ramsey wrote this on May 30 using code in"
    header['HISTORY'] = "m16_deepdive.get_excitation_temperature_12co"
    header['HISTORY'] = 'and updated it April 19, 2023 to use Marcs uncertainties'
    hdu = fits.PrimaryHDU(data=peak_temperature.array, header=header)
    hdu.writeto(savename)


def calculate_co_column_density():
    """
    May 30, 2022
    Putting this all together and trying to calculate 13CO column density
    I could extend this to C18O later, but I'll start with the easier one

    Updated Nov 29, 2022:
    Propagating statistical uncertanties through the calculation

    Updated Feb 8, 2023:
    Using mu=1.33 and rho_H2 = mu * 2 * Hmass * n (instead of independent H2_mass_permole number)

    Equations:
    N(13CO) = 4pi epsilon0 *
        (3 kB / 8pi^3 nu S mu^2) *
        (Qrot / g) *
        (exp(Eu / kTex)) *
        integrated intensity
    where
        S = Ju / (2Ju + 1)
        g = 2Ju + 1
        Qrot = (kTex / hB0) + 1/3
    and constants
        Eu = 5.28880 K
        B0 = 55101.01 MHz
        mu = 0.11046 Debye
        nu = 110.20135400 GHz
        Ju = 1
    """
    # Build up all the constants
    # Already defined in astropy.constants
    # const.k_B, const.eps0, const.h
    #
    B0 = 55101.01 * u.MHz
    Eu = 5.28880 * u.K
    mu = 0.11046 * u.Debye
    nu = 110.20135400 * u.GHz
    Ju = 1.
    g = 2.*Ju + 1
    S = Ju/g
    # Prefactors (after cancelling a factor of 4pi from top and bottom)
    prefactor_numerator = const.eps0 * 3 * const.k_B
    prefactor_denominator = 2 * np.pi**2 * nu * S * mu**2
    # Load in Tex and integrated intensity
    Tex_unitless, Texhdr = fits.getdata(catalog.utils.search_for_file("bima/12co10_19-27.3_peak.fits"), header=True)
    err_Tex = u.Quantity(extract_noise_from_hdr(Texhdr))
    # Tex more often used as kTex (and put units)
    Tex = Tex_unitless*u.K

    fn_13co = catalog.utils.search_for_file("bima/13co10_19-27.integrated.marcs_version.fits")


    integrated_intensity_unitless, intT_hdr = fits.getdata(fn_13co, header=True)
    beam_13co = cube_utils.Beam.from_fits_header(intT_hdr)
    err_intT = u.Quantity(extract_noise_from_hdr(intT_hdr))
    integrated_intensity = integrated_intensity_unitless*u.K*kms
    # Rotational partition function
    Qrot = (const.k_B * Tex / (const.h * B0)).decompose() + (1./3.)
    err_Qrot = (const.k_B * err_Tex / (const.h * B0)).decompose() # constant falls off from derivative
    # exponential term
    exp_term = np.exp(Eu / Tex)
    err_exp_term = err_Tex * exp_term * Eu/(Tex**2) # d(e^(a/x)) = (a dx / x^2) e^(a/x)
    # All together
    N13CO = ((prefactor_numerator/prefactor_denominator) * (Qrot/g) * exp_term * integrated_intensity).to(u.cm**-2)
    # Uncertainty! d(cxyz) = cyz dx + cxz dy + cxy dz. But you gotta do quadrature sum instead of regular sum
    # Collected all constants (prefactor_numerator/prefactor_denominator and 1/g) at the end, outside the derivatives and quad sum
    helper_1 = (Qrot * exp_term * err_intT)**2
    helper_2 = (Qrot * err_exp_term * integrated_intensity)**2
    helper_3 = (err_Qrot * exp_term * integrated_intensity)**2
    err_N13CO = (np.sqrt(helper_1 + helper_2 + helper_3) * (prefactor_numerator / prefactor_denominator) / g).to(u.cm**-2)


    # Mask on integrated intensity error
    masking_by_error = True
    if masking_by_error:
        unmasked_N13CO = N13CO.copy()
        unmasked_err_N13CO = err_N13CO.copy()
        masking_by_error_coeff = 1.
        N13CO[integrated_intensity_unitless < masking_by_error_coeff*err_intT.to_value()] = np.nan
        err_N13CO[integrated_intensity_unitless < masking_by_error_coeff*err_intT.to_value()] = np.nan
    else:
        unmasked_N13CO = None


    N12CO = N13CO * ratio_12co_to_13co
    NH2 = N12CO / ratio_12co_to_H2

    err_N12CO = err_N13CO * ratio_12co_to_13co
    err_NH2 = err_N12CO / ratio_12co_to_H2

    if unmasked_N13CO is not None:
        unmasked_NH2 = unmasked_N13CO * ratio_12co_to_13co / ratio_12co_to_H2
        unmasked_err_NH2 = unmasked_err_N13CO * ratio_12co_to_13co / ratio_12co_to_H2
    else:
        unmasked_NH2 = None
        unmasked_err_NH2 = None

    if False:
        crop = { # i, j
            'p1a': ((378, 478), (227, 355)),
            'p1b': ((260, 371), (117, 246)),
            'p2_head': ((276, 343), (278, 388)),
            'p3_head': ((196, 245), (329, 378)),
            'blob': ((170, 293), (381, 487)),
            'full': ((None, None), (None, None)),
        }
        selected_cutout = 'p1a'
        cutout = (slice(*crop[selected_cutout][0]), slice(*crop[selected_cutout][1]))
        NH2_cropped = NH2[cutout]
        wcs_cropped = WCS(intT_hdr)[cutout]
    elif False:
        selected_box_type = 'threads' # or pillars
        if selected_box_type == 'pillars':
            boxes_reg_list = regions.Regions.read(catalog.utils.search_for_file("catalogs/p123_boxes.reg"))
            selected_box = 'Pillar 1'
        elif selected_box_type == 'threads':
            boxes_reg_list = regions.Regions.read(catalog.utils.search_for_file("catalogs/thread_boxes.reg"))
            selected_box = 'western'
        boxes_reg_dict = {reg.meta['text']: reg for reg in boxes_reg_list}
        box_mask = boxes_reg_dict[selected_box].to_pixel(WCS(intT_hdr)).to_mask().to_image(NH2.shape)
        NH2_cropped = NH2.copy()
        NH2_cropped[(box_mask < 1)] = np.nan
        if selected_box_type == 'pillars' and selected_box[-1] == '3':
            NH2_cropped[178:235, 379:413] = np.nan
        wcs_cropped = WCS(intT_hdr)

    # from .dust_mass import get_physical_area_pixel
    # pixel_area = get_physical_area_pixel(NH2, wcs_object, los_distance_M16.to(u.pc).to_value())
    # This and the method we use below (misc_utils.get_pixel_scale) are the same within 1e-16
    """
    Save a FITS file of:
    13CO column density
    12CO column density implied from that
    H2 column density implied from that
    H2 mass per pixel
    """
    wcs_object = WCS(intT_hdr)

    pixel_scale = misc_utils.get_pixel_scale(wcs_object)
    pixel_area = (pixel_scale * (los_distance_M16/u.radian))**2
    err_pixel_area = 2 * (pixel_scale/u.radian)**2 * los_distance_M16 * err_los_distance_M16

    particle_mass = 2*mean_molecular_weight_neutral*Hmass # molecular H; 2*mu*mH
    mass_per_pixel_map = (pixel_area * NH2 * particle_mass).to(u.solMass)
    # Include both error from column density as well as from LOS distance
    err_mass_per_pixel_raw = np.sqrt((pixel_area * err_NH2 * particle_mass)**2 + (err_pixel_area * NH2 * particle_mass)**2).to(u.solMass)
    pixels_per_beam = (beam_13co.sr / pixel_scale**2).decompose()
    # sqrt(oversample_factor) to correct for correlated pixels
    err_mass_per_pixel = np.sqrt(pixels_per_beam) * err_mass_per_pixel_raw

    def make_and_fill_header():
        # fill header with stuff, make it from WCS
        hdr = wcs_object.to_header()
        hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        hdr['CREATOR'] = f"Ramsey, {__file__}.calculate_co_column_density"
        hdr['HISTORY'] = f"12CO/H2 = {ratio_12co_to_H2:.2E}"
        hdr['HISTORY'] = f"12C/13C = {ratio_12co_to_13co:.2f}"
        hdr['HISTORY'] = f"Hmass = {Hmass:.3E}"
        hdr['HISTORY'] = f"mean molecular weight = {mean_molecular_weight_neutral:.2f}"
        hdr['HISTORY'] = f"adopted particle mass = {particle_mass:.2E}"
        hdr['HISTORY'] = f"pixel scale = {pixel_scale.to(u.arcsec):.3E}"
        hdr['HISTORY'] = f"pixel area = {pixel_area.to(u.pc**2):.3E}"
        hdr['HISTORY'] = f"sqrt(pixels/beam) oversample = {np.sqrt(pixels_per_beam):.2f}"
        hdr['HISTORY'] = f"LOS distance = {los_distance_M16.to(u.pc):.2f}"
        hdr['HISTORY'] = "Using Marcs 13co10 moment, which is less noisy"
        hdr['HISTORY'] = "Also using Marcs channel RMS values for 12 and 13CO"
        if masking_by_error:
            hdr['HISTORY'] = f"Masking by {masking_by_error_coeff:.1f} X integrated intensity error"
        return hdr

    savedir = os.path.dirname(catalog.utils.search_for_file("bima/13co10_19-27.3_integrated.fits"))
    savename = os.path.join(savedir, "13co10_column_density_and_more_with_uncertainty_v3.fits")

    phdu = fits.PrimaryHDU()

    header1 = make_and_fill_header()
    header1['EXTNAME'] = "13COcoldens"
    header1['BUNIT'] = str(N13CO.unit)
    hdu_13co = fits.ImageHDU(data=N13CO.to_value(), header=header1)

    header2 = make_and_fill_header()
    header2['EXTNAME'] = "12COcoldens"
    header2['BUNIT'] = str(N12CO.unit)
    hdu_12co = fits.ImageHDU(data=N12CO.to_value(), header=header2)

    header3 = make_and_fill_header()
    header3['EXTNAME'] = "H2coldens"
    header3['BUNIT'] = str(NH2.unit)
    header3['COMMENT'] = "This is MOLECULAR hydrogen (H2)"
    hdu_H2 = fits.ImageHDU(data=NH2.to_value(), header=header3)

    header4 = make_and_fill_header()
    header4['EXTNAME'] = "mass"
    header4['BUNIT'] = str(mass_per_pixel_map.unit)
    header4['COMMENT'] = "mass is per pixel on this image"
    hdu_mass = fits.ImageHDU(data=mass_per_pixel_map.to_value(), header=header4)


    header5 = make_and_fill_header()
    header5['EXTNAME'] = "err_13COcoldens"
    header5['BUNIT'] = str(err_N13CO.unit)
    hdu_e13co = fits.ImageHDU(data=err_N13CO.to_value(), header=header5)

    header6 = make_and_fill_header()
    header6['EXTNAME'] = "err_12COcoldens"
    header6['BUNIT'] = str(err_N12CO.unit)
    hdu_e12co = fits.ImageHDU(data=err_N12CO.to_value(), header=header6)

    header7 = make_and_fill_header()
    header7['EXTNAME'] = "err_H2coldens"
    header7['BUNIT'] = str(err_NH2.unit)
    header7['COMMENT'] = "This is MOLECULAR hydrogen (H2)"
    hdu_eH2 = fits.ImageHDU(data=err_NH2.to_value(), header=header7)

    header8 = make_and_fill_header()
    header8['EXTNAME'] = "err_mass"
    header8['BUNIT'] = str(err_mass_per_pixel.unit)
    header8['COMMENT'] = "mass is per pixel on this image"
    hdu_emass = fits.ImageHDU(data=err_mass_per_pixel.to_value(), header=header8)



    list_of_hdus = [phdu, hdu_13co, hdu_12co, hdu_H2, hdu_mass,
        hdu_e13co, hdu_e12co, hdu_eH2, hdu_emass]

    if masking_by_error:
        header1a = make_and_fill_header()
        header1a['EXTNAME'] = "13COcoldens_all"
        header1a['BUNIT'] = str(unmasked_N13CO.unit)
        header1a['COMMENT'] = "all values"
        hdu_13co_all = fits.ImageHDU(data=unmasked_N13CO.to_value(), header=header1a)

        header2a = make_and_fill_header()
        header2a['EXTNAME'] = "H2coldens_all"
        header2a['BUNIT'] = str(unmasked_NH2.unit)
        header2a['COMMENT'] = "all values"
        hdu_H2_all = fits.ImageHDU(data=unmasked_NH2.to_value(), header=header2a)

        header3a = make_and_fill_header()
        header3a['EXTNAME'] = "err_H2coldens_all"
        header3a['BUNIT'] = str(unmasked_err_NH2.unit)
        header3a['COMMENT'] = "all values"
        hdu_eH2_all = fits.ImageHDU(data=unmasked_err_NH2.to_value(), header=header3a)

        list_of_hdus.extend([hdu_13co_all, hdu_H2_all, hdu_eH2_all])


    hdul = fits.HDUList(list_of_hdus)
    hdul.writeto(savename, overwrite=True)

    # plt.show()


def calculate_pillar_lifetimes_from_columndensity():
    """
    August 8, 2022
    Use the 13CO column densities and a rough estimate of photoevaporative flow
    mass loss rate to make a lifetime map
    """
    co_column_fn = catalog.utils.search_for_file("bima/13co10_column_density_and_more_with_uncertainty_v2.fits")
    NH2, NH2_hdr = fits.getdata(co_column_fn, header=True, extname='H2coldens')
    NH2 = NH2 * u.cm**-2
    # Convert 13CO to H2 column using numbers from Tiwari 2021
    # NH2 = (N13CO * ratio_12co_to_13co / ratio_12co_to_H2) * u.cm**-2
    """
    Estimate mass loss rate (in terms of H2 molecules / time)
    From Gorti & Hollenbach 2002, mass loss rate dm/dt = A * rho_base * v_f
    For our purposes, dH2 / dt / dA = n_base * v_f
    Assume n_base (physical density at base of flow) and v_f (flow velocity)

    Whether it's H or H2 is hazy but it's just a factor of 2
    """
    n_base = 2e4 * u.cm**-3 # used to use 1e4, I think it's got to be closer to 2
    v_f = 0.6*kms # used to use 1.0 km/s, but I know more now. Using both sides flow, since column will go out towards and away
    H2_loss_rate_per_area = n_base * v_f
    # Divide the H2 column map by this H2 loss rate per area
    lifetime_map = (NH2 / H2_loss_rate_per_area).to(u.Myr)
    print(lifetime_map.unit)
    fig = plt.figure(figsize=(10, 8))
    plt.subplot(111, projection=WCS(NH2_hdr))
    plt.imshow(lifetime_map.to_value(), origin='lower', vmin=0, cmap='terrain')
    plt.colorbar(label='Lifetime upper limit (Myr)')
    plt.title("Photoevaporative flow lifetime limits from N($^{13}$CO)")
    plt.xlabel("RA")
    plt.ylabel("Dec")
    # 2022-08-08, 2023-02-08
    fig.savefig("/home/ramsey/Pictures/2023-02-08/lifetime_map_n13co_color.png",
        metadata=catalog.utils.create_png_metadata(title='lifetime upper limits from N(13co), n_base = 10^4cm-3 v_f = 0.3km/s',
            file=__file__, func='calculate_pillar_lifetimes_from_columndensity'))


def calculate_cii_column_density(filling_factor=1.0):
    """
    November 3, 2022
    Following Okada 2015 Sec 3.3 (pg 10)
    Several equations and some rules on how to assume Tex
    """
    lsm = 8
    cii_cube = cps2.cutout_subcube(length_scale_mult=lsm)
    # cii_cube = cii_cube - cps2.get_cii_background()[:, np.newaxis, np.newaxis]

    cii_cube = cii_cube.with_mask(cii_cube > 6*u.K).with_fill_value(0*u.K)
    # print(cii_cube.filled_data[:])
    # cii_cube[cii_cube < 1*u.K] = 0*u.K

    channel_noise = cube_utils.onesigmas['cii'] * u.K

    print(cii_cube.shape)
    rest_freq = cii_cube.header['RESTFREQ'] * u.Hz
    freq_axis = cii_cube.spectral_axis.to(u.THz, equivalencies=cii_cube.velocity_convention(rest_freq))

    hnu_kB = const.h * rest_freq / const.k_B
    print("T_0 = E_u / k_B = ", hnu_kB.decompose())
    g0, g1 = 2, 4 # lower, upper
    A10 = 10**(-5.63437) / u.s # Einstein A

    peak_T_map = cii_cube.max(axis=0).quantity


    # Can change this; it doesn't make a huge difference between 0.5 and 1, but below 0.5 you get some major differences (high column density)
    # filling_factor = 1.0 # this is an argument now

    """
    !!!!!!!!!!!!!!!!!!!!!!!!
    to switch from tau = 1 to constant Tex, switch assumed_optical_depth = 10+ (optically thick but not too high or there are floating point errors)
    and comment in the line that makes Tex map uniform
    """
    assumed_optical_depth = 1.3 # The tau value for equation 2 if we assume optical depth and solve for Tex at each pixel
    # tau = 1.3 is the latest "upper limit" based on no detection of 13cii with 1 K noise and 40 K 12CII and the latest 12/13 ratio of ~45 (11/18/22)
    # the tau upper limit is calculated by hand (see my notes, or Guevara paper)
    Tex_map = (hnu_kB / np.log((1 - np.exp(-assumed_optical_depth))*(filling_factor * hnu_kB / peak_T_map) + 1)).decompose()

    original_Tex_map = Tex_map.copy()
    fixed_Tex_val = np.nanmax(Tex_map)
    Tex_map[:] = fixed_Tex_val

    # Error on Tex
    # d/dx (a / log(b/x + 1)) = ab / (x(b+x)log*2((b+x)/x))
    helper_a = hnu_kB
    helper_b = (1 - np.exp(-assumed_optical_depth))*filling_factor*hnu_kB
    err_Tex_map = (channel_noise * (helper_a * helper_b) / (Tex_map * (helper_b + Tex_map) * np.log((helper_b/Tex_map) + 1)**2)).decompose()

    #########################
    # dont need to change much below this
    ####################

    # This is how Tex will often be used, and it needs the extra spectral dimension at axis=0
    hnukBTex = hnu_kB/Tex_map[np.newaxis, :]
    err_hnukBTex = (err_Tex_map * hnu_kB / Tex_map**2)[np.newaxis, :]

    exp_hnukBTex = np.exp(hnukBTex)
    err_exp_hnukBTex = err_hnukBTex * exp_hnukBTex # d(e^(a/x)) = (a dx / x^2) e^(a/x)

    # partition function?
    Z = g0 + g1*np.exp(-hnukBTex) # hnu_kB = Eu/kB since ground is 0 energy (might also be ok if not, but it's definitely ok in this case)
    err_Z = g1 * err_hnukBTex * np.exp(-hnukBTex)

    # optical depth in a given channel
    channel_tau = -1*np.log(1 - ((cii_cube.filled_data[:] / (filling_factor * hnu_kB)) * (exp_hnukBTex - 1))) # 3d cube
    print(channel_tau.unit)

    # Uncertainty on optical depth in channel
    helper_a = (exp_hnukBTex - 1) / (filling_factor * hnu_kB)
    err_channel_tau_from_Tb = channel_noise * helper_a / (1 - helper_a*cii_cube.filled_data[:])
    del helper_a
    # reset the definition of "a"! not the same!
    helper_a = (cii_cube.filled_data[:] / (filling_factor * hnu_kB))
    helper_numerator = helper_a * err_exp_hnukBTex
    helper_denominator = 1. - helper_a*(exp_hnukBTex - 1)
    err_channel_tau_from_Tex = helper_numerator / helper_denominator
    # quick analysis shows approximately equal contributions from each source of uncertainty
    err_channel_tau = np.sqrt(err_channel_tau_from_Tex**2 + err_channel_tau_from_Tb**2).decompose()
    # relatively small percentage of channel_tau values


    # Column density in a given channel
    column_constants = (8*np.pi * (rest_freq / const.c)**2) / (g1*A10)
    channel_column = (
        column_constants * channel_tau * Z * (exp_hnukBTex / (1 - np.exp(-hnukBTex)))
    ).decompose()

    # Uncertainty on column density in channel
    helper_1 = (err_channel_tau * Z * (exp_hnukBTex / (1 - np.exp(-hnukBTex))))**2
    helper_2 = (channel_tau * err_Z * (exp_hnukBTex / (1 - np.exp(-hnukBTex))))**2
    helper_3 = (channel_tau * Z * (err_exp_hnukBTex * exp_hnukBTex * (exp_hnukBTex - 2) / (exp_hnukBTex - 1)**2.))**2
    # Quick analysis shows channel_tau error dominates: factor of 40 over Z err, but only factor of 4 over Tex err
    err_channel_column = (np.sqrt(helper_1 + helper_2 + helper_3) * column_constants).decompose()


    integrated_column_map = np.trapz(channel_column[::-1, :, :], x=freq_axis[::-1], axis=0).to(u.cm**-2)
    # Let's just do quadrature sum * dnu for the integral uncertainty propagation
    dnu = np.median(np.diff(freq_axis[::-1]))
    err_integrated_column_map = (np.sqrt(np.sum(err_channel_column**2, axis=0))*dnu).to(u.cm**-2)
    # looking like a 10% error


    integrated_H_column_map = integrated_column_map / Cp_H_ratio
    err_integrated_H_column_map = err_integrated_column_map / Cp_H_ratio

    particle_mass = Hmass * mean_molecular_weight_neutral
    integrated_mass_column_map = integrated_H_column_map * particle_mass
    err_integrated_mass_column_map = err_integrated_H_column_map * particle_mass

    pixel_scale = misc_utils.get_pixel_scale(cii_cube[0, :, :].wcs)
    pixel_area = (pixel_scale * (los_distance_M16/u.radian))**2
    err_pixel_area = 2 * (pixel_scale/u.radian)**2 * los_distance_M16 * err_los_distance_M16

    integrated_mass_pixel_column_map = (integrated_mass_column_map * pixel_area).to(u.solMass)
    # Include error from column density and from LOS distance
    err_integrated_mass_pixel_column_map_raw = np.sqrt((err_integrated_mass_column_map * pixel_area)**2 + (integrated_mass_column_map * err_pixel_area)**2).to(u.solMass)
    pixels_per_beam = (cii_cube.beam.sr / pixel_scale**2).decompose()
    # sqrt(oversample_factor) to correct for correlated pixels
    err_integrated_mass_pixel_column_map = np.sqrt(pixels_per_beam) * err_integrated_mass_pixel_column_map_raw



    def make_and_fill_header():
        # fill header with stuff, make it from WCS
        hdr = wcs_flat.to_header()
        hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        hdr['CREATOR'] = f"Ramsey, {__file__}"
        hdr['HISTORY'] = "Using calculate_cii_column_density.py"
        hdr['HISTORY'] = f"Fixed Tex {fixed_Tex_val:.2f} max Tex calculated using tau={assumed_optical_depth}"
        hdr['HISTORY'] = f"Cutout with length scale {lsm}"
        hdr['HISTORY'] = f"C+/H = {Cp_H_ratio:.2E}"
        hdr['HISTORY'] = f"Hmass = {Hmass:.3E}"
        hdr['HISTORY'] = f"mean molecular weight = {mean_molecular_weight_neutral:.2f}"
        hdr['HISTORY'] = f"adopted particle mass = {particle_mass:.2E}"
        hdr['HISTORY'] = f"pixel scale = {pixel_scale.to(u.arcsec):.3E}"
        hdr['HISTORY'] = f"pixel area = {pixel_area.to(u.pc**2):.3E}"
        hdr['HISTORY'] = f"sqrt(pixels/beam) oversample = {np.sqrt(pixels_per_beam):.2f}"
        hdr['HISTORY'] = f"filling factor = {filling_factor:.2f}"

        # hdr['HISTORY'] = "TEST3: lsm8, Tex variable"
        return hdr

    phdu = fits.PrimaryHDU()
    wcs_flat = cii_cube[0, :, :].wcs

    header1 = make_and_fill_header()
    header1['EXTNAME'] = "C+coldens"
    header1['BUNIT'] = str(integrated_column_map.unit)
    hdu_NCp = fits.ImageHDU(data=integrated_column_map.to_value(), header=header1)

    header2 = make_and_fill_header()
    header2['EXTNAME'] = "mass"
    header2['BUNIT'] = str(integrated_mass_pixel_column_map.unit)
    header2['COMMENT'] = "mass is per pixel on this image"
    hdu_mass = fits.ImageHDU(data=integrated_mass_pixel_column_map.to_value(), header=header2)

    header3 = make_and_fill_header()
    header3['EXTNAME'] = "varyingTex"
    header3['BUNIT'] = str(original_Tex_map.unit)
    header3['COMMENT'] = "This is !!NOT!! the Tex used to calculate column density"
    header3['COMMENT'] = "The fixed Tex (see above) is the max of this image"
    hdu_Tex = fits.ImageHDU(data=original_Tex_map.to(u.K).to_value(), header=header3)

    header4 = make_and_fill_header()
    header4['EXTNAME'] = "Hcoldens"
    header4['BUNIT'] = str(integrated_H_column_map.unit)
    header4['COMMENT'] = "mass is per pixel on this image"
    hdu_NH = fits.ImageHDU(data=integrated_H_column_map.to_value(), header=header4)

    pdrt_density = 2e4 * u.cm**-3
    los_distance_image = (integrated_H_column_map / pdrt_density).to(u.pc)

    header5 = make_and_fill_header()
    header5['EXTNAME'] = "scale_distance"
    header5['BUNIT'] = str(los_distance_image.unit)
    header5['COMMENT'] = f"calculated using PDRT density {pdrt_density:.1E}"
    hdu_distance = fits.ImageHDU(data=los_distance_image.to_value(), header=header5)


    # error maps
    header6 = make_and_fill_header()
    header6['EXTNAME'] = "err_C+coldens"
    header6['BUNIT'] = str(err_integrated_column_map.unit)
    header6['COMMENT'] = "uncertainty propagated"
    hdu_eNCp = fits.ImageHDU(data=err_integrated_column_map.to_value(), header=header6)

    header7 = make_and_fill_header()
    header7['EXTNAME'] = "err_mass"
    header7['BUNIT'] = str(err_integrated_mass_pixel_column_map.unit)
    header7['COMMENT'] = "uncertainty propagated"
    hdu_emass = fits.ImageHDU(data=err_integrated_mass_pixel_column_map.to_value(), header=header7)

    header8 = make_and_fill_header()
    header8['EXTNAME'] = "err_Hcoldens"
    header8['BUNIT'] = str(err_integrated_H_column_map.unit)
    header8['COMMENT'] = "uncertainty propagated"
    hdu_eNH = fits.ImageHDU(data=err_integrated_H_column_map.to_value(), header=header8)


    hdul = fits.HDUList([phdu, hdu_NCp, hdu_NH, hdu_mass, hdu_distance, hdu_Tex,
        hdu_eNCp, hdu_emass, hdu_eNH])
    savename = cube_utils.os.path.join(cps2.cube_info['dir'], f"Cp_coldens_and_mass_lsm{lsm}_ff{filling_factor:.1f}_with_uncertainty.fits")
    print(savename)
    hdul.writeto(savename, overwrite=True)

    # plt.subplot(111)
    # plt.imshow(integrated_mass_pixel_column_map.to_value(), origin='lower')
    # plt.show()

    # plt.subplot(221)
    # plt.imshow(peak_T_map.to_value(), origin='lower')
    # plt.title("Peak $T$")
    # plt.subplot(222)
    # plt.imshow(Tex_map.to_value(), origin='lower')
    # plt.title("$T_{\\rm ex}$")
    # # plt.subplot(223)
    # # plt.imshow(integrated_column_map.to_value(), origin='lower')
    # # plt.title("integrated CII column density")
    # plt.subplot(223)
    # plt.imshow(integrated_column_map.to_value() / 8.5e-5, origin='lower')
    # plt.title("integrated H column density (H nuc / cm2)")
    #
    # plt.subplot(224)
    # plt.imshow((integrated_column_map / (8.5e-5 * 2e4*u.cm**-3)).to(u.pc).to_value(), origin='lower')
    # plt.title("size scale map (pc)")
    # plt.show()


def calculate_dust_column_densities(v=1):
    """
    November 15, 2022
    Use the tau_160 maps to make N(H) maps
    From the RCW 49 paper, Cext,160 / H = 1.9 x 10^-25 cm2/H
    I have versions using 70--250 and 70--160, I'll check both to compare
    Nov 18: added a mass/pixel map so I can integrate in DS9

    Jan 26, 2023: using the newly minted upper and lower tau limits, make
    some new column density and mass/pixel maps.
    upper and lower limits in independent files so that I can run them thru
    the mass estimator separately

    As of Feb 2023, use the calculate_dust_column_densities_and_masses_with_error()
    instead of this when possible. Monte Carlo uncertainty estimation
    """
    # raise RuntimeError("Already ran on November 21, 2022")
    cutout_center_coord = SkyCoord("18:18:55.9969 -13:50:56.169", unit=(u.hourangle, u.deg), frame=FK5)
    cutout_size = (575*u.arcsec, 658*u.arcsec)
    if v == 1:
        # 250 micron version; this uses 70,160,250 and has the resolution of 250 (like 18 arcsec or something)
        fn_v1 = catalog.utils.search_for_file('herschel/M16_2p_3BAND_beta2.0.fits')
        tau_v1, hdr_v1 = fits.getdata(fn_v1, extname='solutiontau', header=True)
        tau_large = 10**tau_v1
        wcs_obj_large = WCS(hdr_v1)
        fn_stub = fn_v1
        summary_stub = '70-160-250'
        comment_stub = 'Calculated using mantipython'
        suffix = ''
    elif v > 1: # haven't assigned above 5
        if v == 2:
            suffix = ''
        elif v >= 3:
            # this version has error bars, but v==3 will get "best values"
            suffix = '_fluxbgsub_with_uncertainty'

        fn_v2 = catalog.utils.search_for_file(f'herschel/T-tau_colorsolution{suffix}.fits')
        with fits.open(fn_v2) as hdul:
            if v == 3:
                extname = 'tau'
                suffix = suffix.replace('_with_uncertainty', '')
            elif v == 4:
                extname = 'tau_LO'
                suffix = suffix.replace('_with_uncertainty', '_LO')
            elif v == 5:
                extname = 'tau_HI'
                suffix = suffix.replace('_with_uncertainty', '_HI')
            tau_v2, hdr_v2 = hdul[extname].data, hdul[extname].header
            if 'mask' in hdul:
                # Use the mask we made, it's good
                mask = hdul['mask'].data > 0.5 # convert float 1s and 0s to bool
            else:
                # Use the T image to mask the tau image
                mask = np.isfinite(hdul['T'].data) & np.isfinite(tau_v2) & (hdul['T'].data > 0) & (tau_v2 > 0)

        tau_v2[~mask] = np.nan
        tau_large = tau_v2
        wcs_obj_large = WCS(hdr_v2)
        fn_stub = fn_v2
        summary_stub = '70-160'
        comment_stub = 'Calculated using the T/tau color solution'

    tau_cutout = Cutout2D(tau_large, cutout_center_coord, cutout_size, wcs=wcs_obj_large)
    tau = tau_cutout.data
    wcs_obj = tau_cutout.wcs

    savename = f"coldens_{summary_stub}{suffix}.fits"
    savename = os.path.join(os.path.dirname(fn_stub), savename)

    Cext160 = 1.9e-25 * u.cm**2

    new_hdr = wcs_obj.to_header()
    new_hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    new_hdr['CREATOR'] = f"Ramsey, {__file__}"
    new_hdr['HISTORY'] = f"Created from bands {summary_stub}"
    new_hdr['HISTORY'] = "Resolution is that of longest wavelength band"
    new_hdr['HISTORY'] = f'From file {fn_stub}'
    new_hdr['HISTORY'] = comment_stub
    new_hdr['HISTORY'] = f"Cext160/H = {Cext160:.2E}"
    new_hdr['COMMENT'] = "Column density is H nucleus (divide by 2 for H2)"

    phdu = fits.PrimaryHDU()

    new_hdr2 = new_hdr.copy()

    N_H = (tau / Cext160).to(u.cm**-2)
    new_hdr['EXTNAME'] = "Hcoldens"
    new_hdr['BUNIT'] = str(N_H.unit)
    hdu = fits.ImageHDU(data=N_H.to_value(), header=new_hdr)

    pixel_scale = misc_utils.get_pixel_scale(wcs_obj)

    pixel_area = (pixel_scale * (los_distance_M16/u.radian))**2
    err_pixel_area = 2 * (pixel_scale/u.radian)**2 * los_distance_M16 * err_los_distance_M16 # d((p * L  / 1rad)^2) = p^2 * L dL / (1rad)^2

    particle_mass = Hmass * mean_molecular_weight_neutral
    mass_per_pixel_map = (pixel_area * N_H * particle_mass).to(u.solMass)
    err_mass_per_pixel_map = (err_pixel_area * N_H * particle_mass).to(u.solMass)

    new_hdr2['EXTNAME'] = "mass"
    new_hdr2['BUNIT'] = str(mass_per_pixel_map.unit)
    new_hdr2['HISTORY'] = f"Hmass = {Hmass:.3E}"
    new_hdr2['HISTORY'] = f"mean molecular weight = {mean_molecular_weight_neutral:.2f}"
    new_hdr2['HISTORY'] = f"adopted particle mass = {particle_mass:.2E}"
    new_hdr2['HISTORY'] = f"pixel scale = {pixel_scale.to(u.arcsec):.3E}"
    new_hdr2['HISTORY'] = f"pixel area = {pixel_area.to(u.pc**2):.3E}"
    new_hdr2['HISTORY'] = f"LOS distance = {los_distance_M16.to(u.pc):.2f} +/- {err_los_distance_M16.to(u.pc):.2f}"
    hdu2 = fits.ImageHDU(data=mass_per_pixel_map.to_value(), header=new_hdr2)

    new_hdr3 = new_hdr2.copy()
    new_hdr3['EXTNAME'] = "err_mass"
    hdu3 = fits.ImageHDU(data=err_mass_per_pixel_map.to_value(), header=new_hdr3)

    hdul = fits.HDUList([phdu, hdu, hdu2, hdu3])
    hdul.writeto(savename, overwrite=False)


"""
Next couple functions are in service of the dust FIR -> mass Monte Carlo thing
"""

def helper_make_dust_t_tau_splines():
    """
    Jan 27, 2023
    Helper function for calculate_dust_column_densities_and_masses_with_error
    Make the splines for 70/160 ratio to temperature and temperature to 160 flux
    Returns the two splines as a tuple
    Astropy units are not supported, but temperatures are in K and flux density
    in MJy/sr
    Copying code from g0_dust.fir_intensity_2 and color_temperature_uncertainty.ipynb
    """
    # Set up PACS detectors
    bands = [70, 160]
    detectors = {band: detector for band, detector in zip(bands, instrument.get_instrument(bands))}
    # Set up model arrays
    model_T_arr = np.arange(1, 200, 0.1) # units of K
    model_bandpass_br_ratio = np.zeros_like(model_T_arr)
    notau_160intensity = np.zeros_like(model_T_arr) # not calling it "zero tau" anymore
    args1 = (-8., dust.TauOpacity(2.)) # for the regular Greybody
    args2 = (0, dust.TauOpacity(2.)) # for the ThinGreybody
    for i, t in enumerate(model_T_arr):
        # First ratio->T
        g = greybody.Greybody(t, *args1)
        model_bandpass_br_ratio[i] = detectors[70].detect(g) / detectors[160].detect(g)
        # Then T -> no tau 160 flux
        notau_160intensity[i] = detectors[160].detect(greybody.ThinGreybody(t, *args2))
    model_bandpass_br_spline = UnivariateSpline(model_bandpass_br_ratio, model_T_arr, s=0)
    notau_160I_spline = UnivariateSpline(model_T_arr, notau_160intensity, s=0)
    return model_bandpass_br_spline, notau_160I_spline


def calculate_dust_column_densities_and_masses_with_error(nsamples=10, debug=False):
    """
    January 27, 2023
    Derive H column density and pillar mass from the FIR dust fluxes.
    This combines work from g0_dust.fir_intensity_2,
    m16_deepdive.calculate_dust_column_densities,
    and m16_deepdive.estimate_uncertainty_mass_and_coldens
    because I want to propagate uncertainties from flux all the way to mass
    and use a Monte Carlo sampling to do so. I can't have them be separate
    functions, I have to do it all at once

    This procedure has a few segments:
    1) zero-point calibrate emission and establish flux uncertainties
    2) create color ratio and convert to T and tau
    3) convert tau to optical depth and mass/pixel
    4) integrate mass/pixel over pillars to find total mass
    The established flux uncertainties will be sampled a large (~100-1000)
    number of times. The flux maps will be expanded out in a 3rd dimension
    and the samples will be added to the entire map (*not* pixel-by-pixel,
    they are systematic).
    This should correctly propagate the errors from flux to column density
    and mass.
    Hopefully this takes less than a couple hours... (starting at 12:54pm)
    (i dont remember when i first finished this, I think Jan 30?)

    :param nsamples: Choose number of realizations (should be >100, maybe 1000 if that's fast enough)
        10-40 is good for testing. 400 takes about 1G of RAM and only ~5-10 seconds
    """
    # Seed a RNG
    rng = np.random.default_rng(1312)
    print("NSAMPLES", nsamples)

    # for writing out filenames
    suffix = ""

    # Step 1: Load and zero-point correct the observations.
    # This stuff is copied out of g0_dust.fir_intensity_2 for the most part (I rewrote it to look nicer the other day)
    corrections, err_corrections = {}, {}
    # Negative offsets, subtracting local background near pillars
    # These are sampled from the "north" regions and do noth include the "south" sample
    corrections[70] = -2836.24 # +/- 417.06
    corrections[160] = -1365.62 # +/- 229.34
    # Error bars
    err_corrections[70] = 417.06
    err_corrections[160] = 229.34
    # Cutout params (use by default)
    use_cutout = True
    center = (2867, 1745-30)
    size = (1192//4, 873//4)
    # Path names
    # I did the reproc160 on October 7, 2022 (previously I had only done 250)
    pacs_obs_dir = catalog.utils.search_for_file("herschel/processed/1342218995_reproc160")
    # ie = 'image' or 'error'
    make_pacs_fn = lambda band, ie : os.path.join(pacs_obs_dir, f"PACS{band}um-{ie}-remapped-conv.fits")
    bands = [70, 160]
    band_image_samples = {} # zero-point corrected (and sampled along new 0 axis)
    band_best_images = {} # 2D arrays, "best" values (no uncertainty)
    slices, wcs_obj = None, None
    flux_mask = None # mask based on flux that excludes background around pillars
    for i, band in enumerate(bands):
        # Load image
        img, hdr = fits.getdata(make_pacs_fn(band, 'image'), header=True)
        # Cutout and save cutout info
        if slices is None:
            img_cutout = Cutout2D(img, center, size[::-1], wcs=WCS(hdr))
            slices = img_cutout.slices_original
            wcs_obj = img_cutout.wcs
            img = img_cutout.data
        else:
            img = img[slices]
        # Collect all uncertainties
        # Flux calibraton error: Get 5% of image before zero-point correction
        # Background subtraction error: single number for all map
        onesigma_err_sys = np.sqrt((img*0.05)**2 + err_corrections[band]**2) # combine them so we only sample once
        onesigma_err_stat = fits.getdata(make_pacs_fn(band, 'error'))[slices]
        """
        We can use a *single* Gaussian-sampled number for BOTH fluxcal and bg for the entire map (per realization)
        For statistical, we will use a map's worth of Gaussian sampled numbers per realization
        """
        sample_sys = rng.normal(size=(nsamples, 1, 1)) # realizations X single number per realization
        sample_stat = rng.normal(size=(nsamples, *img.shape)) # realizations X 2D map shape
        sampled_err_sys = onesigma_err_sys * sample_sys
        sampled_err_stat = onesigma_err_stat * sample_stat

        # Add the error realizations to the data (and add the zero-point correction while things are still 2D)
        sampled_img = (img + corrections[band]) + sampled_err_sys + sampled_err_stat # realizations X 2D map shape
        band_image_samples[band] = sampled_img

        # Save the regular corrected 2D images as "best" values
        band_best_images[band] = img + corrections[band]

        # Also, make that mask of 3x the systematic error in the 160 band
        # And add in 70um above 0
        flux_cutoff = 3
        if band == 70:
            assert flux_mask is None
            flux_cutoff_70 = 0
            if flux_cutoff_70 != flux_cutoff:
                print("70um FLUX CUTOFF", flux_cutoff_70, "x sigma")
                suffix += f"_70gt{flux_cutoff_70:1d}"
            flux_mask = (img + corrections[band]) > flux_cutoff_70*onesigma_err_sys
            # Doing >3sigma to 70um really loses us a lot of Pillar 3. But the alternative is huge error bars and uncertainty due to low temperatures
        if band == 160:
            assert flux_mask is not None
            flux_mask &= (img + corrections[band]) > flux_cutoff*onesigma_err_sys

        if debug and False:
            fig = plt.figure(f"Uncertainty {band}um", figsize=(16, 8))
            ax1 = plt.subplot(231)
            im1 = ax1.imshow(onesigma_err_sys, origin='lower')
            ax1.set_title(f"Systematic $\\sigma$")

            ax2 = plt.subplot(232)
            ax2.imshow(img, origin='lower')
            ax2.set_title("Data values")
            ax2.contour(flux_mask.astype(float), levels=[0.5], colors='k')

            ax3 = plt.subplot(233)
            im3 = ax3.imshow(onesigma_err_stat, origin='lower')
            ax3.set_title("Statistical $\\sigma$")

            ax4 = plt.subplot(234)
            ax4.imshow(sampled_err_sys[0], origin='lower')
            ax4.set_title(f"Systematic sample 0")

            ax5 = plt.subplot(235)
            ax5.imshow(sampled_img[0], origin='lower')
            ax5.set_title(f"Data sample 0")

            ax6 = plt.subplot(236)
            ax6.imshow(sampled_err_stat[0], origin='lower', vmin=np.median(np.abs(sampled_err_stat[0])*-3), vmax=np.median(np.abs(sampled_err_stat[0])*3))
            ax6.set_title("Statistical sample 0")

            plt.show()
    del sampled_err_sys, sampled_err_stat, img

    # An array we can simply multiply to make NaNs. Broadcasts easily, unlike bool indexing
    bad_flux_nan_mask = np.ones_like(flux_mask.astype(float))
    bad_flux_nan_mask[~flux_mask] = np.nan

    # Get the splines
    br_t_spline, t_160_spline = helper_make_dust_t_tau_splines()
    # Get the ratio
    observed_br_ratio_samples = band_image_samples[70] / band_image_samples[160] # Monte Carlo samples
    best_observed_br_ratio = band_best_images[70] / band_best_images[160] # "best" value
    # Use the splines, get T and tau
    t_solution_samples = br_t_spline(observed_br_ratio_samples)
    best_t_solution = br_t_spline(best_observed_br_ratio)

    plot_tau_and_cold_pixels = True
    if debug and plot_tau_and_cold_pixels:
        cold_cutoff = 10
        cold_pixels = np.sum(t_solution_samples<cold_cutoff, axis=0)/nsamples
        cold_pixels *= bad_flux_nan_mask
    tau_solution_samples = band_image_samples[160] / t_160_spline(t_solution_samples)
    best_tau_solution = band_best_images[160] / t_160_spline(best_t_solution)
    del band_image_samples, band_best_images, t_solution_samples, best_t_solution

    if debug and plot_tau_and_cold_pixels:
        tau_solution_median = np.median(tau_solution_samples, axis=0)
        tau_solution_std = np.std(tau_solution_samples, axis=0)
        tau_solution_max = np.max(tau_solution_samples, axis=0)
        tau_solution_median *= bad_flux_nan_mask
        tau_solution_std *= bad_flux_nan_mask
        tau_solution_max *= bad_flux_nan_mask

        tau_Av_conversion_denom = (1.9e-25 * 1.9e21)
        tau_solution_median /= tau_Av_conversion_denom
        tau_solution_std /= tau_Av_conversion_denom
        tau_solution_max /= tau_Av_conversion_denom

        fig = plt.figure("Sampled $\\tau$ statistics", figsize=(16, 10))
        ax1 = plt.subplot(221)
        im = ax1.imshow(tau_solution_median, origin='lower', vmin=0, vmax=150)
        ax1.set_title("Median")
        fig.colorbar(im, ax=ax1, label='Av (mag)')

        ax2 = plt.subplot(222)
        im = ax2.imshow(tau_solution_std, origin='lower', vmin=0, vmax=15)
        ax2.set_title("StdDev")
        fig.colorbar(im, ax=ax2, label='Av (mag)')

        ax3 = plt.subplot(223)
        im = ax3.imshow(cold_pixels, origin='lower', vmin=0, vmax=0.05)
        ax3.set_title(f"Cold (<{cold_cutoff}) pixels")
        fig.colorbar(im, ax=ax3, label='cold pixel fraction (0 is never cold, 1 is always cold)')

        ax4 = plt.subplot(224)
        im = ax4.imshow(tau_solution_max, origin='lower', vmin=0, vmax=300)
        ax4.set_title("Maximum")
        fig.colorbar(im, ax=ax4, label='Av (mag)')

        plt.tight_layout()
        # 2023-01-30, 2023-02-01,09
        fig.savefig(f"/home/ramsey/Pictures/2023-02-09/cold_pixels_{nsamples}{suffix}.png",
            metadata=catalog.utils.create_png_metadata(title=f"nsamples {nsamples}, cutoffs 70/160 {flux_cutoff_70} / {flux_cutoff}",
                file=__file__, func="calculate_dust_column_densities_and_masses_with_error"))
    # Progress as of EOD Friday jan 27
    # Starting back up Monday Jan 30 1:36 PM
    # Next step: Convert to mass/area, integrate over pixels, convert pixels to area (another systematic uncertainty)
    # This stuff is copied from m16_deepdive.calculate_dust_column_densities

    Cext160 = 1.9e-25 * u.cm**2
    N_H_samples = (tau_solution_samples / Cext160).to(u.cm**-2)
    del tau_solution_samples
    best_N_H = (best_tau_solution / Cext160).to(u.cm**-2)
    del best_tau_solution
    # Pixel area and uncertainty
    pixel_scale = misc_utils.get_pixel_scale(wcs_obj)
    pixel_area = (pixel_scale * (los_distance_M16/u.radian))**2.
    err_pixel_area = 2 * (pixel_scale/u.radian)**2 * los_distance_M16 * err_los_distance_M16

    # !!!! still need to do this
    # # TODO: calculate distance uncertainty effect on mass, compare to the others
    # also would be nice to save these stats as a file
    # the quantile thing will let me easily estimate a "symmetric" error

    particle_mass = Hmass * mean_molecular_weight_neutral

    stop_and_save_fits = False
    if stop_and_save_fits:
        # Do not calculate masses; just save the "best" and median and "standard deviation" mass and column density planes
        median_N_H = np.median(N_H_samples, axis=0)
        N_H_16, N_H_84 = misc_utils.flquantiles(N_H_samples, 6)
        N_H_avg_err = (N_H_84 - N_H_16)/2
        median_mass_dpix = (median_N_H * particle_mass*pixel_area).to(u.solMass).to_value()
        mass_dpix_16, mass_dpix_84 = (N_H_16*particle_mass*pixel_area).to(u.solMass).to_value(), (N_H_84*particle_mass*pixel_area).to(u.solMass).to_value()
        mass_dpix_avg_err = (mass_dpix_84 - mass_dpix_16)/2

        best_mass_dpix = (best_N_H * particle_mass * pixel_area).to(u.solMass).to_value()

        phdu = fits.PrimaryHDU()
        hdr = wcs_obj.to_header()
        def hdu_generator(data, extname):
            if hasattr(data, 'unit'):
                data = data.to_value()
            new_hdr = hdr.copy()
            new_hdr['EXTNAME'] = extname
            new_hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
            new_hdr['CREATOR'] = f"Ramsey, {__file__}"
            new_hdr['COMMENT'] = "Column density is H nucleus (divide by 2 for H2)"
            new_hdr['COMMENT'] = f"mean molecular weight H {mean_molecular_weight_neutral}"
            new_hdr['COMMENT'] = f"adopted particle mass {particle_mass:.3E}"
            new_hdr['COMMENT'] = f"pixel scale, area {pixel_scale:.2E}, {pixel_area:.2E}"
            new_hdr['COMMENT'] = f"pct err pixel area: {(err_pixel_area/pixel_area).decompose().to_value() *100:.4f} %"
            new_hdr['COMMENT'] = f"mask using cutoffs 70/160 {flux_cutoff_70} / {flux_cutoff}"
            return fits.ImageHDU(data=data, header=new_hdr)
        hdu_list = [phdu,
            hdu_generator(best_N_H, 'Hcoldens_best'),
            hdu_generator(median_N_H, 'Hcoldens_median'), hdu_generator(N_H_16, 'Hcoldens_16'), hdu_generator(N_H_84, 'Hcoldens_84'), hdu_generator(N_H_avg_err, 'Hcoldens_avgerr'),
            hdu_generator(best_mass_dpix, 'mass_best'),
            hdu_generator(median_mass_dpix, 'mass_median'), hdu_generator(mass_dpix_16, 'mass_16'), hdu_generator(mass_dpix_84, 'mass_84'), hdu_generator(mass_dpix_avg_err, 'mass_avgerr'),
            hdu_generator(100*mass_dpix_avg_err/best_mass_dpix, 'mass_avgerr_pct'),
        ]
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(f"/home/ramsey/Documents/Research/Feedback/m16_data/herschel/coldens_70-160_sampled_{nsamples}{suffix}.fits",
            overwrite=True)
        return
    else:
        # computationally expensive and unnecessary if just saving
        mass_per_area_samples = (N_H_samples * particle_mass).to(u.solMass/u.cm**2)
        best_mass_per_area = (best_N_H * particle_mass).to(u.solMass/u.cm**2)







    ########################### toggle whether it's P1a/P1b/P2/P3 or heads and necks are separated
    entire_pillar_mass = False
    ############################
    if entire_pillar_mass:
        reg_filename_short = "catalogs/mass_boxes_v2.reg" # v2: updated to be tighter around the pillars & mask
    else:
        reg_filename_short = "catalogs/p123_boxes_head_body_withlabels_v3.reg"
        suffix += "_head-neck-boxes"
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    reg_dict = {reg.meta['text']: reg for reg in reg_list if 'noise' not in reg.meta['text']}

    plot_hists = True
    if debug and plot_hists:
        fig = plt.figure("Mass and N_H distribution", figsize=(12, 8))
        ax_col = plt.subplot(211)
        ax_col.set_xlabel("cm$^{-2}$")
        ax_col.set_title("Maximum N(H)")
        ax_mass = plt.subplot(212)
        ax_mass.set_xlabel("Solar masses")
        ax_mass.set_title("Mass")
    for reg in reg_dict:
        print(f"REGION {reg}")
        pix_reg = reg_dict[reg].to_pixel(wcs_obj)
        reg_mask = pix_reg.to_mask().to_image(wcs_obj.array_shape).astype(bool).reshape(1, *wcs_obj.array_shape)
        combined_mask = reg_mask & flux_mask # shape should be (1, imgI, imgJ) for obvious broadcasting to sampled img shape
        print(f"Mask shape {combined_mask.shape}", end="; ")
        # Due to restrictions on array bool indexing, I can't just index the 3D array with a 2D boolean array
        # I also want to keep the 3D structure intact so I can keep each realization separate
        # So instead of indexing, I will multiply by the masks
        col_values_samples = N_H_samples * combined_mask
        mass_values_samples = mass_per_area_samples * combined_mask

        best_col_values = best_N_H * combined_mask[0, :, :]
        best_mass_values  = best_mass_per_area * combined_mask[0, :, :]
        n_pixels = combined_mask.sum()
        print(f"npixels {n_pixels}")

        # Keep location, we will reuse for samples
        best_max_col_loc = np.unravel_index(np.argmax(best_col_values), best_col_values.shape)
        best_max_col = best_col_values[best_max_col_loc]
        best_mass = (np.sum(best_mass_values)*pixel_area).to(u.solMass)
        best_mass_err = (np.sum(best_mass_values)*err_pixel_area).to(u.solMass)

        max_col_samples = col_values_samples[(slice(None), *best_max_col_loc)]
        mass_samples = (np.sum(mass_values_samples, axis=(1, 2)) * pixel_area).to(u.solMass)
        mass_err_samples = (np.sum(mass_values_samples, axis=(1, 2)) * err_pixel_area).to(u.solMass)

        print(f"There are {np.sum(np.isnan(max_col_samples))} bad samples")
        print()
        print("Column density")
        print(f"Best val {best_max_col:.2E}")
        col_median = np.nanmedian(max_col_samples)
        col_std = np.nanstd(max_col_samples)
        col_16, col_84 = misc_utils.flquantiles(max_col_samples, 6) # 6 is a good approximation to 16,84 (16.7, 83.3)
        print(f"Median val {col_median:.2E}")
        print(f"StdDev {col_std:.2E}")
        print(f"16,84: {col_median-col_16:.2E},{col_84-col_median:.2E}. Avg: {(col_84-col_16)/2:.2E}")
        """
        There must be some very large column densities, perhaps due to near-0 temperatures?
        Yeah that seems to be it. Shaking the flux around when it's already close to backgroud
        will produce low temperatures and high column densities. It's all sort of continuous from low to reasonable temperatures,
        so there's no natural place to decide what is "unreasonable" (i.e., 5? 10? 15? 20? also introduces a bias on one side of the MC)
        """
        print()
        print("Mass")
        print(f"Best val {best_mass:.2f}")
        mass_median = np.nanmedian(mass_samples)
        mass_std = np.nanstd(mass_samples)
        mass_16, mass_84 = misc_utils.flquantiles(mass_samples, 6)
        print(f"Median val {mass_median:.2f}")
        print(f"StdDev {mass_std:.2f}")
        print(f"16,84: {mass_median-mass_16:.2f},{mass_84-mass_median:.2f}. Avg: {(mass_84-mass_16)/2:.2f}")

        if debug and plot_hists:
            b = 32
            col_range = (0, 4e23)
            mass_range = (0, 210)
            hist_col, _, p0 = ax_col.hist(max_col_samples.to_value(), label=reg, alpha=0.6, histtype='stepfilled', bins=b, range=col_range)
            hist_mass, _, p1 = ax_mass.hist(mass_samples.to_value(), label=reg, alpha=0.6, histtype='stepfilled', bins=b, range=mass_range)
            color = p0[0].get_facecolor()[:3]

            ax_col.errorbar([col_median.to_value()], [np.max(hist_col)*1.1], xerr=[col_std.to_value()], marker='o', ecolor=color, mec=color, mfc='none', capsize=3)
            ax_col.plot([best_max_col.to_value()], [np.max(hist_col)*1.1], marker='x', color=color)
            ax_col.plot([col_16.to_value(), col_84.to_value()], [np.max(hist_col)*1.15]*2, marker='+', linestyle='--', color=color)

            ax_mass.errorbar([mass_median.to_value()], [np.max(hist_mass)*1.1], xerr=[mass_std.to_value()], marker='o', ecolor=color, mec=color, mfc='none', capsize=3)
            ax_mass.plot([best_mass.to_value()], [np.max(hist_mass)*1.1], marker='x', color=color)
            ax_mass.plot([mass_16.to_value(), mass_84.to_value()], [np.max(hist_mass)*1.15]*2, marker='+', linestyle='--', color=color)

            ax_col.set_xlim(col_range)
            ax_mass.set_xlim(mass_range)

        print()
        print("Mass ERR")
        print(f"Best val {best_mass_err:.2f}")
        print(f"Median val {np.nanmedian(mass_err_samples):.2f}")
        print(f"StdDev {np.nanstd(mass_err_samples):.2f}")
        print(f"percent err: {100*(err_pixel_area/pixel_area).decompose()}")
        print("="*12)
        print()
    if debug and plot_hists:
        ax_col.legend()
        plt.tight_layout()
        # plt.subplots_adjust(top=0.99, bottom=0.03)
        # 2023-01-30, 2023-02-01,08,09
        fig.savefig(f"/home/ramsey/Pictures/2023-02-09/mass_uncertainty_{nsamples}{suffix}.png",
            metadata=catalog.utils.create_png_metadata(title=f"nsamples {nsamples}, cutoffs 70/160 {flux_cutoff_70} / {flux_cutoff}",
                file=__file__, func="calculate_dust_column_densities_and_masses_with_error"))

        # Make histograms of the values, see what the asymmetry is like


def calculate_galactocentric_distance_and_12c13c_ratio():
    """
    November 18, 2022
    Use the best constants we have to get galactocentric distance and 12/13 C
    ratio
    """
    coord = SkyCoord("18:18:55.1692 -13:50:08.828", frame=FK5, unit=(u.hourangle, u.deg)) # P1b
    l = coord.galactic.l.rad
    b = coord.galactic.b.rad
    galactocentric_distance_of_sun = 8.1 * u.kpc # +- 0.1 kpc, Bobylev + Bajkova 2021
    # switch to notation of Brand & Blitz 1993 Equation 2
    d = los_distance_M16
    R = galactocentric_distance_of_sun
    term_1 = (d*np.cos(b))**2
    term_2 = R**2
    term_3 = -2 * R*d * np.cos(b) * np.cos(l)
    galactocentric_M16 = np.sqrt(term_1 + term_2 + term_3)
    print(f"Galactocentric radius of M16: {galactocentric_M16.to(u.kpc):.2f}")

    # 12 to 13 C ratio, Yan et al 2019
    ratio = 5.08*galactocentric_M16.to(u.kpc).to_value() + 11.86
    print(f"12C/13C = {ratio:.2f}")



def estimate_uncertainty_mass_and_coldens(tracer='cii', setting=2):
    """
    TODO!!!!!!!!!!!!!!!!! (dec 13 2022)
    use setting=2 to add _fluxbgsub to dust (not dust250)
    see what the results are
    """

    """
    November 21, 2022
    Per Marc's comment on my section draft, I will make uncertainty estimates
    for mass and column density. I have taken regions for each
    """
    print("MASS AND COLUMN DENSITIES BASED ON", tracer)
    filenames_dict = {'cii': "sofia/Cp_coldens_and_mass_lsm6_ff1.0_with_uncertainty_v2.fits",
        'co': "bima/13co10_column_density_and_more_with_uncertainty_v3.fits",
        # 'dust': "herschel/coldens_70-160.fits", # not using these
        # 'dust250': "herschel/coldens_70-160-250.fits",
    }
    column_extnames = {'cii': 'Hcoldens', 'co': 'H2coldens', 'dust': 'Hcoldens', 'dust250': 'Hcoldens'}
    column_err_extnames = {'cii': 'err_Hcoldens', 'co': 'err_H2coldens'}
    mass_extname = 'mass'
    mass_err_extname = 'err_mass'

    ########################### toggle whether it's P1a/P1b/P2/P3 or heads and necks are separated
    entire_pillar_mass = False
    ############################
    if entire_pillar_mass:
        reg_filename_short = "catalogs/mass_boxes_v2.reg"
    else:
        reg_filename_short = "catalogs/p123_boxes_head_body_withlabels_v3.reg"
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    reg_dict = {reg.meta['text']: reg for reg in reg_list if 'noise' not in reg.meta['text']}

    fn = filenames_dict[tracer]

    # Flux background subtraction (can also check upper and lower error bounds)
    if setting > 1 and tracer == 'dust':
        print("Using the version with FLUX BACKGROUND SUBTRACTION")

        if setting == 2:
            suffix = '_fluxbgsub.fits'
        elif setting == 3:
            print("*"*12 + "LOWER ERROR BAR" + "*"*12)
            suffix = '_fluxbgsub_LO.fits'
        elif setting == 4:
            print("*"*12 + "UPPER ERROR BAR" + "*"*12)
            suffix = '_fluxbgsub_HI.fits'
        else:
            raise NotImplementedError(f"settings above 4 not approved (user setting = {setting})")

        fn = fn.replace('.fits', suffix)
    else:
        suffix = ""

    with fits.open(catalog.utils.search_for_file(fn)) as hdul:
        column_map = hdul[column_extnames[tracer]].data
        hdr = hdul[column_extnames[tracer]].header
        mass_map = hdul[mass_extname].data
        if tracer in column_err_extnames:
            err_column_map = hdul[column_err_extnames[tracer]].data
        else:
            err_column_map = None
        err_mass_map = hdul[mass_err_extname].data # all data have mass error maps now, from LOS uncertainty
    finite_mask = np.isfinite(column_map)

    wcs_object = WCS(hdr)

    plt.figure(figsize=(10,10))
    ax_col = plt.subplot(221, projection=wcs_object)
    plt.imshow(column_map, origin='lower')
    plt.title("N(Hydrogen)")
    ax_mass = plt.subplot(223, projection=wcs_object)
    plt.imshow(mass_map, origin='lower')
    plt.title("Mass/pixel")
    axes_hist = [plt.subplot(222), plt.subplot(224)]

    col_mean_vals = {}
    col_stddev_vals = {}
    col_mean_stat_unc = {}
    mass_mean_vals = {}
    mass_stddev_vals = {}
    mass_mean_stat_unc = {}

    no_background_subtract = ((tracer=='co') or ('dust' in tracer))

    noise_box_lists = [cps2.get_background_regions('north'), cps2.get_background_regions('south')]
    noise_box_lists_labels = ['north', 'south']
    noise_box_colors = [marcs_colors[0], marcs_colors[1]]

    if no_background_subtract:
        pass
    else:
        for i, noise_reg_list in enumerate(noise_box_lists):

            pix_reg_list = [reg.to_pixel(wcs_object) for reg in noise_reg_list]
            mask_list = []
            for pix_reg in pix_reg_list:
                ax_col.add_artist(pix_reg.as_artist(fill=False, edgecolor=noise_box_colors[i]))
                ax_mass.add_artist(pix_reg.as_artist(fill=False, edgecolor=noise_box_colors[i]))

                mask_list.append(pix_reg.to_mask().to_image(wcs_object.array_shape)) # you have to use .to_image() since .to_mask() returns a RegionMask object, which behaves strangely
            noise_mask = np.any(mask_list, axis=0)

            col_values = column_map[noise_mask & finite_mask]
            mass_values = mass_map[noise_mask & finite_mask]
            if err_column_map is None:
                err_col_values = 0
            else:
                err_col_values = err_column_map[noise_mask & finite_mask]
            err_mass_values = err_mass_map[noise_mask & finite_mask]
            n_pixels = len(col_values)

            mean_col = np.mean(col_values)
            std_col = np.std(col_values)
            # error on mean: (1/N) * sum(e**2 for e in errors)
            mean_col_stat_err = np.sqrt(np.sum(err_col_values**2)) / n_pixels

            sum_bg_mass = np.sum(mass_values)
            mean_bg_mass = np.mean(mass_values)
            std_bg_mass = np.std(mass_values)
            mean_bg_mass_stat_err = np.sqrt(np.sum(err_mass_values**2)) / n_pixels

            col_mean_vals[noise_box_lists_labels[i]] = mean_col
            col_stddev_vals[noise_box_lists_labels[i]] = std_col
            col_mean_stat_unc[noise_box_lists_labels[i]] = mean_col_stat_err
            mass_mean_vals[noise_box_lists_labels[i]] = mean_bg_mass
            mass_stddev_vals[noise_box_lists_labels[i]] = std_bg_mass
            mass_mean_stat_unc[noise_box_lists_labels[i]] = mean_bg_mass_stat_err

            print(f"NOISE FROM {noise_box_lists_labels[i]}")

            print("Pixels in mask (real)", n_pixels)
            print("Pixels in image", column_map.size)

            print(f"Mean/STD column: {mean_col:.2E} +/- {std_col:.2E}(sys) +/- {mean_col_stat_err:.2E}(stat)")
            print(f"Sum background: {sum_bg_mass:.1f} +/- {std_bg_mass*n_pixels:.3f}(sys) +/- {mean_bg_mass_stat_err*n_pixels:.3f}(stat)")
            print(f"Mean bg mass/pixel: {mean_bg_mass:.2E} +/- {std_bg_mass:.2E}(sys) +/- {mean_bg_mass_stat_err:.2E}(stat)")

            axes_hist[0].hist(col_values, bins=32, histtype='step', density=True, label=noise_box_lists_labels[i])
            axes_hist[1].hist(mass_values, bins=32, histtype='step', density=True, label=noise_box_lists_labels[i])

            print('='*8)

        axes_hist[0].legend()
        # 2022-11-21, 2022-12-06, 2022-12-13, 2023-01-16,23;02-09
        plt.savefig(f"/home/ramsey/Pictures/2023-04-19/coldens_and_mass_noise_{tracer}{suffix}.png",
            metadata=catalog.utils.create_png_metadata(title=f'reg from {reg_filename_short}',
                file=__file__, func='estimate_uncertainty_mass_and_coldens'))

    print("\n\n")

    for reg in reg_dict:
        if 'noise' in reg:
            # Pillars only
            continue
        print(f"REGION {reg}")
        pix_reg = reg_dict[reg].to_pixel(wcs_object)
        mask = pix_reg.to_mask().to_image(wcs_object.array_shape).astype(bool)

        col_values = column_map[mask & finite_mask]
        mass_values = mass_map[mask & finite_mask]

        if err_column_map is None:
            err_col_values = 0
        else:
            err_col_values = err_column_map[mask & finite_mask]
        err_mass_values = err_mass_map[mask & finite_mask]

        max_col_loc = np.argmax(col_values)
        max_col = col_values[max_col_loc]
        max_col_stat_unc = err_col_values[max_col_loc] if err_column_map is not None else 0

        sum_mass = np.sum(mass_values)
        mass_stat_unc = np.sqrt(np.sum(err_mass_values**2))
        n_pixels = len(mass_values)

        select_index = ('south' if ('p1b' in reg.lower()) else 'north')

        if no_background_subtract:
            col_mean_bg = 0
            col_sys_unc = 0
            col_mean_bg_stat_unc = 0
            mass_pixel_bg = 0
            mass_pixel_sys_unc = 0
            mass_pixel_bg_stat_unc = 0
        else:
            col_mean_bg = col_mean_vals[select_index]
            col_sys_unc = col_stddev_vals[select_index]
            col_mean_bg_stat_unc = col_mean_stat_unc[select_index]
            mass_pixel_bg = mass_mean_vals[select_index]
            mass_pixel_sys_unc = mass_stddev_vals[select_index]
            mass_pixel_bg_stat_unc = mass_mean_stat_unc[select_index]

        # if N x Mass is the formula, d/dMass is N, so N x sigma is the error
        mass_bg = n_pixels * mass_pixel_bg
        mass_sys_unc = n_pixels * mass_pixel_sys_unc
        mass_bg_stat_unc = n_pixels * mass_pixel_bg_stat_unc

        subtracted_column = max_col - col_mean_bg
        subtracted_column_sys_uncertainty = col_sys_unc*np.sqrt(2)
        subtracted_column_stat_uncertainty = np.sqrt(max_col_stat_unc**2 + col_mean_bg_stat_unc**2)

        subtracted_mass = sum_mass - mass_bg
        subtracted_mass_sys_uncertainty = mass_sys_unc*np.sqrt(2)
        subtracted_mass_stat_uncertainty = np.sqrt(mass_stat_unc**2 + mass_bg_stat_unc**2)

        print(f"Max Col (no sub): {max_col:.2E} +/- {col_sys_unc:.2E}(sys) +/- {max_col_stat_unc:.2E}(stat)")
        print(f"Max Col (corrected): {subtracted_column:.2E} +/- {subtracted_column_sys_uncertainty:.2E}(sys) +/- {subtracted_column_stat_uncertainty:.2E}(stat)")
        print(f"Mass (no sub): {sum_mass:.2f} +/- {mass_sys_unc:.2f}(sys) +/- {mass_stat_unc:.2f}(stat)")
        print(f"Mass (corrected) {subtracted_mass:.2f} +/- {subtracted_mass_sys_uncertainty:.2f}(sys) +/- {subtracted_mass_stat_uncertainty:.2f}(stat)")
        print(f"Background used: {mass_bg:.2f} +/- {mass_sys_unc:.2f}(sys) +/- {mass_bg_stat_unc:.2f}(stat)")


        column_modifier = 1e21
        # N(H) vs N(H2). dust and cii are in N(H), convert to N(H2) equivalent
        if (tracer=='cii') or ('dust' in tracer):
            column_modifier *= 2
        print("-"*5)
        print(f"{subtracted_column/column_modifier:4.1f} {subtracted_column_stat_uncertainty/column_modifier:.1f} (sta) | {subtracted_mass:.2f} {subtracted_mass_stat_uncertainty:.2f} (sta) {subtracted_mass_sys_uncertainty:.2f} (sys)")
        print(f"sys {subtracted_column_sys_uncertainty/column_modifier:.2f}")
        print("-"*5)

        print("="*10)
        print("\n")

    # plt.show()

"""
The column density figure for the paper will be made in m16_pictures.column_density_figure() (2022-11-22)
"""

def lifetime_pressure_velocitydispersion_tradeoff(n, selected_pillar):
    """
    February 6, 2023
    Pillar lifetime depends on CII velocity dispersion, but so does turbulent (and magnetic) pressure
    I will make an image here to understand the tradeoff between the two

    alpha is the fraction (quadrature) of the linewidth which is turbulent. (1-alpha) gives flow velocity
    sigma_total^2 = sigma_turb^2 + sigma_flow^2
    sigma_total^2 = alpha*sigma_total^2 + (1-alpha)*sigma_total^2
    sigma_turb^2 = a*sigma_total^2 and so on

    Ptot = Pturb + Pmag
    Pturb = sigma^2 * rho
    Pmag = sigma^2 * rho * (1/2)Q^2/sigma_th^2
    Ptot = sigma^2 * rho * (1 + (1/2)Q^2/sigma_th^2)
    Q = 0.5 (commonly, according to Pattle 2018)
    sigma_th = 14.4 degrees (Pattle 2018 observational result)
    The constant (1/2)Q^2/sigma_th^2 is approximately 1.979
    So Ptot = sigma^2 * rho * 2.979


    Update, Feburary 21, 2023
    Xander explained that the B field measurement by Pattle relies on the turbulent velocity dispersion
    measured in the same place as the sigma_th angle dispersion. So I can't just swap in the CII
    velocity dispersion.
    Best option is just to scale the B field down with the mass density rho, and if it's not enough,
    say that the B field might have relatively weakened in the molecular gas due to ambipolar diffusion.
    i.e. (I think) the C+ ions/electrons in the atomic gas probably couple that gas to a B field better
    than in the molecular

    """
    # (1 Gauss / (1 cm^−(1/2) * g^(1/2) *  s^−1))
    cgs_to_gauss = (u.Gauss / (u.cm**(-1/2) * u.g**(1/2) * u.s**-1))


    #### check what B field needed for 1-3 x 10^7 K cm-3
    def reverse_engineer_B_field(p):
        print(f"For pressure P = {p:.1E}, ", end='')
        b = ((p*8*np.pi*const.k_B)**(1/2) * cgs_to_gauss).to(u.microGauss)
        print(f"B = {b:.2f}")
    reverse_engineer_B_field(3e6*u.K/u.cm**3)
    reverse_engineer_B_field(1e7*u.K/u.cm**3)
    reverse_engineer_B_field(2e7*u.K/u.cm**3)
    reverse_engineer_B_field(3e7*u.K/u.cm**3)
    print()


    def calc_B_field_Pattle(nH2, sigma_v, mmw=1.4):
        """
        Implementing the equation for B field using Pattle's numbers but allowing
        mean molecular weight, sigma_v and nH2 to change
        I will use MMW = 1.33 but I want to check equations using theirs, 1.4
        """
        Q = 0.5
        sigma_th = (14.4*u.deg).to(u.rad).to_value()
        rho = (2 * nH2 * mmw * Hmass).to(u.g/u.cm**3)
        return (Q * np.sqrt(4 * np.pi * rho) * (sigma_v / sigma_th) * cgs_to_gauss).to(u.microGauss)

    def calc_turbulent_pressure(nH2, sigma_v):
        """
        Now default to mmw=1.33
        """
        return ((2 * nH2 * mean_molecular_weight_neutral * Hmass) * sigma_v**2 / const.k_B).to(u.K * u.cm**-3)

    b_170ug = calc_B_field_Pattle(5e4 * u.cm**-3, 0.5 * kms)
    print(f"This should be ~170uG: {b_170ug:.1f}")

    nH2_lo = 1.3e5
    nH2_hi = 1.3e5

    b_molecular_lo = calc_B_field_Pattle(nH2_lo * u.cm**-3, 0.6 * kms, mmw=mean_molecular_weight_neutral)
    b_molecular_hi = calc_B_field_Pattle(nH2_hi * u.cm**-3, 0.6 * kms, mmw=mean_molecular_weight_neutral)
    print(f"This is my best number for molecular gas: {b_molecular_lo:.1f} -- {b_molecular_hi:.1f}")

    def calc_Bpressure_Pattle(B_field):
        return ((B_field/cgs_to_gauss)**2 / (8*np.pi * const.k_B)).to(u.K * u.cm**-3)

    pB_mol_lo = calc_Bpressure_Pattle(b_molecular_lo)
    pB_mol_hi = calc_Bpressure_Pattle(b_molecular_hi)
    print(f"Molecular B pressures: {pB_mol_lo:.2E} -- {pB_mol_hi:.2E}")
    p_therm_mol_lo = 25 * nH2_lo
    p_therm_mol_hi = 25 * nH2_hi
    p_turb_mol_lo = calc_turbulent_pressure(nH2_lo*u.cm**-3, 0.6*kms)
    p_turb_mol_hi = calc_turbulent_pressure(nH2_hi*u.cm**-3, 0.6*kms)
    print(f"Molecular thermal pressure: {p_therm_mol_lo:.1E} -- {p_therm_mol_hi:.1E} ")
    print(f"Molecular turbulent pressure: {p_turb_mol_lo:.1E} -- {p_turb_mol_hi:.1E}")

    p_tot_mol_lo = (pB_mol_lo.to_value() + p_turb_mol_lo.to_value() + p_therm_mol_lo) / 1e6
    p_tot_mol_hi = (pB_mol_hi.to_value() + p_turb_mol_hi.to_value() + p_therm_mol_hi) / 1e6

    print(f"Total molecular pressures: {p_tot_mol_lo:.1f} -- {p_tot_mol_hi:.1f}")

    p_atom_lo = pB_mol_lo * (n/(2*nH2_lo))
    p_atom_hi = pB_mol_hi * (n/(2*nH2_hi))
    # print(f"Atomic pressures: {p_atom_lo:.2E} -- {p_atom_hi:.2E}")

    # n/2 because I baked in the 2xmH for molecular H2 into that function
    b_atom = calc_B_field_Pattle(n/2 * u.cm**-3, 0.6*kms, mmw=mean_molecular_weight_neutral)
    pB_atom = calc_Bpressure_Pattle(b_atom)
    print(f"Atomic B values: {b_atom:.1f}, {pB_atom:.2E}")



    """
    There is a unit issue in the pressure expression; check on Wolfram that my combination of P_B(Bfield) has valid units
    It works it's just the Gaussian units thing
    """


    def sigma_turb(alpha, sigma_total):
        return np.sqrt(alpha) * sigma_total

    def sigma_flow(alpha, sigma_total):
        return np.sqrt(1 - alpha) * sigma_total

    # rho is mass density
    n = n * u.cm**-3 # or 2e4
    # Neutral mass density
    rho = (n*mean_molecular_weight_neutral*Hmass).to(u.g/u.cm**3)

    def turb_pressure(alpha, sigma_total):
        # Combining magnetic and turbulent pressure, which have the same dependence on the quantity rho*sigma^2
        return (rho * sigma_turb(alpha, sigma_total)**2 / const.k_B).to(u.K / u.cm**3)


    p_turb_atomic = (rho * (1.3*kms)**2 / const.k_B).to(u.K / u.cm**3)
    print(f"Atomic turbulent pressure: {p_turb_atomic:.2E}")



    pillar_properties = { # area (pc2), mass (solMass from CO)
        'P1a-head': (0.17886, 64.12), 'P2-head': (0.07557, 11.32), 'P3-head': (0.02191, 4.27)
    }
    def mdot_and_pillar_lifetime(alpha, sigma_total, pillar_label):
        # Return both so we can make 2 plots
        area_pc2, mass_solMass = pillar_properties[pillar_label]
        area = area_pc2 * u.pc**2
        mass = mass_solMass * u.solMass
        mass_loss_rate = (sigma_flow(alpha, sigma_total) * rho * area / 2.).to(u.solMass / u.Myr)
        lifetime = (mass / mass_loss_rate).to(u.Myr)
        return mass_loss_rate, lifetime

    alpha_range = np.arange(0, 1, 0.05)

    fig = plt.figure(figsize=(10, 9))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    transparency = 0.2
    p_therm_lo = n.to_value()*100/1e6
    p_therm_hi = n.to_value()*250/1e6
    print(f"Atomic thermal pressure {p_therm_lo} -- {p_therm_hi}")
    print(f"Atomic total pressure {(p_turb_atomic+pB_atom).to_value()/1e6 + p_therm_lo:.1f} -- {(p_turb_atomic+pB_atom).to_value()/1e6 + p_therm_hi:.1f}")
    pB_atom_val = pB_atom.to_value()/1e6

    colors = marcs_colors[:3]
    # selected_pillar = "P2-head"

    for i, sigma_total in enumerate([1.0, 1.1, 1.3][::-1]*kms):
        label = "$\\sigma_{\\rm tot} =$ " + f"{sigma_total:.2f}"
        ax1.plot(alpha_range, sigma_turb(alpha_range, sigma_total).to_value(), color=colors[i], label=label)
        ax1.plot(alpha_range, sigma_flow(alpha_range, sigma_total).to_value(), color=colors[i], linestyle='--')

        p_turb = turb_pressure(alpha_range, sigma_total).to_value()/1e6
        ax2.fill_between(alpha_range, p_therm_lo+pB_atom_val+p_turb, y2=p_therm_hi+pB_atom_val+p_turb, color=colors[i], alpha=transparency)

        mass_loss_rate, lifetime = mdot_and_pillar_lifetime(alpha_range, sigma_total, selected_pillar)
        ax3.plot(alpha_range, mass_loss_rate.to_value(), color=colors[i])
        ax4.plot(alpha_range, lifetime.to_value(), color=colors[i])

    ax1.legend()

    ax1.set_title(f"bottom plots using {selected_pillar}")
    ax2.set_title(f"Density n={n:.1E}")

    ax2.set_ylim([0, 40])
    ax2.axhspan(p_tot_mol_lo, p_tot_mol_hi, color=marcs_colors[5], alpha=transparency, label='$P_{{\\rm H}_2}$') # fill region
    ax2.axhspan(18, 36, color=marcs_colors[6], alpha=transparency, label='$P_{\\rm HII}$') # fill region
    ax2.axhline(pB_atom_val, color=marcs_colors[5], alpha=transparency, label='$P_{{\\rm HI,B}}$')
    ax2.axhspan(p_therm_lo + pB_atom_val, p_therm_hi + pB_atom_val, color=marcs_colors[7], alpha=transparency, label='$P_{{\\rm HI,B}} + P_{{\\rm HI,therm}}$')
    ax2.legend(loc='upper left')

    ax3.set_xlabel("$\\alpha$")
    ax4.set_xlabel("$\\alpha$")
    ax1.set_ylabel("1D Velocity dispersion $\\sigma$ (km s-1)")
    ax2.set_ylabel("Total non-thermal pressure (cm-3)")
    ax3.set_ylabel(f"{selected_pillar}" + " $M_{\\odot}$ (solMass Myr-1)")
    ax3.set_ylim([0, 100])
    ax4.set_ylabel(f"{selected_pillar} Pillar lifetime (Myr)")
    ax4.axhspan(1, 3, color=marcs_colors[5], alpha=transparency)
    ax4.set_ylim([0, 8])
    # 2023-02-06,21, 03-16,25
    fig.savefig(f"/home/ramsey/Pictures/2023-03-25/pressure_mdot_tradeoff_{selected_pillar}_{n.to_value():.1E}.png",
        metadata=catalog.utils.create_png_metadata(title=f"B pressure scaled by density only; {selected_pillar}; n={n:.1E}",
            file=__file__, func="lifetime_pressure_velocitydispersion_tradeoff"))


persisent_regions_list = None # Save the list for future uses in same run, avoid loading from disk repeatedly
def get_samples_at_locations(img, wcs_obj):
    """
    February 9, 2023
    Use the regions in pillar123_pointsofinterest_v2 and grab a value at each location from the given map.
    Return as dictionary using region names as keys. Img can be array or quantity, as long as WCS works.
    If img=='coords' (set to a string, instead of a Quantity or array), then returns the coordinates as nicely formatted strings. wcs_obj will not be used in this case.
    """
    global persisent_regions_list
    if persisent_regions_list is None:
        reg_list = regions.Regions.read(catalog.utils.search_for_file("catalogs/pillar123_pointsofinterest_v2.reg"))
        persisent_regions_list = reg_list
    else:
        reg_list = persisent_regions_list
    return_dict = {}
    if isinstance(img, str) and img == 'coords':
        for reg in reg_list:
            print(reg.center.frame)
            return_dict[reg.meta['text']] = reg.center.transform_to('icrs').to_string(style='hmsdms')
    else:
        for reg in reg_list:
            coords = tuple(round(x) for x in reg.to_pixel(wcs_obj).center.xy[::-1]) # xy[::-1] = ij
            try:
                return_dict[reg.meta['text']] = img[coords]
            except IndexError:
                return_dict[reg.meta['text']] = np.nan * img.unit
    return return_dict


def column_of_table_sample_peak_brightness_temperatures(line_stub):
    """
    February 9, 2023
    Get peak brightness temperatures for a given line
    Also test out the get_samples_at_locations() function before I use it on column density
    """
    cube_obj = cube_utils.CubeData(line_stub)
    try:
        max_cube = cube_obj.data.max(axis=0)
    except ValueError:
        cube_obj.data.allow_huge_operations=True
        print("DOING HUGE OPERATION ON ", line_stub)
        max_cube = cube_obj.data[5:-5].max(axis=0)
    # Convert the 2D max array, which should be much faster than converting the 3D cube
    max_cube_values = cube_obj.convert_to_K(value=max_cube)
    return get_samples_at_locations(max_cube_values, max_cube.wcs)


def table_sample_peak_brightness_temperatures():
    """
    February 9, 2023
    Get brightness temperatures for every line towards regions, save table (csv)
    Looks like this works easily! Need to fill out the line list and rerun for the final table, but the written table is very clean and will be easy to open in Calc
    """
    line_list = ['cii', 'oi', '12co10', '13co10', 'c18o10', '12co32', '13co32', 'co65', 'hcn', 'hcop', 'cs', 'n2hp']
    # line_list = ['13co10', 'c18o10', '12co32']
    # line_list = ['cs', 'n2hp']
    uncertainty_list = []
    super_dict = {}

    # get_samples_at_locations('coords', None)
    # return

    for line_stub in line_list:
        super_dict[cube_utils.cubenames[line_stub]] = column_of_table_sample_peak_brightness_temperatures(line_stub)
        uncertainty_list.append(f"{cube_utils.onesigmas[line_stub]*u.K:.1f}")

    df = pd.DataFrame.from_dict(super_dict).applymap(lambda x: f"{x:.1f}")
    df['Coordinates'] = pd.Series(get_samples_at_locations('coords', None))
    df = df[['Coordinates'] + [x for x in df.columns if x!='Coordinates']]
    df.loc['T_RMS'] = [''] + uncertainty_list

    # 2023-02-09, 03-28,29,31, 04-12,20,23
    save_path = "/home/ramsey/Pictures/2023-04-23/max_brightness_temperatures"
    df.to_csv(save_path+".csv")
    table_as_latex = df.to_latex().replace('nan K', '')
    with open(save_path+".txt", 'w') as f:
        f.write(table_as_latex)



def table_sample_column_densities():
    """
    February 10, 2023
    Repeat the table_sample_peak_brightness_temperatures() method on
    the column density and error maps
    There is background correction and systematic error to add before these make it to the paper,
    but I will do that in a spreadsheet! This will just be direct samples.
    Values are N(H) for C+ and N(H2) for CO. I will make the conversion to N(H)/2 later
    """
    filenames_dict = {'cii': "sofia/Cp_coldens_and_mass_lsm6_ff1.0_with_uncertainty.fits",
        'co': "bima/13co10_column_density_and_more_with_uncertainty_v3.fits",
        'dust': "herschel/coldens_70-160_sampled_1000.fits"}
    column_extnames = {'cii': 'Hcoldens', 'co': 'H2coldens_all', 'dust': 'Hcoldens_best'}
    column_err_extnames = {'cii': 'err_Hcoldens', 'co': 'err_H2coldens_all', 'dust': 'Hcoldens_avgerr'}

    super_dict = {}
    for line_stub in filenames_dict:
        with fits.open(catalog.utils.search_for_file(filenames_dict[line_stub])) as hdul:
            column_map = hdul[column_extnames[line_stub]].data
            wcs_obj = WCS(hdul[column_extnames[line_stub]].header)
            err_column_map = hdul[column_err_extnames[line_stub]].data
        # TODO: Finish this! use get_samples_at_locations() to get samples and make table
        super_dict[line_stub] = get_samples_at_locations(column_map, wcs_obj)
        super_dict[line_stub+"_err"] = get_samples_at_locations(err_column_map, wcs_obj)
    # then copy the csv table into excel and add in background correction, do errors, etc
    df = pd.DataFrame.from_dict(super_dict).applymap(lambda x: f"{x:.2E}")
    # print(df)
    # 2023-02-11, 04-19,
    df.to_csv("/home/ramsey/Pictures/2023-04-19/column_densities_lsm6_old.csv")


def remake_leflochlazareff_Bfield_graph():
    """
    March 27, 2023
    Remake Figures 10-11, Lefloch and Lazareff 1994. Shows gravitational collapse
    parameter space: clump mass and ionizing flux.
    I'm remaking it with a more accurate B field pressure term
    """
    ionizing_photons = 2e50 / u.s
    d_stars_lims = (1.5*u.pc, 2.5*u.pc)
    photon_rate_to_flux = lambda dist : ((ionizing_photons / (4 * np.pi * dist**2)) / (1e7 / (u.s * u.cm**2))).decompose()
    ionizing_photon_flux_lims = np.log10(list(photon_rate_to_flux(dist) for dist in d_stars_lims))
    masses = np.log10(([119, 115, 21]*u.solMass).to_value())
    beta = 10
    T = 20 # K
    ionizing_photon_flux_array = np.linspace(-3, 6, 50)
    phi = np.log(10**ionizing_photon_flux_array)

    # IF: photon flux mainly consumed in ionization front
    # IBL: photon flux mainly consumed in recombinations in outflowing gas
    IBL_IF_boundary_mass = 2.42 - 2*ionizing_photon_flux_array + np.log10(beta * T / 100.)
    IBL_IF_boundary_mass2 = 2.42 - 2*phi + np.log(beta * T / 100.)

    plt.plot(ionizing_photon_flux_array, IBL_IF_boundary_mass, color='red') # plot is in LOG10 and so is the math
    # plt.plot(phi, IBL_IF_boundary_mass2, color='red', linestyle=':') # plot is in LOG_E and so is the math
    plt.plot(ionizing_photon_flux_array, np.log10(np.exp(IBL_IF_boundary_mass2)), color='red', linestyle='--') # plot is in LOG10 but math is in LOG_E (worst option but looks closest to their plots)

    ibl_collapse = 1.21  - (1./3)*ionizing_photon_flux_array + (7./3)*np.log10(beta*T/100.)
    ibl_collapse2 = 1.21  - (1./3)*phi + (7./3)*np.log(beta*T/100.)

    plt.plot(ionizing_photon_flux_array, ibl_collapse, color='blue')  # plot is in LOG10 and so is the math
    # plt.plot(phi, ibl_collapse2, color='blue', linestyle=':') # plot is in LOG_E and so is the math
    plt.plot(ionizing_photon_flux_array, np.log10(np.exp(ibl_collapse2)), color='blue', linestyle='--') # plot is in LOG10 but math is in LOG_E (worst option but looks closest to their plots)

    plt.ylim([-5, 5])
    plt.xlim([-2, 6])

    plt.axvspan(*ionizing_photon_flux_lims, alpha=0.2, color='k')
    for m in masses:
        plt.axhline(m, alpha=0.3, color='k')


    plt.show()






if __name__ == "__main__":

    ...
    # calculate_dust_column_densities_and_masses_with_error(nsamples=50, debug=True)

    # for line_stub in ['12co32', '12co10CONV']:
    #     prepare_pdrt_tables(line_stub, reg_filename="catalogs/pillar123_pointsofinterest_v2.reg", convert_units=True)
    # prepare_pdrt_tables_fir(reg_filename="catalogs/pillar123_pointsofinterest_v2.reg")
    # prepare_pdrt_tables_g0(reg_filename="catalogs/pillar123_pointsofinterest_v2.reg")

    # table_sample_peak_brightness_temperatures()
    # table_sample_column_densities()

    # for p in ['P1a-head', 'P2-head', 'P3-head']:
    #     for n in (1e4, 2e4):
    # lifetime_pressure_velocitydispersion_tradeoff(1e4, 'P1a-head')
    # lifetime_pressure_velocitydispersion_tradeoff(1.8e4, 'P1a-head')

    # calculate_co_column_density()
    # calculate_pillar_lifetimes_from_columndensity()
    # calculate_cii_column_density(filling_factor=1.0)
    # calculate_dust_column_densities(v=3)

    # estimate_uncertainty_mass_and_coldens(tracer='cii')
    # estimate_uncertainty_mass_and_coldens(tracer='co')
    # estimate_uncertainty_mass_and_coldens(tracer='dust')

    # for s in (None, 'W', 'E', 'N', 'S'):
    #     if s is None:
    #         s = ''
    #     else:
    #         s = ' ' + s
    #     fit_molecular_components_with_gaussians('peak'+s)

    # fit_molecular_components_with_gaussians('bluest component', cii=1, regrid=1)

    # kwargs = dict(pillar='1b', select=1)

    # fit_molecular_and_cii_with_gaussians(2, lines=['12co10CONV', 'hcopCONV', 'cii'], **kwargs)
    # fit_molecular_and_cii_with_gaussians(2, lines=['13co10CONV', 'hcnCONV', 'csCONV'], **kwargs)
    # fit_molecular_and_cii_with_gaussians(2, lines=['12co32', 'co65CONV', 'n2hpCONV'], **kwargs)

    # fit_molecular_and_cii_with_gaussians(1, lines=['cs', 'hcop', 'cii'], pillar=2, select='head')
    # fit_molecular_and_cii_with_gaussians(1, lines=['cs', 'hcop', 'cii'], pillar=3)
    # fit_molecular_and_cii_with_gaussians(1, lines=['csCONV', 'hcopCONV', 'cii'], pillar=2, select=1)
    # fit_molecular_and_cii_with_gaussians(1, lines=['csCONV', 'hcopCONV', 'cii'], pillar=2, select=2)


    # fit_spectrum_detailed('13co32', n_components=2, pillar='1b', reg_number=5)
    # fit_spectrum_detailed('12co32', n_components=3, pillar=1, reg_number=8)
    # fit_spectrum_detailed('hcnCONV', n_components=2, pillar=1, reg_number=3)
    # fit_spectrum_detailed('13co10CONV', n_components=2, pillar=1, reg_number=4)
    # fit_spectrum_detailed('cii', n_components=2, pillar=1, reg_number=3)

    # generate_n2hp_frequency_axis(debug=True)
    # fit_n2hp_peak(5)
    # save_n2hp_full_spectra()
    # save_n2hp_new_vaxis_cube()
    # for line in ['hcn', 'cs', 'n2hp']:
    #     save_hcop_and_cs_moment_imgs(line)

    # easy_pv_2() # most recently uncommented until Apr 19, 2022

    # make_3d_fit_viz_in_2d(3, line='hcop', version=2) # working on this april 19, 2022, sept 1, 2022
    """
    July 25, 2023 note: hcop: use version 2. cii: use version 1 (version just controls what models u load)
    """
    for i in range(1, 5):
        make_3d_fit_viz_in_2d(i, line='hcop', version=2) # working on this april 19, 2022, sept 1, 2022

    # correlations_between_carma_molecule_intensities('cs', 'n2hp') # working on this april 20, 2022

    # plot_selected_hcop_spectra_fits()
