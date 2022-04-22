"""
A designated place for even MORE nice M16 images
Descended from m16_investigation and m16_pictures

Created: July 9, 2021
Starting with investigating the "light feature" over P2 and repeating some
transverse PV diagrams
I hope the paper is within reach. Maybe august?? Rise and grind

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

from math import ceil
from scipy import signal
from scipy.interpolate import UnivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.table import Table, QTable

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
    ax_img.text(pv_path._coords[1].ra.deg, pv_path._coords[1].dec.deg - 4*u.arcsec.to(u.deg), f'Offset = {pv_path_length.to_value():.1f}\"', color='red', fontsize=10, va='top', ha='center', transform=ax_img.get_transform('world'))
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


def prepare_pdrt_tables():
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
    """

    # vel_lims = (20*kms, 24*kms) # P2 regions
    vel_lims = (20*kms, 28*kms) # P1 regions

    cii_cube = cps2.cutout_subcube(length_scale_mult=4, reg_index=2)
    cii_restfreq = cii_cube.header['RESTFRQ'] * u.Hz
    cii_mom0 = cii_cube.spectral_slab(*vel_lims).moment0()
    del cii_cube

    cii_vel_to_freq_equiv = u.doppler_optical(cii_restfreq)
    cii_vel_to_freq_f = lambda v: v.to(u.Hz, equivalencies=cii_vel_to_freq_equiv)
    cii_dv = (cii_vel_to_freq_f(vel_lims[0]) - cii_vel_to_freq_f(vel_lims[1]))
    ### that's a conversion factor from km/s (over the moment0 integration limits) to Hz
    """
    2022-01-17 comment: I'm not sure if the moment integration limits are the
    important spectral quantity here. I think K km/s is kinda just saying
    K * Hz but with km/s instead, so the relevant quantity would be the channel
    width integrated over in the moment calculation... I should revisit this!
    TODO: revisit K km/s -> CGI unit conversion, remake pdrt tables
    """

    co10_cube = cube_utils.CubeData("bima/M16_12CO1-0_14x14.fits").convert_to_K().data
    co10_restfreq = co10_cube.header['RESTFRQ'] * u.Hz
    co10_mom0 = co10_cube.spectral_slab(*vel_lims).moment0().to(u.K*kms)
    del co10_cube

    reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/pdrt/pdrt_test_pillar1.reg"))
    for i, reg in enumerate(reg_list):
        pixreg = reg.to_pixel(cii_mom0.wcs)
        reg_mask = pixreg.to_mask().to_image(cii_mom0.shape)

        cii_values = cii_mom0[np.where(reg_mask==1)]
        cii_values = (cii_values/kms)
        cii_values = cii_values.to(u.Jy/u.sr, equivalencies=u.brightness_temperature(cii_restfreq))
        cii_values = (cii_values * cii_dv).to(u.erg / (u.s * u.sr * u.cm**2))
        cii_ID = ['CII_158']*len(cii_values)
        cii_uncertainty = [10.]*len(cii_values) # using 10%
        ### CII noise RMS is about 1; but this DOESNT account for the moment0 part!!!! needs a sqrt(N) term...

        t = QTable([cii_values, cii_uncertainty, cii_ID], names=('data', 'uncertainty', 'identifier'), meta={'name': 'CII_158'}, units={'uncertainty': '%'})
        t.write(f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt/cii_pillar1_{i}.txt", format='ipac')

        co10_vel_to_freq_equiv = u.doppler_optical(co10_restfreq)
        co10_vel_to_freq_f = lambda v: v.to(u.Hz, equivalencies=co10_vel_to_freq_equiv)
        co10_dv = co10_vel_to_freq_f(vel_lims[0]) - co10_vel_to_freq_f(vel_lims[1])

        co10_reproj = reproject_interp((co10_mom0.to_value(), co10_mom0.wcs), cii_mom0.wcs, shape_out=cii_mom0.shape, return_footprint=False)
        co10_values = co10_reproj[np.where(reg_mask==1)] * co10_mom0.unit
        co10_values = (co10_values/kms)
        co10_values = co10_values.to(u.Jy/u.sr, equivalencies=u.brightness_temperature(co10_restfreq))
        co10_values = (co10_values * co10_dv).to(u.erg / (u.s * u.sr * u.cm**2))

        cii_to_co10_values = cii_values/co10_values

        cii_to_co10_ID = ["CII_158/CO_10"]*len(cii_values) # CHECK THIS
        cii_to_co10_uncertainty = cii_uncertainty # also use 10%
        # co10_uncertainty = [4]*len(cii_to_co10_ID) # CHECK THIS TOO; needs sqrt(N) term.... but N depends on number of channels.. CO10 RMS is around 4

        t = QTable([cii_to_co10_values, cii_to_co10_uncertainty, cii_to_co10_ID], names=('data', 'uncertainty', 'identifier'), meta={'name': 'CII_158/CO_10'}, units={'uncertainty': '%'})
        t.write(f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt/cii_to_co10_pillar1_{i}.txt", format='ipac')
    """
    TODO: convert K km/s to erg cm-2 s-1 sr-1 ????????
    Solved:
    >>> from astropy import units as u
    >>> v = 3*u.km/u.s
    >>> restfreq = (158*u.micron).to(u.GHz, equivalencies=u.spectral())
    >>> restfreq
    <Quantity 1897.42062025 GHz>

    >>> vel_to_freq = u.doppler_optical(restfreq)
    >>> (1*u.km/u.s).to(u.GHz, equivalencies=vel_to_freq)
    <Quantity 1897.41429116 GHz>
    >>> (50*u.km/u.s).to(u.GHz, equivalencies=vel_to_freq)
    <Quantity 1897.10421733 GHz>
    >>> vel_to_freq_f = lambda v : v.to(u.Hz, equivalencies=vel_to_freq)

    >>> kms = u.km/u.s
    >>> vel_to_freq_f(24*kms) - vel_to_freq_f(20*kms)
    <Quantity -25312740.45532227 Hz>
    >>> (vel_to_freq_f(20*kms) - vel_to_freq_f(24*kms)).to(u.GHz)
    <Quantity 0.02531274 GHz>
    >>> (vel_to_freq_f(20*kms) - vel_to_freq_f(24*kms))
    <Quantity 25312740.45532227 Hz>

    >>> T = 10*u.K
    >>> (T).to(u.Jy/u.sr, equivalencies=u.brightness_temperature(restfreq))
    <Quantity 1.106112e+12 Jy / sr>
    >>> jysr = (T).to(u.Jy/u.sr, equivalencies=u.brightness_temperature(restfreq))
    >>> jysr
    <Quantity 1.106112e+12 Jy / sr>
    >>> (vel_to_freq_f(20*kms) - vel_to_freq_f(24*kms))
    <Quantity 25312740.45532227 Hz>
    >>> dv = (vel_to_freq_f(20*kms) - vel_to_freq_f(24*kms))
    >>> (jysr*dv).to(u.erg / (u.s * u.sr * u.cm**2))
    <Quantity 0.00027999 erg / (cm2 s sr)>
    based on the pdrtpy-nb/notebooks/rcw49_nc_cii158.tab, this is of the right order of magnitude
    ok go forth and prosper
    """


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


def fit_molecular_and_cii_with_gaussians():
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
    """
    # reg_filename_short = "catalogs/pillar1_emissionpeaks.hcopregrid.moreprecise.reg" # order appears to be [HCO+, CII]
    # reg_filename_short = "catalogs/p1_threads_pathsandpoints.reg" # order appears to be North-E, North-W, South-E, South-W
    # reg_filename_short = "catalogs/pillar1_pointsofinterest.reg" # order is wide profile, blue tail, north of west thread, bluest component, top of western thread (where IRAC4 peaks), above western thread (where IRAC4 is dark)
    # reg_filename_short = "catalogs/pillar1_peak_degeneracyboundary.reg" # order is 3 component peak, 2 component peak
    reg_filename_short = "catalogs/pillar1_pointsofinterest_v2.reg" # using the first and second, which is wide profile and 3c peak (E of center)
    sky_regions = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    # now supporting multiple selected_regions
    selected_region = list(sky_regions[:2])
    if isinstance(selected_region, list):
        pixel_name = "-and-".join([reg.meta['label'].replace(" ", '-') for reg in selected_region])
    else:
        pixel_name = selected_region.meta['label'].replace(" ", '-')
    # TODO left off here using this to show that the 3c-fixed HCO+ fit makes a little bit of sense with CII if you do the rgb theory

    # cii_cube = cube_utils.CubeData("sofia/M16_CII_pillar1_BGsubtracted.fits").data
    cii_cube = cps2.cutout_subcube(length_scale_mult=4.0)
    cii_cube = cii_cube - cps2.get_cii_background()[:, np.newaxis, np.newaxis]
    regrid = True # Just in case I want to switch back to regular resolution? doesn't hurt
    fn = f"carma/M16.ALL.hcop.sdi.cm.subpv{'.SOFIAbeam.regrid' if regrid else ''}.fits"
    hcop_cube = cube_utils.CubeData(fn).convert_to_K().data.with_spectral_unit(kms)
    hcop_flat_wcs = hcop_cube[0, :, :].wcs

    if isinstance(selected_region, list):
        selected_pixel = [tuple(round(x) for x in reg.to_pixel(hcop_flat_wcs).center.xy[::-1]) for reg in selected_region]
    else:
        selected_pixel = tuple(round(x) for x in selected_region.to_pixel(hcop_flat_wcs).center.xy[::-1])


    # selected_region = 1
    # selected_pixel = (31, 28) # 31, 28 is good for exploring that blue tail thing on Eastern thread
    # pixel_name = "_DEBUGPIXEL_01_"

    # pixel_coords = [tuple(round(x) for x in reg.to_pixel(hcop_flat_wcs).center.xy[::-1]) for reg in sky_regions] # converted to (i, j) tuples
    # selected_pixel = pixel_coords[0]

    # Start with one pixel, just fit that first
    assert regrid
    if isinstance(selected_region, list):
        cii_spectrum = [cii_cube[(slice(None), *px_coords)].to_value() for px_coords in selected_pixel]
        hcop_spectrum = [hcop_cube[(slice(None), *px_coords)].to_value() for px_coords in selected_pixel]
    else:
        cii_spectrum = cii_cube[(slice(None), *selected_pixel)].to_value()
        hcop_spectrum = hcop_cube[(slice(None), *selected_pixel)].to_value()
    cii_x = cii_cube.spectral_axis.to_value()
    hcop_x = hcop_cube.spectral_axis.to_value()
    # Set up plots
    fig = plt.figure(figsize=(15, 10))
    if isinstance(selected_region, list):
        ax_cii_img = plt.subplot2grid((2, 3), (0, 2))
        ax_hcop_img = plt.subplot2grid((2, 3), (1, 2))
        ax_cii_spec = [plt.subplot2grid((2, 3), (0, 0)), plt.subplot2grid((2, 3), (0, 1))]
        ax_hcop_spec = [plt.subplot2grid((2, 3), (1, 0)), plt.subplot2grid((2, 3), (1, 1))]
    else:
        ax_cii_img = plt.subplot2grid((2, 3), (0, 0))
        ax_hcop_img = plt.subplot2grid((2, 3), (1, 0))
        ax_cii_spec = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        ax_hcop_spec = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    # Plot images
    vel_lims = (23, 26)
    ax_cii_img.imshow(cii_cube.spectral_slab(*(v*kms for v in vel_lims)).moment0().to_value(), origin='lower', cmap='plasma')
    ax_hcop_img.imshow(hcop_cube.spectral_slab(*(v*kms for v in vel_lims)).moment0().to_value(), origin='lower', cmap='plasma')

    if isinstance(selected_region, list):
        # [cps2.plot_box(ax_cii_img, *((x-0.5, x+0.5) for x in px_coords), (0, 0)) for px_coords in selected_pixel]
        # [cps2.plot_box(ax_hcop_img, *((x-0.5, x+0.5) for x in px_coords), (0, 0)) for px_coords in selected_pixel]
        pad = 1.5
        for ax in [ax_cii_img, ax_hcop_img]:
            for idx, (y, x) in enumerate(selected_pixel):
                ax.plot([x], [y], 'o', markersize=5, color='k')
                dx = pad if idx else -pad
                dy = pad
                ax.text(x+dx, y+dy, str(idx+1), color='k', fontsize=12, ha='center', va='center')
    else:
        cps2.plot_box(ax_cii_img, *((x-0.5, x+0.5) for x in selected_pixel), (0, 0))
        cps2.plot_box(ax_hcop_img, *((x-0.5, x+0.5) for x in selected_pixel), (0, 0))

    # Do noise stuff
    noise_cii = 1 # 1 K has been my estimate for a while
    noise_hcop = 0.12 # estimated from the cube, lower than the original 0.5 due to smoothing to CII beam and rebinning to CII channels
    if isinstance(selected_region, list):
        [cps2.plot_noise_and_vlims(ax, noise_cii, vel_lims) for ax in ax_cii_spec]
        [cps2.plot_noise_and_vlims(ax, noise_hcop, vel_lims) for ax in ax_hcop_spec]
        for ax in (ax_cii_spec + ax_hcop_spec):
            ax.set_xlim(15, 30)
    else:
        cps2.plot_noise_and_vlims(ax_cii_spec, noise_cii, vel_lims)
        cps2.plot_noise_and_vlims(ax_hcop_spec, noise_hcop, vel_lims)
        for ax in [ax_cii_spec, ax_hcop_spec]:
            ax.set_xlim(20, 30)

    # Set up Gaussian model, start with one component
    # Decide which things are fixed
    fixedstd = True
    tiestd = True
    untieciistd = False
    fixed_cii_std = True
    fixedmean = True
    stddev_hcop = 0.55
    g0 = cps2.models.Gaussian1D(amplitude=10, mean=23.5, stddev=stddev_hcop,
        bounds={'amplitude': (0, None), 'mean': (20, 30)})
    g0.mean = 24
    g1 = g0.copy()
    g1.mean = 25
    g2 = g0.copy()
    g2.mean = 26
    g = g0 + g1 + g2
    # Now do the fitting
    fitter = cps2.fitting.LevMarLSQFitter(calc_uncertainties=True)
    if tiestd:
        cps2.tie_std_models(g)
    if fixedstd:
        cps2.fix_std(g)
    ndim_hcop = len(get_fittable_param_names(g))

    def fit_and_plot(g, spec_hcop, spec_cii, spec_ax_hcop, spec_ax_cii):
        """
        Helper function to help when there's two regions
        """
        g_fit_hcop = fitter(g, hcop_x, spec_hcop, weights=np.full(spec_hcop.size, 1.0/noise_hcop))
        if fixedmean:
            g = g_fit_hcop.copy()
            # g.mean_0 = 23.5
            # g.mean_1 = 25.50
            cps2.fix_mean(g)
        if untieciistd:
            cps2.tie_std_models(g, untie=True)
        else:
            cps2.tie_std_models(g)
        if not fixed_cii_std:
            cps2.unfix_std(g)
        else:
            for m in cps2.iter_models(g):
                m.stddev = 1.
            cps2.fix_std(g)
        ndim_cii = len(get_fittable_param_names(g))
        g_fit_cii = fitter(g, cii_x, spec_cii, weights=np.full(spec_cii.size, 1.0/noise_cii))
        # Plot the fits
        cps2.plot_everything_about_models(spec_ax_cii, cii_x, spec_cii, g_fit_cii, noise=noise_cii, dof=(cii_x.size - ndim_cii))
        cps2.plot_everything_about_models(spec_ax_hcop, hcop_x, spec_hcop, g_fit_hcop, noise=noise_hcop, dof=(hcop_x.size - ndim_hcop))

    if isinstance(selected_region, list):
        for i in range(2):
            fit_and_plot(g.copy(), hcop_spectrum[i], cii_spectrum[i], ax_hcop_spec[i], ax_cii_spec[i])
    else:
        fit_and_plot(g, hcop_spectrum, cii_spectrum, ax_hcop_spec, ax_cii_spec)

    if isinstance(selected_region, list):
        for ax in ax_hcop_spec + ax_cii_spec + [ax_cii_img, ax_hcop_img]:
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_tick_params(direction='in', which='both')
            ax.yaxis.set_tick_params(direction='in', which='both')
        for line_name, ax in zip(['CII', 'HCO+'], [ax_cii_img, ax_hcop_img]):
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.text(0.9, 0.9, f'{line_name}', fontsize=15, ha='center', va='center', transform=ax.transAxes)
        for ax in ax_cii_spec:
            ax.xaxis.set_ticklabels([])
        for ax in [ax_cii_spec[1], ax_hcop_spec[1]]:
            ax.yaxis.set_ticklabels([])
        for idx, ax in enumerate(ax_cii_spec):
            ax.text(0.9, 0.9, f'CII\n{idx+1}', fontsize=15, ha='center', va='center', transform=ax.transAxes)
            ax.set_ylim([-3, 30])
        for idx, ax in enumerate(ax_hcop_spec):
            ax.text(0.9, 0.9, f'HCO+\n{idx+1}', fontsize=15, ha='center', va='center', transform=ax.transAxes)
            ax.set_ylim([-1, 15])
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)

    fixedstd_stub = f"_fixedstd{stddev_hcop:04.2f}" if fixedstd else ''
    tiestd_stub = f"_hcopUNtiedstd" if not tiestd else ''
    untieciistd_stub = f"_ciiUNtiedstd" if untieciistd else ''
    fixedciistd_stub = "_ciifixedstd" if fixed_cii_std else ''
    fixedmean_stub = f"_fixedciimean" if fixedmean else '_fittingciimean'
    # plt.savefig(f'/home/ramsey/Pictures/2021-12-21-work/fit_{g.n_submodels}molecular_components_and_CII_{pixel_name}_{fixedstd_stub}{tiestd_stub}{untieciistd_stub}{fixedmean_stub}.png')
    fig.savefig(f"/home/ramsey/Pictures/2022-01-20-work/fit_{g.n_submodels}_hcop_cii_{pixel_name}_{fixedstd_stub}{tiestd_stub}{untieciistd_stub}{fixedciistd_stub}{fixedmean_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=f'regions from {reg_filename_short}',
            file=__file__, func='fit_molecular_and_cii_with_gaussians'))
    # plt.show()



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



def make_3d_fit_viz_in_2d(n_submodels=3, line='hcop', version='3'):
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
    ii, jj = tuple(x.ravel() for x in np.mgrid[0:shape[0], 0:shape[1]])
    i_array = []
    j_array = []
    if n_submodels > 1:
        for k in range(n_submodels):
            means.extend(hdul[f'mean_{k}'].data[:].ravel())
            amplitudes.extend(hdul[f'amplitude_{k}'].data[:].ravel())
            i_array.extend(ii)
            j_array.extend(jj)
    else:
        means = list(hdul['mean'].data[:].ravel())
        amplitudes = list(hdul['amplitude'].data[:].ravel())
        i_array = list(ii)
        j_array = list(jj)

    means = np.array(means)
    amplitudes = np.array(amplitudes)
    i_array = np.array(i_array)
    j_array = np.array(j_array)
    if line == 'hcop':
        amp_cutoff = 2.5
    elif line == 'hcop_regrid':
        amp_cutoff = 0.6
    else:
        amp_cutoff = 5
    amp_mask = amplitudes > amp_cutoff # about 5sigma
    means = means[amp_mask]
    amplitudes = amplitudes[amp_mask]
    i_array = i_array[amp_mask]
    j_array = j_array[amp_mask]

    fig = plt.figure()
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224)
    # ax.hist(means, bins=256, range=(20, 30))
    im1 = ax1.hist2d(j_array, means, bins=64)[3]
    fig.colorbar(im1, ax=ax1)
    # ax1.set_xlabel("RA")
    ax1.set_ylabel("Velocity (km/s)")
    # ax.invert_xaxis()

    im2 = ax2.hist2d(j_array, i_array, bins=64)[3]
    fig.colorbar(im2, ax=ax2)
    ax2.set_xlabel("RA")
    ax2.set_ylabel("Dec")

    im3 = ax3.hist2d(means, i_array, bins=64)[3]
    fig.colorbar(im3, ax=ax3)
    ax3.set_xlabel("Velocity (km/s)")
    ax3.set_ylabel("Dec")


    plt.show()
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



if __name__ == "__main__":
    # Amplitudes = [1, 1.1, 1.25, 1.5, 1.7, 2, 2.5, 3, 3.5, 4, 5, 8, 10, 15]
    # Velocities = [0, 0.1, 0.2, 0.5, 0.7, 1, 1.25, 1.5, 1.8, 2, 2.5, 3, 3.5, 4, 5, 8]
    # Sigmas = [1, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    # for j, x in enumerate(Sigmas):
    #     test_fitting_2_gaussians_with_1(j, x, "changingSigma")

    # for s in (None, 'W', 'E', 'N', 'S'):
    #     if s is None:
    #         s = ''
    #     else:
    #         s = ' ' + s
    #     fit_molecular_components_with_gaussians('peak'+s)

    # fit_molecular_components_with_gaussians('bluest component', cii=1, regrid=1)
    # fit_molecular_and_cii_with_gaussians()

    # easy_pv_2() # most recently uncommented until Apr 19, 2022

    # make_3d_fit_viz_in_2d(3, line='hcop', version=2) # working on this april 19, 2022
    correlations_between_carma_molecule_intensities('cs', 'n2hp') # working on this april 20, 2022

    # test_fitting_uncertainties_with_emcee(which_line='cii')
    # investigate_emcee_result('cii')

    # test_fitting_2_gaussians_with_2(snr=50)
    # test_fitting_2G_with_2G_wrapper()

    # plot_selected_hcop_spectra_fits()
