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
    main_pv_fn = "carma/M16.ALL.hcop.sdi.cm.subpv.fits" # WRONG CUBE!!!!!!!!!!!!!!!!!!!!!!!! HCOP!!!!!!!!!!!!!!!!!!
    cube = cube_utils.CubeData(main_pv_fn).data
    # Is that necessary? Can it just wait for the loop?
    reg_filename = catalog.utils.search_for_file("catalogs/p1_IDgradients_thru_head.reg")

    selected_path = 0

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
    # plt.show()

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


def select_pixels_and_models(mol, i, var_mean=False, var_std=False):
    """
    November 5, 2021
    Easy selection of pixels and models
    :param mol: molecular/atomic line name (cii, 12co10, hcop)
    :param i: name of position (totally arbitrary, I decide the name)
    :param test_model: whether to let the mean float in the model
    :param var_std: whether to let the stddev float in the model
    """
    if mol == "cii":
        # Only one position for cii right now
        good_pixel = (37, 42)
        di, dj = 2, 2
        g = None


    elif mol == '12co10':
        ### This was my work for 12CO(1-0). These worked alright but not great for every area

        if i == 'bluest component':
            good_pixel = (466, 275) # good for bluest component
            di, dj = 2, 3
            g = cps2.models.Gaussian1D(amplitude=50, mean=23.8, stddev=1.06,
                bounds={'amplitude': (0, 200)})

        elif i == 'blue thread':
            good_pixel = (405, 287) # blue (W) thread
            di, dj = 2, 3
            g = cps2.models.Gaussian1D(amplitude=50, mean=25.1, stddev=0.95,
                bounds={'amplitude': (0, 200)})

        elif i == 'red main part':
            good_pixel = (408, 243) # red main part
            di, dj = 5, 5
            g = cps2.models.Gaussian1D(amplitude=50, mean=25.8, stddev=0.83,
                bounds={'amplitude': (0, 200)})


    elif mol == 'hcop':

        if i == 'western horn':
            # This is the Western horn component
            good_pixel = (447, 375)
            di, dj = 2, 2
            g = cps2.models.Gaussian1D(amplitude=10.291692169984568, mean=24.440935924615744, stddev=0.4614265241399322,
                bounds={"amplitude": (0, 100), "mean": (23, 27), "stddev": (0.1, 2)})

        elif i == 'bluest component':
            # This is the bluest N-E corner component
            good_pixel = (602, 415)
            di, dj = 1, 1
            g = cps2.models.Gaussian1D(amplitude=5.2758702607467525, mean=23.46286597585026, stddev=0.46, # fitted stddev = 0.4526447822523458
                bounds={"amplitude": (0, 30), "mean": (21, 26), "stddev": (0.1, 2)})

        elif i == 'bluest component 2':
            # Now even further out in the blue component
            good_pixel = (610, 420)
            di, dj = 1, 1
            g = cps2.models.Gaussian1D(amplitude=2.68, mean=23.516, stddev=0.46, # fitted stddev = 0.45380048744753915
                bounds={"amplitude": (0, 30), "mean": (21, 26), "stddev": (0.1, 2)})

        elif i == 'western thread N':
            # Western thread, from a pixel a little above it that shows a clean spectrum
            good_pixel = (544, 448)
            di, dj = 1, 1
            g = cps2.models.Gaussian1D(amplitude=5.305657851279191, mean=24.917103298134492, stddev=0.46, # fitted stddev = 0.44549289151047716
                bounds={"amplitude": (0, 30), "mean": (21, 27), "stddev": (0.1, 2)})

        elif i == 'just off peak':
            # Check out HCO+ (near) peak spectrum to see how its width stacks up with 0.46
            good_pixel = (570, 451)
            di, dj = 1, 1
            g = cps2.models.Gaussian1D(amplitude=20, mean=25, stddev=0.46, # fitted stddev = 0.44549289151047716
                bounds={"amplitude": (0, 30), "mean": (21, 27), "stddev": (0.1, 2)})

        elif i == 'eastern thread N':
            # Eastern thread N sample (trying to avoid the other thing that's there to the west)
            good_pixel = (556, 393)
            di, dj = 4, 4
            g = cps2.models.Gaussian1D(amplitude=4.27, mean=25.85, stddev=0.46, # fitted stddev = 0.72
                bounds={"amplitude": (1, 30), "mean": (25, 26.5), "stddev": (0.1, 2)})
        
        elif i == 'eastern thread S':
            # Eastern thread S sample
            good_pixel = (541, 373)
            di, dj = 6, 6
            g = cps2.models.Gaussian1D(amplitude=3.74, mean=25.76, stddev=0.46, # fitted stddev = 0.64
                bounds={"amplitude": (1, 30), "mean": (25, 26.5), "stddev": (0.1, 2)})

        elif i == 'main red':
            # just north of the Eastern thread, and probably the main red component in the peak
            good_pixel = (564, 425)
            di, dj = 2, 2
            g = cps2.models.Gaussian1D(amplitude=12.4, mean=25.35, stddev=0.46, # fitted stddev = 0.61
                bounds={"amplitude": (1, 30), "mean": (25, 26.5), "stddev": (0.1, 2)})

        elif i == 'main red plus blue':
            # this is a compound model!
            good_pixel = (583, 431)
            di, dj = 3, 3
            g_red = select_pixels_and_models('hcop', 'main red', var_mean=var_mean, var_std=var_std)[2]
            g_blue = select_pixels_and_models('hcop', 'bluest component', var_mean=var_mean, var_std=var_std)[2]
            g = g_red + g_blue
            tie_std_models(g)

    if not var_mean:
        fix_mean(g)
    if not var_std:
        fix_std(g)
    return good_pixel, (di, dj), g


def iter_models(model):
    """
    November 5, 2021
    Convenience function for iterating over models even if it's just one model (usually breaks)
    :param model: an astropy.modeling.models model, compound OR single
    :returns: iterator that will return a single model per iteration
    """
    try:
        return iter(model)
    except:
        return iter((model,))


def tie_std_models(model):
    """
    November 5, 2021
    Convenience function for tying all the stddevs together
    If they're already fixed, it shouldn't have any effect
    If single (not compound) model, no effect
    :param model: an astropy.modeling.models model
    """
    try:
        for i, m in enumerate(model):
            if i == 0:
                pass
            else:
                m.stddev.tied = lambda x: x.stddev_0
    except:
        pass


def fix_mean(model, set_to=True):
    """
    November 9, 2021
    Convenience function for fixing mean parameter for an unknown number
    of composite or single models
    If already fixed, no effect
    :param model: an astropy.modeling.models model
    :param set_to: whether to fix or unfix. Default is fix (mean cannot change)
    """
    for m in iter_models(model):
        m.mean.fixed = set_to


def fix_std(model, set_to=True):
    """
    November 9, 2021
    Convenience function for fixing stddev parameter for an unknown number
    of composite or single models
    If already fixed, no effect
    :param model: an astropy.modeling.models model
    :param set_to: whether to fix or unfix. Default is fix (stddev cannot change)
    """
    for m in iter_models(model):
        m.stddev.fixed = set_to


def make_show_box(show_box_i_lims, show_box_j_lims):
    """
    November 5, 2021
    """
    show_box_i_lo, show_box_i_hi = show_box_i_lims
    show_box_j_lo, show_box_j_hi = show_box_j_lims
    return (slice(show_box_i_lo, show_box_i_hi), slice(show_box_j_lo, show_box_j_hi))



def plot_noise_and_vlims(ax, noise, vel_lims):
    """
    November 5, 2021
    """
    [ax.axhline(sign*noise, color='grey', alpha=0.3, linestyle='--') for sign in (-1, 1)]
    [ax.axvline(v, color='grey', alpha=0.5) for v in vel_lims]


def plot_box(ax, i_lims, j_lims, show_box_lo_lims):
    """
    November 5, 2021
    """
    i_lo, i_hi = i_lims
    j_lo, j_hi = j_lims
    show_box_i_lo, show_box_j_lo = show_box_lo_lims
    box_x = np.array([j_lo, j_hi, j_hi, j_lo, j_lo]) - show_box_j_lo
    box_y = np.array([i_lo, i_lo, i_hi, i_hi, i_lo]) - show_box_i_lo
    ax.plot(box_x, box_y, color='grey')


def plot_noise_img(ax, noise_loc, show_box_lo_lims):
    """
    November 5, 2021
    """
    show_box_i_lo, show_box_j_lo = show_box_lo_lims
    ax.plot([noise_loc[1] - show_box_j_lo], [noise_loc[0] - show_box_i_lo], 'x', color='grey')


def plot_everything_about_models(ax, xaxis, spectrum, model, m_color='r', text_x=0.05, text_y=0.95, dy=-0.05):
    """
    November 5, 2021
    Convenience function for plotting all these models
    :param model: an astropy.modeling.models model
    """
    if spectrum is not None:
        ax.plot(xaxis, spectrum, color='k', linestyle='-', marker='|')
    if model is None:
        return
    fitted_spectrum = model(xaxis)
    ax.plot(xaxis, fitted_spectrum, color=m_color, linestyle='-')
    if spectrum is not None:
        ax.plot(xaxis, spectrum-fitted_spectrum, color='g', alpha=0.6, linestyle='--')
    for i, m in enumerate(iter_models(model)):
        ax.plot(xaxis, m(xaxis), color=m_color, linestyle='--', alpha=0.7)
        ax.axvline(m.mean, color=m_color, linestyle='--', alpha=0.3)
        ax.text(text_x, text_y + dy*(0 + 4*i), f"$A_{i}$ = {m.amplitude.value:5.2f}", transform=ax.transAxes, color=m_color)
        ax.text(text_x, text_y + dy*(1 + 4*i), f"$\mu_{i}$ = {m.mean.value:5.2f}", transform=ax.transAxes, color=m_color)
        ax.text(text_x, text_y + dy*(2 + 4*i), f"$\sigma_{i}$ = {m.stddev.value:5.2f}", transform=ax.transAxes, color=m_color)


def fit_molecular_components_with_gaussians():
    """
    Created October 22, 2021
    Try my hand at fitting with Gaussians again
    This time it's the CO (1-0) data (maybe....13...?)
    Try to find distinct components and see if they can be responsible for the CII profile without
    major velocity shifts
    """
    # cube_co = cube_utils.CubeData("bima/M16_12CO1-0_7x4.fits").convert_to_K().data.with_spectral_unit(kms)
    cube = cube_utils.CubeData("carma/M16.ALL.hcop.sdi.cm.subpv.fits").convert_to_K().data.with_spectral_unit(kms)
    # try with HCO+ smoothed first, then see if it holds up for unsmoothed
    # good_pixel = (446, 304)
    # di, dj = 20, 30

    img_idx = 4

    hcop_regions = ('western horn', 'bluest component', 'bluest component 2', # 0, 1, 2
        'western thread N', 'just off peak', # 3, 4
        'eastern thread N', 'eastern thread S', 'main red', # 5, 6, 7
        'main red plus blue', # 8
        )
    region_name = hcop_regions[8]
    print(region_name)
    good_pixel, (di, dj), g = select_pixels_and_models('hcop', region_name, var_mean=1, var_std=1)
    # # mean = 24.91
    # ## this is the western thread, it seems to always be around 25 km/s
    # good_pixel = (540, 430) # mean = 24.87
    # good_pixel = (527, 410) # 24.96
    # good_pixel = (512, 401) # 25.07
    # good_pixel = (487, 373) # 25.21
    # ## good example of probable multiple components

    # di, dj = 3, 3



    # Identify noise level
    noise_pixel = (644, 318) # noise for 2x2 pixels is ~0.5 K

    vel_lims = (23, 26)

    i_lims = i_lo, i_hi = tuple(good_pixel[0] + sign*di for sign in (-1, 1))
    j_lims = j_lo, j_hi = tuple(good_pixel[1] + sign*dj for sign in (-1, 1))

    noise_i = slice(*(noise_pixel[0] + sign*di for sign in (-1, 1)))
    noise_j = slice(*(noise_pixel[1] + sign*dj for sign in (-1, 1)))
    print(noise_i, noise_j)

    # vel_lims = (23, 24)
    mom0 = cube.spectral_slab(*(v*kms for v in vel_lims)).moment0()
    fig = plt.figure(figsize=(15, 7))


    ax_img = plt.subplot(121)
    show_box_i_lims = show_box_i_lo, show_box_i_hi = 370, 665 # 0, mom0.shape[0]
    show_box_j_lims = show_box_j_lo, show_box_j_hi = 277, 592 # 0, mom0.shape[1]
    show_box_lo_lims = (show_box_i_lo, show_box_j_lo)
    show_box = make_show_box(show_box_i_lims, show_box_j_lims)
    ax_img.imshow(mom0.to_value()[show_box], origin='lower')

    x_axis = cube.spectral_axis.to_value()
    spectrum = cube[:, i_lo:i_hi, j_lo:j_hi].mean(axis=(1, 2))

    ax_spec = plt.subplot(122)
    noise = np.std(cube[:, noise_i, noise_j].mean(axis=(1, 2)).to_value())
    ax_spec.set_ylabel(f"Noise: {noise:.3f}")

    # mark some things on each plot
    plot_noise_and_vlims(ax_spec, noise, vel_lims)
    plot_box(ax_img, i_lims, j_lims, show_box_lo_lims)
    plot_noise_img(ax_img, noise_pixel, show_box_lo_lims)

    mask = spectrum.to_value() > -100

    fitter = cps2.fitting.SLSQPLSQFitter()
    g_fit = fitter(g, x_axis[mask], spectrum.to_value()[mask],
        verblevel=1)
    print(g_fit)
    plot_everything_about_models(ax_spec, x_axis, spectrum.to_value(), g_fit)

    ax_spec.set_title("HCO+ spectrum from within box (see left), with fit")
    ax_img.set_title(f"HCO+ moment 0 between {vel_lims[0]}, {vel_lims[1]} km/s")
    reg_stub = region_name.replace(' ', '-')
    plt.savefig(f'/home/rkarim/Pictures/2021-10-26-work/fit_molecular_components_{reg_stub}.png')
    # plt.show()
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

def test_fitting_2_gaussians_with_2(dv=1, c_A=1, c_std=1, snr=20, numbers_only=False):
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
    y_obs += default_rng(12345).normal(scale=noise_rms, size=y_obs.shape)
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
    ax = plt.subplot(121)
    plt.title("2 Gaussians fit with 2 Gaussian")
    plot_everything_about_models(ax, x_axis, y_obs, g_all, m_color='k', text_x=0.8, text_y=0.3, dy=0.05)
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
    plot_everything_about_models(ax, high_res_x, None, g_fit)

    # plt.plot(high_res_x, g_fit(high_res_x), color='b', linestyle=':', label='Single component fit')
    # plt.axvline(g_fit.mean, color='b', linestyle=':', alpha=0.3)
    residuals = y_obs - g_fit(x_axis)
    # plt.plot(x_axis, residuals, color='grey', linestyle=':', label='Residuals')
    plt.legend()
    plt.show()


def test_fitting_2G_with_2G_wrapper():
    """
    November 10, 2021
    Loop over the previous function and see how things change as parameters and noise changes
    """
    ...


def fit_molecular_and_cii_with_gaussians():
    """
    Created October 27 2021, 25 minutes before my meeting
    Let's do this
    Fit the HCO+ peak with 2 Gaussians, check it on the 13CO(1-0) peak (optional)
    and then check it on the CII peak
    """
    cube = cube_utils.CubeData("carma/M16.ALL.hcop.sdi.cm.subpv.fits").convert_to_K().data.with_spectral_unit(kms)
    cube_cii = cps2.cutout_subcube(length_scale_mult=5)

    # HCO+ peak
    # good_pixel = (570, 451)
    # di, dj = 1, 1

    # more of HCO+ peak
    good_pixel = (575, 447)
    di, dj = 20, 25 #1
    di, dj = 25, 30 #2
    di, dj = 35, 40 #3

    g1 = cps2.models.Gaussian1D(amplitude=20, mean=24, stddev=0.46, # fitted stddev = 0.44549289151047716
        bounds={"amplitude": (0, 100), "mean": (21, 27), "stddev": (0.1, 2)})
    g2 = cps2.models.Gaussian1D(amplitude=20, mean=26, stddev=0.46, # fitted stddev = 0.44549289151047716
        bounds={"amplitude": (0, 100), "mean": (21, 27), "stddev": (0.1, 2)})

    # g3 = cps2.models.Gaussian1D(amplitude=20, mean=25, stddev=0.46, # fitted stddev = 0.44549289151047716
    #     bounds={"amplitude": (0, 100), "mean": (21, 27), "stddev": (0.1, 2)})

    g_all = g1 + g2 #+ g3
    g_all.stddev_1.tied = lambda m: m.stddev_0
    # g_all.stddev_0.fixed = g_all.stddev_1.fixed = g_all.stddev_2.fixed = True
    # g1.stddev.fixed = True
        # Identify noise level
    noise_pixel = (644, 318)
    # noise_pixel = (284, 573) # all these are consistent, noise for 2x2 pixels is ~0.5 K
    # noise_pixel = (200, 510)
    # good_pixel = noise_pixel
    # di, dj = 1, 1


    vel_lims = (23, 26)

    i_lo, i_hi = (good_pixel[0] + sign*di for sign in (-1, 1))
    j_lo, j_hi = (good_pixel[1] + sign*dj for sign in (-1, 1))

    noise_i = slice(*(noise_pixel[0] + sign*di for sign in (-1, 1)))
    noise_j = slice(*(noise_pixel[1] + sign*dj for sign in (-1, 1)))

    # vel_lims = (23, 24)
    mom0 = cube.spectral_slab(*(v*kms for v in vel_lims)).moment0()
    fig = plt.figure(figsize=(15, 7))


    ax_img = plt.subplot(221)
    show_box_i_lo, show_box_i_hi = 370, 665 # 0, mom0.shape[0]
    show_box_j_lo, show_box_j_hi = 277, 592 # 0, mom0.shape[1]
    show_box = (slice(show_box_i_lo, show_box_i_hi), slice(show_box_j_lo, show_box_j_hi))
    ax_img.imshow(mom0.to_value()[show_box], origin='lower')

    x_axis = cube.spectral_axis.to_value()
    spectrum = cube[:, i_lo:i_hi, j_lo:j_hi].mean(axis=(1, 2))
    ax_spec = plt.subplot(222)
    ax_spec.plot(x_axis, spectrum, color='k')
    noise = np.std(cube[:, noise_i, noise_j].mean(axis=(1, 2)).to_value())
    ax_spec.set_ylabel(f"Noise: {noise:.3f}")
    # mark some things on each plot
    [ax_spec.axvline(v, color='grey', alpha=0.5) for v in vel_lims]
    [ax_spec.axhline(sign*noise, color='grey', alpha=0.3, linestyle='--') for sign in (-1, 1)]
    box_x = np.array([j_lo, j_hi, j_hi, j_lo, j_lo]) - show_box_j_lo
    box_y = np.array([i_lo, i_lo, i_hi, i_hi, i_lo]) - show_box_i_lo
    ax_img.plot(box_x, box_y, color='grey')
    ax_img.plot([noise_pixel[1] - show_box_j_lo], [noise_pixel[0] - show_box_i_lo], 'x', color='grey')
    # TODO: plot the spectrum and use modeling to fit gaussian
    mask = spectrum.to_value() > -100

    fitter = cps2.fitting.SLSQPLSQFitter()

    g_fit = fitter(g_all, x_axis[mask], spectrum.to_value()[mask],
        verblevel=1)

    fitted_spectrum = g_fit(x_axis)
    ax_spec.plot(x_axis, fitted_spectrum, color='r', linestyle='--')
    for g in g_fit:
        ax_spec.axvline(g.mean, color='r', linestyle='--', alpha=0.3)
        ax_spec.plot(x_axis, g(x_axis), color='r', linestyle=':', alpha=0.3)
    print(g_fit)
    ax_spec.plot(cube.spectral_axis, spectrum.to_value()-fitted_spectrum, color='k', alpha=0.6, linestyle=':')

    text_x, text_y, dy = 0.05, 0.95, 0.05
    for i, g in enumerate(g_fit):
        ax_spec.text(text_x, text_y - 4*i*dy, f"$A_{i}$ = {g.amplitude.value:5.2f}", transform=ax_spec.transAxes)
        ax_spec.text(text_x, text_y - dy - 4*i*dy, f"$\mu_{i}$ = {g.mean.value:5.2f}", transform=ax_spec.transAxes)
        ax_spec.text(text_x, text_y - dy*2 - 4*i*dy, f"$\sigma_{i}$ = {g.stddev.value:5.2f}", transform=ax_spec.transAxes)
    ax_spec.set_title("HCO+ spectrum from within box (see left), with fit")
    ax_img.set_title(f"HCO+ moment 0 between {vel_lims[0]}, {vel_lims[1]} km/s")


    ax_img_cii = plt.subplot(223)
    ax_spec_cii = plt.subplot(224)

    mom0_cii = cube_cii.spectral_slab(*(v*kms for v in vel_lims)).moment0()
    ax_img_cii.imshow(mom0_cii.to_value(), origin='lower')
    x_axis = cube_cii.spectral_axis.to_value()

    good_pixel_cii = (37, 42)
    di, dj = 2, 2

    i_lo, i_hi = (good_pixel_cii[0] + sign*di for sign in (-1, 1))
    j_lo, j_hi = (good_pixel_cii[1] + sign*dj for sign in (-1, 1))

    spectrum = cube_cii[:, i_lo:i_hi, j_lo:j_hi].mean(axis=(1, 2))
    cii_bg_spectrum, artists = cps2.get_cii_background(cube_cii, return_artist=True, ec='w', linestyle='--', alpha=0.3, fill=False)

    ax_spec_cii.plot(x_axis, spectrum, color='k', alpha=0.3)
    ax_spec_cii.plot(x_axis, cii_bg_spectrum, color='k', linestyle='--', alpha=0.3)
    spectrum = spectrum - cii_bg_spectrum
    ax_spec_cii.plot(x_axis, spectrum, color='k')

    cii_noise = np.std(spectrum[cube_cii.spectral_axis < 17*kms].to_value())
    [ax_spec_cii.axhline(sign*cii_noise, color='grey', alpha=0.3, linestyle='--') for sign in (-1, 1)]

    [ax_spec_cii.axvline(v, color='grey', alpha=0.5) for v in vel_lims]
    box_x = np.array([j_lo, j_hi, j_hi, j_lo, j_lo]) - 0
    box_y = np.array([i_lo, i_lo, i_hi, i_hi, i_lo]) - 0
    ax_img_cii.plot(box_x, box_y, color='grey')

    g_cii_guess = g_fit.copy()
    g_cii_guess.stddev_0.fixed = g_cii_guess.stddev_1.fixed = True#g_cii_guess.stddev_2.fixed = False
    g_cii_guess.mean_0.fixed = g_cii_guess.mean_1.fixed = True # g_cii_guess.mean_2.fixed = True

    g_cii_guess.stddev_1.tied = lambda m : m.stddev_0
    # g_cii_guess.stddev_2.tied = lambda m : m.stddev_0

    g_fit_cii = fitter(g_cii_guess, x_axis, spectrum.to_value(), verblevel=1)
    fitted_cii_spectrum = g_fit_cii(x_axis)
    ax_spec_cii.plot(x_axis, fitted_cii_spectrum, color='r', linestyle='-')
    ax_spec_cii.plot(x_axis, spectrum.to_value() - fitted_cii_spectrum, color='g', linestyle=':')
    for g in g_fit_cii:
        ax_spec_cii.axvline(g.mean, color='r', linestyle='--', alpha=0.3)
        ax_spec_cii.plot(x_axis, g(x_axis), color='r', linestyle=':', alpha=0.3)
    for i, g in enumerate(g_fit_cii):
        ax_spec_cii.text(text_x, text_y - 4*i*dy, f"$A_{i}$ = {g.amplitude.value:5.2f}", transform=ax_spec_cii.transAxes)
        ax_spec_cii.text(text_x, text_y - dy - 4*i*dy, f"$\mu_{i}$ = {g.mean.value:5.2f}", transform=ax_spec_cii.transAxes)
        ax_spec_cii.text(text_x, text_y - dy*2 - 4*i*dy, f"$\sigma_{i}$ = {g.stddev.value:5.2f}", transform=ax_spec_cii.transAxes)

    ax_spec.set_xlim([16, 35])
    ax_spec_cii.set_xlim([16, 35])
    for bg_artist in artists:
        ax_img_cii.add_artist(bg_artist)

    plt.savefig(f'/home/rkarim/Pictures/2021-11-03-work/fit_molecular_components_and_CII_2g_fixedwidth_3.png')
    # plt.show()


def save_bgsub_cii():
    """
    November 4, 2021
    Save a background-subtracted version of the CII cube (small cutout)
    Use all 4 regions from catalogs/pillar_background_sample_multiple_4.reg
    """
    cii_bg_spectrum = cps2.get_cii_background()
    cii_cube = cps2.cutout_subcube(length_scale_mult=4)
    cii_cube = cii_cube - cii_bg_spectrum[:, np.newaxis, np.newaxis]
    # cii_cube.write(catalog.utils.m16_data_path + "sofia/M16_CII_pillar1_BGsubtracted.fits")
    ### Already ran this on Nov 4, 2021



if __name__ == "__main__":
    # Amplitudes = [1, 1.1, 1.25, 1.5, 1.7, 2, 2.5, 3, 3.5, 4, 5, 8, 10, 15]
    # Velocities = [0, 0.1, 0.2, 0.5, 0.7, 1, 1.25, 1.5, 1.8, 2, 2.5, 3, 3.5, 4, 5, 8]
    # Sigmas = [1, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    # for j, x in enumerate(Sigmas):
    #     test_fitting_2_gaussians_with_1(j, x, "changingSigma")

    # fit_molecular_components_with_gaussians()
    test_fitting_2_gaussians_with_2()
