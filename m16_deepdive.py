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
    """
    # This PV will be in color and perhaps also in contour
    main_pv_fn = "sofia/M16_CII_U.fits"
    cube = cube_utils.CubeData(main_pv_fn).data
    # Is that necessary? Can it just wait for the loop?
    reg_filename = catalog.utils.search_for_file("catalogs/filament_p23south.reg")
    pv_path = pvdiagrams.path_from_ds9(reg_filename, 0, width=15*u.arcsec)
    sl = pvextractor.extract_pv_slice(cube.spectral_slab(17*kms, 24*kms), pv_path)
    sl_wcs = WCS(sl.header)
    ax_sl = plt.subplot(111, projection=sl_wcs)
    ax_sl.imshow(sl.data, origin='lower', aspect=2)
    ax_sl.coords[1].set_format_unit(u.km/u.s)
    ax_sl.coords[1].set_major_formatter('x.xx')
    ax_sl.coords[0].set_format_unit(u.arcsec)
    ax_sl.coords[0].set_major_formatter('x.xx')
    ax_sl.contour(sl.data, colors='k')
    plt.show()




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


def fit_co10_components_with_gaussians():
    """
    Created October 22, 2021
    Try my hand at fitting with Gaussians again
    This time it's the CO (1-0) data (maybe....13...?)
    Try to find distinct components and see if they can be responsible for the CII profile without
    major velocity shifts
    """
    cube_co = cube_utils.CubeData("bima/M16_12CO1-0_7x4.fits").convert_to_K().data.with_spectral_unit(kms)
    # try with HCO+ too after we debug
    # good_pixel = (446, 304)
    # di, dj = 20, 30

    # good_pixel = (466, 275) # good for bluest component
    # di, dj = 2, 3
    g1 = cps2.models.Gaussian1D(amplitude=50, mean=23.8, stddev=1.06,
        bounds={'amplitude': (0, 200)})
    g1.mean.fixed = g1.stddev.fixed = True


    # good_pixel = (405, 287) # blue (W) thread
    # di, dj = 2, 3
    g2 = cps2.models.Gaussian1D(amplitude=50, mean=25.1, stddev=0.95,
        bounds={'amplitude': (0, 200)})
    g2.mean.fixed = g2.stddev.fixed = True

    # good_pixel = (408, 243) # red main part
    # di, dj = 5, 5
    g3 = cps2.models.Gaussian1D(amplitude=50, mean=25.8, stddev=0.83,
        bounds={'amplitude': (0, 200)})
    g3.mean.fixed = g3.stddev.fixed = True

    g_all = g1 + g2 + g3


    # test regions
    # good_pixel = (443, 270)
    # good_pixel = (440, 288)
    # good_pixel = (440, 305)
    # good_pixel = (430, 315)
    # good_pixel = (425, 340)
    good_pixel = (393, 275)
    # good_pixel = (417, 241)
    di, dj = 5, 5


    i_lo, i_hi = (good_pixel[0] + sign*di for sign in (-1, 1))
    j_lo, j_hi = (good_pixel[1] + sign*dj for sign in (-1, 1))
    vel_lims = (24.5, 25.5)
    # vel_lims = (23, 24)
    mom0 = cube_co.spectral_slab(*(v*kms for v in vel_lims)).moment0()
    ax_img = plt.subplot(121)
    ax_img.imshow(mom0.to_value(), origin='lower')
    spectrum = cube_co[:, i_lo:i_hi, j_lo:j_hi].mean(axis=(1, 2))
    ax_spec = plt.subplot(122)
    ax_spec.plot(cube_co.spectral_axis, spectrum, color='k')
    # mark some things on each plot
    [ax_spec.axvline(v, color='grey', alpha=0.5) for v in vel_lims]
    ax_img.plot([j_lo, j_hi, j_hi, j_lo, j_lo], [i_lo, i_lo, i_hi, i_hi, i_lo], color='k')
    # TODO: plot the spectrum and use modeling to fit gaussian
    fitter = cps2.fitting.SLSQPLSQFitter()
    mask = spectrum.to_value() > 45
    x_axis = cube_co.spectral_axis.to_value()
    g_fit = fitter(g_all, x_axis[mask], spectrum.to_value()[mask],
        verblevel=1)
    fitted_spectrum = g_fit(x_axis)
    ax_spec.plot(cube_co.spectral_axis, fitted_spectrum, color='r', linestyle='--')
    print(g_fit)
    ax_spec.plot(cube_co.spectral_axis, spectrum.to_value()-fitted_spectrum, color='k', alpha=0.6, linestyle=':')
    plt.show()
    return


if __name__ == "__main__":
    fit_co10_components_with_gaussians()
