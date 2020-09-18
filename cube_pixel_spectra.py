import numpy as np
import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 7}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import sys
import warnings

from scipy.optimize import curve_fit
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.nddata.utils import Cutout2D
import regions
from math import ceil

from . import misc_utils
from . import catalog
from . import cube_utils
from . import pvdiagrams
from . import crosscut

"""
Looking at individual or groups of pixel spectra from cubes.
The goal is to compare line centers, high velocity wings, etc.

Perhaps even modeling the lines? Comparing 12 and 13 C/CO lines?
Created: August 5, 2020
    (while listening to Cristian's SOFIA talk on optically thick CII)
Updated: September 9, 2020
    (after listening to that CO-dark gas SOFIA talk)
    I want to try overlaying other spectral from further out, like the image
    I made in DS9, in 9-09-20-work/
    I also want to try a sort of "spectrum cross cut". Like, same idea as a PV
    diagram, but with more emphasis on the spectra, and averaged over a small
    area
"""


# M16
filenames = [
    "apex/M16_12CO3-2.fits", "apex/M16_13CO3-2.fits",
    "bima/M16_12CO1-0_APEXbeam.fits",
    "sofia/M16_CII_U_APEXbeam.fits",
    ]
fn = filenames[3]

m16_regions_path = f"{catalog.utils.m16_data_path}catalogs/"


def retrieve_region(reg_list, list_index, color=None, width=None):
    reg = reg_list[list_index]
    if color is not None:
        reg.visual['color'] = color
    if width is not None:
        reg.visual['width'] = width
    return reg


def pixel_from_sky(reg_sky, wcs_obj):
    reg_pixel = reg_sky.to_pixel(wcs_obj)
    reg_pixel.visual.update(reg_sky.visual)
    return reg_pixel

def extract_spectrum_from_region(cube, reg_sky):
    subcube = cube.data.subcube_from_regions([reg_sky])
    spectrum = subcube.mean(axis=(1, 2))
    return spectrum


def plot_spectrum_from_region(cube_or_spectral, spectrum_or_region,
    ax=None, label=None):
    """
    As the second argument, can either take the spectrum (already extracted)
    or the region from which a spectrum should be extracted
    """
    if isinstance(spectrum_or_region, regions.SkyRegion):
        spectrum = extract_spectrum_from_region(cube, reg_sky)
    else:
        spectrum = spectrum_or_region
        if hasattr(spectrum, 'unit'):
            spectrum = spectrum.to_value()
    plot_args = (cube.spectral_axis/1e3, spectrum)
    plot_kwargs = {'color': reg_sky.visual['color'], 'linewidth': 0.7, 'label': label}
    if ax is not None:
        ax.plot(*plot_args, **plot_kwargs)
    else:
        plt.plot(*plot_args, **plot_kwargs)
    return spectrum


def fit_gaussian(cube, spectrum, p0=None):
    """
    Fit one gaussian to a spectrum
    p0 should be a list of guesses for:
        mu, sigma, amplitude
    Uses the cube.spectral_axis as the x axis
    :returns: mu, sigma, amplitude
    """
    x = cube.data.spectral_axis.to_value()
    y = spectrum.to_value()
    return curve_fit(misc_utils.gaussian, x, y, p0=p0)[0]


def simple_fit_attempt():
    cube = cube_utils.CubeData(fn)

    reg_list = regions.read_ds9(f"{m16_regions_path}pillar_1_tip.reg")

    pillar_1_tip_sky = retrieve_region(reg_list, 0, color='red', width=2)
    pillar_1_body = retrieve_region(reg_list, 3, color='orange', width=2)
    off_1 = retrieve_region(reg_list, 1, color='grey', width=2)
    off_2 = retrieve_region(reg_list, 2, color='black', width=2)

    tip_approx_center = SkyCoord(pillar_1_tip_sky.vertices.ra.mean(), pillar_1_tip_sky.vertices.dec.mean(), frame=pillar_1_tip_sky.vertices.frame)

    mom0 = cube.data.moment(order=0).to_value()
    mom0_cutout = Cutout2D(mom0, tip_approx_center, [6*u.arcmin]*2, wcs=cube.wcs_flat, mode='partial', fill_value=np.nan)

    mom0_flat = mom0_cutout.data.flatten()
    lo, up = misc_utils.flquantiles(mom0_flat[np.isfinite(mom0_flat)], 25)
    ax1 = plt.subplot(121, projection=mom0_cutout.wcs)
    plt.imshow(mom0_cutout.data, origin='lower', vmin=lo, vmax=up)

    ax2 = plt.subplot(122)
    spectra = []
    for reg_sky, label in zip([pillar_1_tip_sky, pillar_1_body, off_1, off_2], ['pillar tip', 'pillar head', 'off_1', 'off_2']):
        reg_pix = pixel_from_sky(reg_sky, mom0_cutout.wcs)
        ax1.add_artist(reg_pix.as_artist())
        spectra.append(plot_spectrum_from_region(cube, reg_sky, ax=ax2, label=label))

    off_2_spectrum = spectra[3]
    gaussian_params = fit_gaussian(cube, off_2_spectrum, p0=[3e4, 3e3, 5.*np.sqrt(2)*3e3])
    print(gaussian_params)
    ax2.plot(cube.data.spectral_axis/1e3, misc_utils.gaussian(cube.data.spectral_axis.to_value(), *gaussian_params), '--', color='black', linewidth='1', label='fit to off_2')

    plt.xlabel("Velocity (km/s)")
    plt.ylabel("[CII] brightness (K)")
    ax2.set_ylim([-3, 25])
    plt.sca(ax2)
    plt.legend()

    plt.show()


def attempt_spectral_cross_cut():
    warnings.filterwarnings("ignore")
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # color_iter = iter(prop_cycle.by_key()['color'])
    color_iter = iter(['DarkGreen', 'DarkMagenta', 'DarkOrange', 'RoyalBlue'])

    circle_radius = 15. * u.arcsec

    reg_index = 9
    global_center_coord, length_scale, location_generator = pvdiagrams.linear_series_from_description(
        *crosscut.coords_from_region(catalog.utils.search_for_file("catalogs/m16_lines_of_interest.reg"), index=reg_index),
        None, None, pvpath_width=circle_radius*4, points_not_paths=True
    )

    sky_region_list = []
    for loc in location_generator:
        circle_reg = regions.CircleSkyRegion(loc, circle_radius, visual=regions.RegionVisual(color='black', width='1'))
        sky_region_list.append(circle_reg)

    spectral_axes = {}
    spectra = {}
    ptps = []
    colors = {}
    for i in [0, 1, 3]:
        fn = filenames[i]
        cube = cube_utils.CubeData(fn)
        spectral_axes[cube.name()] = cube.data.spectral_axis/1e3
        spectra[cube.name()] = []
        colors[cube.name()] = next(color_iter)
        for circle_reg in sky_region_list:
            spectrum = extract_spectrum_from_region(cube, circle_reg).to_value()
            if 'BIMA' in cube.name():
                spectrum /= 2
            spectra[cube.name()].append(spectrum)
            ptps.append(spectrum.ptp())
        if i != 3:
            del cube, spectrum
            print("(clearing memory)", end=" ")
        else:
            print()


    # Make an image to display
    # img = cube.data.moment(order=0).to_value()
    # w = cube.wcs_flat
    # I could probably use CrossCut for this; I should generalize CrossCut
    img, w = crosscut.DataLayer('5.6 um', "spitzer/SPITZER_I3_6049792_0000_5_E8698528_maic.fits").load()

    img_cutout = Cutout2D(img, global_center_coord, [length_scale*2]*2, wcs=w, mode='partial', fill_value=np.nan)
    img_cutout_flat = img_cutout.data.flatten()
    lo, up = misc_utils.flquantiles(img_cutout_flat[np.isfinite(img_cutout_flat)], 25)
    del img_cutout_flat
    fig = plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(121, projection=img_cutout.wcs)
    plt.imshow(img_cutout.data, origin='lower', vmin=lo, vmax=up)
    reg_artists = []
    first = True
    for circle_reg in sky_region_list:
        reg_pix = pixel_from_sky(circle_reg, img_cutout.wcs)
        if first:
            reg_pix.visual['color'] = 'red'
            first = False
        ax1.add_artist(reg_pix.as_artist())
    ax1.set_title(f"IRAC 5.6 um image. Spectra averaged in r = {circle_radius.to(u.arcsec):.1f} circles")

    """
    Now to plot all the spectra
    I have all the ptps of the spectra, so I can figure out a good
    separation between them. I am planning doing "top down", and I can run a
    faint axhline through the zero point of each
    """

    max_ptp = max(ptps)
    round_to = 5
    vertical_allotment = ceil((max_ptp * 1.02)/round_to)*5
    n_regs = len(sky_region_list)
    total_height = vertical_allotment * n_regs

    ax2 = plt.subplot(122)
    cube_plots = {}
    first = True
    for i in range(n_regs):
        current_height = vertical_allotment * (n_regs - i - 1)
        hline_color = 'k'
        if first:
            hline_color = 'r'
            first = False
        plt.axhline(y=current_height, color=hline_color, alpha=0.4, lw=0.7, linestyle='-')
        for cube_stub in spectra:
            if spectra[cube_stub][i] is None:
                continue
            spectral_axis = spectral_axes[cube_stub]
            spectrum = spectra[cube_stub][i] + current_height
            p = plt.plot(spectral_axis, spectrum, color=colors[cube_stub], linewidth=0.7)
            if 'BIMA' in cube_stub:
                cube_stub += ' x 0.5'
            cube_plots[cube_stub] = p
    plt.legend(labels=list(cube_plots.keys()), handles=list(x.pop() for x in cube_plots.values()), loc='upper left')
    plt.xlim([5, 40])
    fn = f"/home/ramsey/Pictures/9-17-20-mtg/spectrum_series_reg-{reg_index}.png"
    plt.savefig(fn)


if __name__ == "__main__":
    attempt_spectral_cross_cut()
