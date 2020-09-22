import numpy as np
import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 7}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import sys
import warnings
import time
import datetime

from scipy.optimize import curve_fit
from scipy import signal

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.modeling import models, fitting
from astropy import convolution

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
    """
    Extract average spectrum over region
    """
    subcube = cube.data.subcube_from_regions([reg_sky])
    spectrum = subcube.mean(axis=(1, 2))
    return spectrum

def extract_spectrum_at_point(cube, coord):
    """
    Extract spectrum from one specific point (no summing/averaging)
    """
    pi, pj = cube.wcs_flat.world_to_array_index(coord)
    spectrum = cube.data.unmasked_data[:, pi, pj] * cube.data.unit
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
    plot_args = (cube_or_spectral.spectral_axis/1e3, spectrum)
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
    Uses the cube.spectral_axis in km/s as the x axis
    :returns: mu, sigma, amplitude
    """
    x = cube.data.spectral_axis.to(u.km/u.s).to_value()
    y = spectrum.to_value()
    return curve_fit(misc_utils.gaussian, x, y, p0=p0)[0]


def fit_gaussians(cube_or_spectral, spectrum, n=2, p0=None):
    """
    Fit n gaussians to a spectrum
    p0 should be a list of guesses for:
        mu, sigma, amplitude
    Repeat that pattern for each gaussian
    cube or spectral_axis for the first argument
    Uses the cube.spectral_axis in km/s as the x axis
    """
    if isinstance(cube_or_spectral, cube_utils.CubeData):
        x = cube_or_spectral.data.spectral_axis.to(u.km/u.s).to_value()
    else:
        x = cube_or_spectral
        if hasattr(spectrum, 'unit'):
            x = x.to_value()
    if hasattr(spectrum, 'unit'):
        y = spectrum.to_value()
    else:
        y = spectrum
    """
    Probably should use astropy.modeling, but I don't know it well enough yet
    """
    def fit_n_gaussians(x, *args):
        result = np.zeros_like(x)
        for i in range(n):
            i_params = args[i*3:(i+1)*3]
            result += misc_utils.gaussian(x, *i_params)
        return result

    return curve_fit(fit_n_gaussians, x, y, p0=p0)[0], fit_n_gaussians


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
    gaussian_params = fit_gaussian(cube, off_2_spectrum, p0=[30, 3, 5.*np.sqrt(2*np.pi)*3])
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
        None, None, pvpath_width=100*u.arcsec, points_not_paths=True
    )

    sky_region_list = []
    loc_list = []
    for loc in location_generator:
        loc_list.append(loc)
        circle_reg = regions.CircleSkyRegion(loc, circle_radius, visual=regions.RegionVisual(color='black', width='1'))
        sky_region_list.append(circle_reg)

    spectral_axes = {}
    spectra = {}
    ptps = []
    colors = {}
    max_beam = None
    for i in [3]:
        fn = filenames[i]
        cube = cube_utils.CubeData(fn)
        spectral_axes[cube.name()] = cube.data.spectral_axis.to(u.km/u.s)
        spectra[cube.name()] = []
        colors[cube.name()] = next(color_iter)
        if max_beam is None:
            max_beam = cube.data.beam
        else:
            max_beam = max((cube.data.beam, max_beam), key=lambda x: x.major)
        for loc in loc_list:
            spectrum = extract_spectrum_at_point(cube, loc).to_value()
            # if 'BIMA' in cube.name():
            #     spectrum /= 2
            spectra[cube.name()].append(spectrum)
            ptps.append(spectrum.ptp())
        if i != 3:
            del cube, spectrum
            print("(clearing memory)", end=" ")
        else:
            print()
    print("max beam:", max_beam)

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

    beam_patch = max_beam.ellipse_to_plot(*img_cutout.wcs.world_to_pixel(loc_list[0]), misc_utils.get_pixel_scale(img_cutout.wcs))
    beam_patch.set_facecolor('None')
    beam_patch.set_edgecolor('red')
    ax1.add_artist(beam_patch)

    loc_coords = SkyCoord(loc_list)
    ax1.plot(loc_coords.ra.deg, loc_coords.dec.deg, '.', color='k', transform=ax1.get_transform('world'))
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
            # if 'BIMA' in cube_stub:
            #     cube_stub += ' x 0.5'
            cube_plots[cube_stub] = p
    plt.legend(labels=list(cube_plots.keys()), handles=list(x.pop() for x in cube_plots.values()), loc='upper left')
    plt.xlim([5, 40])
    fn = f"/home/ramsey/Pictures/9-17-20-mtg/spectrum_series_reg-{reg_index}.png"
    # plt.savefig(fn)
    plt.show()


def second_fit_attempt():
    warnings.filterwarnings("ignore")
    color_iter = iter(['DarkGreen', 'DarkMagenta', 'DarkOrange', 'RoyalBlue'])
    color_model_iter = iter(['LimeGreen', 'Magenta', 'Orange', 'Blue'])

    reg_index = 0
    global_center_coord, length_scale, location_generator = pvdiagrams.linear_series_from_description(
        *crosscut.coords_from_region(catalog.utils.search_for_file("catalogs/m16_lines_of_interest.reg"), index=reg_index),
        None, None, pvpath_width=10*u.arcsec, points_not_paths=True
    )

    loc_list = list(location_generator)[::4]

    spectral_axes = {}
    spectra = {}
    results = {}
    fitted_spectra = {}
    residuals = {}
    colors = {}
    max_beam = None


    fn = filenames[3]
    cube = cube_utils.CubeData(fn)
    spectral_axis = cube.data.spectral_axis.to(u.km/u.s)
    cube_name = cube.name()
    spectral_axes[cube_name] = spectral_axis
    spectra[cube_name] = []
    colors[cube_name] = (next(color_iter), next(color_model_iter))
    for loc in loc_list:
        spectrum = extract_spectrum_at_point(cube, loc).to_value()
        spectra[cube_name].append(spectrum)
        if max_beam is None:
            max_beam = cube.data.beam
        else:
            max_beam = max((cube.data.beam, max_beam), key=lambda x: x.major)

    tpi = np.sqrt(2*np.pi)
    p0 = [
        3, 24, 15,
        25, 26, 1,
        1, 29, 1,
    ]
    results[cube_name] = []
    fitted_spectra[cube_name] = []
    residuals[cube_name] = []
    ptps = []
    fitter = fitting.SLSQPLSQFitter()
    g_init = (models.Gaussian1D(*p0[0:3], bounds={'amplitude': (0, 3), 'mean': (18, 32), 'stddev': (6, None)}, name='ionized')
        + models.Gaussian1D(*p0[3:6], bounds={'amplitude': (0, 50), 'mean': (22, 27.5), 'stddev': (0.1, 3)}, name='pillar')
        + models.Gaussian1D(*p0[6:9], bounds={'amplitude': (0, 6), 'mean': (27.5, 32), 'stddev': (0.1, 3)}, name='bg'))
    for spectrum in spectra[cube_name]:
        # result, f = fit_gaussians(cube, spectrum, n=3, p0=p0)
        g_fit = fitter(g_init, spectral_axis.to_value(), spectrum, verblevel=0)
        results[cube_name].append(g_fit)

        fitted = g_fit(spectral_axis.to_value())
        fitted_spectra[cube_name].append(fitted)

        residuals[cube_name].append(spectrum - fitted)
        ptps.append(spectrum.ptp())
    print(f_init)
    # All the reference image code
    # img, w = crosscut.DataLayer('5.6 um', "spitzer/SPITZER_I3_6049792_0000_5_E8698528_maic.fits").load()
    img, w = crosscut.DataLayer("CII", "sofia/M16_CII_U.fits", cube=True, alpha=0.7, vlims=(5, 40)).load()
    img_cutout = Cutout2D(img, global_center_coord, [length_scale*2]*2, wcs=w, mode='partial', fill_value=np.nan)
    img_cutout_flat = img_cutout.data.flatten()
    lo, up = misc_utils.flquantiles(img_cutout_flat[np.isfinite(img_cutout_flat)], 25)
    del img_cutout_flat
    fig = plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(121, projection=img_cutout.wcs)
    plt.imshow(img_cutout.data, origin='lower', vmin=lo, vmax=up)
    beam_patch = max_beam.ellipse_to_plot(*img_cutout.wcs.world_to_pixel(loc_list[0]), misc_utils.get_pixel_scale(img_cutout.wcs))
    beam_patch.set_facecolor('None')
    beam_patch.set_edgecolor('red')
    ax1.add_artist(beam_patch)
    loc_coords = SkyCoord(loc_list)
    ax1.plot(loc_coords.ra.deg, loc_coords.dec.deg, '.', color='k', transform=ax1.get_transform('world'))
    ax1.set_title("IRAC 5.6 um image with spectrum sample locations overlaid")


    max_ptp = max(ptps)
    round_to = 5
    vertical_allotment = ceil((max_ptp * 1.02)/round_to)*5
    n_regs = len(loc_list)
    total_height = vertical_allotment * n_regs
    ax2 = plt.subplot(143)
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
            p = plt.plot(spectral_axis, spectrum, color=colors[cube_stub][0], linewidth=0.7)
            # if 'BIMA' in cube_stub:
            #     cube_stub += ' x 0.5'
            cube_plots[cube_stub] = p
            fitted = fitted_spectra[cube_name][i]
            x = spectral_axis.to_value()
            complete_model = results[cube_name][i]
            plt.plot(spectral_axis, complete_model['ionized'](x) + current_height, '--', color=colors[cube_name][1], linewidth=0.7)
            plt.plot(spectral_axis, complete_model['pillar'](x) + current_height, '--', color=colors[cube_name][1], linewidth=0.7)
            plt.plot(spectral_axis, complete_model['bg'](x) + current_height, '--', color=colors[cube_name][1], linewidth=0.7)

    plt.legend(labels=list(cube_plots.keys()), handles=list(x.pop() for x in cube_plots.values()), loc='upper left')
    plt.xlim([5, 40])

    ax3 = plt.subplot(144)
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
            residual = residuals[cube_name][i] + current_height
            fitted = fitted_spectra[cube_name][i] + current_height
            plt.plot(spectral_axis, fitted, color=colors[cube_stub][1], linewidth=0.7)
            plt.plot(spectral_axis, residual, color=colors[cube_stub][0], linewidth=0.7)

    plt.xlim([5, 40])
    # plt.savefig(fn)
    plt.show()


def area_fit_attempt():
    warnings.filterwarnings("ignore")

    reg_index = 0
    global_center_coord, length_scale, location_generator = pvdiagrams.linear_series_from_description(
        *crosscut.coords_from_region(catalog.utils.search_for_file("catalogs/m16_lines_of_interest.reg"), index=reg_index),
        None, None, pvpath_width=10*u.arcsec, points_not_paths=True
    )

    img, w = crosscut.DataLayer("CII", "sofia/M16_CII_U.fits", cube=True, alpha=0.7, vlims=(5, 40)).load()
    img_cutout = Cutout2D(img, global_center_coord, [length_scale*2]*2, wcs=w, mode='partial', fill_value=np.nan)

    fn = filenames[3]
    cube = cube_utils.CubeData(fn)
    spectral_axis = cube.data.spectral_axis.to(u.km/u.s).to_value()
    cube_name = cube.name()

    # Make a subcube using those slices. This has good WCS (even though it doesn't look like it)
    subcube = cube.data[:, img_cutout.slices_original[0], img_cutout.slices_original[1]]
    # Hamming smooth the spectrum
    smooth_kernel = convolution.kernels.CustomKernel(signal.hamming(7))
    subcube = subcube.spectral_smooth(smooth_kernel)

    subcube_array = subcube.unmasked_data[:].to_value().reshape((len(spectral_axis), img_cutout.data.size)).T
    residuals_array = np.zeros_like(subcube_array)
    models_array = np.zeros_like(subcube_array)
    print(subcube_array.shape)

    tpi = np.sqrt(2*np.pi)
    fitter = fitting.SLSQPLSQFitter()
    default_pillar_low_A = 0
    pillar_A_nonexistant_cutoff = 6
    default_bg2_low_A = 0
    bg2_A_nonexistant_cutoff = 3
    g_init = (models.Gaussian1D(3, 24, 15, bounds={'amplitude': (0, 3), 'mean': (20, 32), 'stddev': (8, 15)}, name='ionized')
        + models.Gaussian1D(25, 26, 1, bounds={'amplitude': (default_pillar_low_A, 50), 'mean': (22, 27.5), 'stddev': (1, 2.5)}, name='pillar')
        + models.Gaussian1D(1, 29, 1, bounds={'amplitude': (0, 6), 'mean': (27.5, 32), 'stddev': (0.7, 3)}, name='bg')
        + models.Gaussian1D(1, 39, 1, bounds={'amplitude': (0, 8), 'mean': (35, 45), 'stddev': (0.7, 4)}, name='bg2'))
    # Manually set all the bounding boxes, since it can't be a parameter to Gaussian1D.__init__
    g_init['ionized'].bounding_box=(10, 40)
    g_init['pillar'].bounding_box=(18, 35)
    g_init['bg'].bounding_box=(26, 35)
    g_init['bg2'].bounding_box=(30, 50)
    param_names = g_init.param_names
    param_units = ["K", "km / s", "km / s"]*g_init.n_submodels # assume all gaussians
    results = []

    # Some stuff for logging progress
    timing_t0 = time.time()
    print(f"Started at {datetime.datetime.now(datetime.timezone.utc).astimezone().ctime()}")
    last_count = 0
    count_coeff = 100./img_cutout.data.size
    for i in range(img_cutout.data.size):
        spectrum = subcube_array[i, :]
        # Handle the pillar peak being too small when it's clearly there in the data
        g_init.amplitude_1.min = default_pillar_low_A
        if np.nanmax(spectrum) > pillar_A_nonexistant_cutoff:
            g_init.amplitude_1.min = pillar_A_nonexistant_cutoff
        # Handle the 40 km/s peak being too small when it's clear in data
        g_init.amplitude_3.min = default_bg2_low_A
        if np.nanmax(spectrum[spectral_axis > 35]) > bg2_A_nonexistant_cutoff:
            g_init.amplitude_3.min = bg2_A_nonexistant_cutoff
        # Fit!
        g_fit = fitter(g_init, spectral_axis, spectrum, verblevel=0)
        results.append(g_fit.param_sets[:, 0])
        fitted_spectrum = g_fit(spectral_axis)
        residuals_array[i, :] = spectrum - fitted_spectrum
        models_array[i, :] = fitted_spectrum
        current_count = int(round(i * count_coeff))
        if current_count > last_count:
            sys.stdout.write(f"{current_count:4d} percent done\n")
            sys.stdout.flush()
            last_count = current_count
    results = np.array(results).T.reshape((len(param_names), img_cutout.shape[0], img_cutout.shape[1]))
    residuals_array = residuals_array.T.reshape((len(spectral_axis), img_cutout.shape[0], img_cutout.shape[1]))
    models_array = models_array.T.reshape((len(spectral_axis), img_cutout.shape[0], img_cutout.shape[1]))

    timing_t1 = time.time()
    print(f"Finished at {datetime.datetime.now(datetime.timezone.utc).astimezone().ctime()}")
    print(f"Time elapsed: {str(datetime.timedelta(seconds=(timing_t1-timing_t0)))}")

    filename_stub = "models/gauss_fit_4G_v3"
    to_header = lambda : img_cutout.wcs.to_header()
    phdu = fits.PrimaryHDU(header=to_header())
    phdu.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    phdu.header['CREATOR'] = f"Ramsey, {__file__}"
    hdu_list = [phdu]
    for i in range(len(param_names)):
        hdu = fits.ImageHDU(data=results[i], header=to_header())
        hdu.header['EXTNAME'] = param_names[i]
        hdu.header['BUNIT'] = param_units[i]
        hdu_list.append(hdu)
    hdu_data = fits.ImageHDU(data=img_cutout.data, header=to_header())
    hdu_data.header['EXTNAME'] = "CII integrated (5, 40)"
    hdu_data.header['BUNIT'] = "K km / s"
    hdu_list.append(hdu_data)
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(cube_utils.os.path.join(cube.directory, f"{filename_stub}.param.fits"), overwrite=True)

    phdu = fits.PrimaryHDU(data=residuals_array, header=subcube.wcs.to_header())
    phdu.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    phdu.header['CREATOR'] = f"Ramsey, {__file__}"
    phdu.header['BUNIT'] = 'K'
    phdu.header['COMMENT'] = 'Data - Model residuals'
    phdu.writeto(cube_utils.os.path.join(cube.directory, f"{filename_stub}.resid.fits"), overwrite=True)

    phdu = fits.PrimaryHDU(data=models_array, header=subcube.wcs.to_header())
    phdu.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    phdu.header['CREATOR'] = f"Ramsey, {__file__}"
    phdu.header['BUNIT'] = 'K'
    phdu.header['COMMENT'] = 'Model intensity'
    phdu.writeto(cube_utils.os.path.join(cube.directory, f"{filename_stub}.model.fits"), overwrite=True)

    print("Done!")


def examine_area_solution():
    cube = cube_utils.CubeData(filenames[3])
    filename_stub = "models/gauss_fit_4G_v2"
    filename_stub = cube_utils.os.path.join(cube.directory, filename_stub)
    residuals_cube = cube_utils.SpectralCube.read(f"{filename_stub}.resid.fits")
    model_intensity_cube = cube_utils.SpectralCube.read(f"{filename_stub}.model.fits")
    param_array = []
    param_names = []
    integrated = None
    cutout_wcs = None
    with fits.open(f"{filename_stub}.param.fits") as hdul:
        for hdu in hdul:
            if hdu.data is None:
                cutout_wcs = WCS(hdu.header)
            elif 'integrated' in hdu.header['EXTNAME']:
                integrated = hdu.data
            else:
                param_array.append(hdu.data)
                param_names.append(hdu.header['EXTNAME'])
    param_array = np.array(param_array)

    def g_model(*args):
        return models.Gaussian1D(*args[0:3], name='ionized') + models.Gaussian1D(*args[3:6], name='pillar') + models.Gaussian1D(*args[6:9], name='bg') + models.Gaussian1D(*args[9:12], name='bg2')

    def get_spectrum_at(i, j):
        """
        i and j are array coordinates from the cutout cube
        """
        coord = cutout_wcs.array_index_to_world(i, j)
        i, j = cube.wcs_flat.world_to_array_index(coord)
        return cube.data[:, i, j]

    def get_gmodel_at(i, j):
        return g_model(*param_array[:, i, j])

    def plot_spectrum(ax, spectrum, **kwargs):
        ax.plot(spectrum.spectral_axis.to(u.km/u.s), spectrum.to_value(), **kwargs)

    def plot_model(ax, any_cube, model, **kwargs):
        x = any_cube.spectral_axis.to(u.km/u.s).to_value()
        ax.plot(x, model(x), **kwargs)

    def plot_model_separately(ax, any_cube, model, **kwargs):
        x = any_cube.spectral_axis.to(u.km/u.s).to_value()
        for i in range(model.n_submodels):
            ax.plot(x, model[i](x), **kwargs)

    i, j = 6, 18
    i, j = 20, 17
    i, j = 18, 22
    i, j = 8, 22
    i, j = 3, 8
    i, j = 3, 5
    i, j = 21, 3
    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].imshow(integrated, origin='lower')
    axes[0].plot([j], [i], 'x', color='r')

    spectrum1 = residuals_cube[:, i, j] + model_intensity_cube[:, i, j]
    plot_spectrum(axes[1], spectrum1, color='k', linewidth=0.7)
    spectrum2 = get_spectrum_at(i, j)
    plot_spectrum(axes[1], spectrum2, color='r', linewidth=0.6, linestyle='--', alpha=0.4)
    plot_spectrum(axes[1], residuals_cube[:, i, j], color='k', linewidth=0.7, alpha=0.7)
    g_fit = get_gmodel_at(i, j)
    print(g_fit)
    plot_model_separately(axes[1], spectrum1, g_fit, color='g', linestyle='-.', linewidth=0.7)
    # plot_model(axes[1], spectrum1, g_fit['pillar'], color='r')
    plt.show()

if __name__ == "__main__":
    examine_area_solution()
