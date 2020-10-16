import numpy as np
import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 7}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import sys
import warnings
import datetime
import time

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.modeling import models, fitting
# from astropy import convolution

import regions
from math import ceil

from . import misc_utils
from . import catalog
from . import cube_utils
from . import pvdiagrams
from . import crosscut
# This is where I "started" this task, some functions might be useful
from . import cube_pixel_spectra as cps1
"""

A sequel to cube_pixel_spectra.py. Looking closer at the M16 line data,
fitting things to the spectra, etc.
The goal remains the same, I'm just still working on it.

Created: September 25, 2020
    (while listening to the Front Bottoms on a pleasant, rainy Friday evening)
Major reorganization October 14, 2020
    shortly after realizing I missed the AAS abstract deadline yesterday!
"""
__author__ = "Ramsey Karim"

cube_info = {}

def cutout_subcube(length_scale_mult=2, data_filename=None, reg_filename=None,
    length_scale=None, global_center_coord=None, reg_index=0):
    """
    This is just the first few lines of cps1.area_fit_attempt
    I think this subcube will be useful, so I'll keep using it.
    This is the same grid that I fit all those Gaussians to.
    length_scale_mult was 2 in that grid; I can change it here
    :param length_scale_mult: multiplier for the length_scale, wherever that
        comes from. Default is 2
    :param data_filename: filename for the cube. Can be absolute or under one
        of the "_data" directories in Feedback/. Default is M16 SOFIA CII
    :param reg_filename: filename for DS9 .reg file to use for center coordinate
        and length_scale. If None, then length_scale and global_center_coord
        should be given. This will override those, if given.
    :param reg_index: index for the region you want to reference in reg_filename
    :param length_scale: length scale (Quantity or Angle, angular unit) for the
        cutout. Multiplied by length_scale_mult when used for the cutout.
        If tuple, gives (y, x) size. If reg_filename is given, then this
        is ignored.
    :param global_center_coord: center coordinate (SkyCoord) for the cutout.
        If reg_filename is given, then this is ignored.
    """
    warnings.filterwarnings("ignore")
    if data_filename is None:
        data_filename = "sofia/M16_CII_U.fits"
    if reg_filename is None:
        try:
            assert global_center_coord is not None
            assert length_scale is not None
        except AssertionError as e:
            # We want to warn the user but still drop down to the "reg_filename is not None" clause with the default filename
            print(f"You need to either give reg_filename or both global_center_coord and length_scale: {e}")
            reg_filename = "catalogs/m16_lines_of_interest.reg"
            print(f"Using the default filename for now: {reg_filename}")
    # This next if block will run if 1) reg_filename was set or 2) the other two keywords were left out
    if reg_filename is not None:
        global_center_coord, length_scale, _ = pvdiagrams.linear_series_from_description(
            *crosscut.coords_from_region(catalog.utils.search_for_file(reg_filename), index=reg_index),
            None, None, pvpath_width=10*u.arcsec, points_not_paths=True
        )

    img, w = crosscut.DataLayer("", data_filename, cube=True, alpha=0.7, vlims=(5, 40)).load()
    img_cutout = Cutout2D(img, global_center_coord, [length_scale*length_scale_mult]*2, wcs=w, mode='partial', fill_value=np.nan)

    cube = cube_utils.CubeData(data_filename)
    cube_info['dir'] = cube.directory

    # Make a subcube using those slices. This has good WCS (even though it doesn't look like it)
    subcube = cube.data[:, img_cutout.slices_original[0], img_cutout.slices_original[1]]
    return subcube.with_spectral_unit(u.km/u.s)


def mask_above_xpower(cube, xpower, additional_cutoff=0):
    """
    General function to mask a cube by some fraction of its peak power in each
    spectrum. Also can impose an additional mask on that fraction of the peak.

    For example, if xpower=2 and additional_cutoff=6, then each spectrum will
    be masked out below half power in that spectrum. Also, spectra whose half
    power is less than 6 (flux units) will be completely masked out.

    SUPER IMPORTANT: to actually use the masked thing, you need to use
    filled_data.
    Methods of SpectralCube seems to work ok, like moment(), they use the mask
    without explicitly being told to
    """
    half_power = np.max(cube, axis=0)/xpower
    mask = (cube > half_power) & (half_power > additional_cutoff*cube.unit)
    masked_cube = cube.with_mask(mask)
    return masked_cube


def mask_with_best_setting(cube):
    """
    Use mask_above_xpower with the current best recipe
    """
    if 'apex' in cube_info['dir']:
        return mask_above_xpower(cube, 2, additional_cutoff=0)
    elif 'sofia' in cube_info['dir']:
        return mask_above_xpower(cube, 3, additional_cutoff=2.5)


def prepare_initial_conditions(unmasked_cube, masked_cube):
    """
    Return a dictionary with useful initial condition arrays, like moment1 and
    stuff.
    :param unmasked_cube: the UNMASKED cube
    :param masked_cube: the ALREADY masked cube, masked however you want, the
        way you'll use it for fitting
    :returns: dict with keys that describe everything in it
        full_power_loc is in units of km/s, though just float array
    """
    # "Manual" mean guess
    full_power_loc = unmasked_cube.spectral_axis[unmasked_cube.argmax(axis=0)].to_value()
    # SpectralCube mean guess
    moment1 = masked_cube.moment(order=1).to_value()
    # Amplitude
    full_power = unmasked_cube.max(axis=0).to_value()
    # Standard deviation
    linewidth_manual = linewidth_fwhm(unmasked_cube)
    linewidth_spectralcube = masked_cube.linewidth_fwhm().to_value()
    # Moment 0
    moment0 = masked_cube.moment(order=0).to_value()
    # Spectral Axis
    spectral_axis = unmasked_cube.spectral_axis.to_value()
    # Gather in dictionary
    return_dict = {
        'full_power_loc': full_power_loc,
        'moment0': moment0, 'moment1': moment1,
        'linewidth_manual': linewidth_manual, 'linewidth_spectralcube': linewidth_spectralcube,
        'full_power': full_power, 'spectral_axis': spectral_axis,
    }
    return return_dict


def initialize_gaussian(init_conds, g, ij):
    """
    :param init_conds: dictionary returned by prepare_initial_conditions
    :param g: a Gaussian1D object to reuse. Can be None, in which case a new
        one is created
    :param ij: tuple array indices
    """
    i, j = ij
    # Amplitude
    A = init_conds['full_power'][i, j]
    A_bounds = (A*0.95, A*1.05)
    # Standard deviation
    std_manual = init_conds['linewidth_manual'][i, j] / 2.355
    std_spcube = init_conds['linewidth_spectralcube'][i, j] / 2.355
    if std_manual < 0.3 or std_manual > 2.5:
        std = 1.5
    else:
        std = std_manual
    std_bounds = (0.5, 3)
    # Mean
    mean_manual = init_conds['full_power_loc'][i, j]
    mean_spcube = init_conds['moment1'][i, j]
    if np.abs(mean_spcube - mean_manual) < min(std, 1.5):
        # Well behaved case
        mean = mean_spcube
    else:
        # Intervention case
        mean = mean_manual
    mean_bounds = (mean*0.95, mean*1.05)
    if g is None:
        g = models.Gaussian1D(A, mean, std, bounds={
            'amplitude': A_bounds,
            'mean': mean_bounds,
            'stddev': std_bounds,
        })
    else:
        g.amplitude = A
        g.amplitude.bounds = A_bounds
        g.mean = mean
        g.mean.bounds = mean_bounds
        g.stddev = std
        g.stddev.bounds = std_bounds
    return g


def fit_gaussian(init_conds, masked_cube, g, ij, fitter, verblevel=1):
    """
    :param init_conds: return dict of prepare_initial_conditions
    :param masked_cube: already masked SpectralCube
    :param g: Gaussian1D already initialized
    :param ij: tuple array indices
    :param fitter: some kind of astropy.modeling.fitting fitter
    :returns: fitted Gaussian1D and the resulting model array
    """
    i, j = ij
    masked_spectrum_val = masked_cube.filled_data[:, i, j].to_value()
    masked_spectrum_mask = masked_cube.get_mask_array()[:, i, j]
    new_spectrum_mask = identify_longest_run(masked_spectrum_mask)
    masked_spectrum_val[~new_spectrum_mask] = np.nan

    # astropy.modeling does not like NaNs!!!
    finite_mask = np.isfinite(masked_spectrum_val)
    if np.sum(finite_mask.astype(int)) > 3:
        fit_x, fit_y = init_conds['spectral_axis'][finite_mask], masked_spectrum_val[finite_mask]
        weights = np.abs(fit_y)
        weights[weights < 1.3] = 1.3
        weights = (weights/np.max(weights))/1.3
        g_fit = fitter(g, fit_x, fit_y, weights=weights, verblevel=verblevel)
        g_fit_array = g_fit(init_conds['spectral_axis'])
        return g_fit, g_fit_array, masked_spectrum_val
    else:
        nan_array = np.full(masked_cube.shape[0], np.nan)
        return None, nan_array, nan_array

def fit_live_interactive(cube):
    # INTERACTIVE

    masked_cube = mask_with_best_setting(cube)
    init_conds = prepare_initial_conditions(cube, masked_cube)
    spectral_axis = init_conds['spectral_axis']

    plt.ion()
    fig = plt.figure(figsize=(6, 3.5))
    ax_manual = plt.subplot2grid((2, 5), (0, 0), colspan=2, fig=fig, projection=cube.wcs, slices=('x', 'y', 50))
    ax_spcube = plt.subplot2grid((2, 5), (1, 0), colspan=2, fig=fig, projection=cube.wcs, slices=('x', 'y', 50), sharex=ax_manual, sharey=ax_manual)
    ax_manual.tick_params(axis='x', labelbottom=False)
    ax_spectr = plt.subplot2grid((2, 5), (0, 2), colspan=3, rowspan=2, fig=fig)

    ax_manual.imshow(init_conds['moment0'], origin='lower')
    ax_manual.set_title("Moment 0 (masked)")
    ax_spcube.imshow(init_conds['moment1'], origin='lower')
    ax_spcube.set_title("Moment 1 (masked)")

    ax_spectr.plot(spectral_axis, masked_cube.mean(axis=(1, 2)).to(u.K).to_value(), linewidth=0.7)
    ax_spectr.set_xlim((spectral_axis[0], spectral_axis[-1]))

    fitter = fitting.SLSQPLSQFitter()
    plot_info_dict = {'x1': None, 'x2': None}

    def onclick(event):
        try:
            j, i = int(round(event.xdata)), int(round(event.ydata))
        except Exception as e:
            print(f"something went wrong... {e}")
            return
        if event.button == 1 and (event.inaxes is ax_manual or event.inaxes is ax_spcube):
            ax_spectr.clear()
            if plot_info_dict['x1'] is not None:
                ax_manual.lines.remove(plot_info_dict['x1'])
                ax_spcube.lines.remove(plot_info_dict['x2'])

            plot_info_dict['x1'], = ax_manual.plot([j], [i], 'x', color='red')
            plot_info_dict['x2'], = ax_spcube.plot([j], [i], 'x', color='red')

            g_init = initialize_gaussian(init_conds, None, (i, j))
            """
            astropy modeling does NOT like NaNs. That's weird! They should!
            """
            g_fit, g_fit_array, masked_spectrum_val = fit_gaussian(init_conds, masked_cube, g_init, (i, j), fitter)
            if g_fit is not None:
                print("Parameter [MIN : init_val/fitted_val : MAX]")
                print(f"Ampl [{g_fit.amplitude.min:6.2f} {g_init.amplitude.value:6.2f}/{g_fit.amplitude.value:6.2f} {g_fit.amplitude.max:6.2f}]")
                print(f"Mean [{g_fit.mean.min:6.2f} {g_init.mean.value:6.2f}/{g_fit.mean.value:6.2f} {g_fit.mean.max:6.2f}]")
                print(f"Stdd [{g_fit.stddev.min:6.2f} {g_init.stddev.value:6.2f}/{g_fit.stddev.value:6.2f} {g_fit.stddev.max:6.2f}]")
            else:
                print("No fit was made")

            spectrum = cube[:, i, j].to_value()
            ax_spectr.plot(spectral_axis, spectrum, color='k', linewidth=0.7, label='Data', marker='o', alpha=0.2, markersize=3)
            ax_spectr.plot(spectral_axis, masked_spectrum_val, color='Indigo', marker='x', linewidth=2, linestyle='dotted', alpha=0.9, label='Data fitted to')

            ax_spectr.plot(spectral_axis, g_fit_array, color='orange', linestyle='dotted', linewidth=1, label="Fitted", alpha=0.9)
            ax_spectr.plot(spectral_axis, spectrum - g_fit_array, color='k', linewidth=0.7, label='Residuals', alpha=0.6)

            # Metric; sum over this for a number that, if large, means the fit isn't good
            metric = -(spectrum - g_fit_array)*np.sign(spectrum)
            metric[metric < 0] = 0
            # ax_spectr.plot(spectral_axis, metric, color='Indigo', marker='+', alpha=0.3, label='metric')

            ax_spectr.set_xlabel("v (km/s)")
            ax_spectr.set_ylabel("T (K)")
            # ax_spectr.set_xlim((round(mean_spcube)-5, round(mean_spcube)+5))
            ax_spectr.legend()
        elif event.button == 3:
            plt.ioff()
            plt.close()

    return fig.canvas.mpl_connect('button_press_event', onclick)



def linewidth_fwhm(cube, masked_already=False, flux_cutoff=6.5):
    """
    cube is a SpectralCube, not already (extensively) masked
    flux_cutoff is a lower limit on flux (in K) that will be imposed along with
        half-max. This will completely eradicate spectra with half-power fluxes
        below this value. However, those pixels may not have clean line spectra
        fit for FWHM estimates
    """
    # Make a half-power mask for each spectrum
    # Apply this mask to all spatial and spectral pixels
    # This selects out the main peak (above the flux_cutoff)
    if not masked_already:
        if 'apex' in cube_info['dir']:
            masked_cube = mask_above_xpower(cube, 2, additional_cutoff=0)
        elif 'sofia' in cube_info['dir']:
            masked_cube = mask_above_xpower(cube, 2, additional_cutoff=flux_cutoff)
    else:
        masked_cube = cube
    underlying_mask = masked_cube.get_mask_array()
    spectral_axis = cube.spectral_axis.to_value()
    ##### Now, use the apply_function_parallel_spectral method to get FWHMs
    result = np.apply_along_axis(fwhm_from_mask, 0, underlying_mask, spectral_axis=spectral_axis)
    return result


def fwhm_from_mask(spec_mask, spectral_axis=None):
    """
    spec_mask is a boolean array of the same shape representing the mask
    spectral_axis is a float array (NOT QUANTITY) representing the x axis

    returns one single value which is the FWHM of the longest continuous
    array of unmasked flux in units of the spectral axis of the cube

    The algorithm I used is from https://stackoverflow.com/a/1066838

    This is probably not necessary for our spectra; the runs of flux
    above half-max will NEVER hit the borders in physically interesting
    cases. If they do, it's noise. I could even mask them out prior to this
    function call.
    Skipping this step will help keep indexing simpler and might increase speed,
    but the solution is less "pure".
    # spec_mask = np.hstack(([False], spec_mask, [False]))
    """
    if not np.any(spec_mask):
        return 0.
    elif np.all(spec_mask):
        return 0.
    elif np.sum(spec_mask.astype(int)) > 50:
        return 0.
    if spectral_axis is None:
        xarr = np.arange(spec_mask.size)
    else:
        xarr = spectral_axis
    spec_mask = np.hstack(([False], spec_mask, [False]))
    diffs = np.diff(spec_mask.astype(int))
    # Starts are the indices of "False" right BEFORE the True
    run_starts, = np.where(diffs > 0)
    # Ends are the LAST indices of "True" BEFORE the False
    run_ends, = np.where(diffs < 0)
    # Lengths are calculated from the x axis (spectral_axis)
    run_lengths = xarr[run_ends-1] - xarr[run_starts]
    max_length = run_lengths.max()
    return max_length


def identify_longest_run(spec_mask, spectral_axis=None):
    """
    Similar to fwhm_from_mask, but just identify the longest run of True values
    in the array, and make a mask of those
    """
    spec_mask = np.hstack(([False], spec_mask, [False]))
    if not np.any(spec_mask):
        return 0.
    if spectral_axis is None:
        xarr = np.arange(spec_mask.size)
    else:
        xarr = spectral_axis
    diffs = np.diff(spec_mask.astype(int))
    # Starts are the indices of "False" right BEFORE the True
    run_starts, = np.where(diffs > 0)
    run_starts += 1 # Bump up to first True
    # Ends are the LAST indices of "True" BEFORE the False
    run_ends, = np.where(diffs < 0)
    run_ends += 1 # Bump up to first False
    # Lengths are calculated from the x axis (spectral_axis)
    run_lengths = xarr[run_ends] - xarr[run_starts]
    max_loc = run_lengths.argmax()
    # Subtract 2 from size to account for first and last False padding
    return_mask = np.full(spec_mask.size-2, False, dtype=bool)
    # Subtract 1 from indices to account for padded False at beginning
    return_mask[run_starts[max_loc]-1:run_ends[max_loc]-1] = True
    return return_mask



def subtract_hybrid_gaussian(cube):
    """
    Use the "hybrid" gaussian to subtract the main peak
    Hybrid takes the mean from spectral_cube and the fwhm from manual method
    """
    pass

def fit_image_to_file(cube):
    """
    Do the big fit but only on stuff above half power, and only one Gaussian1D
    """
    masked_cube = mask_with_best_setting(cube)
    init_conds = prepare_initial_conditions(cube, masked_cube)
    spectral_axis = init_conds['spectral_axis']
    fitter = fitting.SLSQPLSQFitter()
    g_init = models.Gaussian1D(25, 26, 1, bounds={'amplitude': (0, 50), 'mean': (22, 28), 'stddev': (1, 3.5)})
    n_params = len(g_init.param_names)

    results = np.zeros((n_params*2, *masked_cube.shape[1:]))
    residuals_array = np.zeros(masked_cube.shape)
    models_array = np.zeros(masked_cube.shape)

    # Some stuff for logging progress
    timing_t0 = time.time()
    print(f"Started at {datetime.datetime.now(datetime.timezone.utc).astimezone().ctime()}")
    last_count = 0
    count_coeff = 100./(masked_cube.shape[1]*masked_cube.shape[2])
    # Use one for-loop and iterate over a "flattened" image
    for flat_idx in range(masked_cube.shape[1]*masked_cube.shape[2]):
        # Convert the flat index to the 2D image index
        i, j = np.unravel_index(flat_idx, masked_cube.shape[1:])

        g_init = initialize_gaussian(init_conds, g_init, (i, j))
        g_fit, fitted_spectrum, _ = fit_gaussian(init_conds, masked_cube, g_init, (i, j), fitter, verblevel=0)

        results[:n_params, i, j] = g_init.param_sets[:, 0]
        if g_fit is not None:
            results[n_params:, i, j] = g_fit.param_sets[:, 0]
        else:
            results[n_params:, i, j] = np.nan
        residuals_array[:, i, j] = cube[:, i, j].to_value() - fitted_spectrum
        models_array[:, i, j] = fitted_spectrum
        # Logging again
        current_count = int(round(flat_idx * count_coeff))
        if current_count > last_count:
            sys.stdout.write(f"{current_count:4d} percent done\n")
            sys.stdout.flush()
            last_count = current_count

    timing_t1 = time.time()
    print(f"Finished at {datetime.datetime.now(datetime.timezone.utc).astimezone().ctime()}")
    print(f"Time elapsed: {str(datetime.timedelta(seconds=(timing_t1-timing_t0)))}")

    filename_stub = "models/gauss_fit_above_1G_v3"
    param_units = ['K', 'km / s', 'km / s']
    wcs_flat = cube.moment(order=0).wcs
    to_header = lambda : wcs_flat.to_header()
    phdu = fits.PrimaryHDU(header=to_header())
    phdu.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    phdu.header['CREATOR'] = f"Ramsey, {__file__}"
    phdu.header['COMMENT'] = f"Fit with best masking settings right now."
    phdu.header['COMMENT'] = "Using a confusing weight scheme.."
    phdu.header['COMMENT'] = "Cutout with length_scale_mult 4"
    hdu_list = [phdu]
    for i in range(n_params):
        hdu = fits.ImageHDU(data=results[i], header=to_header())
        hdu.header['EXTNAME'] = g_init.param_names[i] + "_INIT"
        hdu.header['BUNIT'] = param_units[i]
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(data=results[i+n_params], header=to_header())
        hdu.header['EXTNAME'] = g_init.param_names[i] + "_FIT"
        hdu.header['BUNIT'] = param_units[i]
        hdu_list.append(hdu)
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(cube_utils.os.path.join(cube_info['dir'], f"{filename_stub}.param.fits"), overwrite=True)

    phdu = fits.PrimaryHDU(data=residuals_array, header=cube.wcs.to_header())
    phdu.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    phdu.header['CREATOR'] = f"Ramsey, {__file__}"
    phdu.header['BUNIT'] = 'K'
    phdu.header['COMMENT'] = 'Data - Model residuals'
    phdu.writeto(cube_utils.os.path.join(cube_info['dir'], f"{filename_stub}.resid.fits"), overwrite=True)

    phdu = fits.PrimaryHDU(data=models_array, header=cube.wcs.to_header())
    phdu.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    phdu.header['CREATOR'] = f"Ramsey, {__file__}"
    phdu.header['BUNIT'] = 'K'
    phdu.header['COMMENT'] = 'Model intensity'
    phdu.writeto(cube_utils.os.path.join(cube_info['dir'], f"{filename_stub}.model.fits"), overwrite=True)

    print("Done!")


def investigate_fit():
    """
    Investiate the fit made in the previous function
    """

    filename_stub = "models/gauss_fit_above_1G_v3"
    param_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".param.fits")
    resid_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".resid.fits")
    model_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".model.fits")

    hdul = fits.open(param_fn)
    print(list(hdu.header['EXTNAME'] for hdu in hdul if 'EXTNAME' in hdu.header))

    resid_cube = cube_utils.SpectralCube.read(resid_fn)
    model_cube = cube_utils.SpectralCube.read(model_fn)

    spectral_axis = resid_cube.spectral_axis.to(u.km/u.s)

    vlo, vhi = 25, 30
    resid_mom0 = resid_cube.spectral_slab(vlo*u.km/u.s, vhi*u.km/u.s).moment(order=0).to(u.K*u.km/u.s).to_value()

    plt.ion()
    fig = plt.figure(figsize=(6, 3.5))
    ax_img = plt.subplot2grid((1, 5), (0, 0), colspan=2, fig=fig, projection=resid_cube.wcs, slices=('x', 'y', 0))
    ax_spectr = plt.subplot2grid((1, 5), (0, 2), colspan=3, fig=fig)

    im = ax_img.imshow(resid_mom0, origin='lower', vmin=0, vmax=35)
    cbar = fig.colorbar(im, ax=ax_img)
    cax = cbar.ax
    ax_img.set_title(f"Integrated residuals, [{vlo:4.1f}, {vhi:4.1f}] km/s")

    plot_info_dict = {'x1': None, 'xij': None, 'currently_selecting': False}

    def onclick(event):
        try:
            j, i = int(round(event.xdata)), int(round(event.ydata))
        except Exception as e:
            print(f"something went wrong... {e}")
            return
        if event.button == 1 and event.inaxes is ax_img:
            ax_spectr.clear()
            if plot_info_dict['x1'] is not None:
                ax_img.lines.remove(plot_info_dict['x1'])
            plot_info_dict['x1'], = ax_img.plot([j], [i], 'x', color='red')
            plot_info_dict['xij'] = (i, j)
            A = hdul['amplitude_INIT'].data[i, j]
            mean = hdul['mean_INIT'].data[i, j]
            stddev = hdul['stddev_INIT'].data[i, j]
            resid_spectr = resid_cube[:, i, j]
            model_spectr = model_cube[:, i, j]
            ax_spectr.plot(spectral_axis, resid_spectr+model_spectr, color='k', label='original', linewidth=0.7, marker='o', alpha=0.2, markersize=3)
            ax_spectr.plot(spectral_axis, model_spectr, color='Indigo', label='model', linewidth=0.7, alpha=0.6)
            ax_spectr.plot(spectral_axis, resid_spectr, color='orange', label='resid', linewidth=0.7, alpha=0.7)
            ax_spectr.legend()
            # ax_spectr.set_xlim()
            ax_spectr.set_xlabel("v (km/s)")
            ax_spectr.set_ylabel("T (K)")
            plot_info_dict['currently_selecting'] = False
        elif event.button == 1 and event.inaxes is ax_spectr:
            if not plot_info_dict['currently_selecting']:
                plot_info_dict['vel_bound'] = event.xdata
                plot_info_dict['intensity_bound'] = event.ydata
                plot_info_dict['currently_selecting'] = True
            else:
                velocity_bound2 = event.xdata
                intensity_bound2 = event.ydata
                velocity_bound1 = plot_info_dict.pop('vel_bound')
                intensity_bound1 = plot_info_dict.pop('intensity_bound')
                ilo = min(intensity_bound1, intensity_bound2)
                ihi = max(intensity_bound1, intensity_bound2)
                vlo = min(velocity_bound1, velocity_bound2)
                vhi = max(velocity_bound1, velocity_bound2)
                # Convert from K to (average) K km/s
                vspan = vhi - vlo
                ilo *= vspan
                ihi *= vspan
                # Integrate
                print(f"Integrated between {vlo:.1f} and {vhi:.1f} km/s. Intensity limits: {ilo:.1f}, {ihi:.1f}")
                resid_mom0 = resid_cube.spectral_slab(vlo*u.km/u.s, vhi*u.km/u.s).moment(order=0).to(u.K*u.km/u.s).to_value()
                ax_img.clear()
                cax.clear()
                im = ax_img.imshow(resid_mom0, origin='lower', vmin=ilo, vmax=ihi)
                fig.colorbar(im, cax=cax)
                ax_img.set_title(f"Integrated residuals, [{vlo:4.1f}, {vhi:4.1f}] km/s")
                if plot_info_dict['x1'] is not None:
                    i, j = plot_info_dict['xij']
                    plot_info_dict['x1'], = ax_img.plot([j], [i], 'x', color='red')
                plot_info_dict['currently_selecting'] = False
        elif event.button == 3:
            hdul.close()
            plt.ioff()
            plt.close()
    return fig.canvas.mpl_connect('button_press_event', onclick)


def make_wing_moments(cube):
    """
    Check emission between [20, 21.5] and [28.5, 30]
    """
    kms = u.km/u.s
    vel_bounds = 20*kms, 30*kms
    original_mom0 = cube.spectral_slab(*vel_bounds).moment(order=0).to(u.K*u.km/u.s).to_value()
    contour_args = (original_mom0,)
    contour_kwargs = dict(levels=[30, 60, 90, 120, 150], linewidths=0.5, colors='k', alpha=0.5)

    filename_stub = "models/gauss_fit_above_1G_v3"
    param_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".param.fits")
    resid_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".resid.fits")
    model_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".model.fits")
    resid_cube = cube_utils.SpectralCube.read(resid_fn)

    blue_slab = resid_cube.spectral_slab(20*kms, 25*kms)
    red_slab = resid_cube.spectral_slab(25*kms, 30*kms)
    blue_integrated = blue_slab.moment(order=0).to_value()
    red_integrated = red_slab.moment(order=0).to_value()
    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(121, projection=cube.wcs, slices=('x', 'y', 50))
    ax1.imshow(blue_integrated, origin='lower', vmin=0)
    ax1.set_title("Blue wing, integrated [20, 25] km/s")
    ax1.contour(*contour_args, **contour_kwargs)

    ax2 = plt.subplot(122, projection=cube.wcs, slices=('x', 'y', 50), sharex=ax1, sharey=ax1)
    ax2.imshow(red_integrated, origin='lower', vmin=0)
    ax2.set_title("Red wing, integrated [25, 30] km/s")
    ax2.contour(*contour_args, **contour_kwargs)
    plt.show()


def calculate_noise(cube):
    """
    There seems to be some tile-dependence of the noise.
    The noise in the pillars seems to be about 1.3 K
    """
    noise_cube = cube.spectral_slab(-5*u.km/u.s, 10*u.km/u.s)
    noise_rms = noise_cube.std(axis=0)
    fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection=noise_cube.wcs, slices=('x', 'y', 0)))
    im = axes[1].imshow(noise_rms.to_value(), origin='lower')
    axes[0].imshow(cube.moment(order=0).to_value(), origin='lower')
    fig.colorbar(im, ax=axes[1])
    plt.show()


def check_if_wings_trace_peak_emission(cube):
    """
    I need the original cube as arg, and then I can load the fitted stuff
    """
    kms = u.km/u.s
    vel_bounds = 20*kms, 30*kms
    original_mom0 = cube.spectral_slab(*vel_bounds).moment(order=0).to(u.K*u.km/u.s).to_value()
    contour_args = (original_mom0,)
    contour_kwargs = dict(levels=[30, 60, 90, 120, 150], linewidths=0.5, colors='k', alpha=0.5)

    filename_stub = "models/gauss_fit_above_1G_v3"
    param_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".param.fits")
    resid_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".resid.fits")
    model_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".model.fits")
    resid_cube = cube_utils.SpectralCube.read(resid_fn)
    resid_mom0 = resid_cube.spectral_slab(*vel_bounds).moment(order=0).to(u.K*u.km/u.s).to_value()

    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(121, projection=cube.wcs, slices=('x', 'y', 50))
    ax2 = plt.subplot(122, projection=cube.wcs, slices=('x', 'y', 50), sharex=ax1, sharey=ax1)

    im = ax1.imshow(resid_mom0/original_mom0, origin='lower', vmin=-0.2, vmax=0.3)
    ax1.set_title("Residuals/Original moment 0 ([20, 30] km/s)")
    ax1.contour(*contour_args, **contour_kwargs)
    fig.colorbar(im, ax=ax1)

    metric = (-resid_cube.spectral_slab(*vel_bounds).unmasked_data[:]*np.sign(cube.spectral_slab(*vel_bounds).unmasked_data[:])).to_value()
    metric[metric < 0] = 0
    metric = metric.sum(axis=0)
    im = ax2.imshow(metric, origin='lower', vmin=0, vmax=40)
    ax2.set_title("Sum of negative residuals (x -1) [20, 30]")
    ax2.contour(*contour_args, **contour_kwargs)
    fig.colorbar(im, ax=ax2)

    plt.show()


if __name__ == "__main__":
    subcube = cutout_subcube(length_scale_mult=4)#, data_filename="apex/M16_13CO3-2.fits")
    subcube = subcube.spectral_slab(0*u.km/u.s, 45*u.km/u.s)

    # try_mask_above_half_power(subcube, xpower=2)
    fit_live_interactive(subcube)
    # make_wing_moments(subcube)
    # fit_image_to_file(subcube)
    # investigate_fit()

    # calculate_noise(subcube)
