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

    img, w = crosscut.DataLayer("CII", data_filename, cube=True, alpha=0.7, vlims=(5, 40)).load()
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
    """
    half_power = np.max(cube, axis=0)/xpower
    mask = (cube > half_power) & (half_power > additional_cutoff*cube.unit)
    masked_cube = cube.with_mask(mask)
    return masked_cube


def try_mask_above_half_power(cube, xpower=2):
    """
    Per Lee's recommendation in our Sept 25, 2020 Friday meeting
    For each spatial pixel, I will mask out spectral pixels that are below
    the half-power (or 1/3, or something) level of that spectrum.
    Then, I can make a moment 0 or 1 or whatever image with that.

    SpectralCube offers pretty good masking capability, so I'll make the most
    of that.

    xpower is something to DIVIDE the max by for the  mask. If you want below
    half power masked out, then xpower = 2. If you want below 1/3, then it's 3
    """
    kms = u.km/u.s
    kkms = u.K*u.km/u.s
    moment = 2
    moment_units = [kkms, kms, kms*kms][moment]
    masked_cube = mask_above_xpower(cube, xpower, additional_cutoff=6.5)
    fig = plt.figure(figsize=(10, 8))
    # 'x' 'y' 50 came from astropy documentation; 50 i think doesn't specifically matter
    ax0 = plt.subplot2grid((2, 2), (0, 0), fig=fig, projection=cube.wcs, slices=('x','y', 50))
    ax1 = plt.subplot2grid((2, 2), (0, 1), fig=fig, projection=cube.wcs, slices=('x','y', 50))
    ax2 = plt.subplot2grid((2, 2), (1, 0), fig=fig)#, projection=cube.wcs, slices=(50, 50, 'x'))
    ax3 = plt.subplot2grid((2, 2), (1, 1), fig=fig)#, projection=cube.wcs, slices=(50, 50, 'x'))
    # for ax in (ax2, ax3):
    #     ra, dec, vel = ax.coords
    #     vel.set_format_unit(kms)
    img = cube.moment(order=moment).to(moment_units).to_value()
    im = ax0.imshow((img), origin='lower')
    fig.colorbar(im, ax=ax0)
    img = masked_cube.moment(order=moment).to(moment_units).to_value()
    im = ax1.imshow((img), origin='lower', vmax=4)
    fig.colorbar(im, ax=ax1)
    ax2.plot(cube.spectral_axis.to_value(), cube.mean(axis=(1, 2)).to(u.K).to_value(), linewidth=0.7)
    ax3.plot(masked_cube.spectral_axis.to_value(), masked_cube.mean(axis=(1, 2)).to(u.K).to_value(), linewidth=0.7)
    ax3.set_xlim(ax2.get_xlim())
    ax2.set_title("Mean spectrum, unmasked")
    ax3.set_title(f"Mean spectrum, masked above 1/{xpower} power")
    plt.show()


def fit_live_interactive(cube):
    # INTERACTIVE
    masked_cube = mask_above_xpower(cube, 3, additional_cutoff=2.5)
    full_power_loc = cube.argmax(axis=0)
    full_power = cube.max(axis=0).to_value()
    linewidth_manual = linewidth_fwhm(cube)

    linewidth_spectralcube = masked_cube.linewidth_fwhm().to_value()
    moment1 = masked_cube.moment(order=1).to_value()
    moment0 = masked_cube.moment(order=0).to_value()

    spectral_axis = masked_cube.spectral_axis.to_value()

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax_manual = plt.subplot2grid((2, 5), (0, 0), colspan=2, fig=fig, projection=cube.wcs, slices=('x', 'y', 50))
    ax_spcube = plt.subplot2grid((2, 5), (1, 0), colspan=2, fig=fig, projection=cube.wcs, slices=('x', 'y', 50), sharex=ax_manual, sharey=ax_manual)
    ax_manual.tick_params(axis='x', labelbottom=False)
    ax_spectr = plt.subplot2grid((2, 5), (0, 2), colspan=3, rowspan=2, fig=fig)

    ax_manual.imshow(moment0, origin='lower')
    ax_manual.set_title("Moment 0 (masked)")
    ax_spcube.imshow(moment1, origin='lower')
    ax_spcube.set_title("Moment 1 (masked)")

    ax_spectr.plot(spectral_axis, masked_cube.mean(axis=(1, 2)).to(u.K).to_value(), linewidth=0.7)

    fitter = fitting.SLSQPLSQFitter()
    xs = {'x1': None, 'x2': None}

    def onclick(event):
        try:
            j, i = int(round(event.xdata)), int(round(event.ydata))
        except Exception as e:
            print(f"something went wrong... {e}")
            return
        if event.button == 1 and (event.inaxes is ax_manual or event.inaxes is ax_spcube):
            ax_spectr.clear()
            if xs['x1'] is not None:
                ax_manual.lines.remove(xs['x1'])
                ax_spcube.lines.remove(xs['x2'])

            xs['x1'], = ax_manual.plot([j], [i], 'x', color='red')
            xs['x2'], = ax_spcube.plot([j], [i], 'x', color='red')

            A = full_power[i, j]

            mean_manual = spectral_axis[full_power_loc[i, j]]
            mean_spcube = moment1[i, j]

            std_manual = linewidth_manual[i, j]/2.355
            std_spcube = linewidth_spectralcube[i, j]/2.355

            g_manual = models.Gaussian1D(A, mean_manual, std_manual, name='manual')
            g_spcube = models.Gaussian1D(A, mean_spcube, std_spcube, name='spectral_cube')
            g_hybrid = models.Gaussian1D(A, mean_spcube, std_manual, name='hybrid')
            g_manual_array = g_manual(spectral_axis)
            g_spcube_array = g_spcube(spectral_axis)
            g_hybrid_array = g_hybrid(spectral_axis)

            # g_manual.amplitude.min = A*0.99
            # g_manual.amplitude.max = A*1.03
            if np.abs(mean_spcube - mean_manual) < min(std_manual, 1.5):
                mean = mean_spcube
            else:
                mean = mean_manual
            g_manual.mean = mean
            # g_manual.mean.min = mean*0.95
            # g_manual.mean.max = mean*1.05

            # Standard deviation
            std = std_manual
            if std < 0.5 or std > 3:
                std = 1.5
            g_manual.stddev = std
            # g_manual.stddev.min = std - 0.8
            # g_manual.stddev.max = std + 0.8
            masked_spectrum_val = masked_cube.filled_data[:, i, j].to_value()
            """
            astropy modeling does NOT like NaNs. That's weird! They should!
            """
            fit_x, fit_y = spectral_axis[np.isfinite(masked_spectrum_val)], masked_spectrum_val[np.isfinite(masked_spectrum_val)]
            weights = np.abs(fit_y)
            weights[weights < 1.3] = 1.3
            weights = (weights/np.max(weights))/1.3
            g_fit_werr = fitter(g_manual, fit_x, fit_y, verblevel=1, weights=weights) # 1.3 from an estimate of < 10 km/s
            g_fit_werr_array = g_fit_werr(spectral_axis)

            spectrum = cube[:, i, j]
            ax_spectr.plot(spectral_axis, spectrum, color='k', linewidth=0.7, label='Data', marker='o', alpha=0.2, markersize=3)
            ax_spectr.plot(spectral_axis, masked_spectrum_val, color='Indigo', marker='x', linewidth=2, linestyle='dotted', alpha=0.9, label='Data fitted to')

            # ax_spectr.plot(spectral_axis, g_manual_array, color='r', linewidth=0.7, label=g_manual.name)
            # ax_spectr.plot(spectral_axis, g_spcube_array, color='b', linewidth=0.7, label=g_spcube.name)
            # ax_spectr.plot(spectral_axis, g_hybrid_array, color='purple', linewidth=0.7, label=g_hybrid.name)

            ax_spectr.plot(spectral_axis, g_fit_werr_array, color='orange', linestyle='dotted', linewidth=1, label="Fitted", alpha=0.9)
            ax_spectr.plot(spectral_axis, spectrum.to_value() - g_fit_werr_array, color='k', linewidth=0.7, label='Residuals', alpha=0.6)

            # Metric; sum over this for a number that, if large, means the fit isn't good
            metric = -(spectrum.to_value() - g_fit_werr_array)*np.sign(spectrum.to_value())
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
    if spectral_axis is None:
        xarr = np.arange(spec_mask.size)
    else:
        xarr = spectral_axis
    diffs = np.diff(spec_mask.astype(int))
    # Starts are the indices of "False" right BEFORE the True
    run_starts, = np.where(diffs > 0)
    # Ends are the LAST indices of "True" BEFORE the False
    run_ends, = np.where(diffs < 0)
    # Lengths are calculated from the x axis (spectral_axis)
    run_lengths = xarr[run_ends] - xarr[run_starts+1]
    max_length = run_lengths.max()
    # return np.full_like(spec, max_length)
    return max_length


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
    spectral_axis = cube.spectral_axis.to_value()
    fitter = fitting.SLSQPLSQFitter()
    g_init = models.Gaussian1D(25, 26, 1, bounds={'amplitude': (0, 50), 'mean': (22, 28), 'stddev': (1, 3.5)})
    xpower, additional_cutoff = 3, 2.5
    masked_cube = mask_above_xpower(cube, xpower, additional_cutoff=additional_cutoff)

    full_power_loc = masked_cube.argmax(axis=0)
    full_power = masked_cube.max(axis=0).to_value()
    moment1 = masked_cube.moment(order=1).to_value()
    linewidth_manual = linewidth_fwhm(cube)

    results = np.zeros((len(g_init.param_names)*2, *masked_cube.shape[1:]))
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
        spectrum = masked_cube.filled_data[:, i, j].to_value()

        # Get some initial guesses and inform the boundaries
        # Amplitude
        A = full_power[i, j]
        g_init.amplitude = A
        g_init.amplitude.min = A*0.99
        g_init.amplitude.max = A*1.03

        # Standard deviation
        std = linewidth_manual[i, j]/2.355
        if std < 0.5 or std > 3:
            std = 1.5
        g_init.stddev = std
        g_init.stddev.min = std - 0.8
        g_init.stddev.max = std + 0.8

        # Mean
        mean = moment1[i, j]
        mean_manual = spectral_axis[full_power_loc[i, j]]
        if np.abs(mean - mean_manual) > min(std, 1.5):
            mean = mean_manual
        g_init.mean = mean
        g_init.mean.min = mean*0.95
        g_init.mean.max = mean*1.05

        g_fit = fitter(g_init, spectral_axis, spectrum, verblevel=0, weights=(spectrum/(5*1.3))) # is arbitrary, just normalizing to SNR=5
        results[:len(g_init.param_names), i, j] = g_init.param_sets[:, 0]
        results[len(g_init.param_names):, i, j] = g_fit.param_sets[:, 0]
        fitted_spectrum = g_fit(spectral_axis)
        residuals_array[:, i, j] = masked_cube[:, i, j].to_value() - fitted_spectrum
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

    filename_stub = "models/gauss_fit_above_1G_v2"
    param_units = ['K', 'km / s', 'km / s']
    wcs_flat = cube.moment(order=0).wcs
    to_header = lambda : wcs_flat.to_header()
    phdu = fits.PrimaryHDU(header=to_header())
    phdu.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    phdu.header['CREATOR'] = f"Ramsey, {__file__}"
    phdu.header['COMMENT'] = f"Fit above 1/{xpower:.1f} x max power, which must also be > {additional_cutoff:.1f}."
    phdu.header['COMMENT'] = "Using weights of spectrum/1.3 (1.3 K rms estimate)"
    phdu.header['COMMENT'] = "Cutout with length_scale_mult 8"
    hdu_list = [phdu]
    for i in range(len(g_init.param_names)):
        hdu = fits.ImageHDU(data=results[i], header=to_header())
        hdu.header['EXTNAME'] = g_init.param_names[i] + "_INIT"
        hdu.header['BUNIT'] = param_units[i]
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(data=results[i+len(g_init.param_names)], header=to_header())
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
    phdu.header['COMMENT'] = f"Fit above 1/{xpower:.1f} x max power"
    phdu.header['COMMENT'] = "Cutout with length_scale_mult 8"
    phdu.writeto(cube_utils.os.path.join(cube_info['dir'], f"{filename_stub}.resid.fits"), overwrite=True)

    phdu = fits.PrimaryHDU(data=models_array, header=cube.wcs.to_header())
    phdu.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    phdu.header['CREATOR'] = f"Ramsey, {__file__}"
    phdu.header['BUNIT'] = 'K'
    phdu.header['COMMENT'] = 'Model intensity'
    phdu.header['COMMENT'] = f"Fit above 1/{xpower:.1f} x max power"
    phdu.header['COMMENT'] = "Cutout with length_scale_mult 8"
    phdu.writeto(cube_utils.os.path.join(cube_info['dir'], f"{filename_stub}.model.fits"), overwrite=True)

    print("Done!")


def investigate_fit():
    """
    Investiate the fit made in the previous function
    """

    filename_stub = "models/gauss_fit_above_1G_v2"
    param_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".param.fits")
    resid_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".resid.fits")
    model_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".model.fits")

    hdul = fits.open(param_fn)
    print(list(hdu.header['EXTNAME'] for hdu in hdul if 'EXTNAME' in hdu.header))

    resid_cube = cube_utils.SpectralCube.read(resid_fn)
    model_cube = cube_utils.SpectralCube.read(model_fn)

    spectral_axis = resid_cube.spectral_axis.to(u.km/u.s)

    vlo, vhi = 25, 30
    resid_int = resid_cube.spectral_slab(vlo*u.km/u.s, vhi*u.km/u.s).moment(order=0).to(u.K*u.km/u.s)

    plt.ion()
    fig = plt.figure()
    ax_img = plt.subplot2grid((1, 5), (0, 0), colspan=2, fig=fig, projection=resid_cube.wcs, slices=('x', 'y', 0))
    ax_spectr = plt.subplot2grid((1, 5), (0, 2), colspan=3, fig=fig)

    im = ax_img.imshow(resid_int.to_value(), origin='lower', vmin=0, vmax=35)
    fig.colorbar(im, ax=ax_img)
    ax_img.set_title(f"integrated residuals, [{vlo}, {vhi}] km/s")

    xs = {'x1': None}

    def onclick(event):
        try:
            j, i = int(round(event.xdata)), int(round(event.ydata))
        except Exception as e:
            print(f"something went wrong... {e}")
            return
        if event.button == 1 and event.inaxes is ax_img:
            ax_spectr.clear()
            if xs['x1'] is not None:
                ax_img.lines.remove(xs['x1'])
            xs['x1'], = ax_img.plot([j], [i], 'x', color='red')
            A = hdul['amplitude_INIT'].data[i, j]
            mean = hdul['mean_INIT'].data[i, j]
            stddev = hdul['stddev_INIT'].data[i, j]
            resid_spectr = resid_cube[:, i, j]
            model_spectr = model_cube[:, i, j]
            ax_spectr.plot(spectral_axis, model_spectr, label='model', linewidth=0.7)
            ax_spectr.plot(spectral_axis, resid_spectr, label='resid', linewidth=0.7)
            ax_spectr.plot(spectral_axis, resid_spectr+model_spectr, color='k', label='original', linewidth=0.7, marker='o', alpha=0.5)
            ax_spectr.legend()
            # ax_spectr.set_xlim()
            ax_spectr.set_xlabel("v (km/s)")
            ax_spectr.set_ylabel("T (K)")

    # hdul.close()
    return fig.canvas.mpl_connect('button_press_event', onclick)

def make_wing_moments(cube):
    """
    Check emission between [20, 21.5] and [28.5, 30]
    """
    kms = u.km/u.s
    integrated = cube.spectral_slab(21.5*kms, 28.5*kms).moment(order=0).to_value()
    contour_args = (integrated,)
    contour_kwargs = dict(levels=[30, 60, 90, 120, 150], linewidths=0.5, colors='k', alpha=0.5)

    blue_slab = cube.spectral_slab(20*kms, 21.5*kms)
    red_slab = cube.spectral_slab(28.5*kms, 30*kms)
    blue_integrated = blue_slab.moment(order=0).to_value()
    red_integrated = red_slab.moment(order=0).to_value()
    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(121, projection=cube.wcs, slices=('x', 'y', 50))
    ax1.imshow(np.arcsinh(blue_integrated), origin='lower')
    ax1.set_title("Blue wing, integrated [20, 21.5] km/s")
    ax1.contour(*contour_args, **contour_kwargs)

    ax2 = plt.subplot(122, projection=cube.wcs, slices=('x', 'y', 50), sharex=ax1, sharey=ax1)
    ax2.imshow(np.arcsinh(red_integrated), origin='lower')
    ax2.set_title("Red wing, integrated [28.5, 30] km/s")
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


if __name__ == "__main__":
    subcube = cutout_subcube(length_scale_mult=4)
    # try_mask_above_half_power(subcube, xpower=2)
    fit_live_interactive(subcube)
    # make_wing_moments(subcube)
    # fit_image_to_file(subcube)

    # calculate_noise(subcube)

    # investigate_fit()
