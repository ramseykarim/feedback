import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 8}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys
import warnings
import datetime
import time

from scipy import signal
from scipy.interpolate import UnivariateSpline

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.modeling import models, fitting
from astropy import convolution
from reproject import reproject_interp

import regions
from math import ceil

from . import misc_utils
from . import catalog
from . import cube_utils
from . import crosscut
pvdiagrams = crosscut.pvdiagrams
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
make_vel_stub = lambda x : f"[{x[0].to_value():.1f}, {x[1].to_value():.1f}] {x[0].unit}"


def cutout_subcube(length_scale_mult=2, data_filename=None, reg_filename=None,
    length_scale=None, global_center_coord=None, reg_index=0, return_cutout=False):
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

    cube = cube_utils.CubeData(data_filename)
    cube.convert_to_K()
    cube_info['dir'] = cube.directory
    if length_scale_mult is None:
        return cube.data.with_spectral_unit(u.km/u.s)

    if length_scale_mult is not None:
        try:
            img = cube.data.moment0()
            w = img.wcs
            img = img.to_value()
            # img, w = crosscut.DataLayer("", data_filename, cube=True, alpha=0.7, vlims=(5, 40)).load()
            img_cutout = Cutout2D(img, global_center_coord, [length_scale*length_scale_mult]*2, wcs=w, mode='partial', fill_value=np.nan)
        except:
            print("(cutout_subcube) failed to get WCS info for the cube")
            img = w = img_cutout = None

    # just return the Cutout2D
    if return_cutout:
        return img_cutout

    if img_cutout is not None:
        # Make a subcube using those slices. This has good WCS (even though it doesn't look like it)
        subcube = cube.data[:, img_cutout.slices_original[0], img_cutout.slices_original[1]]
        cube_info['cutout'] = img_cutout # save the slices
    else:
        subcube = cube.data
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


def mask_with_best_setting(cube, dataset='sofia'):
    """
    Use mask_above_xpower with the current best recipe
    """
    if dataset == 'apex':
        return mask_above_xpower(cube, 2, additional_cutoff=0)
    elif dataset == 'sofia':
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
    # Moment 0 and 2D WCS
    moment0 = masked_cube.moment(order=0)
    wcs_flat = moment0.wcs
    moment0 = moment0.to_value()
    # Spectral Axis
    spectral_axis = unmasked_cube.spectral_axis.to_value()
    # Gather in dictionary
    return_dict = {
        'full_power_loc': full_power_loc,
        'moment0': moment0, 'moment1': moment1,
        'wcs_flat': wcs_flat,
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


def initialize_double_gaussian(init_conds, g, ij):
    """
    :param init_conds: dictionary returned by prepare_initial_conditions
    :param g: a SINGLE Gaussian1D object, ALREADY FITTED, to base initial
        conditions on
    :param ij: tuple array indices
    """
    i, j = ij
    # Amplitude
    A = g.amplitude/2
    A_bounds = (A*0.85, A*1.15)
    # Standard deviation
    std_manual = init_conds['linewidth_manual'][i, j] / 2.355
    std_spcube = init_conds['linewidth_spectralcube'][i, j] / 2.355
    if std_manual < 0.3 or std_manual > 2.5:
        std = 1.5
    else:
        std = std_manual
    std_bounds = (0.2, 3)
    # Mean
    mean_0 = g.mean - 1.4
    mean_1 = g.mean + 1
    mean_bounds = (g.mean*0.85, g.mean*1.15)
    # Create the double Gaussian
    doubleG = models.Gaussian1D() + models.Gaussian1D()
    doubleG.amplitude_0 = A
    doubleG.amplitude_1 = A
    # doubleG.amplitude_1.tied = lambda m: m.amplitude_0
    doubleG.amplitude_0.bounds = A_bounds
    doubleG.amplitude_1.bounds = A_bounds
    doubleG.mean_0 = mean_0
    doubleG.mean_1 = mean_1
    doubleG.mean_0.bounds = mean_bounds
    doubleG.mean_1.bounds = mean_bounds
    doubleG.stddev_0 = std
    doubleG.stddev_1 = std
    doubleG.stddev_0.bounds = std_bounds
    doubleG.stddev_1.bounds = std_bounds
    return doubleG


def fit_gaussian(init_conds, masked_cube, g, ij, fitter, double=False, template=False, cube_is_masked=True, noise=None):
    """
    :param init_conds: return dict of prepare_initial_conditions
    :param masked_cube: already masked SpectralCube
    :param g: Gaussian1D already initialized
        Could be a composite model if you are using template=True
    :param ij: tuple array indices
    :param fitter: some kind of astropy.modeling.fitting fitter
    :param double: fit 2 Gaussian1Ds near the peak of the single Gaussian
        Ignored if template=True
    :param template: use the input 'g' as a template and don't modify it.
        This implies 'g' may be a composite model already.
        If True, renders 'double' irrelevant
    :param cube_is_masked: if True, jump through the masking hoops.
        If False, just fit the pixel
    :param noise: Could be None, a single value, or an array
        If None, then this function will give more weight to larger values.
        If scalar or array:
            scalar -> array of the scalar value
            These values are interpreted as Gaussian uncertainties on the data
            in the same units as the data. They will be passed to the fitter
            as weights in the form 1/noise (recommend by: https://docs.astropy.org/en/stable/modeling/example-fitting-line.html#fit-using-uncertainties)
    :returns: fitted Gaussian1D and the resulting model array
    """
    i, j = ij
    masked_spectrum_val = masked_cube.filled_data[:, i, j].to_value()
    if np.sum(np.isfinite(masked_spectrum_val).astype(int)) < 3:
        nan_array = np.full(masked_cube.shape[0], np.nan)
        return None, nan_array, nan_array
    if cube_is_masked:
        masked_spectrum_mask = masked_cube.get_mask_array()[:, i, j]
        new_spectrum_mask = identify_longest_run(masked_spectrum_mask)
        masked_spectrum_val[~new_spectrum_mask] = np.nan

    # astropy.modeling does not like NaNs!!!
    finite_mask = np.isfinite(masked_spectrum_val)
    if np.sum(finite_mask.astype(int)) > 3:
        fit_x, fit_y = init_conds['spectral_axis'][finite_mask], masked_spectrum_val[finite_mask]
        if noise is None:
            weights = np.abs(fit_y)
            weights[weights < 1.3] = 1.3
            weights = (weights/np.max(weights))/1.3
        else:
            if isinstance(noise, np.ndarray):
                weights = 1.0 / noise[finite_mask]
            else:
                weights = np.full(fit_x.size, 1.0/noise)

        g_fit = fitter(g, fit_x, fit_y, weights=weights)

        if double and not template:
            doubleG = initialize_double_gaussian(init_conds, g_fit, ij)
            g_fit = fitter(doubleG, fit_x, fit_y, weights=weights)

        g_fit_array = g_fit(init_conds['spectral_axis'])
        return g_fit, g_fit_array, masked_spectrum_val
    else:
        nan_array = np.full(masked_cube.shape[0], np.nan)
        return None, nan_array, nan_array


"""
=======================
New stuff for fitting live!!!
=======================
"""

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
        return select_pixels_and_models('hcop-cii', i, var_mean=var_mean, var_std=var_std)



    elif mol == '12co10':
        ### This was my work for 12CO(1-0). These worked alright but not great for every area

        if i == 'bluest component':
            good_pixel = (466, 275) # good for bluest component
            di, dj = 2, 3
            g = models.Gaussian1D(amplitude=50, mean=23.8, stddev=1.06,
                bounds={'amplitude': (0, 200)})

        elif i == 'blue thread':
            good_pixel = (405, 287) # blue (W) thread
            di, dj = 2, 3
            g = models.Gaussian1D(amplitude=50, mean=25.1, stddev=0.95,
                bounds={'amplitude': (0, 200)})

        elif i == 'red main part':
            good_pixel = (408, 243) # red main part
            di, dj = 5, 5
            g = models.Gaussian1D(amplitude=50, mean=25.8, stddev=0.83,
                bounds={'amplitude': (0, 200)})


    elif mol[:4] == 'hcop':

        if i == 'western horn':
            # This is the Western horn component
            if mol[-4:] == '-cii':
                good_pixel = (12, 24)
                di, dj = 0, 0
            else:
                good_pixel = (447, 375)
                di, dj = 2, 2
            g = models.Gaussian1D(amplitude=10.291692169984568, mean=24.440935924615744, stddev=0.46, #0.4614265241399322
                bounds={"amplitude": (0, 100), "mean": (23, 27), "stddev": (0.1, 2)})

        elif i == 'bluest component':
            # This is the bluest N-E corner component
            if mol[-4:] == '-cii':
                good_pixel = (35, 30)
                di, dj = 0, 0
            else:
                good_pixel = (602, 415)
                di, dj = 1, 1
            g = models.Gaussian1D(amplitude=5.2758702607467525, mean=23.46286597585026, stddev=0.46, # fitted stddev = 0.4526447822523458
                bounds={"amplitude": (0.6, 30), "mean": (22.5, 25), "stddev": (0.1, 2)})

        elif i == 'bluest component 2':
            # Now even further out in the blue component
            good_pixel = (610, 420)
            di, dj = 1, 1
            g = models.Gaussian1D(amplitude=2.68, mean=23.516, stddev=0.46, # fitted stddev = 0.45380048744753915
                bounds={"amplitude": (0, 30), "mean": (21, 26), "stddev": (0.1, 2)})

        elif i == 'western thread N':
            # Western thread, from a pixel a little above it that shows a clean spectrum
            good_pixel = (544, 448)
            di, dj = 1, 1
            g = models.Gaussian1D(amplitude=5.305657851279191, mean=24.917103298134492, stddev=0.46, # fitted stddev = 0.44549289151047716
                bounds={"amplitude": (0, 30), "mean": (21, 27), "stddev": (0.1, 2)})

        elif i == 'just off peak':
            # Check out HCO+ (near) peak spectrum to see how its width stacks up with 0.46
            good_pixel = (570, 451)
            di, dj = 1, 1
            g = models.Gaussian1D(amplitude=20, mean=25, stddev=0.46, # fitted stddev = 0.44549289151047716
                bounds={"amplitude": (0, 30), "mean": (21, 27), "stddev": (0.1, 2)})

        elif i == 'eastern thread N':
            # Eastern thread N sample (trying to avoid the other thing that's there to the west)
            good_pixel = (556, 393)
            di, dj = 4, 4
            g = models.Gaussian1D(amplitude=4.27, mean=25.85, stddev=0.46, # fitted stddev = 0.72
                bounds={"amplitude": (1, 30), "mean": (25, 26.5), "stddev": (0.1, 2)})

        elif i == 'eastern thread S':
            # Eastern thread S sample
            good_pixel = (541, 373)
            di, dj = 6, 6
            g = models.Gaussian1D(amplitude=3.74, mean=25.76, stddev=0.46, # fitted stddev = 0.64
                bounds={"amplitude": (1, 30), "mean": (25, 26.5), "stddev": (0.1, 2)})

        elif i == 'main red':
            # just north of the Eastern thread, and probably the main red component in the peak
            if mol[-4:] == '-cii':
                good_pixel = (27, 30)
                di, dj = 0, 0
            else:
                good_pixel = (564, 425)
                di, dj = 2, 2
            g = models.Gaussian1D(amplitude=12.4, mean=25.35, stddev=0.46, # fitted stddev = 0.61
                bounds={"amplitude": (1, 30), "mean": (25, 26.5), "stddev": (0.1, 2)})

        elif i == 'east of peak':
            # this is a compound model!
            if mol[-4:] == '-cii':
                good_pixel = (33, 32)
                di, dj = 0, 0
            else:
                good_pixel = (583, 431)
                di, dj = 3, 3
            g_red = select_pixels_and_models('hcop', 'main red', var_mean=var_mean, var_std=var_std)[2]
            g_blue = select_pixels_and_models('hcop', 'bluest component', var_mean=var_mean, var_std=var_std)[2]
            g = g_red + g_blue
            tie_std_models(g)

        elif i[:4] == 'peak' and len(i) < 7:
            # there are four options for the pixel here
            if i == 'peak':
                # main peak
                if mol[-4:] == '-cii':
                    good_pixel = (30, 35)
                else:
                    good_pixel = (572, 450)
            elif i[-1] == 'N':
                good_pixel = (588, 466)
            elif i[-1] == 'E':
                good_pixel = select_pixels_and_models(mol, 'east of peak', var_mean=var_mean, var_std=var_std)[0]
            elif i[-1] == 'W':
                good_pixel = (569, 465)
            elif i[-1] == 'S':
                good_pixel = (561, 446)
            if mol[-4:] == '-cii':
                di, dj = 0, 0
            else:
                di, dj = 3, 3
            g0 = models.Gaussian1D(amplitude=7, mean=23.58, stddev=0.46,
                bounds={'amplitude': (0, None), 'mean': (20, 30)}) # 23-24 or so, based on (23.2, 23.9)
            g1 = models.Gaussian1D(amplitude=7, mean=24.57, stddev=0.46,
                bounds={'amplitude': (0, None), 'mean': (20, 30)}) # 24-25, based on (24.4, 24.9)
            g2 = models.Gaussian1D(amplitude=7, mean=25.43, stddev=0.46,
                bounds={'amplitude': (0, None), 'mean': (20, 30)}) # 25-26, based on (25.3, 25.6)
            g = g0 + g1 + g2
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
    It assumes the submodels are Gaussian1D (or at least have a 'mean' parameter)
    It will return the iterable in INCREASING ORDER OF MEAN (line center)
    :param model: an astropy.modeling.models model, compound OR single
    :returns: iterator that will return a single model per iteration;
        models will be in increasing order of Gaussian mean parameter value
    """
    try:
        sorted_by_means = sorted(iter(model), key=lambda x: x.mean)
        # Put the zero-amplitude components last
        nonzero_components = []
        zero_components = []
        for m in sorted_by_means:
            if m.amplitude.value == 0:
                zero_components.append(m)
            else:
                nonzero_components.append(m)
        return nonzero_components + zero_components
    except:
        return iter((model,))


def tie_std_models(model, untie=False):
    """
    November 5, 2021
    Convenience function for tying all the stddevs together
    If they're already fixed, it shouldn't have any effect
    If single (not compound) model, no effect (unless untie==True)
    :param model: an astropy.modeling.models model
    :param untie: if True, unties the models (even if single)
    """
    try:
        for i, m in enumerate(model):
            if untie:
                m.stddev.tied = False
            else:
                if i == 0:
                    pass
                else:
                    m.stddev.tied = lambda x: x.stddev_0
    except:
        if untie:
            try:
                model.stddev.tied = False
            except:
                pass
        else:
            pass


def fix_mean(model, set_to=True):
    """
    November 9, 2021
    Convenience function for fixing mean parameter for an unknown number
    of composite or single models
    If already fixed, no effect
    Moved from m16_deepdive.py on Nov 11
    :param model: an astropy.modeling.models model
    :param set_to: whether to fix or unfix. Default is fix (mean cannot change)
    """
    for m in iter_models(model):
        m.mean.fixed = set_to


def unfix_mean(model):
    """
    December 10, 2021
    In case I forget that I have a set_to option in fix_mean. More intuitive.
    Runs fix_mean(model, set_to=FALSE)
    :param model: an astropy.modeling.models model
    """
    fix_mean(model, set_to=False)


def fix_std(model, set_to=True):
    """
    November 9, 2021
    Convenience function for fixing stddev parameter for an unknown number
    of composite or single models
    If already fixed, no effect
    Moved from m16_deepdive.py on Nov 11
    :param model: an astropy.modeling.models model
    :param set_to: whether to fix or unfix. Default is fix (stddev cannot change)
    """
    for m in iter_models(model):
        m.stddev.fixed = set_to


def unfix_std(model):
    """
    December 10, 2021
    In case I forget that I have a set_to option in fix_std. More intuitive.
    Runs fix_std(model, set_to=FALSE)
    :param model: an astropy.modeling.models model
    """
    fix_std(model, set_to=False)


def make_show_box(show_box_i_lims, show_box_j_lims):
    """
    November 5, 2021
    Moved from m16_deepdive.py on Nov 11
    """
    show_box_i_lo, show_box_i_hi = show_box_i_lims
    show_box_j_lo, show_box_j_hi = show_box_j_lims
    return (slice(show_box_i_lo, show_box_i_hi), slice(show_box_j_lo, show_box_j_hi))



def plot_noise_and_vlims(ax, noise, vel_lims):
    """
    November 5, 2021
    Moved from m16_deepdive.py on Nov 11
    """
    if noise is not None:
        [ax.axhline(sign*noise, color='grey', alpha=0.3, linestyle='--') for sign in (-1, 1)]
    if vel_lims is not None:
        [ax.axvline(v, color='grey', alpha=0.5) for v in vel_lims]


def plot_box(ax, i_lims, j_lims, show_box_lo_lims):
    """
    November 5, 2021
    Moved from m16_deepdive.py on Nov 11
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
    Moved from m16_deepdive.py on Nov 11
    """
    show_box_i_lo, show_box_j_lo = show_box_lo_lims
    ax.plot([noise_loc[1] - show_box_j_lo], [noise_loc[0] - show_box_i_lo], 'x', color='grey')


def plot_everything_about_models(ax, xaxis, spectrum, model, m_color='r', text_x=0.05, text_y=0.93, dy=-0.05, noise=None, dof=None):
    """
    November 5, 2021
    Convenience function for plotting all these models
    Moved from m16_deepdive.py on Nov 11
    August 19, 2022: added the peak velocity to the sidebar printout
    :param model: an astropy.modeling.models model
    """
    if spectrum is not None:
        ax.plot(xaxis, spectrum, color='k', linestyle='-', marker='|')
        # Calculate and print the peak velocity (subject to noise spikes, but better than nothing)
        peak_velocity = xaxis[np.argmax(spectrum)]
        ax.text(text_x, text_y, "$v_{\\rm peak}$ = " + f"{peak_velocity:5.2f}", transform=ax.transAxes, color='k')
        text_y += dy # increment text_y for the rest of the printouts
        # Put a vertical dashed line through the peak velocity
        ax.axvline(peak_velocity, color='k', linestyle='--', alpha=0.3)

    if model is None:
        return
    fitted_spectrum = model(xaxis)
    ax.plot(xaxis, fitted_spectrum, color=m_color, linestyle='-', alpha=0.8)
    if spectrum is not None:
        ax.plot(xaxis, spectrum-fitted_spectrum, color='g', alpha=0.6, linestyle='--')
    for i, m in enumerate(iter_models(model)):
        component_is_nonzero = (m.amplitude.value != 0)
        if component_is_nonzero:
            ax.plot(xaxis, m(xaxis), color=m_color, linestyle='--', alpha=0.6)
            ax.axvline(m.mean.value, color=m_color, linestyle='--', alpha=0.3)
        amplitude_unc_txt = f"$\pm${m.amplitude.std:.3f}" if m.amplitude.std is not None else ""
        mean_unc_txt = f"$\pm${m.mean.std:.3f}" if m.mean.std is not None else ""
        stddev_unc_txt = f"$\pm${m.stddev.std:.3f}" if m.stddev.std is not None else ""
        alpha = 1 if component_is_nonzero else 0.15
        ax.text(text_x, text_y + dy*(1 + 4*i), f"$\mu_{i}$ = {m.mean.value:5.2f}{mean_unc_txt}", transform=ax.transAxes, color=m_color, alpha=alpha)
        ax.text(text_x, text_y + dy*(2 + 4*i), f"$\sigma_{i}$ = {m.stddev.value:5.2f}{stddev_unc_txt}", transform=ax.transAxes, color=m_color, alpha=alpha)
        ax.text(text_x, text_y + dy*(0 + 4*i), f"$A_{i}$ = {m.amplitude.value:5.2f}{amplitude_unc_txt}", transform=ax.transAxes, color=m_color, alpha=alpha)
    if noise is not None:
        chisq = np.sum((spectrum-fitted_spectrum)**2 / noise**2)
        if dof is not None:
            chisq_stub = "$\\chi^{2}$/dof"
            chisq = chisq/dof
        else:
            chisq_stub = "chisq"
        ax.text(text_x, text_y + dy*(0 + 4*(i+1)), f"{chisq_stub} = {chisq:.2f}", transform=ax.transAxes, color=m_color)


def fit_live_interactive(cube, template_model=None, double=False, mask=True, noise=None,
        ylim_min=-3, ylim_max=23, n_params=None, live_intercept=None, invert_xaxis=False):
    """
    INTERACTIVE fitting and plotting
    :param cube: the cube to fit to
    :param template_model: an astropy.modeling.models model to use as an initial
        input for fitting. If you use this option, this function won't do any
        of it's "clever" boundary or initial guess stuff. It'll just use your
        guess and its bounds and stuff.
        Will render "double" argument irrelevant
    :param double: whether to fit a single or double Gaussian model. If you
        set template_model, then this argument isn't used at all.
    :param mask: whether to mask the spectra with an intensity cutoff
        If True, then some areas with low power won't be able to be fit at all.
    :param noise: arg to pass thru to fit_gaussian; see that description
        If this is a 2-element tuple, will calculate noise from that pixel location
        in the unmasked cube
    :param n_params: number of fitted params in the template_model. This function
        won't detect it automatically, so if you want a reduced chi squared
        calculated, you have to input it yourself
    """

    if mask:
        masked_cube = mask_with_best_setting(cube)
    else:
        masked_cube = cube
    init_conds = prepare_initial_conditions(cube, masked_cube)
    spectral_axis = init_conds['spectral_axis']

    if isinstance(noise, (tuple, list)) and len(noise) == 2:
        # this is referring to a pixel in the cube
        noise_spectrum = cube[:, noise[0], noise[1]].to_value()
        noise = np.std(noise_spectrum)
        print(f"USING NOISE = {noise:.3f} K")

    plt.ion()
    fig = plt.figure(figsize=(6, 3.5))
    ax_img = plt.subplot2grid((2, 5), (0, 0), colspan=2, fig=fig, projection=cube.wcs, slices=('x', 'y', 50))
    ax_img2 = plt.subplot2grid((2, 5), (1, 0), colspan=2, fig=fig, projection=cube.wcs, slices=('x', 'y', 50), sharex=ax_img, sharey=ax_img)
    ax_img.tick_params(axis='x', labelbottom=False)
    ax_spectr = plt.subplot2grid((2, 5), (0, 2), colspan=3, rowspan=2, fig=fig)

    im = ax_img.imshow(init_conds['moment0'], origin='lower')
    cbar = fig.colorbar(im, ax=ax_img)
    cax = cbar.ax
    ax_img.set_title("Moment 0 (masked)")
    im = ax_img2.imshow(cube.spectral_slab(24*u.km/u.s, 26*u.km/u.s).moment0().to_value(), origin='lower')
    cbar2 = fig.colorbar(im, ax=ax_img2)
    cax2 = cbar2.ax
    ax_img2.set_title("Moment 0 [24, 26] km/s")

    ax_spectr.plot(spectral_axis, masked_cube.mean(axis=(1, 2)).to(u.K).to_value(), linewidth=0.7)
    ax_spectr.set_xlim((spectral_axis[0], spectral_axis[-1]))
    if invert_xaxis:
        ax_spectr.invert_xaxis()

    fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
    plot_info_dict = {'x1': None, 'x2': None, 'currently_selecting': False}

    def onclick(event):
        try:
            j, i = int(round(event.xdata)), int(round(event.ydata))
            if live_intercept['ij'] is not None:
                print(f"Clicked {i}, {j} but ignoring those to plot the intercepted ", end="")
                i, j = live_intercept['ij']
                live_intercept['ij'] = None
                print(f"{i}, {j}.")
        except Exception as e:
            print(f"something went wrong... {e}")
            return
        if event.button == 1 and (event.inaxes is ax_img or event.inaxes is ax_img2):
            ax_spectr.clear()
            if plot_info_dict['x1'] is not None:
                ax_img.lines.remove(plot_info_dict['x1'])
                ax_img2.lines.remove(plot_info_dict['x2'])

            plot_info_dict['x1'], = ax_img.plot([j], [i], 'x', color='red')
            plot_info_dict['x2'], = ax_img2.plot([j], [i], 'x', color='red')
            plot_info_dict['xij'] = (i, j)
            print(f"Fitting (i, j): ({i}, {j})")

            if template_model is None:
                g_init = initialize_gaussian(init_conds, None, (i, j))
            else:
                g_init = template_model
            """
            astropy modeling does NOT like NaNs. That's weird! They should!
            """
            g_fit, g_fit_array, masked_spectrum_val = fit_gaussian(init_conds, masked_cube, g_init, (i, j), fitter,
                double=double, template=(template_model is not None),
                cube_is_masked=mask, noise=noise)

            if g_fit is not None:
                print(g_fit)
                print(g_fit.cov_matrix)
                # print("Parameter [MIN : init_val/fitted_val : MAX]")
                # print(f"Ampl [{g_fit.amplitude.min:6.2f} {g_init.amplitude.value:6.2f}/{g_fit.amplitude.value:6.2f} {g_fit.amplitude.max:6.2f}]")
                # print(f"Mean [{g_fit.mean.min:6.2f} {g_init.mean.value:6.2f}/{g_fit.mean.value:6.2f} {g_fit.mean.max:6.2f}]")
                # print(f"Stdd [{g_fit.stddev.min:6.2f} {g_init.stddev.value:6.2f}/{g_fit.stddev.value:6.2f} {g_fit.stddev.max:6.2f}]")
            else:
                print("No fit was made")

            try:
                float(noise)
                plot_noise_and_vlims(ax_spectr, noise, [24, 26])
            except:
                pass


            spectrum = cube[:, i, j].to_value()
            ax_spectr.plot(spectral_axis, spectrum, color='k', linewidth=0.7, label='Data', marker='o', alpha=0.2, markersize=2)
            if (n_params is not None) and (noise is not None):
                chisq_kwargs = dict(noise=noise, dof=(g_fit_array.size - n_params))
            else:
                chisq_kwargs = {}
            plot_everything_about_models(ax_spectr, spectral_axis, spectrum, g_fit, **chisq_kwargs)

            # Metric; sum over this for a number that, if large, means the fit isn't good
            metric = -(spectrum - g_fit_array)*np.sign(spectrum)
            metric[metric < 0] = 0
            # ax_spectr.plot(spectral_axis, metric, color='Indigo', marker='+', alpha=0.3, label='metric')

            ax_spectr.set_xlabel("v (km/s)")
            ax_spectr.set_ylabel("T (K)")
            ax_spectr.set_xlim((18, 32)) # (0, 45) for CII
            ax_spectr.set_ylim((ylim_min, ylim_max)) # 23 for HCOP, 35 for CII
            ax_spectr.legend()
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
                mom0_unmasked = cube.spectral_slab(vlo*u.km/u.s, vhi*u.km/u.s).moment(order=0).to(u.K*u.km/u.s).to_value()
                ax_img.clear()
                cax.clear()
                im = ax_img.imshow(mom0_unmasked, origin='lower', vmin=ilo, vmax=ihi)
                ax_img.set_title(f"Integrated cube [{vlo:4.1f}, {vhi:4.1f}] km/s")
                fig.colorbar(im, cax=cax)
                if plot_info_dict['x1'] is not None:
                    i, j = plot_info_dict['xij']
                    plot_info_dict['x1'], = ax_img.plot([j], [i], 'x', color='red')
                plot_info_dict['currently_selecting'] = False
        elif event.button == 3:
            plt.ioff()
            plt.close()

    return fig.canvas.mpl_connect('button_press_event', onclick)



def linewidth_fwhm(cube, masked_already=False, flux_cutoff=6.5, dataset='sofia'):
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
        if dataset == 'apex':
            masked_cube = mask_above_xpower(cube, 2, additional_cutoff=0)
        elif dataset == 'sofia':
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


def fit_image_to_file(cube, double=False, template_model=None, mask=True, noise=None,
    skip_low_emission=False):
    """
    Do the big fit
    :param cube: the cube to fit
    :param double: single or doubel Gaussian
        only used if template_model is None
    :param template_model: an astropy.modeling.models model to use as the
        initial guess for fitting. Overrides "double".
        Presumed to be a single or composite model composed only of
        Gaussian1D components, so 3 parameters per component
    :param mask: fit only stuff above half power
        else, fit the entire spectrum for every pixel
    :param noise: arg to pass thru to fit_gaussian; see that description
        If this is a 2-element tuple, will calculate noise from that pixel location
        in the unmasked cube
        (this is the exact same procedure as fit_live_interactive)
    """
    if mask:
        masked_cube = mask_with_best_setting(cube)
    else:
        masked_cube = cube
    init_conds = prepare_initial_conditions(cube, masked_cube)
    spectral_axis = init_conds['spectral_axis']
    cube_dv = np.abs((spectral_axis[1] - spectral_axis[0]))
    fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
    if template_model is None:
        g_init = models.Gaussian1D(7, 25, 0.46, bounds={'amplitude': (0, 50), 'mean': (22, 28), 'stddev': (1, 3.5)})
        n_params = len(g_init.param_names)
        if double:
            n_params *= 2
    else:
        g_init = template_model
        n_params = len(g_init.param_names)

    if isinstance(noise, (tuple, list)) and len(noise) == 2:
        # this is referring to a pixel in the cube
        noise_spectrum = cube[:, noise[0], noise[1]].to_value()
        noise = np.std(noise_spectrum)

    results = np.zeros((n_params, *masked_cube.shape[1:]))
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

        if skip_low_emission:
            # Check if there's significant emission in this spectrum
            integrated_emission = np.sum(masked_cube.unmasked_data[:, i, j].to_value())
            moment_noise = noise * cube_dv * np.sqrt(masked_cube.shape[0])
            if integrated_emission < 3*moment_noise:
                results[:, i, j] = np.nan
                residuals_array[:, i, j] = np.nan
                models_array[:, i, j] = np.nan
                continue

        if template_model is None:
            g_init = initialize_gaussian(init_conds, g_init, (i, j))
        g_fit, fitted_spectrum, _ = fit_gaussian(init_conds, masked_cube, g_init, (i, j), fitter,
            double=double, template=(template_model is not None),
            cube_is_masked=mask, noise=noise)

        if g_fit is not None:
            if template_model is None:
                if double:
                    # Put the thinner Gaussian first
                    individual_models = sorted(g_fit, key=lambda m: m.stddev)
                    results[:n_params//2, i, j] = individual_models[0].param_sets[:, 0]
                    results[n_params//2:, i, j] = individual_models[1].param_sets[:, 0]
                else:
                    results[:, i, j] = g_fit.param_sets[:, 0]
            else:
                # template model; just put the components in the (correct) order
                for m_idx, m in enumerate(iter_models(g_fit)):
                    nparams_1 = len(m.param_names) # single model number of params
                    results[(nparams_1*m_idx):(nparams_1*(m_idx+1)), i, j] = m.param_sets[:, 0]
        else:
            results[:, i, j] = np.nan
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

    filename_stub = "models/gauss_fit_13co10_3G_v1"
    param_units = ['K', 'km / s', 'km / s'] * ((int(double) + 1) if template_model is None else g_init.n_submodels)
    wcs_flat = cube.moment(order=0).wcs
    to_header = lambda : wcs_flat.to_header()
    phdu = fits.PrimaryHDU(header=to_header())
    phdu.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    phdu.header['CREATOR'] = f"Ramsey, {__file__}"
    phdu.header['COMMENT'] = f"Fit with best masking settings right now."
    # phdu.header['COMMENT'] = "Using a confusing weight scheme.."
    phdu.header['COMMENT'] = "Cutout with length_scale_mult 2"
    hdu_list = [phdu]
    for i in range(n_params):
        hdu = fits.ImageHDU(data=results[i], header=to_header())
        hdu.header['EXTNAME'] = g_init.param_names[i]
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


def investigate_fit(cube, double=False, template_model=None, filename_stub=None,
        ylim_min=-3, ylim_max=25, show='resid'):
    """
    Investiate the fit made in the previous function

    The "cube" argument is only used to plot contours
    """
    kms = u.km/u.s
    pillar_1_highlight = cube.spectral_slab(20*kms, 27*kms).moment0()
    contour_args = (pillar_1_highlight.to_value(),)
    contour_kwargs = dict(linewidths=1, colors='k', alpha=0.9)

    if filename_stub is None:
        if double and template_model is None:
            filename_stub = "models/gauss_fit_hcop_2G_v2"
        elif template_model is not None:
            filename_stub = "models/gauss_fit_hcop_3G_v2"
        else:
            filename_stub = "models/gauss_fit_hcop_1G_v2"
    param_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".param.fits")
    resid_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".resid.fits")
    model_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".model.fits")

    hdul = fits.open(param_fn)
    print(list(hdu.header['EXTNAME'] for hdu in hdul if 'EXTNAME' in hdu.header))

    resid_cube = cube_utils.SpectralCube.read(resid_fn)
    model_cube = cube_utils.SpectralCube.read(model_fn)

    spectral_axis = resid_cube.spectral_axis.to(u.km/u.s).to_value()

    vlo, vhi = 25, 30
    if show == 'resid':
        img_mom0 = resid_cube.spectral_slab(vlo*kms, vhi*kms).moment(order=0).to(u.K*kms).to_value()
    else:
        img_mom0 = cube.spectral_slab(vlo*kms, vhi*kms).moment(order=0).to(u.K*kms).to_value()

    plt.ion()
    fig = plt.figure(figsize=(6, 3.5))
    ax_img = plt.subplot2grid((1, 5), (0, 0), colspan=2, fig=fig, projection=resid_cube.wcs, slices=('x', 'y', 0))
    ax_spectr = plt.subplot2grid((1, 5), (0, 2), colspan=3, fig=fig)


    im = ax_img.imshow(img_mom0, origin='lower', vmin=0)

    ax_img.contour(*contour_args, **contour_kwargs)
    cbar = fig.colorbar(im, ax=ax_img)
    cax = cbar.ax
    ax_img.set_title(f"Integrated {'residuals' if show=='resid' else 'observed intensity'}, [{vlo:4.1f}, {vhi:4.1f}] km/s")

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
            resid_spectr = resid_cube[:, i, j]
            model_spectr = model_cube[:, i, j]
            ax_spectr.plot(spectral_axis, resid_spectr+model_spectr, color='k', label='original', linewidth=0.7, marker='o', alpha=0.2, markersize=3)
            if double and template_model is None:
                g_thin = models.Gaussian1D(*(hdul[x].data[i, j] for x in ('amplitude_0', 'mean_0', 'stddev_0')))
                g_thick = models.Gaussian1D(*(hdul[x].data[i, j] for x in ('amplitude_1', 'mean_1', 'stddev_1')))
                ax_spectr.plot(spectral_axis, g_thin(spectral_axis.to_value()),color='Indigo', label='thin comp', linewidth=0.7, alpha=0.5, linestyle='--')
                ax_spectr.plot(spectral_axis, g_thick(spectral_axis.to_value()),color='MediumOrchid', label='thick comp', linewidth=0.7, alpha=0.5, linestyle='--')
            elif template_model is not None:
                # assume all Gaussian1Ds
                components = []
                for k in range(template_model.n_submodels):
                    components.append(models.Gaussian1D(*(hdul[x].data[i, j] for x in (f'amplitude_{k}', f'mean_{k}', f'stddev_{k}'))))
                g_all = sum(components[1:], components[0])
                plot_everything_about_models(ax_spectr, spectral_axis, None, g_all)
            ax_spectr.plot(spectral_axis, model_spectr, color='Indigo', label='model', linewidth=0.7, alpha=0.6)
            ax_spectr.plot(spectral_axis, resid_spectr, color='orange', label='resid', linewidth=0.7, alpha=0.7)
            ax_spectr.legend()
            ax_spectr.set_ylim((ylim_min, ylim_max)) # 35 for CII
            if spectral_axis.min() > 16:
                ax_spectr.set_xlim((21, 29))
            else:
                ax_spectr.set_xlim((15, 35))
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
                if show == 'resid':
                    img_mom0 = resid_cube.spectral_slab(vlo*u.km/u.s, vhi*u.km/u.s).moment(order=0).to(u.K*u.km/u.s).to_value()
                else:
                    img_mom0 = cube.spectral_slab(vlo*kms, vhi*kms).moment(order=0).to(u.K*kms).to_value()
                ax_img.clear()
                cax.clear()
                im = ax_img.imshow(img_mom0, origin='lower', vmin=ilo, vmax=ihi)
                ax_img.contour(*contour_args, **contour_kwargs)
                fig.colorbar(im, cax=cax)
                ax_img.set_title(f"Integrated {'residuals' if show=='resid' else 'observed intensity'}, [{vlo:4.1f}, {vhi:4.1f}] km/s")
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

    blue_interval = (20*kms, 25*kms)
    blue_str = f"[{blue_interval[0]:.0f}, {blue_interval[1]:.0f}]"
    red_interval = (25.5*kms, 29*kms)
    red_str = f"[{red_interval[0]:.0f}, {red_interval[1]:.0f}]"

    blue_slab = cube.spectral_slab(*blue_interval)
    red_slab = cube.spectral_slab(*red_interval)
    blue_integrated = blue_slab.moment(order=0).to_value()
    red_integrated = red_slab.moment(order=0).to_value()

    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(121, projection=cube.wcs, slices=('x', 'y', 50))
    ax1.imshow(blue_integrated, origin='lower', vmin=0)
    ax1.set_title(f"Blue wing, integrated {blue_str} km/s")
    ax1.contour(*contour_args, **contour_kwargs)

    ax2 = plt.subplot(122, projection=cube.wcs, slices=('x', 'y', 50), sharex=ax1, sharey=ax1)
    ax2.imshow(red_integrated, origin='lower', vmin=0)
    ax2.set_title(f"Red wing, integrated {red_str} km/s")
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


class ImgContourPair:

    default_alpha = 0.9
    smooth_spatial_ = None

    def __init__(self, moment_map, name, levels=None, color='k'):
        self.moment_map = moment_map
        self.name = name
        self.wcs = moment_map.wcs
        self.levels = levels
        self.color = color

    def img(self):
        return self.moment_map.to_value()

    def carg(self):
        return (ImgContourPair.smooth_spatial_(self.moment_map.to_value(), self.wcs),)

    def ckwarg(self):
        kwargs = {}
        if self.levels is not None:
            kwargs['levels'] = self.levels
        if self.color is not None:
            kwargs['colors'] = self.color
        kwargs['alpha'] = ImgContourPair.default_alpha
        kwargs['linewidths'] = 1
        return kwargs



def check_if_wings_trace_peak_emission(cube):
    """
    I need the original cube as arg, and then I can load the fitted stuff
    Revamped based on a pretty sweet DS9 viz I did (Oct 22, 2020)
    """
    kms = u.km/u.s
    vel_bounds = 20*kms, 30*kms
    smooth_beam = cube_utils.Beam(11*u.arcsec)
    ImgContourPair.smooth_spatial_ = lambda img, w: smooth_spatial(img, w, cube, smooth_beam)


    pillar_1_highlight_vlims = (24*kms, 27*kms)
    pillar_1_highlight = cube.spectral_slab(*pillar_1_highlight_vlims).moment0()
    pillar_1_highlight = ImgContourPair(pillar_1_highlight, make_vel_stub(pillar_1_highlight_vlims), levels=[x*1.5 for x in [20, 30, 40, 50, 60]], color='k')

    background_35_highlight_vlims = (32*kms, 36*kms)
    background_35_highlight = cube.spectral_slab(*background_35_highlight_vlims).moment0()
    background_35_highlight = ImgContourPair(background_35_highlight, make_vel_stub(background_35_highlight_vlims), levels=[11, 20, 35], color='r')

    background_30_highlight_vlims = (28*kms, 29*kms)
    # background_30_highlight_vlims = (29*kms, 30*kms)
    background_30_highlight = cube.spectral_slab(*background_30_highlight_vlims).moment0()
    background_30_highlight = ImgContourPair(background_30_highlight, make_vel_stub(background_30_highlight_vlims), levels=[8,10,12], color='orange')

    # original_mom0 = cube.spectral_slab(*vel_bounds).moment(order=0).to(u.K*u.km/u.s).to_value()

    """ CAN ONLY DO THESE IF I RAN MODEL FITS FOR THEM
    filename_stub = "models/gauss_fit_above_1G_v4_smooth"
    param_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".param.fits")
    resid_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".resid.fits")
    model_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".model.fits")
    resid_cube = cube_utils.SpectralCube.read(resid_fn)
    red_wing_highlight = resid_cube.spectral_slab(27*kms, 30*kms).moment0().to(u.K*u.km/u.s)
    red_wing_highlight = ImgContourPair(red_wing_highlight, "Red Residual Wing", levels=[6, 9], color='w')
    # resid_mom0 = resid_cube.spectral_slab(*vel_bounds).moment(order=0).to(u.K*u.km/u.s).to_value()

    with fits.open(param_fn) as hdul:
        std_fitted = hdul['stddev_FIT'].data

    # metric = (-resid_cube.spectral_slab(*vel_bounds).unmasked_data[:]*np.sign(cube.spectral_slab(*vel_bounds).unmasked_data[:])).to_value()
    # metric[metric < 0] = 0
    # metric = metric.sum(axis=0)
    # ############### add this in
    """

    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(121, projection=cube.wcs, slices=('x', 'y', 0))
    ax2 = plt.subplot(122, projection=cube.wcs, slices=('x', 'y', 0))

    def plot_axes(img1, v1, img2, v2, *additional_contours):
        im = ax1.imshow(img1.img(), origin='lower', vmin=v1[0], vmax=v1[1], cmap='viridis')
        # im = ax1.imshow(std_fitted, origin='lower', vmin=1.2, vmax=3, cmap='seismic')
        # ax1.set_title("1")
        # ax1.contour(*contour_args, **contour_kwargs)
        ax1.contour(*img1.carg(), **img1.ckwarg())
        handles1 = [mpatches.Patch(color=img1.color, label=img1.name)]
        fig.colorbar(im, ax=ax1)
        ax1.legend(handles=handles1, loc='lower right')
        ax1.set_title(img1.name)

        im = ax2.imshow(img2.img(), origin='lower', vmin=v2[0], vmax=v2[1], cmap='viridis')
        handles2 = []
        for c_img in additional_contours:
            ax2.contour(*c_img.carg(), **c_img.ckwarg())
            handles2.append(mpatches.Patch(color=c_img.color, label=c_img.name))
        ax2.contour(*img2.carg(), **img2.ckwarg())
        handles2.append(mpatches.Patch(color=img2.color, label=img2.name))
        fig.colorbar(im, ax=ax2)
        ax2.legend(handles=handles2, loc='lower right')
        ax2.set_title(img2.name)

    # plot_axes(pillar_1_highlight, (5, 80), red_wing_highlight, (0, 12), pillar_1_highlight)
    plot_axes(pillar_1_highlight, (5, 80*1.5), background_30_highlight, (4, 17), pillar_1_highlight) #, background_35_highlight
    plt.tight_layout()
    plt.savefig("/home/ramsey/Pictures/2021-06-03-work/redshifted_wing_co_2.png")
    # plt.show()


def investigate_template_model_fit(n_submodels=3, line='hcop', version='3'):
    """
    November 17, 2021
    Check out distribution of line centers and stuff
    """
    if line[:4] == 'hcop':
        directory = "carma"
        if version is None:
            if line == 'hcop':
                version = 2
            elif line == 'hcop_regrid':
                version = 3
        # vel_coeff just stretches out the velocity axis, like an aspect ratio
        vel_coeff = 30
    elif line[:3] == 'cii':
        directory = 'sofia'
        if version is None:
            version = 1
        vel_coeff = 4
    elif line == '13co10':
        directory = 'bima'
        if version is None:
            version = 1
        vel_coeff = 30
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
    if False:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.hist(means, bins=256, range=(20, 30))
        plt.show()
    elif True:
        from mayavi import mlab
        mlab.figure(bgcolor=(0.2, 0.2, 0.2), fgcolor=(0.93, 0.93, 0.93), size=(800, 700))
        mlab.axes(ranges=[0, shape[1], 0, shape[0], 20, 30],
            xlabel='j (ra)', ylabel='i (dec)', zlabel='velocity (km/s)', nb_labels=10,
            line_width=19)
        kwargs = dict(mode='cube', colormap='jet',
            scale_mode='none', scale_factor=0.7, opacity=0.2)
        mlab.points3d(j_array, i_array, -1*means*vel_coeff, amplitudes, **kwargs)
        mlab.show()



"""
FIT BOTH CUBES
"""
def fit_multicube_live_interactive(cubecii, cube12co, cube13co):
    # INTERACTIVE

    cubes = [cubecii, cube12co, cube13co]
    names = ["sofia_CII", "apex_12CO", "apex_13CO"]
    names_to_print = [None]*3
    colors = ['DarkGreen', 'DarkMagenta', 'DarkOrange', 'RoyalBlue']
    colors_model = ['LimeGreen', 'Magenta', 'Orange', 'Blue']
    masked_cubes = []
    init_cond_list = []
    spectral_axes = []

    for name, cube in zip(names, cubes):
        masked_cube = mask_with_best_setting(cube, dataset=name.split("_")[0])
        masked_cubes.append(masked_cube)
        init_conds = prepare_initial_conditions(cube, masked_cube)
        init_cond_list.append(init_conds)
        spectral_axes.append(init_conds['spectral_axis'])

    plt.ion()
    fig = plt.figure(figsize=(6, 3.5))

    ax_cii = plt.subplot2grid((3, 5), (0, 0), colspan=2, fig=fig, projection=cubes[0].wcs, slices=('x', 'y', 0))
    ax_12co = plt.subplot2grid((3, 5), (1, 0), colspan=2, fig=fig, projection=cubes[1].wcs, slices=('x', 'y', 0))
    ax_13co = plt.subplot2grid((3, 5), (2, 0), colspan=2, fig=fig, projection=cubes[2].wcs, slices=('x', 'y', 0))
    img_axes = [ax_cii, ax_12co, ax_13co]

    ax_cii.tick_params(axis='x', labelbottom=False)
    ax_12co.tick_params(axis='x', labelbottom=False)
    ax_spectr = plt.subplot2grid((3, 5), (0, 2), colspan=3, rowspan=3, fig=fig)


    for idx, ax in enumerate(img_axes):
        ax.imshow(cubes[idx].spectral_slab(20*u.km/u.s, 30*u.km/u.s).moment0().to_value(), origin='lower')
        name = names[idx].upper().replace("_", " ")
        names_to_print[idx] = name
        ax.set_title(f"{name} Moment0 [20,30] km/s")
        # Also plot the spectrum from the cube
        ax_spectr.plot(spectral_axes[idx], masked_cubes[idx].mean(axis=(1, 2)).to(u.K).to_value(), linewidth=0.7, color=colors[idx], label=name)
    ax_spectr.set_xlim((spectral_axes[0][0], spectral_axes[0][-1]))
    ax_spectr.legend()

    fitter = fitting.SLSQPLSQFitter()
    plot_info_dict = [None]*3

    def onclick(event):
        try:
            j, i = int(round(event.xdata)), int(round(event.ydata))
        except Exception as e:
            print(f"something went wrong... {e}")
            return
        if event.button == 1 and (event.inaxes in img_axes):
            ax_spectr.clear()
            if plot_info_dict[0] is not None:
                for idx, ax in enumerate(img_axes):
                    try:
                        ax.lines.remove(plot_info_dict[idx])
                    except:
                        pass

            selected_info = init_cond_list[img_axes.index(event.inaxes)]
            selected_coord = selected_info['wcs_flat'].array_index_to_world(i, j)
            array_indices = [None]*3

            for idx, ax in enumerate(img_axes):
                if event.inaxes is ax:
                    plot_info_dict[idx], = ax.plot([j], [i], 'x', color='red')
                    array_indices[idx] = (i, j)
                else:
                    i1, j1 = init_cond_list[idx]['wcs_flat'].world_to_array_index(selected_coord)
                    plot_info_dict[idx], = ax.plot(j1, i1, 'x', color='red')
                    array_indices[idx] = (int(i1), int(j1))

                masked_cube = masked_cubes[idx]
                init_conds = init_cond_list[idx]

                g_init = initialize_gaussian(init_conds, None, array_indices[idx])
                """
                astropy modeling does NOT like NaNs. That's weird! They should!
                """
                g_fit, g_fit_array, masked_spectrum_val = fit_gaussian(init_conds, masked_cube, g_init, array_indices[idx], fitter)
                if g_fit is not None:
                    print(f"({names_to_print[idx]}) Parameter [MIN : init_val/fitted_val : MAX]")
                    print(f"Ampl [{g_fit.amplitude.min:6.2f} {g_init.amplitude.value:6.2f}/{g_fit.amplitude.value:6.2f} {g_fit.amplitude.max:6.2f}]")
                    print(f"Mean [{g_fit.mean.min:6.2f} {g_init.mean.value:6.2f}/{g_fit.mean.value:6.2f} {g_fit.mean.max:6.2f}]")
                    print(f"Stdd [{g_fit.stddev.min:6.2f} {g_init.stddev.value:6.2f}/{g_fit.stddev.value:6.2f} {g_fit.stddev.max:6.2f}]")
                else:
                    print(f"({names_to_print[idx]}) No fit was made")

                spectrum = cubes[idx][:, array_indices[idx][0], array_indices[idx][1]].to_value()
                spectral_axis = spectral_axes[idx]
                ax_spectr.plot(spectral_axis, spectrum, color=colors[idx], linewidth=0.7, label=f'Data: {names_to_print[idx]}', marker='o', alpha=0.2, markersize=3)
                ax_spectr.plot(spectral_axis, masked_spectrum_val, color=colors[idx], marker='x', linewidth=2, linestyle='dotted', alpha=0.9, label='Data fitted to')

                ax_spectr.plot(spectral_axis, g_fit_array, color=colors_model[idx], linestyle='dotted', linewidth=1, label="Fitted", alpha=0.9)
                ax_spectr.plot(spectral_axis, spectrum - g_fit_array, color=colors_model[idx], linewidth=0.7, label='Residuals', alpha=0.6)

                # Metric; sum over this for a number that, if large, means the fit isn't good
                metric = -(spectrum - g_fit_array)*np.sign(spectrum)
                metric[metric < 0] = 0
                # ax_spectr.plot(spectral_axis, metric, color='Indigo', marker='+', alpha=0.3, label='metric')

            ax_spectr.set_xlabel("v (km/s)")
            ax_spectr.set_ylabel("T (K)")
            ax_spectr.set_xlim((15, 35))
            ax_spectr.legend()
        elif event.button == 3:
            plt.ioff()
            plt.close()

    return fig.canvas.mpl_connect('button_press_event', onclick)


def compare_cii_co_contours(cubecii, cube12co, cube13co):
    cubes = [cubecii, cube12co, cube13co]
    names = ["sofia_CII", "apex_12CO", "apex_13CO"]
    names_to_print = [None]*3
    colors = ['DarkGreen', 'DarkOrchid', 'DarkOrange']

    kms = u.km/u.s
    vel_bounds = 23*kms, 27*kms
    vel_str = f"[{vel_bounds[0]:.0f}, {vel_bounds[1]:.0f}]"

    moment0s = [None]*3
    contour_args_list = [None]*3
    contour_kwargs_list = [None]*3
    cii_wcs = [None]
    for idx, cube in enumerate(cubes):
        moment0 = cube.spectral_slab(*vel_bounds).moment0().to(u.K*kms)
        if idx == 0:
            moment0s[idx] = moment0.to_value()
            cii_wcs[0] = moment0.wcs
        else:
            moment0_reproj = reproject_interp((moment0.to_value(), moment0.wcs), cii_wcs[0], moment0s[0].shape, return_footprint=False)
            moment0s[idx] = moment0_reproj
        contour_args_list[idx] = (moment0s[idx],)
        lw = 1.
        alpha = 1.
        contour_kwargs_list[idx] = dict(linewidths=lw, colors=colors[idx], alpha=alpha)
        # if names[idx] == 'sofia_CII':
        #     contour_kwargs_list[idx]['levels'] = [30, 60, 90, 120, 150]
        # elif names[idx] == 'apex_12CO':
        #     contour_kwargs_list[idx]['levels'] = [15, 30, 45, 60, 75, 90]
        # elif names[idx] == 'apex_13CO':
        #     contour_kwargs_list[idx]['levels'] = [10, 20, 30, 40]

    selected_index = 0
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot(111, projection=cubes[selected_index].wcs, slices=('x', 'y', 0))
    im = ax.imshow(moment0s[selected_index], origin='lower', vmin=0, vmax=170)
    ax.set_title(f"{names[selected_index]} Moment 0 {vel_str} km/s")
    for idx, carg, ckwarg in zip(range(3), contour_args_list, contour_kwargs_list):
        if idx == 0:
            continue
        ax.contour(*carg, **ckwarg)
    fig.colorbar(im, ax=ax)

    plt.savefig("/home/ramsey/Pictures/10-21-20-mtg/cii_co_contours.png")


def stack_pillar_spectra(cube):
    """
    I need the original cube as arg, and then I can load the fitted stuff
    I want to specifically look at the double-peaked residual spectrum
    """
    kms = u.km/u.s
    vel_bounds = 20*kms, 30*kms

    pillar_1_highlight = cube.spectral_slab(25*kms, 27*kms).moment0()
    contour_args = (pillar_1_highlight.to_value(),)

    # original_mom0 = cube.spectral_slab(*vel_bounds).moment(order=0).to(u.K*u.km/u.s).to_value()

    filename_stub = "models/gauss_fit_above_1G_v4_smooth"
    param_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".param.fits")
    resid_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".resid.fits")
    model_fn = cube_utils.os.path.join(cube_info['dir'], filename_stub+".model.fits")
    resid_cube = cube_utils.SpectralCube.read(resid_fn)
    # red_wing_highlight = resid_cube.spectral_slab(27*kms, 30*kms).moment0().to(u.K*u.km/u.s)
    # cargs2 = (red_wing_highlight.to_value(),)
    # ckwargs2 = dict(levels=[6, 9], linewidths=1, colors='w', alpha=1)
    # # resid_mom0 = resid_cube.spectral_slab(*vel_bounds).moment(order=0).to(u.K*u.km/u.s).to_value()
    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0), projection=resid_cube.wcs, slices=('x', 'y', 0))
    ax2 = plt.subplot2grid((2, 2), (1, 0), projection=resid_cube.wcs, slices=('x', 'y', 0))
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    if 'sofia' in cube_info['dir']:
        pillar_mask = contour_args[0] > 50
    elif 'apex' in cube_info['dir']:
        pillar_mask = contour_args[0] > 25
    masked_resid_1 = resid_cube.with_mask(pillar_mask)
    stacked_resid_1 = masked_resid_1.mean(axis=(1, 2))
    ax1.imshow(pillar_mask, origin='lower')
    ax1.set_title("1")
    if 'sofia' in cube_info['dir']:
        pillar_mask_2 = (contour_args[0] > 30) ^ pillar_mask
    elif 'apex' in cube_info['dir']:
        pillar_mask_2 = (contour_args[0] > 15) ^ pillar_mask
    masked_resid_2 = resid_cube.with_mask(pillar_mask_2)
    stacked_resid_2 = masked_resid_2.mean(axis=(1, 2))
    ax2.imshow(pillar_mask_2, origin='lower')
    ax2.set_title("2")

    x = cube.spectral_axis.to_value()
    splines_list = []
    centered_spectra = []
    with fits.open(param_fn) as hdul:
        mean_fitted = hdul['mean_FIT'].data
    centered_x = x - np.nanmean(mean_fitted[pillar_mask])
    for i, j in zip(*np.where(pillar_mask)):
        spline = UnivariateSpline(x - mean_fitted[i, j], resid_cube[:, i, j].to_value(), s=0)
        # splines_list.append(spline)
        centered_spectrum = spline(centered_x)
        # centered_spectra.append(centered_spectrum)
        if (np.any(centered_spectrum < -1.5) and 'sofia' in cube_info['dir']) or (np.any(centered_spectrum < -5) and 'apex' in cube_info['dir']):
            continue
        ax3.plot(centered_x, centered_spectrum, color='k', alpha=0.2, lw=0.4)
    ax3.plot(centered_x, stacked_resid_1.to_value(), label='1', color='g')
    ax3.plot(x - np.nanmean(mean_fitted[pillar_mask_2]), stacked_resid_2.to_value(), label='2', color='b')
    ax3.legend()
    plt.show()


def get_all_subcubes(**kwargs):
    """
    As of Nov 11, jerry rigged to take one extra filename (like bima) and return a longer tuple
    As of Jan 25 (2021), wondering why the first call to cutout_subcube was ABOVE
        the if/else block that popped "data_filename" off the kwarg dict?
    As of Jan 26, made data_filename just intercept the FIRST load call. extra_filename
        gets the extra file. Moved first call back to top, maybe that was why...
    """
    if 'extra_filename' in kwargs:
        extra_fn = kwargs.pop('extra_filename')
    else:
        extra_fn = None
    subcube_cii = cutout_subcube(**kwargs)
    if 'data_filename' in kwargs:
        kwargs.pop('data_filename')
    if kwargs['length_scale_mult'] <= 4:
        stub = "_cutout"
    else:
        stub = ""
    subcube_12co = cutout_subcube(**kwargs, data_filename=f"apex/M16_12CO3-2_truncated{stub}.fits")
    subcube_13co = cutout_subcube(**kwargs, data_filename=f"apex/M16_13CO3-2_truncated{stub}.fits")
    subcubes = (subcube_cii, subcube_12co, subcube_13co)
    if extra_fn is not None:
        subcubes = subcubes + (cutout_subcube(**kwargs, data_filename=extra_fn),)
    return subcubes


def smooth(cube, width=7):
    """
    Smooth a cube with a 7-spaxel long Hamming window
    :param cube: SpectralCube instance
    :returns: SpectralCube instance, same shape and everything, just smoothed
    """
    # Hamming smooth the spectrum
    smooth_kernel = convolution.kernels.CustomKernel(signal.hamming(width))
    return cube.spectral_smooth(smooth_kernel)


def smooth_spatial(img_to_convolve, img_wcs, cube, target_beam):
    conv_beam = target_beam.deconvolve(cube.beam)
    # print(f"Convolving from {str(cube.beam)} to {str(target_beam)} using {str(conv_beam)}")
    return convolution.convolve(img_to_convolve, conv_beam.as_kernel(misc_utils.get_pixel_scale(img_wcs)), boundary='extend')


def get_cii_background(cii_cube=None, return_artist=False, **kwargs):
    """
    Load up the standard CII background spectrum
    Unfortunately totally messes up your noise estimates (I don't want to make that calculation)
    :param cii_cube: SpectralCube, a CII cube from which to take the background spectrum.
        If left None, loads up the standard CII cube and takes the spectrum.
    :param return_artist: return a matplotlib artist for the background patch
    :param kwargs: sent to PixelRegion.as_artist()
    :returns: Quantity array spectrum (, list(Artist) for background patch)
    """
    if cii_cube is None:
        cii_cube = cutout_subcube(length_scale_mult=None)
    bg_reg = regions.read_ds9(catalog.utils.search_for_file("catalogs/pillar_background_sample_multiple_4.reg"))
    cii_bg_spectrum = cii_cube.subcube_from_regions(bg_reg).mean(axis=(1, 2))
    kwargs.setdefault('fill', False)
    kwargs.setdefault('color', 'k')
    if return_artist:
        artists = []
        for reg in bg_reg:
            artists.append(reg.to_pixel(cii_cube[0, :, :].wcs).as_artist(**kwargs))
        return cii_bg_spectrum, artists
    else:
        return cii_bg_spectrum


def test_cii_background():
    """
    #######################
    USEFUL DEBUG FUNCTION!!!! CHECK IF BACKGROUND RUNS THE SAME ON JUPITER AND
    LAPTOP!!
    #######################
    Nov 19, 2021
    Unnerving discrepancy between running the background subtraction on my
    laptop vs desktop, so I need to investigate that...
    The solution was an astropy/regions update. Weird!
    Jan 14, 2022: I verified again that this produces the same result on both
        jupiter and my laptop. I also checked that the astropy versions were
        both as up-to-date as possible (laptop runs py3.9 so astropy is 5.0,
        jupiter runs py3.7 so astropy is 4.3.1) and that is appparently good
        enough
        I want to create a plot similar to this to show the background spectra
        but I will do it in m16_pictures since this is a good function to leave
        alone for future debugging.
    """
    cii_cube = cutout_subcube(length_scale_mult=6)
    cii_bg_spectrum, artists = get_cii_background(cii_cube=cii_cube, return_artist=True)
    ax1 = plt.subplot(121, projection=cii_cube[0, :, :].wcs)
    plt.imshow(cii_cube.moment0().to_value(), origin='lower')
    for a in artists:
        ax1.add_artist(a)
    ax2 = plt.subplot(122)
    plt.plot(cii_cube.spectral_axis, cii_bg_spectrum)
    peak_loc = cii_cube.spectral_axis[np.argmax(cii_bg_spectrum)]
    plt.axvline(peak_loc.to_value(), color='r')
    plt.title(f"Peak: {peak_loc.to(u.km/u.s).to_value():.2f} km/s")
    plt.show()


def bin_edges_helper(center, width):
    """
    Dec 6, 2021
    Return left and right edges
    """
    return center - width/2, center + width/2


def rebin_channels(centers_old, centers_new, values_old):
    """
    Dec 6, 2021
    Rebin a spectrum to another set of bins. The new bins should be wider.
    """
    n_bins_new = centers_new.size
    width_old = centers_old[1] - centers_old[0]
    width_new = centers_new[1] - centers_new[0]
    left_edges_old, right_edges_old = bin_edges_helper(centers_old, width_old)
    left_edges_new, right_edges_new = bin_edges_helper(centers_new, width_new)
    values_new = np.full(n_bins_new, np.nan)
    for bin_idx_new, bin_center_new in enumerate(centers_new):
        if (left_edges_new[bin_idx_new] < left_edges_old[0]):
            continue
        if (right_edges_new[bin_idx_new] > right_edges_old[-1]):
            break
        fully_included_bins = (left_edges_new[bin_idx_new] < left_edges_old) & (right_edges_new[bin_idx_new] > right_edges_old)
        sum_value = np.sum(values_old[fully_included_bins])
        averaged_bins = np.sum(fully_included_bins)
        # partial bins
        leftmost_bin_idx, rightmost_bin_idx = np.where(fully_included_bins)[0][[0, -1]]
        # left partial bin
        left_partial_bin_idx = leftmost_bin_idx - 1
        fraction_contained = 1 - (left_edges_new[bin_idx_new] - left_edges_old[left_partial_bin_idx])/width_old
        sum_value += fraction_contained * values_old[left_partial_bin_idx]
        averaged_bins += fraction_contained
        # right partial bin
        right_partial_bin_idx = rightmost_bin_idx + 1
        fraction_contained = (right_edges_new[bin_idx_new] - left_edges_old[right_partial_bin_idx])/width_old
        sum_value += fraction_contained * values_old[right_partial_bin_idx]
        averaged_bins += fraction_contained
        # finalize
        values_new[bin_idx_new] = sum_value / averaged_bins
    return values_new


def test_rebin():
    """
    Dec 6, 2021
    debug the above code
    """
    m = models.Gaussian1D()
    x = np.arange(-3, 3, 0.1)
    y = m(x)
    plt.plot(x, y, 'o', fillstyle='none', label='high resolution')
    x2 = np.arange(-4, 4, 0.5)
    y2 = m(x2)
    plt.plot(x2, y2, 'x', label='low resolution truth')
    y_rebin = rebin_channels(x, x2, y)
    plt.plot(x2, y_rebin, '+', label='low resolution rebin')
    plt.legend()
    plt.show()


def regrid_hcop():
    """
    Dec 6, 2021
    Use the .reproject method of SpectralCube to grid HCO+ to the CII grid
    Use the CII grid at length scale 4

    Finished using this on Dec 7, 2021
    There are some documentation inconsistencies and weird bugs in spectral_cube
    that made this process rather arduous.
    I think some of it might be that I initialized the VRAD SpectralCube using
    data=hcop_cube.unmasked_data, which provides a "view" not an array. Maybe
    unmasked_data[:] would be better? But this whole thing takes so long to
    run that I don't want to test that theory.
    January 20, 2022: bug fix about shapes not matching. See below
    """
    if False:
        cii_cube = cutout_subcube(length_scale_mult=4.)
        hcop_cube_obj = cube_utils.CubeData("carma/M16.ALL.hcop.sdi.cm.subpv.SOFIAbeam.fits")
        hcop_cube = hcop_cube_obj.data

        # The reproject function apparently messes with the spectral axis too, so...
        # get the delta_velocity of HCO+ ; this is how they do it in spectral_cube
        boxsmooth_filename = hcop_cube_obj.full_path.replace('.fits', '.boxsmooth.fits')
        if not os.path.exists(boxsmooth_filename):
            hcop_dv = np.mean(np.diff(hcop_cube.spectral_axis))
            cii_dv = np.mean(np.diff(cii_cube.spectral_axis))
            mean_filter_width = (cii_dv/hcop_dv).decompose().to_value()
            print(mean_filter_width)
            mean_filter_width = np.round(mean_filter_width, 4)
            print(mean_filter_width)
            from astropy.convolution import Box1DKernel
            mean_filter = Box1DKernel(mean_filter_width)
            print(mean_filter.array)
            hcop_cube = hcop_cube.spectral_smooth(mean_filter)
            hcop_cube = hcop_cube.spectral_interpolate(cii_cube.spectral_axis.to(hcop_cube.spectral_axis.unit))
            hcop_cube.write(boxsmooth_filename, format='fits')
        else:
            """
            I saved a version of this on Jan 20 2022 since the above code takes
            like a full 5 minutes to run, and the rest of the reproject code
            take just a few seconds.
            """
            hcop_cube = cube_utils.SpectralCube.read(boxsmooth_filename)

        # now finish it
        hdr = hcop_cube.wcs.to_header()
        hdr['CTYPE3'] = 'VRAD'
        w = WCS(hdr)
        hcop_cube = cube_utils.SpectralCube(data=hcop_cube.unmasked_data[:], wcs=w, meta=hcop_cube.meta)
        hcop_cube = hcop_cube.reproject(cii_cube.header)
        """
        Bug fix January 20, 2022
        # hcop_cube = hcop_cube.minimal_subcube()
        # print(hcop_cube.shape[1:])
        I had minimal_subcube in there (idr why) but the interesting thing is,
        HCO+ actually doesn't cover the top edge of this reprojected frame.
        So minimal_subcube shaves off the top row of data, making the shapes of
        HCO+ regrid and CII(length_scale_mult=4) not match.
        Luckily, it's the top row, so the coordinates still match and everything
        works unless you give it i=48 as a coordinate
        I will just leave the NaN row in there as a reminder of this
        """
        hcop_cube = hcop_cube.spectral_slab(17*u.km/u.s, 30*u.km/u.s)
        savename = hcop_cube_obj.full_path.replace('.fits', '.fullregrid_v2.fits')
        hcop_cube.write(savename, format='fits', overwrite=True)
        # Forgot to trim off the extra channels on either end
    else:
        raise RuntimeError("I already ran this on Dec 7, 2021 and Jan 20, 2022!")



if __name__ == "__main__":

    # subcube = cutout_subcube(length_scale_mult=4, data_filename="apex/M16_12CO3-2_truncated_cutout.fits")

    # test_cii_background()

    if True:
        #### HCOP
        # regrid = False
        # if regrid:
        #     subcube = cutout_subcube(length_scale_mult=None, data_filename="carma/M16.ALL.hcop.sdi.cm.subpv.SOFIAbeam.regrid.fits")
        #     noise = 0.12
        #     stddev = 0.55
        # else:
        #     subcube = cutout_subcube(length_scale_mult=1.5, data_filename=f"carma/M16.ALL.hcop.sdi.cm.subpv.fits",
        #         reg_filename="catalogs/p1_IDgradients_thru_head.reg", reg_index=2)
        #     noise = 0.546
        #     stddev = 0.46
        #### 12CO10
        # subcube = cutout_subcube(length_scale_mult=3, data_filename="bima/M16_12CO1-0_7x4.Kkms.fits", reg_filename="catalogs/p1_IDgradients_thru_head.reg", reg_index=2)
        # noise = 6.2 # units: K. Checked this on 2022-01-11, 2022-04-21
        # stddev = 0.46
        #### 13CO10
        subcube = cutout_subcube(length_scale_mult=2, data_filename="bima/M16.BIMA.13co1-0.fits", reg_filename="catalogs/p1_IDgradients_thru_head.reg", reg_index=2)
        noise = 2.6 # units: K. Checked this on 2022-04-21
        stddev = 0.46
        #### CII
        # subcube = cutout_subcube(length_scale_mult=1.5, reg_filename="catalogs/p1_IDgradients_thru_head.reg", reg_index=2)
        # cii_bg_spectrum = get_cii_background()
        # subcube = subcube - cii_bg_spectrum[:, np.newaxis, np.newaxis]
        # test_cii_background()

    ###### length_scale_mult = 0.0125 is good for testing HCOP; gives 4 pixels
    ###### length_scale_mult = 1 is good for running pillar head fits

    # subcube = cutout_subcube(length_scale_mult=4)

    # subcube = smooth(subcube)

    # subcubes = [smooth(c) for c in get_all_subcubes()]
    # try_mask_above_half_power(subcube, xpower=2)

    # check_if_wings_trace_peak_emission(subcube)

    if True:
        #############
        ##### TEMPLATE MODEL SETUP
        #############
        ### template model from those presets in the function
        # g_init = select_pixels_and_models('hcop', 'peak', var_mean=1, var_std=0)[2]
        # g_init = g_init + g_init[2].copy()

        ### template model by hand
        g0 = models.Gaussian1D(amplitude=10, mean=24.5, stddev=stddev,
            bounds={'amplitude': (0, None), 'mean': (20, 30), 'stddev': (0.3, 1.3)})
        g1 = g0.copy()
        g1.mean = 25.5

        g2 = g0.copy()
        g2.mean = 23

        g_init = g0 + g1 + g2
        # fix_std(g_init)
        tie_std_models(g_init)
        print(g_init)

        #### for CII:
        # for g in g_init:
        #     g.amplitude.bounds = (0.05, 100)
        # g_init.stddev_0 = 1.1
        #############

    # fit_image_to_file(subcube, mask=False, template_model=g_init, noise=noise, skip_low_emission=True)


    # live_intercept_dict = {'ij': None}
    # fit_live_interactive(subcube, mask=False, template_model=g_init, noise=noise,
    #     ylim_min=-10, ylim_max=50, n_params=7, live_intercept=live_intercept_dict) # noise from: (125, 32) at length_scale_mult=1

        ## HCOP noise: 0.546, CII noise: ~1, HCOP at CII grid noise: 0.12


    # investigate_fit(subcube, double=False, template_model=g_init,
    #     filename_stub="models/gauss_fit_13co10_3G_v1",
    #     ylim_max=50, show='mom0')

    investigate_template_model_fit(3, line='13co10', version=1)

    # regrid_hcop()

    # stack_pillar_spectra(subcube)
    # fit_multicube_live_interactive(*subcubes)
    # compare_cii_co_contours(*subcubes)
    # calculate_noise(subcube)
