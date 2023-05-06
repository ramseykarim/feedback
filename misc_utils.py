from numbers import Number

import numpy as np

from astropy import units as u
from astropy.nddata.utils import Cutout2D
import regions

"""
General use utilitues for working with images or spectra
Mostly math/physics based and not instrument specific
Created: September 29, 2019
"""
__author__ = "Ramsey Karim"


def convolve_helper(image, kernel):
    ft = np.fft.fft2(image) * np.fft.fft2(kernel)
    result = np.fft.ifft2(ft)
    return np.real(np.fft.fftshift(result, axes=(-2, -1)))


def convolve_properly(image, kernel):
    """
    Convolve image with kernel
    Preserve NaNs
    Also mitigate edge effects / normalization from NaN correction
    Convolves using convolve helper (check that implementation for details)
    :param image: 2d array image
    :param kernel: 2d array kernel, must be same shape as image
    :return: 2d array convolved result matching shape of image
    """
    image = image.copy()
    nan_mask = np.isnan(image)
    image[nan_mask] = 0.
    result = convolve_helper(image, kernel)
    # now account for edge effects / normalization
    image[~nan_mask] = 1.
    norm = convolve_helper(image, kernel)
    image[:] = 1.
    norm /= convolve_helper(image, kernel)
    result /= norm
    result[nan_mask] = np.nan
    return result

def gaussian(x, mu, sigma, amplitude):
    """
    Exactly what it looks like. Good for curve fitting or convolution.
    :param x: independent variable x array
    :param mu: mean of gaussian
    :param sigma: standard deviation of gaussian
    :param amplitude: amplitude coefficient of gaussian
    :return: gaussian curve with array shape of x argument
    """
    coefficient = amplitude / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mu) ** 2 / (2 * sigma * sigma))
    return coefficient * np.exp(exponent)

def polynomial(x, fit):
    """
    Polynomial given x array/scalar and coefficients
    :param x: x array or scalar
    :param fit: coefficient sequence. Assumes "np.polyfit" coefficient ordering, so 0th coeff is for highest order
    :return: x-shaped array (or scalar float) of polynomial y values
    """
    deg = len(fit) - 1
    if hasattr(x, 'ndim') and x.ndim > 0:
        solution = np.zeros(x.shape)
    else:
        solution = 0.
    for i, coeff in enumerate(fit):
        solution += coeff * x**(deg - i)
    return solution


def flquantiles(x, q):
    """
    Get values of first and last q-quantiles of x values.
    If x is multi-D, only works if first axis (0) is sample value axis.
    :param x: sample values
    :param q: number of quantiles. Should be >2.
    :return: tuple(first, last) where first and last have dtype of x[i]
    """
    sorted_x = np.sort(x, axis=0)
    first_quant = sorted_x[len(x) // q]
    last_quant = sorted_x[(q-1)*len(x) // q]
    return (first_quant, last_quant)


def check_stretch(stretch):
    """
    Sanitize the visual stretch command, raise a RuntimeError if it's not valid
    :param stretch: either a string key to the valid_stretches dictionary
        defined here, or a callable function that can operate on numbers
    """
    valid_stretches = {'linear': lambda x: x, 'log': np.log10, 'arcsinh': np.arcsinh, 'sqrt': np.sqrt}
    if stretch in valid_stretches:
        return valid_stretches[stretch]
    elif callable(stretch):
        try:
            result = stretch(np.ones((2, 2), dtype=np.float64))
            assert result.shape == (2, 2)
        except AssertionError:
            raise RuntimeError("Your stretch function changes the data shape.")
        except Exception as e:
            raise RuntimeError(f"Your stretch function doesn't work right: {e}")
        else:
            return stretch
    else:
        raise RuntimeError(f"Not a valid stretch: {stretch}")


def get_pixel_scale(wcs_obj):
    """
    Get the pixel scale, assuming same for X and Y.
    :param wcs_obj: the WCS object for a FITS image
    :return: angular pixel scale as Quantity
    """
    ps = np.mean(np.abs(np.diag(wcs_obj.pixel_scale_matrix))) * u.deg
    c0, c1 = wcs_obj.pixel_to_world(0, 0), wcs_obj.pixel_to_world(1, 0)
    return c0.separation(c1)


def is_number(x):
    return isinstance(x, Number)


def minimum_valid_cutout(img):
    """
    Isolate a sub-array from an array which has a padding of unnecessary or
    invalid values. This function is general-purpose and can be thought of
    as 1) the opposite of the np.pad function or 2) similar to the Cutout2D
    function in astropy.
    You pass in a boolean array where there are True values surrounded by
    False values. The True vales should be confined to a small space, though
    they don't necessarily have to cover a perfect rectangle or be uniform.
    This function will return the array slices necessary to capture all the True
    values and discard as many False edge values as possible.
    The worst case scenario for this function is an isolated True value far
    from the main cluster of True values; try to avoid that.
    :param img: a boolean array where False values are unnecessary.
        Ideally, the True values are limited to a single small rectangular
        (aligned with array axes) region in which there are few, if any, False
        values.
    :returns: tuple(slice, slice) which can be applied to the original image
        to obtain the valid subregion. Similar to the "slices" attribute
        from astropy's Cutout2D.

    2021-10-11: Moved from m16_investigation.py to misc_utils.py
    """
    return tuple(slice(np.min(x), np.max(x)+1) for x in np.where(img))


def identify_longest_run(bool_array, xaxis=None, return_mask=False):
    """
    Identify the longest "run" of True values in the array and return the
    starting and ending indices, or a mask which only includes the longest run
    of True values from bool_array.
    I got the solution for this off StackExchange or something, but didn't write
    the link down...
    I think this code is from 2020, but I couldn't be sure.
    Updated and moved here September 15, 2022

    Example:
    If we have a boolean array
     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    The longest run of True (1) values is indices 9 through 14 (inclusive).
    This program would return (9, 14), the indices of the first and last True.
    If you wanted a valid array slice, you'd have to add 1 to 14 yourself.
    If return_mask=True, then the resulting mask would be
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    Note that we use the array indices to measure length in the example above,
    implicitly assuming that each array value is spaced uniformly.
    If you supply via the kwarg xaxis a 1D array the same length as bool_array
    whose values represent a "location" for each bool_array element, then this
    function will take that into account while calculating length.
    If we had the array:
     0  1  2  3  4  5  6  7  8  9 10
    [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
    and its xaxis
    22 23 25 30 38 40 41 42 43 44 45
    then we no longer see that the run of 3 True values is the longest run,
    since those only span a length of 2 units, but instead the run of 2 True
    values which spans a length of 8 units.
    The indices returned by this function would be (3, 4) in this case.

    :param bool_array: a 1D boolean array whose longest run of True values we
        want identified.
    :param xaxis: (optional) a 1D float-like array which gives locations of
        the bool_array values along some axis. If this argument is not given,
        bool_array values are assumed to be evenly spaced.
        If the supplied xaxis is evenly spaced (i.e. something that could come
        from np.arange) then the answer will be no different than leaving
        xaxis equal to None (default).
        Must be same length as bool_array and must be sorted.
    :param return_mask: rather than returning a tuple of indices, return a
        boolean array the same length as bool_array in which the only True
        values are in the longest run.
    :returns: tuple(int, int) of the indices of the first and last True values
        in the longest run of True values. To construct a valid array slice for
        the run, you would have to add 1 to the ending index.
        If return_mask==True, then 1D boolean array the same shape as
        bool_array as described above.
        If two runs are tied for longest, the earlier run will be returned.
        If there are no True values in bool_array, returns None.
    """
    # Pad with False to make sure every "run" starts and ends within the array
    padded_bool_array = np.hstack(([False], bool_array, [False]))
    if not np.any(padded_bool_array):
        return None
    if xaxis is None:
        xaxis = np.arange(padded_bool_array.size)
    else:
        raise NotImplementedError("xaxis argument not supported yet.")
        # I have to pad the xaxis with something that makes sense, or just do
        # indexing tricks.
    # Find where the values switch (runs start or end)
    diffs = np.diff(padded_bool_array.astype(int))
    # Starts are the indices of "False" right BEFORE the True
    run_starts, = np.where(diffs > 0)
    run_starts += 1 # Bump up to first True
    # Ends are the LAST indices of "True" BEFORE the False
    run_ends, = np.where(diffs < 0)
    run_ends += 1 # Bump up to first False
    # Lengths are calculated from the x axis (spectral_axis)
    run_lengths = xaxis[run_ends] - xaxis[run_starts]
    max_loc = run_lengths.argmax()
    # Get the indices to return, correct for padding by subtracting 1
    start, end = run_starts[max_loc]-1, run_ends[max_loc]-2
    if return_mask:
        # Subtract 2 from size to account for first and last False padding
        return_mask = np.full(spec_mask.size-2, False, dtype=bool)
        return_mask[start:end+1] = True
        return return_mask
    else:
        return start, end


def cutout2d_from_region(data, wcs_obj, reg_filename, reg_index=0):
    """
    May 4, 2023
    Apply Cutout2D based on a DS9 box region in a .reg file.
    Returns the entire Cutout2D object.
    :param data: np.array input array. Should be 2 dimensional.
        This is passed directly to Cutout2D as its "data" argument, so more
        detail can be found in Cutout2D's documentation.
    :param wcs_obj: WCS describing data.
    :param reg_filename: str or path-like pointing to a DS9 '.reg' file.
        The file must contain a rectangle region at the specified reg_index.
    :param reg_index: int index into region file (optional, default 0)
        Index of the box in the .reg file. Default is 0, which is acceptable
        for single-region files as well as multi-region. Index must point to
        a box region (e.g. Rectangle region in astropy regions).
    :returns: Cutout2D created from data using the WCS and size info from the
        supplied box region.
    """
    box_skyreg = regions.Regions.read(reg_filename)[reg_index]
    # Using sky coordinates to specify Cutout2D
    # center specified as SkyCoord. size given as (ny, nx) = (height, width)
    cutout = Cutout2D(data, box_skyreg.center, (box_skyreg.height, box_skyreg.width), wcs=wcs_obj, mode='partial')
    return cutout
