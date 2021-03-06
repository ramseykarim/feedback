from numbers import Number

import numpy as np

from astropy import units as u

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
