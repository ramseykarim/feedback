"""
Plotting in mayavi, originally in dust_mass.py
Created: May 14, 2020
"""
__author__ = "Ramsey Karim"


import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import WCS

from mayavi import mlab

from dust_mass import prepare_img_for_plot, smooth_image, load_tau


def plot_surface(img, warp_scale=15, colormap='blue-red'):
    """
    Make a surface plot with mayavi
    """
    img = img[::-1, ::-1]

    yy, xx = np.meshgrid(*(np.arange(x) for x in img.shape), indexing='ij')
    # xx, yy = xx.T, yy.T
    mlab.surf(xx.T, yy.T, img.T, warp_scale=warp_scale, colormap=colormap)


@mlab.animate(ui=False, delay=30)
def loop_plots():
    """
    This works fine as long as this generator is assigned to a variable (not garbage collected)
    """
    f = mlab.gcf()
    x_arr = np.linspace(0, 2*np.pi, 50)
    az_list = (np.sin(x_arr) * 15) + 45
    i = 0
    while i < len(az_list) - 1:
        current_az = az_list[i]
        mlab.view(azimuth=current_az, elevation=30)
        f.scene.render()
        # mlab.savefig(f"./figures/dust_surf_anim/anim_tausurf_{i:04d}.png")
        i += 1
        yield


def plot_tau_loop_anim():
    """
    Some plots to illustrate the integrate_shell_by_hand method
    Written 4/22/20
    """
    sizemod = 2
    mlab.figure(size=(400*sizemod, 350*sizemod))
    img = get_tau()
    img = smooth_image(img, kernel_length=10, std=1.5)
    img = prepare_img_for_plot(img, scale=np.log10, low_cutoff=lambda x: np.nanmedian(x)/2)
    plot_surface(img, warp_scale=50)
    # The object "a" needs to NOT be garbage collected. There must be a reference!
    a = loop_plots()
    mlab.show()



def skycoord_from_ds9(ra, dec):
    """
    ra and dec are strings separated already from a DS9 reg description
    """
    return SkyCoord(" ".join((ra, dec)), unit=(u.hourangle, u.deg), frame=FK5)


def gen_bubble_params(reg_string):
    """
    Make a SkyCoord center and Quantity radius from these parameters
    The parameters can be copied from a DS9 .reg description of a Circle region
    example of reg_string: 10:23:56.4097,-57:45:30.609,310.000
    Radius assumed to be in arcsec
    """
    # Split ra, dec, and radius
    ra, dec, radius = reg_string.split(",")
    # Make the center SkyCoord
    center = skycoord_from_ds9(ra, dec)
    # Make the radius Quantity
    # Interestingly, if radius is a str, the result is a Unit...
    radius = float(radius) * u.arcsec
    return center, radius


def calc_line_length(reg_string):
    """
    Calculate line length in physical units using the start and end points
        of a DS9 line region description
    example of a reg_string: 10:23:28.3651,-57:39:55.977,10:23:28.3651,-57:39:55.977
    """
    ra1, dec1, ra2, dec2 = reg_string.split(",")
    c1, c2 = skycoord_from_ds9(ra1, dec1), skycoord_from_ds9(ra2, dec2)
    return c1.separation(c2)


def gauss2d(r, mu, sigma):
    # Radial Gaussian depending only on mean and standard deviation
    # r is a 2d grid of radius
    exponent = (r - mu)**2. / (2 * sigma**2.)
    return np.exp(-1*exponent)

def sphere2d(r, mean_radius, thickness):
    # sphere height on 2d radius grid
    result = np.zeros_like(r)
    # The mean_radius should be the mean of inner and outer radius
    # We want a sphere for the outer radius
    radius = mean_radius + thickness/2.
    result[r < radius] = np.sqrt(radius**2 - r[r < radius]**2)
    return result


def get_draw_params(center, radius, fwhm, height, background, wcs_obj, nanmask=None):
    """
    Center should be SkyCoord, radius and FWHM should be Quantity or Angle
    height and background should all be optical depth at 160 um
    WCS object should describe the target grid on which we draw this
        WCS  object needs to have "array_shape" attribute present
    nanmask (optional) should be True where the grid is NaN; this is just
        to make the final image look more like the template img it is based on
    Returns a dictionary of these parameters and some conversions
    """
    # Get center pixel in I,J
    # Reminder that world_to_pixel gives exact (float) pixel in X,Y indexing
    # and world_to_array_index gives INTEGER array index in I,J indexing
    center_pixel_xy = wcs_obj.world_to_pixel(center) # this is XY
    # Get angular pixel scale
    pixel_scale = np.mean(np.abs(np.diag(wcs_obj.pixel_scale_matrix))) * u.deg
    # Convert angular radius to pixel radius (decompose resolves the units)
    radius_pixel = (radius / pixel_scale).decompose().to_value()
    # Convert angular FWHM to pixel standard deviation
    fwhm_pixel = (fwhm / pixel_scale).decompose().to_value()
    sigma = fwhm_pixel / (2 * np.sqrt(2 * np.log(2)))
    # Make XY grid (should have array_shape, but XY ordering)
    xx, yy = np.meshgrid(*(np.arange(x) for x in wcs_obj.array_shape[::-1]), indexing='xy')
    xx = xx - center_pixel_xy[0]
    yy = yy - center_pixel_xy[1]
    rr = np.sqrt(xx*xx + yy*yy)
    if nanmask is None:
        nanmask = np.zeros(wcs_obj.array_shape).astype(bool)
    param_dict = {
        'center': center,
        'radius': radius,
        'fwhm': fwhm,
        'height': height,
        'background': background,
        'wcs_obj': wcs_obj,
        'nanmask': nanmask,
        'center_pixel_xy': center_pixel_xy,
        'pixel_scale': pixel_scale,
        'radius_pixel': radius_pixel,
        'fwhm_pixel': fwhm_pixel,
        'sigma': sigma,
        'xx': xx,
        'yy': yy,
        'rr': rr,
    }
    return param_dict


def draw_gaussian_ring(param_dict):
    """
    Draw the Gaussian ring that I use as a step in modeling the shell
    """
    result = gauss2d(param_dict['rr'], param_dict['radius_pixel'], param_dict['sigma'])
    result += param_dict['background']
    result[param_dict['nanmask']] = np.nan
    return result


def draw_sphere(param_dict):
    result = sphere2d(param_dict['rr'], param_dict['radius_pixel'], param_dict['fwhm_pixel'])
    result += param_dict['background']
    result[param_dict['nanmask']] = np.nan
    return result


def draw_cored_sphere(param_dict):
    result = draw_sphere(param_dict)
    inside_shell_mask = param_dict['rr'] < param_dict['radius_pixel'] - param_dict['fwhm_pixel']/2.
    result[inside_shell_mask] = param_dict['background']
    return result








"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%& Mayavi plotting &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&

These bubble estimates are from bubbles_hotdust_cii-based.reg (4/24/20)
"""

bubble_hilim = gen_bubble_params("10:23:46.0688,-57:45:57.668,381.637")
bubble_lolim = gen_bubble_params("10:23:56.4097,-57:45:30.609,310.000")

line_descriptions = (
    "10:24:23.8536,-57:43:01.751,10:24:32.5134,-57:42:05.851",
    "10:24:10.2973,-57:41:29.730,10:24:14.5688,-57:40:01.621",
    "10:24:28.2532,-57:46:36.235,10:24:38.1076,-57:46:36.308",
)


def mayavi_plots():
    """
    mayavi plots to illustrate some things
    """
    fwhm = sum(calc_line_length(x) for x in line_descriptions) / len(line_descriptions)
    tau160, tau160_h = load_tau()
    w = WCS(tau160_h)

    param_dict = get_draw_params(*bubble_hilim, fwhm, 0.5, 0.0, w, nanmask=np.isnan(tau160))
    gauss_ring = draw_gaussian_ring(param_dict)
    cs_ring = draw_cored_sphere(param_dict)
    gauss_ring *= np.nansum(cs_ring) / np.nansum(gauss_ring)

    cs_ring = smooth_image(cs_ring, kernel_length=10, std=1.5)
    # gauss_ring = smooth_image(gauss_ring, kernel_length=10, std=1.5)



    diff_ring = gauss_ring - cs_ring
    diff_ring[np.abs(diff_ring) < 1e-2*np.nanmax(cs_ring)] = np.nan


    # lims = dict(vmin=np.nanmin(diff_ring), vmax=np.nanmax(cs_ring), origin='lower')
    # plt.subplot(131)
    # plt.imshow(cs_ring, **lims)
    # plt.subplot(132)
    # plt.imshow(diff_ring, **lims)
    # plt.subplot(133)
    # plt.imshow(gauss_ring, **lims)
    # plt.show()

    plot_surface(diff_ring, warp_scale=1, colormap='blue-red')
    a = loop_plots()
    mlab.show()
