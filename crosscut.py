import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interpn

"""
Functions and script to make a cross-cut plot using several different data sources
"""

def cross_cut(image, wcs, xcut_coords, n_points):
    # Return the values along a linear cut across an 2D image
    # image should be 2d
    # wcs should be for 2d image
    # xcut_coords should be tuple(SkyCoords) start, end
    # n_points across cut, including start & end
    p0, p1 = (wcs.world_to_pixel(c) for c in xcut_coords)
    j_range, i_range = (np.linspace(p0[x], p1[x], n_points) for x in range(2))
    p_range = np.stack([i_range, j_range], axis=1)
    return interpn(tuple(np.arange(x) for x in image.shape), image, p_range, fill_value=np.nan, bounds_error=False, method='linear')

def get_angle_axis(xcut_coords, n_points):
    # Get angle separation from starting coordinate (xcut_coords[0])
    #   for each point along cross cut
    # Returns arcseconds
    ra_range = np.linspace(*(xcut_coords[x].ra.to_value() for x in range(2)), n_points)
    dec_range = np.linspace(*(xcut_coords[x].dec.to_value() for x in range(2)), n_points)
    point_coords = SkyCoord(ra_range, dec_range, unit=u.deg)
    return point_coords.separation(xcut_coords[0]).to(u.arcsec).to_value()

def get_velocity_axis(wcs, vaxis=0):
    # Use WCS to make the velocity axis for a cube
    # wcs is a WCS object
    # NAXIS 3 (axis 0) should be velocity, but can use vaxis to specify
    # Always returns units in km/s
    # It can be assumed that the velocity axis is sorted in increasing order.
    # It would be the fault of the underlying header info if it weren't
    array_coords = np.zeros((wcs.pixel_n_dim, wcs.array_shape[vaxis]))
    array_coords[vaxis, :] = np.arange(wcs.array_shape[vaxis])
    v = wcs.array_index_to_world_values(*array_coords)[wcs.pixel_n_dim - vaxis - 1]
    # Use world_axis_units to convert to km/s
    return v * u.Unit(wcs.world_axis_units[wcs.pixel_n_dim - vaxis - 1]).to("km/s")

def load_image(filename, ext=0):
    # Get image and WCS for a regular 2D image
    img, header = fits.getdata(filename, ext=ext, header=True)
    return img, WCS(header)

def load_cube(filename, vmin, vmax):
    # Get moment-0 image and WCS for a cube, RA-DEC-Velocity
    # Use vmin and vmax to specify limits for moment-0 calculation
    # vmin, vmax are velocity limits in km/s
    # Assumes velocity axis is 0 (NAXIS 3)
    cube, header = fits.getdata(filename, header=True)
    vaxis = get_velocity_axis(WCS(header))
    # If vmin==vmax is in vaxis, then idx_min + 1 == idx_max
    idx_min = np.searchsorted(vaxis, vmin, side='left')
    idx_max = np.searchsorted(vaxis, vmax, side='right')
    img = np.mean(cube[idx_min:idx_max, :, :], axis=0)
    return img, WCS(header, naxis=2)


def load_and_cut(filename, *args):
    # One-shot function for doing all the loading and cutting
    # FILENAME should be the complete path of a FITS file
    # Either 2, 3, or 4 additional args should be given, depending
    #   on whether filename points to a 2D image or a 3D cube
    #   or if the image should be read from a different extension.
    # If 2D: 2 args, coords and n_points for cross_cut function
    # If 3D: 4 args, vmin, vmax for load_cube and then
    #   coords and n_points.
    # If different extension: 3 args, first arg is extension number,
    #   then coords and n_points.
    if len(args) == 4:
        load_args = args[:2] # vmin, vmax
        args = args[2:] # coords, n_points
        img, wcs = load_cube(filename, *load_args)
    elif len(args) == 3:
        load_args = args[0]
        args = args[1:]
        img, wcs = load_image(filename, ext=load_args)
    else:
        img, wcs = load_image(filename)
    return cross_cut(img, wcs, *args)


def normalize_crosscut(xcut):
    # A few operations to comfortably line up all the cross cuts
    # Subtract median
    subtracted = xcut - np.nanmedian(xcut)
    # Get rid of stars (mostly for HST)
    subtracted[subtracted > np.nanstd(subtracted)*5] = np.nan
    # Add back a standard deviation to get it above 0, and normalize
    return (subtracted + np.nanstd(subtracted)) / np.ptp(subtracted[np.isfinite(subtracted)])


if __name__ == "__main__":
    data_path = "/home/rkarim/Research/Feedback/ancillary_data/"

    cross_cuts_coords = {
        0: ("10:24:07.3706 -57:45:04.036", "10:24:39.7421 -57:41:21.431", -4.7, -3.7), # First one I tried
        "WR20b_1": ("10:24:23.3160 -57:48:22.958", "10:24:35.8602 -57:48:51.026", -10.8, -6.8),
        "WR20b_2": ("10:24:19.0298 -57:48:54.625", "10:24:15.1469 -57:50:00.728", 8., 10.4),
        "Wd2_N": ("10:23:55.2786 -57:42:33.456", "10:23:54.8181 -57:41:20.952", -6.6, -5.2),
        "Wd2_N_near": ("10:24:01.4203 -57:43:56.432", "10:24:01.4203 -57:42:43.929", -9.6, -8.2),

    }

    selection = "WR20b_1"
    coord_start_xcut, coord_end_xcut = (SkyCoord(x, unit=(u.hourangle, u.deg)) for x in cross_cuts_coords[selection][:2])
    vlims = cross_cuts_coords[selection][2:]
    
    coords_xcut = (coord_start_xcut, coord_end_xcut)
    n_points = 50
    xcut_args = (coords_xcut, n_points)

    cuts_to_make = {
        # images just need filenames. cubes need velocity limits too.
        # "500 um": "herschel/helpssproc/processed/1342255009/SPIRE500um-image.fits",
        "350 um": "herschel/helpssproc/processed/1342255009/SPIRE350um-image.fits",
        "70 um": "herschel/helpssproc/processed/1342255009/PACS70um-image.fits",
        "843 MHz": "most/J1024M56.FITS",
        "12CO": ("apex/RCW49_12CO.fits",),
        "13CO": ("apex/RCW49_13CO.fits",),
        "CII": ("sofia/rcw49-cii.fits",),
        "8 um": "spitzer/irac/30002561.30002561-28687.IRAC.4.median_mosaic.fits",
        "F814W": "hst/F814W.fits",
        "$\\tau_{d}$": ("herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits", 2),
        "$T_{d}$": ("herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits", 1),
    }

    cuts_to_plot = {}

    for data_name in cuts_to_make:
        if isinstance(cuts_to_make[data_name], str):
            # this is an image
            cuts_to_plot[data_name] = load_and_cut(data_path + cuts_to_make[data_name], *xcut_args)
        elif len(cuts_to_make[data_name]) == 1: # so messy. need object if using this in future.
            # this is a cube
            label = f"{data_name} [{vlims[0]:.1f}, {vlims[1]:.1f}] km/s"
            cuts_to_plot[label] = load_and_cut(data_path + cuts_to_make[data_name][0], *vlims, *xcut_args)
        elif len(cuts_to_make[data_name]) == 2:
            # This is an image to be read from another extension
            # At this point, these should probably be objects.........
            cuts_to_plot[data_name] = load_and_cut(data_path + cuts_to_make[data_name][0], cuts_to_make[data_name][1], *xcut_args)

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    angle_axis = get_angle_axis(*xcut_args)
    for label in cuts_to_plot:
        normed_cut = normalize_crosscut(cuts_to_plot[label])
        plt.plot(angle_axis, normed_cut, label=label, linestyle='-', marker='.')


    plt.ylabel("Normalized intensity")
    plt.xlabel("Distance along cross-cut (arcseconds)")
    plt.ylim([-0.5, 1.4])
    plt.legend()

    img, w = load_image(data_path + cuts_to_make["8 um"])
    plt.subplot(122, projection=w)
    plt.imshow(np.arcsinh(img), origin='lower', vmin=np.arcsinh(11), vmax=np.arcsinh(900))
    plt.plot([coord_start_xcut.ra.to_value(), coord_end_xcut.ra.to_value()], [coord_start_xcut.dec.to_value(), coord_end_xcut.dec.to_value()],
        transform=plt.gca().get_transform('world'), color='r')
    # plt.show()
    plt.savefig(f"/home/rkarim/Pictures/3-10-20-mtg/crosscut_{selection}.png")
