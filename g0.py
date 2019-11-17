# script to find G0 based on OB star positions, luminosities
# created: November 5, 2019
__author__ = "Ramsey Karim"

import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import pandas as pd
from misc_utils import flquantiles

from readstartypes import reduce_catalog_spectral_types, get_catalog_properties_sternberg

data_directory = "../ancillary_data/"

# spitzer irac generator
def irac_data(wl):
    i = [3.6, 4.5, 5.8, 8.0].index(wl) + 1
    fn =  "{:s}spitzer/irac/30002561.30002561-28687.IRAC.{:1d}.median_mosaic.fits".format(data_directory, i)
    data, image_header = fits.getdata(fn, header=True)
    return data, image_header, image_header['WAVELEN']
# herschel spire/pacs generator
def herschel_data(wl):
    i = [70, 160, 250, 350, 500].index(wl)
    folder = ["HPPJSMAPB", "HPPJSMAPR", "extdPSW", "extdPMW", "extdPLW"][i]
    fn = glob.glob(f"{data_directory}herschel/anonymous1571176316/1342255009/level2_5/{folder}/*.fits*").pop()
    with fits.open(fn) as hdul:
        global_header = hdul[0].header
        data = hdul[1].data
        image_header = hdul[1].header
    return data, image_header, global_header['WAVELNTH']

def plot_spire():
    ### READY TO PLOT SPIRE 500 IN GREY
    img_data, img_header, wl = herschel_data(500)
    img_data = np.arcsinh(img_data)
    vmin, vmax = flquantiles(img_data[~np.isnan(img_data)].ravel(), 50)
    print("vlims: {:.2f}, {:.2f}".format(vmin, vmax), img_data.shape)
    plt.subplot(projection=WCS(img_header))
    plt.imshow(img_data, vmin=vmin, vmax=7, cmap='gray_r')
    plt.xlabel("RA")
    plt.ylabel("DEC")
    # CDELT1=-0.003888888888889

def plot_irac():
    ### READY TO PLOT IRAC 3.6um IN GREY
    img_data, img_header, wl = irac_data(3.6)
    img_data = np.arcsinh(img_data)
    vmin, vmax = flquantiles(img_data[~np.isnan(img_data)].ravel(), 50)
    print("vlims: {:.2f}, {:.2f}".format(vmin, vmax), img_data.shape)
    plt.subplot(projection=WCS(img_header))
    plt.imshow(img_data, vmin=vmin, vmax=4.3, cmap='gray_r')
    plt.xlabel("RA")
    plt.ylabel("DEC")

radec_df = pd.read_pickle(f"{data_directory}catalogs/Ramsey/OBradec.pkl")
reduce_catalog_spectral_types(radec_df)
get_catalog_properties_sternberg(radec_df, 'log_L')
get_catalog_properties_sternberg(radec_df, 'Teff')
wd2_center_coord = SkyCoord("10 23 58.1 -57 45 49", unit=(u.hourangle, u.deg))
# Find everything within 11 arcminutes of the center (the rough size of the nebula on the sky)
def within_range(star_row):
    return star_row.coords.separation(wd2_center_coord).arcmin < 11
is_within_range = radec_df.apply(within_range, axis=1)

def get_transform():
    return plt.gca().get_transform('world')

def get_radec(df):
    return df.RAdeg, df.DEdeg

def plot_all_ET_vs_ST():
    plt.figure(figsize=(14, 9))
    plot_spire()
    mask = radec_df.SpectralType != 'ET'
    plt.scatter(*get_radec(radec_df.loc[mask]), marker='x',
        color='blue', transform=get_transform(), label='ST')
    plt.scatter(*get_radec(radec_df.loc[~mask]), marker='x',
        color='red', transform=get_transform(), label='ET')
    plt.legend()
    plt.show()

def plot_nearby_Wd2():
    plt.figure(figsize=(13, 10))
    plot_irac()
    print(radec_df.shape)
    print(is_within_range.sum())
    plt.scatter(*get_radec(radec_df.loc[is_within_range]), marker='x',
        color='blue', transform=get_transform(), label='stars')
    plt.scatter([wd2_center_coord.ra.deg], [wd2_center_coord.dec.deg], marker='o',
        color='red', transform=get_transform(), label='center of Wd2')
    plt.legend()
    plt.show()

def plot_nearby_Wd2_types_JUSTSTARS(extra_filter=True):
    df = radec_df.loc[is_within_range & (radec_df.SpectralType_Number < 20) & (radec_df.Teff > 0) & extra_filter]
    values = df.Teff
    plt.scatter(*get_radec(df), marker='o', s=12, c=values, cmap='Reds_r',
        transform=get_transform(), label='stars')


def plot_nearby_Wd2_types():
    plt.figure(figsize=(13, 10))
    plot_irac()
    plot_nearby_Wd2_types_JUSTSTARS()
    plt.scatter([wd2_center_coord.ra.deg], [wd2_center_coord.dec.deg], marker='x',
        color='blue', transform=get_transform(), label='center of Wd2')
    plt.legend()
    plt.show()


def make_wcs(ref_coord, grid_shape=None, ref_pixel=None, pixel_scale=None):
    """
    ref_pixel should be Numpy array index (0-indexed)
    If grid shape is (10, 10) i.e. (0..9, 0..9) and you want pixel (4, 4)
        i.e. the fifth i,j pixels to be the center, specify (4, 4).
        This function will pass (4+1, 4+1) to WCS to ensure that the fifth
        pixels are chosen in this case.
    pixel_scale can be a Quantity; if it isn't, it's assumed to be in arcmin
    """
    if not isinstance(pixel_scale, u.quantity.Quantity):
        pixel_scale *= u.arcmin
    if ref_pixel is None:
        ref_pixel = tuple(int(x/2) for x in grid_shape)
    kws = {
        'NAXIS': (2, "Number of axes"),
        'NAXIS1': (grid_shape[1], "X/j axis length"),
        'NAXIS2': (grid_shape[0], "Y/i axis length"),
        'RADESYS': ('ICRS', "Interational Celestial Reference System"),
        'CRVAL1': (ref_coord.ra.deg, "[deg] RA of reference point"),
        'CRVAL2': (ref_coord.dec.deg, "[deg] DEC of reference point"),
        'CRPIX1': (ref_pixel[1] + 1, "[pix] Image reference point"),
        'CRPIX2': (ref_pixel[0] + 1, "[pix] Image reference point"),
        'CTYPE1': ('RA---TAN', "RA projection type"),
        'CTYPE2': ('DEC--TAN', "DEC projection type"),
        'PA': (0., "[deg] Position angle of axis 2 (E of N)"),
        'CD1_1': (-pixel_scale.to(u.deg).to_value(), "Transformation matrix"),
        'CD1_2': (0., ""),
        'CD2_1': (0., ""),
        'CD2_2': (pixel_scale.to(u.deg).to_value(), ""),
        'EQUINOX': (2000., "[yr] Equatorial coordinates definition"),
    }
    header = fits.Header()
    # Two lines to avoid some weird bug about reading dictionaries in the constructor
    header.update(kws)
    return WCS(header)

"""
TWO METHODS FOR CALCULATING SEPARATION FROM A SINGLE POINT
pixelgrid is faster (not sure about scaling difference but it's better than linear)
wcssep is more accurate across larger areas, since it deals in great circles instead of planes
Depending on the approximation being made, it may not matter so much
"""

def distance_from_center_pixelgrid(center_coord, distance_los_pc, grid_shape=None, ref_pixel=None, pixel_scale=None):
    w = make_wcs(center_coord, grid_shape, ref_pixel, pixel_scale)
    grid = np.sqrt((np.arange(grid_shape[0]) - ref_pixel[0])[:, np.newaxis]**2 + (np.arange(grid_shape[1]) - ref_pixel[1])[np.newaxis, :]**2)
    grid = np.radians(grid * pixel_scale.to('deg').to_value()) * distance_los_pc
    return grid, w

def distance_from_center_wcssep(center_coord, distance_los_pc, grid_shape=None, ref_pixel=None, pixel_scale=None):
    """
    grid_shape and ref_pixel should be in i,j order (row, col), NOT x,y
    Returns grid with physical distances in pc from center_coord, as well as WCS object for this grid
    """
    w = make_wcs(center_coord, grid_shape, ref_pixel, pixel_scale)
    grid = np.full(grid_shape, np.nan)
    ij_arrays = tuple(idx_grid.ravel() for idx_grid in np.mgrid[tuple(slice(0, shape_i) for shape_i in grid_shape)])
    grid[ij_arrays] = w.array_index_to_world(*ij_arrays).separation(center_coord).to('rad').to_value() * distance_los_pc
    return grid, w

"""
Now need to make arrays for distances from an arbitrary SkyCoord given an existing WCS
"""

def distance_from_point_pixelgrid(point_coord, w, distance_los_pc):
    """
    point_coord is a SkyCoord object
    w is a WCS object
    """
    # grid_shape from w.array_shape
    grid_shape = w.array_shape
    ref_pixel = w.world_to_array_index(point_coord)
    # Get physical separation per pixel along each axis at 0,0 (assume they do not change -- this should be ok for small regions)
    ds_di = w.array_index_to_world(*ref_pixel).separation(w.array_index_to_world(ref_pixel[0]+1, ref_pixel[1])).to('rad').to_value() * distance_los_pc
    ds_dj = w.array_index_to_world(*ref_pixel).separation(w.array_index_to_world(ref_pixel[0], ref_pixel[1]+1)).to('rad').to_value() * distance_los_pc
    grid = np.sqrt((ds_di*(np.arange(grid_shape[0]) - ref_pixel[0]))[:, np.newaxis]**2 + (ds_dj*(np.arange(grid_shape[1]) - ref_pixel[1]))[np.newaxis, :]**2)
    return grid

def distance_from_point_wcssep(point_coord, w, distance_los_pc):
    """
    Again, this is way slower (probably grid.size**2)
    """
    grid = np.full(w.array_shape, np.nan)
    ij_arrays = tuple(idx_grid.ravel() for idx_grid in np.mgrid[tuple(slice(0, shape_i) for shape_i in w.array_shape)])
    grid[ij_arrays] = w.array_index_to_world(*ij_arrays).separation(point_coord).to('rad').to_value() * distance_los_pc
    return grid

args, kwargs = (4.16*1000,), {'pixel_scale': 1*u.arcsec, 'grid_shape':(1500, 1500), 'ref_pixel':(500, 500)}


def test_compare_wcssep_pixelgrid():
    img_data, img_header, wl = herschel_data(500)
    w = WCS(img_header)
    t0 = datetime.datetime.now()
    grid = distance_from_point_pixelgrid(test_point, w, *args)
    t1 = datetime.datetime.now()
    grid2 = distance_from_point_wcssep(test_point, w, *args)
    t2 = datetime.datetime.now()
    print((t1-t0).total_seconds()*1000, "ms")
    print((t2-t1).total_seconds()*1000, "ms")
    plt.subplot(111, projection=w)
    plt.imshow((grid2-grid)/grid2, cmap='gray_r')
    plt.xlabel("RA")
    plt.ylabel("DEC")
    plt.show()



def distance_from_all_points(point_coords, w, distance_los_pc):
    return np.sum(point_coords.apply(lambda x: (.1/distance_from_point_pixelgrid(x, w, distance_los_pc)**2)), axis=0)

def calc_g0(cat, w, distance_los_pc):
    inv_dist = cat.coords.apply(lambda x: (.1/distance_from_point_pixelgrid(x, w, distance_los_pc)**2))
    return np.sum(2.1 * np.exp(cat.log_L) * inv_dist, axis=0)

# img_data, img_header, wl = irac_data(8)
# w = WCS(img_header)
w = make_wcs(wd2_center_coord, pixel_scale=2*u.arcsec, grid_shape=(1500, 1500))
# base_grid, w = distance_from_center_wcssep(wd2_center_coord, *args, **kwargs)
# print(w)
# test_point = w.array_index_to_world(1, 1)

t0 = datetime.datetime.now()
grid = calc_g0(radec_df, w, 4.16*1000)
t1 = datetime.datetime.now()
print((t1-t0).total_seconds()*1000, "ms")
plt.figure(figsize=(13, 10))
plt.subplot(111, projection=w)
plt.imshow(np.log10(grid), cmap='gray')
plot_nearby_Wd2_types_JUSTSTARS()
plt.xlabel("RA")
plt.ylabel("DEC")

plot_nearby_Wd2_types()

plt.show()


