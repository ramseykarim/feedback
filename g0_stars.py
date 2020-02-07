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

import readstartypes
import g0_dust

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
# herschel SED fit on 350 grid cutout generator
def herschel_SED_fit():
    fn = f"{data_directory}herschel/RCW49large_3p_secondCal_sysErr_jac.fits"
    with fits.open(fn) as hdul:
        T = hdul[1].data
        image_header = hdul[1].header
    return T, image_header, 'T'
# herschel spire/pacs processed data generator
def processed_herschel_data(wl):
    band_stub = {70:"PACS70um", 160:"PACS160um", 250:"SPIRE250um", 350:"SPIRE350um", 500:"SPIRE500um"}
    offset = {70: 80, 160: 370}
    folder = f"{data_directory}herschel/helpssproc/processed/1342255009_reproc350/"
    offset_stub = f"-plus{offset[wl]:06d}" if wl in offset else ""
    fn = f"{folder}{band_stub[wl]}-image-remapped-conv{offset_stub}.fits"
    data, image_header = fits.getdata(fn, header=True)
    return data, image_header, image_header['WAVELNTH']    
# sofia data that Maitraiyee emailed me on nov 18 2019
def sofia_data_integrated():
    data, header = fits.getdata(f"{data_directory}sofia/rcw49-cii-int.fits", header=True)
    header.set('NAXIS', value=2)
    for k in header:
        if '3' in k:
            header.remove(k)
    return data[0, :, :], header, 157

def plot_spire(subplot=111):
    ### READY TO PLOT SPIRE 500 IN GREY
    img_data, img_header, wl = herschel_data(500)
    img_data = np.arcsinh(img_data)
    vmin, vmax = flquantiles(img_data[~np.isnan(img_data)].ravel(), 50)
    print("vlims: {:.2f}, {:.2f}".format(vmin, vmax), img_data.shape)
    plt.subplot(subplot, projection=WCS(img_header))
    plt.imshow(img_data, vmin=vmin, vmax=7, cmap='gray_r')
    plt.xlabel("RA")
    plt.ylabel("DEC")
    # CDELT1=-0.003888888888889

def plot_irac(subplot=111):
    ### READY TO PLOT IRAC 3.6um IN GREY
    img_data, img_header, wl = irac_data(3.6)
    img_data = np.arcsinh(img_data)
    vmin, vmax = flquantiles(img_data[~np.isnan(img_data)].ravel(), 50)
    print("vlims: {:.2f}, {:.2f}".format(vmin, vmax), img_data.shape)
    plt.subplot(subplot, projection=WCS(img_header))
    plt.imshow(img_data, vmin=vmin, vmax=4.3, cmap='gray_r')
    plt.xlabel("RA")
    plt.ylabel("DEC")

def plot_sofia_integrated(subplot=111):
    ### READT TO PLOT SOFIA INTEGRATED [CII] INTENSITY (moment 0)
    img_data, img_header, wl = sofia_data_integrated()
    vmin, vmax = 8, 400
    plt.subplot(subplot, projection=WCS(img_header, naxis=2))
    plt.imshow(img_data, vmin=vmin, vmax=vmax, cmap='gray_r')
    plt.xlabel("RA")
    plt.ylabel("DEC")


radec_df = pd.read_pickle(f"{data_directory}catalogs/Ramsey/OBradec.pkl")
readstartypes.reduce_catalog_spectral_types(radec_df)
readstartypes.get_catalog_properties_vacca(radec_df, 'Teff')
readstartypes.get_catalog_properties_vacca(radec_df, 'log_g')
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
        color='blue', transform=get_transform(), label='Wd2 OB Stars')
    plt.scatter([wd2_center_coord.ra.deg], [wd2_center_coord.dec.deg], marker='o',
        color='red', transform=get_transform(), label='center of Wd2')
    plt.legend()
    plt.show()

def plot_nearby_Wd2_types_JUSTSTARS(extra_filter=True):
    df = radec_df.loc[is_within_range & extra_filter] # & (radec_df.SpectralType_Number < 20) & (radec_df.Teff > 0)
    values = df.paramx
    plt.scatter(*get_radec(df), marker='o', s=12, c=values, cmap='cool',
        transform=get_transform(), label='Wd2 OB Stars')

def plot_specific_stars(rows, names):
    for row, name in zip(rows, names):
        plt.plot([row['RAdeg']], [row['DEdeg']], marker='x', markersize=10, transform=get_transform(), label=name)

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
    radfield1d = 1.6e-3 # erg cm-2 s-1
    cm2_to_pc2 = (u.cm.to(u.pc))**2
    # gives inverse distance in cm-2
    inv_dist = cat.coords.apply(lambda x: cm2_to_pc2/(distance_from_point_pixelgrid(x, w, distance_los_pc)**2))
    # gives luminosity in erg s-1
    lum = cat.luminosity.apply(lambda x: x.to(u.erg/u.s).to_value())
    # returns luminosity expressed in terms of avg 1d IRF
    return np.sum(lum * inv_dist, axis=0) / radfield1d

# img_data, img_header, wl = irac_data(8)
# w = WCS(img_header)

# w = make_wcs(wd2_center_coord, pixel_scale=2*u.arcsec, grid_shape=(1500, 1500))
def find_specific_star(spectral_type):
    # Find first star of this spectral type
    star = radec_df.loc[radec_df.SpectralType == spectral_type]
    return star.loc[star.index[0]]

def find_all_stars(condition):
    stars = radec_df.loc[condition]
    return [stars.loc[x] for x in stars.index]

# WN6ha = radec_df.loc[radec_df.SpectralType == 'WN6ha']
# WN6ha = WN6ha.loc[WN6ha.index[0]]

radec_df['paramx'] = radec_df.Teff_V96
radec_df['paramy'] = radec_df.log_g_V96
is_WR = radec_df.SpectralType_Number.isnull() & is_within_range
WR_i = radec_df.loc[is_WR].index[0]
radec_df.loc[WR_i, 'paramx'] = 43000
radec_df.loc[WR_i, 'paramy'] = readstartypes.PoWRGrid.calculate_Rt(19.7, 8.5e-6, 1600, 4)

OBGrid = readstartypes.PoWRGrid('OB')
WRGrid = readstartypes.PoWRGrid('WNE')
def nearest_gridpoint(row):
    if row['SpectralType_ReducedTuple'][0][0] == 'W':
        grid = WRGrid
    else:
        grid = OBGrid
    paramx, paramy = grid.parse_query_params(row['paramx'], row['paramy'])
    return paramx, paramy
radec_df['grid_params'] = radec_df.loc[is_within_range].apply(nearest_gridpoint, axis=1)
radec_df['grid_paramx'] = radec_df.grid_params.loc[is_within_range].apply(lambda x: x[0])
radec_df['grid_paramy'] = radec_df.grid_params.loc[is_within_range].apply(lambda x: x[1])
radec_df.drop(columns='grid_params', inplace=True)
def get_wlflux(row):
    if row['SpectralType_ReducedTuple'][0][0] == 'W':
        grid = WRGrid
    else:
        grid = OBGrid
    return grid.get_model(row['paramx'], row['paramy'])
radec_df['wlflux'] = radec_df.loc[is_within_range].apply(get_wlflux, axis=1)
radec_df['wl'] = radec_df.wlflux.loc[is_within_range].apply(lambda x: x[0])
radec_df['flux'] = radec_df.wlflux.loc[is_within_range].apply(lambda x: x[1])

def plot_model_grids():
    plt.figure(figsize=(16, 9))
    plt.subplot(121)
    OBGrid.plot_grid_space(setup=False, show=False)
    plt.scatter(radec_df.grid_paramx[~is_WR], radec_df.grid_paramy[~is_WR], color='red', marker='x')
    plt.subplot(122)
    WRGrid.plot_grid_space(setup=False, show=False)
    plt.scatter(radec_df.grid_paramx[is_WR], radec_df.grid_paramy[is_WR], color='red', marker='x')
    plt.show()

def plot_all_fuv_spectra():
    plt.figure(figsize=(13, 9))
    radec_df.wlflux.loc[is_within_range].apply(readstartypes.PoWRGrid.plot_spectrum, setup=False, show=False, fuv=True, xunit=u.eV)
    plt.show()

radec_df['luminosity'] = radec_df.wlflux.loc[is_within_range].apply(readstartypes.PoWRGrid.integrate_flux)

def coords_to_string(coord):
    return coord.to_string(style='hmsdms', sep=':').replace(' ', ',')
def gen_region(row):
    log_lum = (np.log10(row.luminosity) - 4.5)*10 + 5
    comment = "# {:s}".format(row.SpectralType_Reduced)
    return "circle({:s},{:.4f}\") {:s}".format(coords_to_string(row.coords), log_lum, comment)


def write_region_file():
    radec_df['luminosity'] = radec_df.luminosity.loc[is_within_range].apply(lambda x: x.to_value())
    with open('figures/Wd2_stars_luminosity.reg', 'w') as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=white dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        for row in radec_df.loc[is_within_range].itertuples():
            f.write(gen_region(row)+'\n')
        f.write('\n')
    print('done')


def save_html():
    columns_to_drop = ['SpectralType_ReducedTuple', 'SpectralType_Number',
        'wlflux','wl','flux', 'coords', 'SpectralType_Adopted',
        'Teff_V96', 'paramy', 'grid_paramx', 'grid_paramy',]
    for c in columns_to_drop:
        radec_df.drop(columns=c, inplace=True)
    radec_df.luminosity.loc[is_within_range] = radec_df.luminosity.loc[is_within_range].apply(lambda x: "{:.2f}".format(np.log10(x.to_value())))
    radec_df.paramx.loc[is_within_range] = radec_df.paramx.loc[is_within_range].apply(lambda x: "{:.2f}".format(x/1000))
    nearby_stars = radec_df.loc[is_within_range]
    nearby_stars = nearby_stars[['RAdeg', 'DEdeg', 'MSP', 'SpectralType', 'SpectralType_Reduced', 'paramx', 'luminosity']]
    nearby_stars.rename(columns={'SpectralType_Reduced': 'SpectralType_Adopted', 'paramx': 'Teff (kK)', 'luminosity': 'log FUV L (Lsun)'}, inplace=True)
    nearby_stars.to_html('Wd2_catalog_FUVflux.html', na_rep='')


def make_g0CII_figure():
    plt.figure(figsize=(16, 8))
    img_data, img_header, wl = sofia_data_integrated()
    w = WCS(img_header, naxis=2)

    t0 = datetime.datetime.now()
    grid = calc_g0(radec_df.loc[is_within_range], w, 4.16*1000)
    print(grid.shape)
    t1 = datetime.datetime.now()
    print((t1-t0).total_seconds()*1000, "ms")
    plt.subplot(121, projection=w)
    plt.imshow(np.log10(grid), cmap='cividis', vmax=5.5)
    plt.colorbar(label='log10 G0')
    plot_nearby_Wd2_types_JUSTSTARS()
    plt.xlabel("RA")
    plt.ylabel("DEC")
    plot_sofia_integrated(subplot=122)
    plt.colorbar()
    plot_nearby_Wd2_types_JUSTSTARS()
    plt.subplots_adjust(top=0.974, bottom=0.061, left=0.05, right=0.95, hspace=0.2, wspace=0.1)
    plt.show()
    # plt.savefig("Wd2_sofia_G0.pdf")

def make_g0hotstars_figure(Tcut_kK, hotter=True, fracmin=0.3, fracmax=0.7):
    """
    About half the global G0 is from stars > ~45kK
    about 60% from > 40kK

    Star 34 (O5V) dominates its own region
    It is also the star with the bow-shock looking arc to the northwest
    """
    distance_to_Wd2 = 4.16*1000
    plt.figure(figsize=(19, 8))
    img_data, img_header, wl = herschel_SED_fit()
    w = WCS(img_header, naxis=2)
    if hotter:
        condition = radec_df.paramx > Tcut_kK*1e3
    else:
        condition = radec_df.paramx < Tcut_kK*1e3
    # print(radec_df.loc[is_hot & is_within_range])
    grid_hot = calc_g0(radec_df.loc[condition & is_within_range], w, distance_to_Wd2)
    grid_all = calc_g0(radec_df.loc[is_within_range], w, distance_to_Wd2)
    grid_diff = grid_hot/grid_all
    pkwargs = dict(cmap='cividis', vmin=2.7, vmax=5.5)
    glt = ">" if hotter else "<"
    for sp, grid, title in zip((131, 132, 133), (grid_hot, grid_diff, grid_all), ("Stars with Teff {:s} {:.0f} kK".format(glt, Tcut_kK), "Difference", "All stars")):
        plt.subplot(sp, projection=w)
        if sp == 132:
            plt.imshow(grid, cmap='magma', vmin=fracmin, vmax=fracmax)
            plt.colorbar(label='Hot star contribution fraction')
        else:
            plt.imshow(np.log10(grid), **pkwargs)
            plt.colorbar(label='log G0')
        plot_nearby_Wd2_types_JUSTSTARS()
        plt.title(title)
        plt.xlabel("RA")
        plt.ylabel("DEC")
    plt.subplots_adjust(top=0.974, bottom=0.019, left=0.054, right=0.985, hspace=0.2, wspace=0.223)
    plt.show()


def save_g0_fits(Tcut_kK=None, hotter=True):
    img_data, img_header, wl = herschel_SED_fit()
    w = WCS(img_header, naxis=2)

    if Tcut_kK is not None:
        if hotter:
            condition = radec_df.paramx > Tcut_kK*1e3
        else:
            condition = radec_df.paramx < Tcut_kK*1e3
        condition = condition & is_within_range
    else:
        condition = is_within_range

    t0 = datetime.datetime.now()
    grid = calc_g0(radec_df.loc[condition], w, 4.16*1000)
    t1 = datetime.datetime.now()
    print((t1-t0).total_seconds()*1000, "ms")

    header = fits.Header({'SIMPLE': True})
    header.update(w.to_header())
    header['BUNIT'] = ("Habing fields", "Data unit")
    header['OBJECT'] = ("RCW49", "Target name")
    header['CREATOR'] = ("Ramsey: {}".format(str(__file__)), "FITS file creator")
    header['DATE'] = (datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(), "File creation date")
    header['COMMENT'] = "Habing field taken to be 1.6e-3 erg cm-2 s-1"
    header['HISTORY'] = "Star catalog synthesized from several literature sources"
    header['HISTORY'] = "Catalog used here can be found as Wd2_catalog_FUVflux.csv"
    header['HISTORY'] = "Used Hershel SPIRE 350um pixel grid"
    fits.writeto("figures/rcw49-g0-stars.fits", grid, header)


def make_g0_starsVdust_figure(Tcut_kK=None, hotter=True):
    distance_to_Wd2 = 4.16*1000
    plt.figure(figsize=(14, 10))
    img_data, img_header, wl = herschel_SED_fit()
    w = WCS(img_header, naxis=2)
    if Tcut_kK is not None:
        glt = ">" if hotter else "<"
        if hotter:
            condition = radec_df.paramx > Tcut_kK*1e3
        else:
            condition = radec_df.paramx < Tcut_kK*1e3
        condition = condition & is_within_range
        stars_txt = "G0 from stars with Teff {:s} {:.0f} kK".format(glt, Tcut_kK)
    else:
        condition = is_within_range
        stars_txt = "G0 from stars"
    # print(radec_df.loc[is_hot & is_within_range])
    grid_stars = calc_g0(radec_df.loc[condition], w, distance_to_Wd2)
    grid_dust = g0_dust.calculate_g0()
    grid_diff = grid_stars/grid_dust
    pkwargs = dict(cmap='cividis', vmin=1, vmax=5)
    for sp, grid, title in zip((221, 223, 222), (grid_stars, grid_diff, grid_dust), (stars_txt, "Difference", "G0 from dust")):
        plt.subplot(sp, projection=w)
        if sp == 223:
            plt.imshow(np.log10(grid), cmap='magma', vmin=-1, vmax=4)
            plt.colorbar(label='log stellar G0 excess')
        else:
            plt.imshow(np.log10(grid), **pkwargs)
            plt.colorbar(label='log G0')
        plot_nearby_Wd2_types_JUSTSTARS()
        plt.title(title)
        plt.xlabel("RA")
        plt.ylabel("DEC")
    plt.subplots_adjust(top=0.974, bottom=0.019, left=0.054, right=0.985, hspace=0.2, wspace=0.223)
    plt.show()



# All stars, real simple
# make_g0CII_figure()

# O5 and earlier
# make_g0hotstars_figure(45)

# The swarm of ETs
# make_g0hotstars_figure(36, hotter=False, fracmin=0.2, fracmax=0.45)

# COMPARE TO DUST
make_g0_starsVdust_figure()

# save_g0_fits()

print(f'edit the end of {__file__} to save a file or make a figure')
