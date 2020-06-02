"""
script to find G0 based on OB star positions, luminosities
created: November 5, 2019
updated: May 5, 2020
"""
__author__ = "Ramsey Karim"

import datetime
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from misc_utils import flquantiles
from catalog_read import load_final_catalog_df
import catalog_spectral
import catalog_utils
import g0_dust

data_directory = catalog_utils.ancillary_data_path

rcw_dist = 4.16*u.kpc


"""
Finish preparing the catalog with physical properties
This step uses catalog_spectral.py functions
"""
def main():
    catalog_df = load_final_catalog_df()
    catalog_df = convert_ST_to_properties(catalog_df)

    # catalog_df = filter_by_within_range(catalog_df)
    catalog_df = filter_by_within_range(catalog_df, radius_arcmin=3.)
    catalog_df = filter_by_within_range(catalog_df, radius_arcmin=6.)
    catalog_df = filter_by_within_range(catalog_df, radius_arcmin=12.)


    # catalog_utils.save_df_html(catalog_df)

    # Option here to copy the improved make_wcs and make_wcs_like functions from mosaic_vlt.py
    # We can just use CII's WCS in the meantime

    # print(len(catalog_df.loc[catalog_df['is_within_12.0_arcmin']]))
    # print(len(catalog_df.loc[catalog_df['is_within_6.0_arcmin']]))
    # print(len(catalog_df.loc[catalog_df['is_within_3.0_arcmin']]))

    cii_mom0, cii_w = catalog_utils.load_cii(2)
    # calc_and_plot_g0(catalog_df, cii_mom0, cii_w)
    # calc_and_plot_mdot(catalog_df, cii_mom0, cii_w)

    ## Experiment to see what WR stars mass loss looks like (May 22, 2020)
    catalog_df = filter_by_only_WR(catalog_df)
    catalog_df = catalog_df.loc[~catalog_df['is_WR']]
    # calc_and_plot_and_save_WR_wind_power(catalog_df, cii_w, rcw_dist, saving=True, radius_arcmin=6.)
    calc_and_plot_mdot(catalog_df, cii_mom0, cii_w)
    return


def convert_ST_to_properties(catalog_df):
    """
    Applies catalog_spectral.STResolver to the catalog created in catalog_read
    :param df: the final catalog from catalog_read
    """
    # Make STResolver objects
    catalog_df['ST_obj'] = catalog_df['Spectral'].apply(catalog_spectral.STResolver)
    # Make the PoWR objects
    powr_tables = {x: catalog_spectral.PoWRGrid(x) for x in ('OB', 'WNE', 'WNL')}
    # Make the Sternberg tables object
    sb_tables = catalog_spectral.S03_OBTables()
    catalog_df['ST_obj'].apply(lambda s: s.link_powr_grids(powr_tables))
    # Use a helper function to do this quickly
    def apply_and_unpack(fn_to_apply_to_STobj, name_of_property):
        """
        Helper function to unpack values and uncertainties
        Modifies catalog_df in place
        """
        tmp = catalog_df['ST_obj'].apply(fn_to_apply_to_STobj)
        catalog_df[name_of_property+'_med'] = tmp.apply(lambda x: x[0])
        catalog_df[name_of_property+'_lo'] = tmp.apply(lambda x: x[1][0])
        catalog_df[name_of_property+'_hi'] = tmp.apply(lambda x: x[1][1])
    # Get FUV flux
    apply_and_unpack(lambda s: s.get_FUV_flux(), 'FUV')
    # Get stellar wind mass loss rate
    apply_and_unpack(lambda s: s.get_mass_loss_rate(sb_tables), 'Mdot')
    # Get stellar wind terminal velocity
    apply_and_unpack(lambda s: s.get_terminal_wind_velocity(sb_tables), 'vinf')
    return catalog_df


def filter_by_within_range(catalog_df, radius_arcmin=6.):
    """
    Creates boolean column, true if object is within radius_arcmin
        of the center of the cluster
    I think the center coordinate was taken from SIMBAD.
    Returns the catalog, but modifies in place
    """
    if not isinstance(radius_arcmin, u.Quantity):
        radius_arcmin = radius_arcmin * u.arcmin
    def within_range(coord):
        return coord.separation(catalog_utils.wd2_center_coord) < radius_arcmin
    catalog_df[f'is_within_{radius_arcmin.to(u.arcmin).to_value():.1f}_arcmin'] = catalog_df['SkyCoord'].apply(within_range)
    return catalog_df


def filter_by_only_WR(catalog_df):
    """
    Creates boolean column, true if object is a WR star
    Designed for testing some stellar wind ideas; WRs should domiante stellar
        wind production
    Returns the catatalog, but modifies in place
    """
    def is_WR(spectral_type_string):
        return 'W' in spectral_type_string
    catalog_df['is_WR'] = catalog_df['Spectral'].apply(is_WR)
    return catalog_df


def calc_g0(catalog_df, wcs_obj, distance_los):
    """
    Create an array of G0 in Habing units (average 1-D IRF) given the catalog
        with FUV fluxes and a WCS object for the array
    Also returns lower, upper limits
    :param catalog_df: needs to have "FUV_med", "FUV_lo", "FUV_hi",
        and "SkyCoord" columns
    :param wcs_obj: a WCS object that describes a region around these stars
    :param distance_los: a Quantity or float distance. If float, assumed to
        be in parsecs
    :returns: val, lo, hi as arrays
    """
    # Habing unit
    radfield1d = 1.6e-3 * u.erg / (u.cm*u.cm * u.s)
    # Make distance array function
    def inv_dist_f(coord):
        return 1./(4*np.pi * catalog_utils.distance_from_point_pixelgrid(coord, wcs_obj, distance_los)**2.)
    # Get inverse distance array AND INCLUDE 4PI (I think)
    # If I rewrote distance_from_point_, I could maybe do this with SkyCoord arrays.
    # As it's written now, this needs to be done as DataFrame.apply to each SkyCoord
    # inv_dist will be a 3D array
    inv_dist = u.Quantity(list(catalog_df['SkyCoord'].apply(inv_dist_f).values))
    """
    This could be a good place to look at the effects of distance uncertainties too
    """
    # Make fuv function to do this quickly
    def sum_fuv(fuv_flux_array):
        """
        :param fuv_flux_array: some kind of Quantity in power units,
            should be the same shape as the inv_dist.shape[0]
        """
        fuv_flux_array = u.Quantity(list(fuv_flux_array.values))
        return (np.sum(inv_dist * fuv_flux_array[:, np.newaxis, np.newaxis], axis=0) / radfield1d).decompose()
    # Get the (median) radiation field value
    fuv_med = sum_fuv(catalog_df['FUV_med'])
    # Get the lower and upper limits based on FUV uncertainty
    fuv_lo = sum_fuv(catalog_df['FUV_lo'])
    fuv_hi = sum_fuv(catalog_df['FUV_hi'])
    # Returns val, (lo, hi)
    return fuv_med, fuv_lo, fuv_hi


def calc_mdot(catalog_df):
    """
    Sum up the mass loss rates of all the objects
    :returns: val, lo, hi
    """
    # Make mdot function to do this quickly
    def sum_mdot(mdot_array):
        return np.sum(u.Quantity(list(mdot_array.values)))
    # Get the (median) value
    mdot_med = sum_mdot(catalog_df['Mdot_med'])
    # Get the lower and upper bounds
    mdot_lo = sum_mdot(catalog_df['Mdot_lo'])
    mdot_hi = sum_mdot(catalog_df['Mdot_hi'])
    return mdot_med, mdot_lo, mdot_hi


def calc_KE(catalog_df):
    """
    Sum up mechanical luminosity, calculated by multiplying 0.5 * mdot * vinf**2
    :returns: val, lo, hi
    """
    # Make function to do this quickly
    def sum_ke(suffix):
        mdot = u.Quantity(list(catalog_df['Mdot'+suffix]))
        vinf = u.Quantity(list(catalog_df['vinf'+suffix]))
        ke_over_time = (mdot * vinf**2 / 2.)
        return np.sum(ke_over_time).to(u.erg / u.s)
    suffixes = ('_med', '_lo', '_hi')
    ke_med, ke_lo, ke_hi = (sum_ke(s) for s in suffixes)
    return ke_med, ke_lo, ke_hi


def calc_and_plot_mdot(catalog_df, cii_mom0, cii_w, plotting=False):
    mdot_med, mdot_lo, mdot_hi = calc_mdot(catalog_df.loc[catalog_df['is_within_3.0_arcmin']])
    ke_med, ke_lo, ke_hi = calc_KE(catalog_df.loc[catalog_df['is_within_3.0_arcmin']])
    dl, du = mdot_med - mdot_lo, mdot_hi - mdot_med
    print(f"MDOT: {mdot_med:.1E}, [-{dl.to_value():.2E}, +{du.to_value():.2E}]")
    dl, du = ke_med - ke_lo, ke_hi - ke_med
    print(f"MECH LUM: {ke_med:.1E} [-{dl.to_value():.2E}, +{du.to_value():.2E}]")
    age = 2.e6*u.year
    dl, du = (mdot_med - mdot_lo)*age, (mdot_hi - mdot_med)*age
    print(f"Mass ejected over {age:.1}: {mdot_med*age:.2f} [-{dl.to_value():.2f}, +{du.to_value():.2f}]")
    dl, du = ((ke_med - ke_lo)*age).to(u.erg), ((ke_hi - ke_med)*age).to(u.erg)
    print(f"Thermal energy over {age:.1}: {(ke_med*age).to(u.erg):.2E} [-{dl.to_value():.2E}, +{du.to_value():.2E}]")
    print(f"{(ke_lo*age).to(u.erg):.2E}, {(ke_hi*age).to(u.erg):.2E}")

    if plotting:
        plt.figure(figsize=(11, 8))
        plt.subplot(111, projection=cii_w)
        plt.title(f"[CII] Moment 0 (-8 to -4 km/s); Mdot = {mdot_med:.1E} [-{dl.to_value():.2E}, +{du.to_value():.2E}]")
        plt.imshow(cii_mom0, origin='lower')
        catalog_utils.plot_coordinates(None, SkyCoord(catalog_df.loc[catalog_df['is_within_3.0_arcmin'], 'SkyCoord'].values), setup=False, show=False)
        plt.savefig("figures/mdot_may29-2020.png")


def calc_and_plot_and_save_WR_wind_power(catalog_df, wcs_obj, distance_los, plotting=False, saving=False, radius_arcmin=12.0):
    """
    This is a very hardcoded function serving a very specific purpose
    I want to see what the wind power from only the WR stars looks like across
        the region
    Created: May 22, 2020
    """
    standard_unit = 1e-38 * 1e-6 * (u.solMass / u.yr) / (u.cm*u.cm)
    catalog_df = catalog_df.loc[~catalog_df['is_WR'] & catalog_df[f'is_within_{radius_arcmin:.1f}_arcmin']]
    # Make distance array function (copied from calc_g0)
    def inv_dist_f(coord):
        return 1./(4*np.pi * catalog_utils.distance_from_point_pixelgrid(coord, wcs_obj, distance_los)**2.)
    inv_dist = u.Quantity(list(catalog_df['SkyCoord'].apply(inv_dist_f).values))
    mdot_array = u.Quantity(list(catalog_df['Mdot_hi'].values))
    mdot_total_grid = (np.sum(inv_dist * mdot_array[:, np.newaxis, np.newaxis], axis=0) / standard_unit).decompose()
    header = {"BUNIT": "10^-6 solMass/yr / cm2", "HISTORY": "all stars, mdot_hi"}
    header = fits.Header(header)
    header.update(wcs_obj.to_header())
    if plotting:
        plt.figure()
        plt.imshow(np.log10(mdot_total_grid), origin='lower', vmin=-2, vmax=2)
        plt.title("WR20a and WR20b mass transfer / area")
        plt.colorbar()
        plt.show()
    if saving:
        fits.writeto(f"{data_directory}catalogs/Ramsey/OB_stars_mdot_within_{radius_arcmin:.1f}arcmin.fits", mdot_total_grid.to_value(), header=header, overwrite=True)



def calc_and_plot_g0(catalog_df, cii_mom0, cii_w):
    fuv_med, fuv_lo, fuv_hi = calc_g0(catalog_df.loc[catalog_df['is_within_6.0_arcmin']], cii_w, rcw_dist)
    coords = SkyCoord(catalog_df.loc[catalog_df['is_within_6.0_arcmin'], 'SkyCoord'].values)

    plt.figure(figsize=(13, 15))

    # CII
    plt.subplot(221, projection=cii_w)
    plt.title(f"[CII] Moment 0 map (-8 to -4 km/s)")
    plt.imshow(cii_mom0, origin='lower')
    plt.colorbar()
    catalog_utils.plot_coordinates(None, coords, setup=False, show=False)

    # Median value
    plt.subplot(222, projection=cii_w)
    plt.title("log(G0) in Habing units (median)")
    plt.imshow(np.log10(fuv_med.value), origin='lower', vmin=2.8, vmax=4.7)
    plt.colorbar()
    catalog_utils.plot_coordinates(None, coords, setup=False, show=False)

    # Low value
    dl = (fuv_med - fuv_lo).value
    plt.subplot(223, projection=cii_w)
    plt.title("log(G0) in Habing units (lower uncertainty)")
    plt.imshow(np.log10(dl), origin='lower', vmin=1.8, vmax=3.7)
    plt.colorbar()
    catalog_utils.plot_coordinates(None, coords, setup=False, show=False)

    # High value
    du = (fuv_hi - fuv_med).value
    plt.subplot(224, projection=cii_w)
    plt.title("log(G0) in Habing units (upper uncertainty)")
    plt.imshow(np.log10(du), origin='lower', vmin=1.8, vmax=3.7)
    plt.colorbar()
    catalog_utils.plot_coordinates(None, coords, setup=False, show=False)
    # plt.show()
    plt.tight_layout()
    plt.savefig("figures/g0_may7-2020.png")


def prepare_and_save_catalog(catalog_df):
    """
    Operates on a copy of the catalog, making python objects into more general
        values (SkyCoords to RA,Dec, Quantities to floats)
    """
    catalog_df = catalog_df.copy()
    columns_to_keep = []
    def convert_to_values(prefix, unit_suffix):
        for suffix in ('_lo', '_med', '_hi'):
            new_colname = prefix+suffix+"_"+unit_suffix
            catalog_df[new_colname] = catalog_df[prefix+suffix].apply(lambda x: x.to_value())
            columns_to_keep.append(new_colname)
    # FUV flux
    convert_to_values('FUV', "solLum")
    # Mdot
    convert_to_values('Mdot', "solMass_yr")
    # v_inf
    convert_to_values('vinf', 'km_s')
    # RA, Dec
    catalog_df['RAdeg'] = catalog_df['SkyCoord'].apply(lambda x: x.ra.deg)
    catalog_df['DEdeg'] = catalog_df['SkyCoord'].apply(lambda x: x.dec.deg)
    columns_to_keep = ['RAdeg', 'DEdeg', 'VPHAS_ID', 'VA_ID', 'TFT_ID', 'Spectral'] + columns_to_keep
    cat_path = f"{data_directory}catalogs/Ramsey/"
    catalog_df[columns_to_keep].to_csv(cat_path+"Wd2_OB_catalog_May-7-2020.csv")
    catalog_df.loc[catalog_df['is_within_6.0_arcmin'], columns_to_keep].to_csv(cat_path+"Wd2_within6arcmin_OB_catalog_May-7-2020.csv")
    catalog_df.loc[catalog_df['is_within_3.0_arcmin'], columns_to_keep].to_csv(cat_path+"Wd2_within3arcmin_OB_catalog_May-7-2020.csv")


"""
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
def original_g0_calculation():
    """
    This is the November 2019 version; it had a lot of left-aligned code,
    so it can't really be imported, which is a bummer
    """
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
        fn = f"{data_directory}herschel/RCW49large_3p.fits"
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

    radec_df.to_html("~/Downloads/test.html")
    sys.exit()

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

    args, kwargs = (4.16*1000,), {'pixel_scale': 1*u.arcsec, 'grid_shape':(1500, 1500), 'ref_pixel':(500, 500)}


    def test_compare_wcssep_pixelgrid():
        img_data, img_header, wl = herschel_data(500)
        w = WCS(img_header)
        t0 = datetime.datetime.now()
        grid = catalog_utils.distance_from_point_pixelgrid(test_point, w, *args)
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
        return np.sum(point_coords.apply(lambda x: (.1/catalog_utils.distance_from_point_pixelgrid(x, w, distance_los_pc)**2)), axis=0)

    def calc_g0(cat, w, distance_los_pc):
        radfield1d = 1.6e-3 # erg cm-2 s-1
        cm2_to_pc2 = (u.cm.to(u.pc))**2
        # gives inverse distance in cm-2
        # SHOULD THERE BE A 4PI HERE???
        inv_dist = cat.coords.apply(lambda x: cm2_to_pc2/(catalog_utils.distance_from_point_pixelgrid(x, w, distance_los_pc)**2))
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
    # These are our parameters for WN6ha
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
    # Editorial note (April 29, 2020): it looks like the "grid_param*" columns were for testing purposes
    # They don't help us find the flux, that can be done with "param*" (see below)
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



    ##### I don't remember what this was for
    # w = WCS(herschel_SED_fit()[1])
    # plt.imshow(catalog_utils.distance_from_point_pixelgrid(wd2_center_coord, w, 4.16*1000), origin='lower')
    # plt.show()
    # sys.exit()

    # All stars, real simple
    # make_g0CII_figure()

    # O5 and earlier
    # make_g0hotstars_figure(45, hotter=False)

    # The swarm of ETs
    # make_g0hotstars_figure(36, hotter=False, fracmin=0.2, fracmax=0.45)

    # COMPARE TO DUST
    make_g0_starsVdust_figure()

    # save_g0_fits()

    print(f'edit the end of {__file__} to save a file or make a figure')

if __name__ == "__main__":
    args = main()
