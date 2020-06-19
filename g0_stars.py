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

from . import misc_utils
from . import catalog

data_directory = catalog.utils.ancillary_data_path

rcw_dist = 4.16*u.kpc


"""
Finish preparing the catalog with physical properties
This step uses catalog_spectral.py functions
"""
def main():
    catalog_df = catalog.parse.load_final_catalog_df()
    catalog_df = convert_ST_to_properties(catalog_df)

    # catalog_df = filter_by_within_range(catalog_df)
    catalog_df = filter_by_within_range(catalog_df, radius_arcmin=3.)
    catalog_df = filter_by_within_range(catalog_df, radius_arcmin=6.)
    catalog_df = filter_by_within_range(catalog_df, radius_arcmin=12.)


    catalog.utils.save_df_html(catalog_df)

    # Option here to copy the improved make_wcs and make_wcs_like functions from mosaic_vlt.py
    # We can just use CII's WCS in the meantime

    # print(len(catalog_df.loc[catalog_df['is_within_12.0_arcmin']]))
    # print(len(catalog_df.loc[catalog_df['is_within_6.0_arcmin']]))
    # print(len(catalog_df.loc[catalog_df['is_within_3.0_arcmin']]))

    cii_mom0, cii_w = catalog.utils.load_cii(2)
    # calc_and_plot_g0(catalog_df, cii_mom0, cii_w)
    # calc_and_plot_mdot(catalog_df, cii_mom0, cii_w)

    ## Experiment to see what WR stars mass loss looks like (May 22, 2020)
    catalog_df = filter_by_only_WR(catalog_df)
    catalog_df = catalog_df.loc[~catalog_df['is_WR']]
    # calc_and_plot_and_save_WR_wind_power(catalog_df, cii_w, rcw_dist, saving=True, radius_arcmin=6.)
    calc_and_plot_mdot(catalog_df, cii_mom0, cii_w)
    return


def convert_catalog_to_CatalogResolver(catalog_df):
    # Make the PoWR objects
    powr_tables = {x: catalog.spectral.powr.PoWRGrid(x) for x in ('OB', 'WNE', 'WNL')}
    # Make the Martins tables object
    cal_tables = catalog.spectral.sttable.STTable(*catalog.spectral.martins.load_tables_df())
    # Make Leitherer tables object
    ltables = catalog.spectral.leitherer.LeithererTable()
    # Create CatalogResolver instance
    catr = catalog.spectral.stresolver.CatalogResolver(catalog_df['Spectral'].values,
        calibration_table=cal_tables, leitherer_table=ltables, powr_dict=powr_tables)
    return catr


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
        return coord.separation(catalog.utils.wd2_center_coord) < radius_arcmin
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


def calc_g0(catr, wcs_obj, distance_los):
    """
    ########################################################
    ########################################################
    ########################################################
    ########################################################
                I left off editing this!!!!!!!
        I should go back, check my work on CatalogResolver,
        update the "finished this" dates eventually (git push)

        On this function, I need to figure out how to do the
        uncertainty. Though not a priority; can just do what
        was doing before since the values are more time
        sensitive than the uncertainties.
        go go go go go!
    ########################################################
    ########################################################
    ########################################################
    ########################################################
    Create an array of G0 in Habing units (average 1-D IRF) given the catalog
        with FUV fluxes and a WCS object for the array
    Also returns uncertainty (lower_bar, upper_bar)
    :param catalog_df: needs to have "SkyCoord" column
    :param catr: CatalogResolver object
    :param wcs_obj: a WCS object that describes a region around these stars
    :param distance_los: a Quantity or float distance. If float, assumed to
        be in parsecs
    :returns: val, lo, hi as arrays
    """
    # Habing unit
    radfield1d = 1.6e-3 * u.erg / (u.cm*u.cm * u.s)
    # Make distance array function
    def inv_dist_f(coord):
        return 1./(4*np.pi * catalog.utils.distance_from_point_pixelgrid(coord, wcs_obj, distance_los)**2.)
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
        return 1./(4*np.pi * catalog.utils.distance_from_point_pixelgrid(coord, wcs_obj, distance_los)**2.)
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



if __name__ == "__main__":
    args = main()
