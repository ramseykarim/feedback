import numpy as np
import pandas as pd
import matplotlib.colors as colors

import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1 # Always get all the rows
import regions

import datetime

import matplotlib.pyplot as plt

from . import catalog

from .g0_stars import print_val_err

"""
Query source catalog
Created: Unknown, at least by late 2020

As of June 2022, this is where I'm doing a lot of the M16 radiation field
calculation. Good organization? probably not. works for now? yes

August 3, 2023: Renamed from queries.py to vizier_queries_m16_g0.py because
the old name is not specific or descriptive. Looking into the Stoop 2023 catalog
this month.
"""
# This was for RCW 49
# catalog_data = Catalogs.query_object("10:24:17.509 -57:45:29.28", radius="0.0122628 deg", catalog="HSC")

def m16_stars():
    """
    (Retroactive guess at creation date): August 4 and October 17 2020
    This is to query Hillenbrand 1993 for M16 stars
    I query the Vizier table using astroquery
    Then I convert it to a pandas DataFrame
    Then I run it through CatalogResolver and get the FUV fluxes
    Then I use matplotlib.colors.LogNorm to convert these to marker sizes
    Finally I use astropy regions to write out a DS9 regions file with sizes
        proportional to the FUV flux

    May 31, 2022: time to make the G0 and ionizing flux maps (TODO)
    Looks like I'm a lot further than I realized, I already made the catalog
    """
    hillenbrand = "J/AJ/106/1906"
    catalog_dict = Vizier.find_catalogs(hillenbrand)
    catalogs = Vizier.get_catalogs(catalog_dict[hillenbrand])
    sptype_catalog = catalogs[1] # returns 2 catalogs
    del catalogs, catalog_dict # save memory
    catalog_df = sptype_catalog.to_pandas(index='ID')
    # Convet spectral type bytes to string
    # catalog_df['SpType'] = catalog_df['SpType'].apply(lambda x: x.decode("utf-8"))
    def extract_coordinate(row):
        """
        Helper function for using these byte things
        """
        ra_str = row['RAJ2000'] #.decode("utf-8")
        dec_str = row['DEJ2000'] #.decode("utf-8")
        return SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg), frame='fk5')
    catalog_df['SkyCoord'] = catalog_df.apply(extract_coordinate, axis=1)
    # catalog_df = catalog_df.drop(columns=["RAJ2000", "DEJ2000"])
    catalog_df_OB = catalog_df[catalog_df['SpType'].apply(lambda s: ('O' in s) or ('B' in s))].copy()
    catalog_df_OB['RA'] = catalog_df_OB['SkyCoord'].apply(lambda c: c.ra.deg).values
    catalog_df_OB['DE'] = catalog_df_OB['SkyCoord'].apply(lambda c: c.dec.deg).values
    catalog_df_OB = catalog_df_OB[['RA', 'DE', 'SkyCoord', 'SpType']] # reduce to important columns

    del catalog_df, sptype_catalog # save memory
    catalog.spectral.stresolver.UNCERTAINTY = False  # toggle the half-type/sampling
    # Make the PoWR objects
    powr_tables = {x: catalog.spectral.powr.PoWRGrid(x) for x in ('OB', 'WNL-H50', 'WNL', 'WNE')}
    # Make the Martins tables object
    cal_tables = catalog.spectral.sttable.STTable(*catalog.spectral.martins.load_tables_df())
    # Make Leitherer tables object
    ltables = catalog.spectral.leitherer.LeithererTable()
    # Create CatalogResolver instance
    catr = catalog.spectral.stresolver.CatalogResolver(catalog_df_OB['SpType'].values,
        calibration_table=cal_tables, leitherer_table=ltables, powr_dict=powr_tables)


    # make and write FUV array
    fuv_flux_array = u.Quantity([x[0] for x in catr.get_array_FUV_flux()])
    fuv_flux_unit = fuv_flux_array.unit
    fuv_flux_array = np.array(fuv_flux_array.to_value())
    catalog_df_OB['log10FUV_flux_'+str(fuv_flux_unit).replace(' ', '')] = np.log10(fuv_flux_array)

    if False:
        center_coord = SkyCoord('18:18:35.9543 -13:45:20.364', unit=(u.hourangle, u.deg), frame='fk5')
        filter_radius = 3.90751*u.arcmin
        catalog_df_OB_write = filter_by_within_range(catalog_df_OB, center_coord, radius_arcmin=filter_radius)
        catalog_df_OB_write = catalog_df_OB_write[catalog_df_OB_write['is_within_3.9_arcmin']]
        catalog_df_OB_write = catalog_df_OB_write.sort_values(by=['log10FUV_flux_solLum'], ascending=False)
        catalog_df_OB_write.to_csv(f"{catalog.utils.m16_data_path}catalogs/hillenbrand_stars_sorted_by_FUV.csv")
        print("Stopping here")
        return


    # make some filters for the catalog so we can try different G0 maps
    filter = 0 # filter 4 (>4.5 and filter radius) is the pillar paper one
    filter_stub = ""
    if filter == 0:
        # no filter
        pass
    if filter == 1 or filter == 4:
        # filter log10FUV_flux_solLum > 4.5 (8 stars) or 5.0 (4 stars)
        catalog_df_OB = catalog_df_OB[catalog_df_OB['log10FUV_flux_solLum'] > 4.5]
        filter_stub += "_fuvlt4.5"
    if filter == 2 or filter == 4:
        # filter < x arcmin from y
        center_coord = SkyCoord('18:18:35.9543 -13:45:20.364', unit=(u.hourangle, u.deg), frame='fk5')
        filter_radius = 3.90751*u.arcmin
        catalog_df_OB = filter_by_within_range(catalog_df_OB, center_coord, radius_arcmin=filter_radius)
        catalog_df_OB = catalog_df_OB[catalog_df_OB['is_within_3.9_arcmin']]
        filter_stub += "_ltxarcmin"
    if filter == 3:
        # filter for just the 2 O5 stars
        catalog_df_OB = catalog_df_OB.loc[[175, 205]]
        filter_stub += "_O5s"
    if filter == 5:
        # 175, 205, 222, 246
        catalog_df_OB = catalog_df_OB.loc[[246]]
        filter_stub += '_justone'

    print(catalog_df_OB)
    catr = catalog.spectral.stresolver.CatalogResolver(catalog_df_OB['SpType'].values,
        calibration_table=cal_tables, leitherer_table=ltables, powr_dict=powr_tables)

    if True:
        print(catr)
        for s in catr.star_list:
            print(s.spectral_types)

    if False:
        # Write out G0 map using these stars
        # almost all of this is copied directly from my example in the scoby-nb repo
        m16_center_coord = SkyCoord("18:18:53.9552 -13:50:45.445", unit=(u.hourangle, u.deg), frame='fk5')
        side_length_angle = 5*250*u.arcsec
        side_length_pixels = 100
        pixel_scale_deg = (side_length_angle / side_length_pixels).to(u.deg).to_value()
        wcs_keywords = {
            'NAXIS': (2, "Number of axes"),
            'NAXIS1': (side_length_pixels, "X (j) axis length"), 'NAXIS2': (side_length_pixels, "Y (i) axis length"),
            'RADESYS': 'FK5',
            'CTYPE1': ('RA---TAN', "RA projection type"), 'CTYPE2': ('DEC--TAN', "DEC projection type"),

            'CRPIX1': (side_length_pixels//2, "[pix] Image reference point"),
            'CRPIX2': (side_length_pixels//2, "[pix] Image reference point"),

            'CRVAL1': (m16_center_coord.ra.deg, "[deg] RA of reference point"),
            'CRVAL2': (m16_center_coord.dec.deg, "[deg] DEC of reference point"),

            'CDELT1': -1*pixel_scale_deg, # RA increases to the left, so CDELT1 is negative
            'CDELT2': pixel_scale_deg,

            'PA': (0, "[deg] Position angle of axis 2 (E of N)"),
            'EQUINOX': (2000., "[yr] Equatorial coordinates definition"),
        }
        image_wcs = WCS(wcs_keywords)

        los_distance = 2.*u.kpc
        def inv_dist_func(coord):
            return 1./(4*np.pi*catalog.utils.distance_from_point_pixelgrid(coord, image_wcs, los_distance)**2.)
        inv_distance_arrays = u.Quantity(list(catalog_df_OB['SkyCoord'].apply(inv_dist_func).values))
        print("Shape of inv dist arrays: ", inv_distance_arrays.shape)
        def illumination_distance(flux_array):
            return inv_distance_arrays*flux_array[:, np.newaxis, np.newaxis]
        fuv_flux_map_val, fuv_flux_map_unc = catr.get_FUV_flux(map_function=illumination_distance)

        radiation_field_1d = 1.6e-3 * u.erg / (u.cm**2 * u.s)
        def fix_units(x):
            return (x / radiation_field_1d).decompose()
        fuv_flux_map_val = fix_units(fuv_flux_map_val)
        fuv_flux_map_unc = tuple(fix_units(x) for x in fuv_flux_map_unc)

        plt.subplot(221)
        plt.imshow(fuv_flux_map_val.to_value(), origin='lower')
        plt.colorbar()
        plt.subplot(223)
        plt.imshow(np.abs(fuv_flux_map_unc[0].to_value()), origin='lower')
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(fuv_flux_map_unc[1].to_value(), origin='lower')
        plt.colorbar()
        plt.show()

        hdr = image_wcs.to_header()
        hdr['AUTHOR'] = "Ramsey Karim"
        hdr['CREATOR'] = "queries.m16_stars"
        hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        hdr['BUNIT'] = 'Habing unit'
        hdr['COMMENT'] = 'G0 map of M16 around pillars'
        hdr['EXTNAME'] = 'median'
        hdu1 = fits.ImageHDU(data=fuv_flux_map_val.to_value(), header=hdr.copy())
        hdr['EXTNAME'] = 'err_lo'
        hdu2 = fits.ImageHDU(data=np.abs(fuv_flux_map_unc[0].to_value()), header=hdr.copy())
        hdr['EXTNAME'] = 'err_hi'
        hdu3 = fits.ImageHDU(data=fuv_flux_map_unc[1].to_value(), header=hdr.copy())
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2, hdu3])
        hdul.writeto(f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/g0_hillenbrand_stars{filter_stub}.fits", overwrite=True)



    if False:
        # Use LogNorm to normalize
        min_nonzero_flux = np.min(fuv_flux_array[fuv_flux_array > 0])
        max_flux = np.max(fuv_flux_array)
        norm = colors.Normalize(vmin=min_nonzero_flux, vmax=max_flux)
        brightness_array = norm(fuv_flux_array)

        # I want angular sizes between 0.1 - 1 arcmin
        size_array = [x * u.arcmin for x in (brightness_array.data * 0.9 + 0.1)]
        catalog_df_OB['markersize'] = size_array
        visual_kwargs = regions.RegionVisual(color='white', linewidth=1)
        region_list = []
        for star in catalog_df_OB.itertuples():
            if star.markersize < 0:
                continue
            circle = regions.CircleSkyRegion(center=star.SkyCoord, radius=star.markersize, visual=visual_kwargs)
            region_list.append(circle)
        regions.write_ds9(region_list, f"{catalog.utils.m16_data_path}catalogs/hillenbrand_stars.reg")
        return None

    if True:
        """
        July 6, 2022
        Copied from g0_stars.py (RCW 49)
        I'm gonna estimate the momentum transfer to the gas / threads
        """
        print(catr)
        mdot_med, mdot_err = catr.get_mass_loss_rate()
        mvflux_med, mvflux_err = catr.get_momentum_flux()
        ke_med, ke_err = catr.get_mechanical_luminosity()
        fuv_tot_med, fuv_tot_err = catr.get_FUV_flux()
        ionizing_tot_med, ionizing_tot_err = catr.get_ionizing_flux()
        print(f"MASS LOSS: {print_val_err(mdot_med, mdot_err)}")
        print(f"MV FLUX:  {print_val_err(mvflux_med, mvflux_err)}")
        print(f"MECH LUM:  {print_val_err(ke_med, ke_err, extra_f=lambda x: x.to(u.erg/u.s))}")
        print(f"FUV LUM:   {print_val_err(fuv_tot_med, fuv_tot_err)}") # extra_f=lambda x: x.to(u.erg/u.s)
        print(f"IONIZING PHOTON FLUX: {print_val_err(ionizing_tot_med, ionizing_tot_err)}") # units should be 1/time
        mass_med, mass_err = catr.get_stellar_mass()
        lum_med, lum_err = catr.get_bolometric_luminosity()
        print(f"STELLAR MASS: {print_val_err(mass_med, mass_err)}")
        print(f"LUMINOSITY:   {print_val_err(lum_med, lum_err)}")
        print(f"MECH/FUV LUM: {(ke_med/fuv_tot_med).decompose():.1E}; MECH/total LUM: {(ke_med/lum_med).decompose():.1E}")

        result_dict = {
            'mdot': (mdot_med, mdot_err), 'mvflux': (mvflux_med, mvflux_err),
            'ke': (ke_med, ke_err), 'fuv_tot': (fuv_tot_med, fuv_tot_err),
            'ionizing_tot': (ionizing_tot_med, ionizing_tot_err),
            'mass': (mass_med, mass_err), 'lum': (lum_med, lum_err),
        }
        return result_dict


def filter_by_within_range(catalog_df, center_coord, radius_arcmin=6.):
    """
    Helper function copied out of g0_stars.py and modified lightly
    Creates boolean column, true if object is within radius_arcmin
        of the center of the cluster
    center_coord must be SkyCoord
    Returns the catalog, but modifies in place
    Catalog must be able to be modified
    """
    if not isinstance(radius_arcmin, u.Quantity):
        radius_arcmin = radius_arcmin * u.arcmin
    def within_range(coord):
        return coord.separation(center_coord) < radius_arcmin
    catalog_df[f'is_within_{radius_arcmin.to(u.arcmin).to_value():.1f}_arcmin'] = catalog_df['SkyCoord'].apply(within_range)
    return catalog_df


def cygx_wright2015():
    """
    Grab the Cyg OB2 stars from Wright et al 2015 (for Kim's paper)
    There are some mismatches between the Berlanas paper and the Wright paper,
    so I want to compare them directly
    Written: November 5, 2020
    """
    wright2015 = "J/MNRAS/449/741"
    catalog_dict = Vizier.find_catalogs(wright2015)
    catalogs = Vizier.get_catalogs(catalog_dict[wright2015])
    sptype_catalog = catalogs[0] # "census" catalog; second is "ref" with references
    del catalogs, catalog_dict
    catalog_df = sptype_catalog.to_pandas()
    catalog_df.index = np.arange(1, len(catalog_df)+1)
    catalog_df['SpType'] = catalog_df['SpType'].apply(lambda x: x.decode("utf-8"))
    catalog_df['SimbadName'] = catalog_df['SimbadName'].apply(lambda x: x.decode("utf-8"))
    catalog_df['RAJ2000'] = catalog_df['RAJ2000'].apply(lambda x: x.decode("utf-8"))
    catalog_df['DEJ2000'] = catalog_df['DEJ2000'].apply(lambda x: x.decode("utf-8"))
    catalog_df.rename(columns={'logL':"LogL", 'SimbadName':"Name"}, inplace=True)
    catalog_df.set_index('Name', inplace=True)
    def extract_coordinate(row):
        """
        Helper function for using these byte things
        """
        ra_str = row['RAJ2000']
        dec_str = row['DEJ2000']
        return SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg), frame='fk5')
    catalog_df['SkyCoord'] = catalog_df.apply(extract_coordinate, axis=1)
    cygx_dir = "/home/ramsey/Documents/Research/Feedback/cygx_data/catalogs/"
    catalog_df[["RAJ2000", "DEJ2000", 'SpType', 'LogL', 'logT']].to_csv(cygx_dir + 'wright.csv')
    catalog_df = catalog_df.drop(columns=["RAJ2000", "DEJ2000"])
    catalog_df = catalog_df[['SkyCoord', 'SpType', 'LogL', 'logT']]
    catalog_df['Teff'] = 10.**catalog_df['logT']
    catalog_df = catalog_df.drop(columns=["logT"])
    catalog_df['origin'] = 'W15'
    catalog_df.to_pickle(cygx_dir + "wright.pkl")


if __name__ == "__main__":
    m16_stars()

"""
Query atomic lines
"""
# from astroquery.atomic import AtomicLineList

# candidates = "S II\nCl II\nP II\nC II\nO I\nO II\nH I\nH II\nHe I\nN I"
# weirdline1 = [7316.47*u.Angstrom, 7323.97*u.Angstrom]  # Likely SII 7319.15 A
# weirdline2 = [7326.47*u.Angstrom, 7333.97*u.Angstrom]  # Maybe SII? Maybe OII or PII?
# weirdline3 = [8577.72*u.Angstrom, 8580.22*u.Angstrom]  # ClII 8579.74 seems most likely
# line_tbl = AtomicLineList.query_object(wavelength_range=weirdline1, wavelength_type='Vacuum',
#     element_spectrum=candidates,
#     depl_factor=1)
