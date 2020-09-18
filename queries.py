import numpy as np
import pandas as pd
import matplotlib.colors as colors

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
import regions

from . import catalog

"""
Query source catalog
"""
# This was for RCW 49
# catalog_data = Catalogs.query_object("10:24:17.509 -57:45:29.28", radius="0.0122628 deg", catalog="HSC")

def m16_stars():
    """
    This is to query Hillenbrand 1993 for M16 stars
    I query the Vizier table using astroquery
    Then I convert it to a pandas DataFrame
    Then I run it through CatalogResolver and get the FUV fluxes
    Then I use matplotlib.colors.LogNorm to convert these to marker sizes
    Finally I use astropy regions to write out a DS9 regions file with sizes
        proportional to the FUV flux
    """
    hillenbrand = "J/AJ/106/1906"
    catalog_list = Vizier.find_catalogs(hillenbrand)
    catalogs = Vizier.get_catalogs(catalog_list[hillenbrand])
    sptype_catalog = catalogs[1] # returns 2 catalogs
    del catalogs, catalog_list # save memory
    catalog_df = sptype_catalog.to_pandas(index='ID')
    # Convet spectral type bytes to string
    catalog_df['SpType'] = catalog_df['SpType'].apply(lambda x: x.decode("utf-8"))
    def extract_coordinate(row):
        """
        Helper function for using these byte things
        """
        ra_str = row['RAJ2000'].decode("utf-8")
        dec_str = row['DEJ2000'].decode("utf-8")
        return SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg), frame='fk5')
    catalog_df['SkyCoord'] = catalog_df.apply(extract_coordinate, axis=1)
    catalog_df = catalog_df.drop(columns=["RAJ2000", "DEJ2000"])
    catalog_df_OB = catalog_df[catalog_df['SpType'].apply(lambda s: ('O' in s) or ('B' in s))]
    catalog_df_OB = catalog_df_OB[['SkyCoord', 'SpType']] # reduce to important columns
    del catalog_df, sptype_catalog # save memory
    # Make the PoWR objects
    powr_tables = {x: catalog.spectral.powr.PoWRGrid(x) for x in ('OB', 'WNL-H50', 'WNL', 'WNE')}
    # Make the Martins tables object
    cal_tables = catalog.spectral.sttable.STTable(*catalog.spectral.martins.load_tables_df())
    # Make Leitherer tables object
    ltables = catalog.spectral.leitherer.LeithererTable()
    # Create CatalogResolver instance
    catr = catalog.spectral.stresolver.CatalogResolver(catalog_df_OB['SpType'].values,
        calibration_table=cal_tables, leitherer_table=ltables, powr_dict=powr_tables)
    fuv_flux_array = [x[0] for x in catr.get_array_FUV_flux()]
    catalog_df_OB['FUV_flux'] = fuv_flux_array
    fuv_flux_array = np.array(u.Quantity(fuv_flux_array).to_value())

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

args = m16_stars()

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
