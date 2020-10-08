import numpy as np
import pandas as pd
import os
import glob

import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 6}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

from . import catalog
from .g0_stars import calc_g0
from .mosaic_vlt import make_wcs

data_dir = os.path.abspath(os.path.join(catalog.utils.feedback_path, "cygx_data/catalogs"))
berlanas_table_filenames = glob.glob(os.path.join(data_dir, "berlanas*"))
berlanas_table_filenames = {
    'newstars': [x for x in berlanas_table_filenames if 'A1' in x].pop(),
    'oldstars': [x for x in berlanas_table_filenames if 'A2' in x].pop(),
}
mytable = os.path.join(data_dir, "cyg-ob2_all.pkl")
mytable_html = os.path.join(data_dir, "cyg-ob2_all.html")

# Helper function for "Sample" column in both tables
def find_cygob2(s):
    return s.strip() == 'Cygnus OB2'


def open_and_reduce_newstars():
    """
    Open Berlanas 2018 Table A1, new stars from that work
    Return a DataFrame with (at least) columns:
        coords: SkyCoord
        Teff: float, K
        LogL: float, log L/Lsun
    """
    # Load everything as string first, since there's junk in there
    newstars = pd.read_csv(berlanas_table_filenames['newstars'], sep=';', comment='#', dtype=str)
    # Drop the unit and hyphen rows
    newstars.drop(index=[0, 1], inplace=True)
    # We won't use the Av column
    newstars.drop(columns=['AV'], inplace=True)
    # Convert these columns from string to float
    for c in ['Teff', 'LogL', '_RA', '_DE']:
        newstars[c] = newstars[c].astype(float)
    # Remove whitespace from SpType
    newstars['SpType'] = newstars['SpType'].apply(lambda x: x.strip())

    # Reduce the table to only the Cyg OB2 entries
    newstars = newstars.loc[newstars['Sample'].apply(find_cygob2)]
    # Drop that column, since they're all the same now
    newstars.drop(columns=['Sample'], inplace=True)

    # Make SkyCoords from the RA and DEC
    def make_skycoords(row):
        return SkyCoord(row['_RA'], row['_DE'], unit=u.deg, frame='fk5')
    newstars['SkyCoord'] = newstars.apply(make_skycoords, axis=1)
    # Drop the RA, DEC columns
    newstars.drop(columns=['_RA', '_DE'], inplace=True)

    # Set the index column to Name, making sure they're unique (should be)
    newstars.set_index('Name', inplace=True, verify_integrity=True)

    # Make a provenance column to keep track of the old and new stars
    newstars['origin'] = 'A1-new'

    return newstars


def open_and_reduce_oldstars():
    """
    Open Berlanas 2018 Table A2, known stars from Comeron & Pasquali 2012
    Only stars from "Cygnus OB2" are included
    Return a DataFrame with (at least) columns:
        coords: SkyCoord
        Teff: float, K
        LogL: float, log L/Lsun
    and Name (usually 2MASS) as the index
    """
    # Load everything as string first, since there's junk in there
    oldstars = pd.read_csv(berlanas_table_filenames['oldstars'], sep=';', comment='#', dtype=str)
    # Drop the unit and hyphen rows
    oldstars.drop(index=[0, 1], inplace=True)

    # Convert these columns from string to float
    for c in ['Teff', 'LogL']:
        oldstars[c] = oldstars[c].astype(float)
    # Remove whitespace from SpType
    oldstars['SpType'] = oldstars['SpType'].apply(lambda x: x.strip())

    # Reduce the table to only the Cyg OB2 entries
    oldstars = oldstars.loc[oldstars['Sample'].apply(find_cygob2)]
    # Drop that column, since they're all the same now
    oldstars.drop(columns=['Sample'], inplace=True)

    # Make SkyCoords from the RA and DEC
    def make_skycoords(row):
        return SkyCoord(row['RAJ2000'], row['DEJ2000'], unit=(u.hourangle, u.deg), frame='fk5')
    oldstars['SkyCoord'] = oldstars.apply(make_skycoords, axis=1)
    # Drop the RA, DEC columns
    oldstars.drop(columns=['RAJ2000', 'DEJ2000'], inplace=True)

    # Set the index column to Name, making sure they're unique (should be)
    oldstars.set_index('Name', inplace=True, verify_integrity=True)

    # Make a provenance column to keep track of the old and new stars
    oldstars['origin'] = 'A2-old'

    return oldstars


def combine_tables():
    """
    Combine tables A1 and A2 from Berlanas 2018 and ensure they don't contain
    duplicates. They don't, I checked, nor should they.
    Save the result as a pickle file. That's the "mytable" variable up top.

    There are ~20 from the "new" table and ~100 from the "old" table
    """
    raise RuntimeError("I ran this already on September 25, 2020")
    newstars = open_and_reduce_newstars()
    oldstars = open_and_reduce_oldstars()

    assert newstars.index.intersection(oldstars.index).empty
    allstars = pd.concat([newstars, oldstars])

    allstars.to_pickle(os.path.join(data_dir, "cyg-ob2_all_NEW.pkl"))

def calc_fluxes(df, plot=False):
    """
    df = pd.read_pickle(mytable)
    Return a CatalogResolver object with correctly linked PoWR tables
    """
    coords = SkyCoord(df.SkyCoord.values)
    catalog.spectral.stresolver.UNCERTAINTY = False

    # Make the PoWR objects
    powr_tables = {x: catalog.spectral.powr.PoWRGrid(x) for x in ('OB', 'WNL-H50', 'WNL', 'WNE')}
    # Make the Martins tables object
    cal_tables = catalog.spectral.sttable.STTable(*catalog.spectral.martins.load_tables_df())
    # Make Leitherer tables object
    ltables = catalog.spectral.leitherer.LeithererTable()
    # Create CatalogResolver instance
    catr = catalog.spectral.stresolver.CatalogResolver(df['SpType'].values,
        calibration_table=cal_tables, leitherer_table=ltables, powr_dict=powr_tables)

    if plot:
        # save these for plotting
        fuv_flux_1, uncertainty = zip(*catr.get_array_FUV_flux())
        fuv_flux_1 = catalog.spectral.stresolver.u.Quantity(fuv_flux_1)

    # Get the T and L values from the DataFrame
    T = df.Teff.values
    L = df.LogL.values
    # Organize into a list of tuples
    TLs = list(zip(T, L))
    # Use that nifty kwargs functionality in CatalogResolver
    catr.link_powr_grids(powr_tables, listof_TL_pair=TLs)
    catr.populate_FUV_flux()
    if not plot:
        return catr
    """
    The rest is just plotting!
    """

    fuv_flux_2, uncertainty = zip(*catr.get_array_FUV_flux())
    fuv_flux_2 = catalog.spectral.stresolver.u.Quantity(fuv_flux_2)

    xf = np.log10(fuv_flux_1.to_value())
    yf = np.log10(fuv_flux_2.to_value())
    # plt.scatter(xf, yf, marker='o')
    xvals = [3, 7]
    yvals = [x - 0.3 for x in xvals]
    # plt.plot(xvals, yvals, '--')

    mask = (yf < xf-0.3)
    xf_f = xf[mask]
    yf_f = yf[mask]

    # Plot L vs T (HR diagram)
    fig1, ax1 = plt.subplots()
    ax1.scatter(T/1000, xf, color='r', marker='o', alpha=0.2, label="(FUV) from spectral types")
    ax1.scatter(T/1000, yf, color='b', marker='x', alpha=0.2, label="(FUV) from T and L from tables")
    xy_pairs = np.array(list(zip(xf, yf)))
    t_pairs = np.array(list(zip(T, T)))/1000
    for i in range(xy_pairs.shape[0]):
        ax1.plot(t_pairs[i], xy_pairs[i], '-', linewidth=0.7, alpha=0.3, color='k')
    ax1.scatter(T/1000, L, color='g', marker='o', alpha=0.2, label='Total L')
    ax1.invert_xaxis()
    ax1.set_xlabel("T (kK)")
    ax1.set_ylabel("log L (L/Lsun)")
    ax1.legend()

    # Plot FUV L vs total L (and x=y line for reference)
    fig2, ax2 = plt.subplots()
    ax2.scatter(L, xf, color='r', marker='o', alpha=0.2, label='from spectral types')
    ax2.scatter(L, yf, color='b', marker='x', alpha=0.2, label='from T and L from tables')
    ax2.plot(xvals, xvals)
    ax2.set_xlabel("log L (L/Lsun)")
    ax2.set_ylabel("log FUV L (L/Lsun)")

    plt.show()
    """
    These plots confirm:
    1) there is some resemblance to an HR diagram, that's good
    2) the NEW FUV L is always LESS THAN the total L. That's important!
        The old one isn't, but that's because we didn't have that information.
    """
    return catr


def create_cygx_wcs(df):
    coords = SkyCoord(df.SkyCoord.values)
    minRA = coords.ra.min()
    maxRA = coords.ra.max()
    minDE = coords.dec.min()
    maxDE = coords.dec.max()
    crval = SkyCoord((minRA + maxRA)/2, (minDE + maxDE)/2, frame=coords.frame)
    grid_shape = (150, 100)
    crpix = (94, 70)
    hdr = make_wcs(ref_coord=crval, ref_pixel=crpix, grid_shape=grid_shape,
        pixel_scale=2*u.arcmin, return_header=True, CREATOR="Ramsey Karim",
        COMMENT="First draft G0 map of Cygnus OB2", DATE='Sept 25, 2020')
    return WCS(hdr), hdr


def make_g0_plot(df, catr, wcs_obj):
    val, uncertainty = calc_g0(df, catr, wcs_obj, 1.4*u.kpc)
    try:
        print(type(val))
        print(val.unit)
        val = val.to_value()
    except:
        print("!!", type(val))
    fig = plt.figure(figsize=(8, 10))
    ax = plt.subplot(111, projection=wcs_obj)
    val_log = np.log10(val)
    im = ax.imshow(val_log, origin='lower', vmin=1, vmax=5, cmap='nipy_spectral')
    c = ax.contour(val, levels=[50, 100, 500, 1000, 5000], linewidths=0.5, colors='k')
    ax.clabel(c, [50, 100, 500, 1000, 5000], inline=True, fontsize=6, fmt='%d')
    fig.colorbar(im, ax=ax)
    ax.set_title("G0 around Cyg OB2 $-$ ROUGH DRAFT")
    fig.savefig(os.path.join(data_dir, "cyg-ob2_g0_2020-09-25.png"))


df = pd.read_pickle(mytable)

# df.drop(columns=['SkyCoord'], inplace=True)
# df['Teff'] = df['Teff']/1000
# df.sort_values(by=['Teff'], inplace=True, ascending=False)
# # df = df.loc[df['Teff'] > 40]
# df.to_html(mytable_html, na_rep="")


catr = calc_fluxes(df)
wcs_obj, hdr = create_cygx_wcs(df)
make_g0_plot(df, catr, wcs_obj)
