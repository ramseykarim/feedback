import numpy as np
import pandas as pd
import os
import glob

import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 12}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

from . import catalog
from .g0_stars import calc_g0
from .mosaic_vlt import make_wcs

distance_los = 1.5*u.kpc

data_dir = os.path.abspath(os.path.join(catalog.utils.feedback_path, "cygx_data/catalogs"))
berlanas_table_filenames = glob.glob(os.path.join(data_dir, "berlanas*"))
berlanas_table_filenames = {
    'newstars': [x for x in berlanas_table_filenames if 'A1' in x].pop(),
    'oldstars': [x for x in berlanas_table_filenames if 'A2' in x].pop(),
}
berlanas_table = os.path.join(data_dir, "berlanas.pkl")
wright_table = os.path.join(data_dir, "wright.pkl")
mytable = os.path.join(data_dir, "cyg-ob2_all.pkl")
mytable_csv = os.path.join(data_dir, "cyg-ob2_all.csv")
mytable_html = os.path.join(data_dir, "cyg-ob2_all.html")
header_file = os.path.join(data_dir, "kim_fits_header.txt")

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
    powr_tables = {x: catalog.spectral.powr.PoWRGrid(x) for x in ('OB', 'WNL-H50', 'WNL', 'WNE', 'WC')}
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
        Q_1, uncertainty = zip(*catr.get_array_ionizing_flux())
        Q_1 = catalog.spectral.stresolver.u.Quantity(Q_1)

    # Get the T and L values from the DataFrame
    T = df.Teff.values
    L = df.LogL.values
    # Organize into a list of tuples
    TLs = list(zip(T, L))

    # Use that nifty kwargs functionality in CatalogResolver
    catr.link_powr_grids(powr_tables, listof_TL_pair=TLs)
    catr.populate_FUV_flux()
    catr.populate_ionizing_flux()
    if not plot:
        return catr
    """
    The rest is just plotting!
    """

    fuv_flux_2, uncertainty = zip(*catr.get_array_FUV_flux())
    fuv_flux_2 = catalog.spectral.stresolver.u.Quantity(fuv_flux_2)
    Q_2, uncertainty = zip(*catr.get_array_ionizing_flux())
    Q_2 = catalog.spectral.stresolver.u.Quantity(Q_2)

    xf = np.log10(fuv_flux_1.to_value())
    yf = np.log10(fuv_flux_2.to_value())
    # plt.scatter(xf, yf, marker='o')
    xvals = [3, 7]
    yvals = [x - 0.3 for x in xvals]
    # plt.plot(xvals, yvals, '--')

    mask = (yf < xf-0.3)
    xf_f = xf[mask]
    yf_f = yf[mask]

    """
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
    """

    def lookup_Q(star):
        # To operate on STResolver
        def find_Q(st_tuple, model_info):
            Q_unit = 1/u.s
            if model_info is None:
                Q = np.nan
            elif star.isWR(st_tuple):
                Q = np.nan # for now
            else:
                Q = 10**(star.calibration_table.lookup_characteristic('Qo', st_tuple))
            return Q*Q_unit
        return star.resolve_uncertainty(star.map_to_components(find_Q, (star.spectral_types, star.powr_models)), nsamples=3)[0]
    Q_martins_array = u.Quantity(catr.map(lookup_Q))

    fig3, axes = plt.subplots(nrows=2, sharex=True)
    ax3, ax3b = axes
    xq = np.log10(Q_1.to_value())
    yq = np.log10(Q_2.to_value())
    print(yq)

    zq = np.log10(Q_martins_array.to_value())
    ax3.scatter(T/1000, xq, color='r', marker='o', alpha=0.2, label='from spectral types')
    ax3.scatter(T/1000, yq, color='b', marker='x', alpha=0.2, label='from T and L')
    ax3.scatter(T/1000, zq, color='g', marker='+', alpha=0.2, label='M05')
    ax3.legend()
    ax3.set_ylabel("Q")
    ax3.invert_xaxis()
    ax3.tick_params('x', labelbottom=False)

    ax3b.scatter(T/1000, xq-yq, color='k', marker='o', alpha=0.2, label='ST to TL')
    ax3b.scatter(T/1000, zq-yq, color='b', marker='x', alpha=0.2, label='M05 to TL')
    ax3b.scatter(T/1000, zq-xq, color='r', marker='+', alpha=0.2, label='M05 to ST')
    ax3b.legend()
    ax3b.set_xlabel("T (kK)")
    ax3b.set_ylabel("ratio (dex)")

    def get_st_number(star):
        # This function to operate on a STResolver object
        # First, map st_to_number onto all the spectral type tuples
        # Then reduce them, taking means instead of sums, and select only the value, not the error
        # Take a small number of samples
        return star.resolve_uncertainty(star.map_to_components(catalog.spectral.parse_sptype.st_to_number, (star.spectral_types,)), dont_add=True, nsamples=3)[0]
    st_numbers = u.Quantity(catr.map(get_st_number)).to_value()

    def lookup_T(star):
        # To operate on STResolver
        def find_T(st_tuple, model_info):
            if model_info is None:
                T = np.nan
            elif star.isWR(st_tuple):
                T = np.nan # for now
            else:
                T = star.calibration_table.lookup_characteristic('Teff', st_tuple)
            return T
        return star.resolve_uncertainty(star.map_to_components(find_T, (star.spectral_types, star.powr_models)), nsamples=3)[0]
    T_martins_array = u.Quantity(catr.map(lookup_T))

    fig4, axes = plt.subplots(nrows=2, sharex=True)
    ax4, ax4b = axes
    ax4.scatter(T/1000, T_martins_array/1000, color='k', marker='o', alpha=0.2)
    ax4.set_xlabel("T [B18] (K)")
    ax4.set_ylabel("T [M05] (K)")
    ax4.plot([0, 40], [0, 40], '--', color='k', alpha=0.3)
    ax4b.scatter(T/1000, st_numbers, color='k', marker='o', alpha=0.2)
    ax4b.set_xlabel("T [B18] (K)")
    ax4b.set_ylabel("Spectral type")
    ax4.invert_xaxis()
    ax4.invert_yaxis()

    # plt.show()
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


def load_cygx_wcs():
    """
    Load from the txt file that Kim sent
    """
    hdr = fits.Header.fromfile(header_file, sep='\n', endcard=False, padding=False)
    additional_kws = dict(COMMENT="Ionizing photon flux map around Cygnus OB2",
        DATE='Oct 7, 2020', CREATOR="Ramsey Karim", BUNIT="cm-2 s-1")
    wcs_obj = WCS(hdr)
    new_hdr = wcs_obj.to_header()
    new_hdr.update(additional_kws)
    return wcs_obj, new_hdr



def make_g0_plot(df, catr, wcs_obj, **kwargs):
    if 'ax' not in kwargs:
        fig = plt.figure(figsize=(8, 10))
        ax = plt.subplot(111, projection=wcs_obj)
    else:
        fig = kwargs.pop('fig')
        ax = kwargs.pop('ax')
    label = kwargs.pop('label', "")
    if 'fuv_or_ionizing' in kwargs and kwargs['fuv_or_ionizing'] == 'ionizing':
        vlims_kw = dict(vmin=8.5, vmax=13.5)
        titlestub = "$q_0$"
        savestub = "q0"
    else:
        vlims_kw = dict(vmin=1, vmax=5)
        titlestub = "$G_0$"
        savestub = "g0"
    val, uncertainty = calc_g0(df, catr, wcs_obj, distance_los, **kwargs)
    try:
        val = val.to_value()
    except:
        print("Val has no .to_value method because its type is ", type(val))
    try:
        uncertainty = u.Quantity(uncertainty)
    except:
        print(f"uncertainty is {type(uncertainty)} but that\'s fine")
    val_log = np.log10(val)
    im = ax.imshow(val_log, origin='lower', cmap='nipy_spectral', **vlims_kw)
    try:
        c = ax.contour(val, levels=[50, 100, 500, 1000, 5000], linewidths=0.5, colors='k')
        ax.clabel(c, [50, 100, 500, 1000, 5000], inline=True, fontsize=6, fmt='%d')
    except Exception as e:
        print("Problem with countours: ", e)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{titlestub} around Cyg OB2 $-${label}")
    fig.savefig(os.path.join(data_dir, f"cyg-ob2_{savestub}_2020-11-06.png"))
    # plt.show()
    return val, uncertainty


def save_g0_fits(array, hdr, **kwargs):
    """
    Already built array and header
    """
    if 'fuv_or_ionizing' in kwargs and kwargs['fuv_or_ionizing'] == 'ionizing':
        savestub = "q0"
    else:
        savestub = "g0"
    hdu = fits.PrimaryHDU(data=array, header=hdr)
    hdu.writeto(os.path.join(data_dir, f"cyg-ob2-{savestub}_2020-11-06.fits"), overwrite=True)
    print("Done")



def crossmatch_wright_and_berlanas(wright_df, berlanas_df):
    """
    Create a combined dataframe from Wright 2015 and Berlanas 2018,2020
    Copied heavily from build_catalog() in catalog.parse
    """
    # First, reset the index so we can see the 0-indexed numerical indices
    wright_df = wright_df.reset_index()
    wright_df['Name'] = wright_df['Name'].apply(lambda s: s.replace('2MASS ', ''))
    berlanas_df = berlanas_df.reset_index()
    # Now make coordinate arrays
    wright_coords = SkyCoord(wright_df['SkyCoord'].values)
    berlanas_coords = SkyCoord(berlanas_df['SkyCoord'].values)
    # Crossmatch each to the other
    # W.match_to_catalog_sky(B) is a W-sized array of B indices
    # Assign these results to columns
    # x, y are stand-in variables so that we can turn the separations into arcseconds

    x, y, _ = wright_coords.match_to_catalog_sky(berlanas_coords)
    wright_df['other_min'], wright_df['other_min_sep_as'] = x, y.arcsec
    _, y, _ = wright_coords.match_to_catalog_sky(berlanas_coords, nthneighbor=2)
    wright_df['next_match_sep_as'] = y.arcsec

    x, y, _ = berlanas_coords.match_to_catalog_sky(wright_coords)
    berlanas_df['other_min'], berlanas_df['other_min_sep_as'] = x, y.arcsec
    _, y, _ = berlanas_coords.match_to_catalog_sky(wright_coords, nthneighbor=2)
    berlanas_df['next_match_sep_as'] = y.arcsec

    # Now vet the matches
    # Make sure the object claiming the match is also the claimed match's match
    def make_vetting_function(this_catalog, other_catalog):
        """
        Generate a vetting function for an arbitrary order of catalogs,
        since we need to do this twice
        """
        def vet_match(row):
            """
            Check a single match (in one direction) with the object from another
            catalog
            This match was from this catalog crossmatched against the other catalog
            In other words, this_catalog.match_to_catalog_sky(other_catalog)
            """
            # Get the match's match idx
            # This is a "this_catalog" index
            other_match_idx = other_catalog.loc[row['other_min'], 'other_min']
            # "row.name" isn't "Name", like Simbad, it's the index!
            return (other_match_idx == row.name)
        return vet_match
    # Create and apply these functions
    vet_B_to_W = make_vetting_function(wright_df, berlanas_df)
    vet_W_to_B = make_vetting_function(berlanas_df, wright_df)
    wright_df['mutual'] = wright_df.apply(vet_B_to_W, axis=1)
    berlanas_df['mutual'] = berlanas_df.apply(vet_W_to_B, axis=1)
    # Include a 0.5 arcsecond cutoff
    wright_df['vetted'] = wright_df['mutual'] & (wright_df['other_min_sep_as']*4 < wright_df['next_match_sep_as']) & (wright_df['other_min_sep_as'] < 5)
    berlanas_df['vetted'] = berlanas_df['mutual'] & (berlanas_df['other_min_sep_as']*4 < berlanas_df['next_match_sep_as']) & (berlanas_df['other_min_sep_as'] < 5)

    # # Flag < 1 arcsecond matches that failed vetting, just to look closer
    # wright_df['flag'] = wright_df['vetted'] & (wright_df['other_min_sep_as'] > 2)
    # berlanas_df['flag'] = berlanas_df['vetted'] & (berlanas_df['other_min_sep_as'] > 2)
    # wright_df.loc[~wright_df['flag'], 'flag'] = np.nan
    # berlanas_df.loc[~berlanas_df['flag'], 'flag'] = np.nan

    # # Make NaNs so they disappear in the HTML table
    # wright_df.loc[~wright_df['mutual'], 'mutual'] = np.nan
    # berlanas_df.loc[~berlanas_df['mutual'], 'mutual'] = np.nan
    # wright_df.loc[~wright_df['vetted'], 'vetted'] = np.nan
    # berlanas_df.loc[~berlanas_df['vetted'], 'vetted'] = np.nan


    """
    Crossmatching completed
    Combine the catalogs!
    I will use berlanas_df as the "base" catalog, adding Wright15 entries to it
    We want to
    1) override A2-old (CP12) with W15, but leave A1-new (B18,20) alone
    2) add all the non-matched Wright15 entries
    """
    # Get the CP12 rows with vetted matches to W15, and get them indexed by W15
    berlanas_df.update(wright_df.loc[berlanas_df.loc[(berlanas_df['origin'] == 'A2-old') & berlanas_df['vetted'], 'other_min']].drop(columns='Name').set_index('other_min'), overwrite=True)
    berlanas_df = berlanas_df.set_index('Name')
    wright_df = wright_df.set_index('Name')
    berlanas_df = berlanas_df.append(wright_df.loc[~wright_df['vetted']], verify_integrity=True)

    # Sort by RA
    berlanas_df['RA'] = berlanas_df['SkyCoord'].apply(lambda c: c.ra.deg)
    berlanas_df.sort_values(by='RA', inplace=True)
    berlanas_df.drop(columns=['RA', 'other_min', 'other_min_sep_as', 'next_match_sep_as', 'vetted', 'mutual'], inplace=True)

    berlanas_df['origin'].where(lambda x: x != 'A1-new', other='B18+20', inplace=True)
    berlanas_df['origin'].where(lambda x: x != 'A2-old', other='CP12', inplace=True)

    return berlanas_df


def save_as_csv(df, stub="all"):
    df = df.reset_index()
    df['RAJ2000'] = df['SkyCoord'].apply(lambda c: c.ra.deg)
    df['DEJ2000'] = df['SkyCoord'].apply(lambda c: c.dec.deg)
    df = df[['RAJ2000', 'DEJ2000', 'SpType', 'Name', 'origin', 'Teff', 'LogL']]
    df.to_csv(mytable_csv.replace('all.csv', stub+'.csv'), na_rep="")

def save_as_html(df, stub="all"):
    df = df.drop(columns=['SkyCoord'])
    df['Teff'] = df['Teff']/1000
    # df.sort_values(by=['Teff'], inplace=True, ascending=False)
    # df = df.loc[df['Teff'] > 40]
    df.to_html(mytable_html.replace('all.html', stub+'.html'), na_rep="")



def generate_and_write_tables():
    """
    I ran all this most recently on Nov 6, 2020
    I commented out the write commands to avoid problems
    The "entire_df" is saved under the "mytable" pkl filename
    """
    wright_df = pd.read_pickle(wright_table)
    berlanas_df = pd.read_pickle(berlanas_table)
    entire_df = crossmatch_wright_and_berlanas(wright_df, berlanas_df)
    # entire_df.to_pickle(mytable)


df = pd.read_pickle(mytable)
catr = calc_fluxes(df, plot=False)
Q = catalog.spectral.stresolver.u.Quantity(tuple(zip(*catr.get_array_ionizing_flux()))[0])
# print(catr.get_ionizing_flux(nsamples=3))
wcs_obj, hdr = load_cygx_wcs()
# fig = plt.figure(figsize=(18, 6))
# ax1 = plt.subplot(131, projection=wcs_obj)
val, uncert = make_g0_plot(df, catr, wcs_obj, extremely_large=True, fuv_or_ionizing='fuv', nsamples=3, catalog_mask=np.isfinite(np.log10(Q.to_value())))
save_g0_fits(val, hdr, fuv_or_ionizing='fuv',)

# df = pd.read_pickle(berlanas_table)
# catr = calc_fluxes(df, plot=False)
# Q = catalog.spectral.stresolver.u.Quantity(tuple(zip(*catr.get_array_ionizing_flux()))[0])
# # print(catr.get_ionizing_flux(nsamples=3))
# wcs_obj, hdr = create_cygx_wcs(df)
# ax3 = plt.subplot(133, projection=wcs_obj)
# val_berlanas, uncert = make_g0_plot(df, catr, wcs_obj, extremely_large=False, fuv_or_ionizing='fuv', nsamples=3, catalog_mask=np.isfinite(np.log10(Q.to_value())), fig=fig, ax=ax3, label=' Berlanas \'18,\'20')
# # save_g0_fits(val, hdr)



if False:
    """
    This is a good example of setting colorbar ticks manually
    """
    ax2 = plt.subplot(132, projection=wcs_obj)
    im = ax2.imshow(np.log10(val / val_berlanas), origin='lower', vmin=-.222, vmax=.222, cmap='seismic')
    cbar = fig.colorbar(im, ax=ax2, ticks=np.log10(np.array([0.6, 0.75, 0.9, 1, 1.1, 1.25, 1.6])))
    cbar.ax.set_yticklabels(['$-$40%', '$-$25%', '$-$10%', '$+$0%', '$+$10%', '$+$25%', '$+$60%'])
    ax2.set_title("Ratio of $G_0$, Complete / Berlanas")

    fig.savefig(os.path.join(data_dir, "cyg-ob2_g0_comparison_2020-11-06.png"))
    plt.show()
