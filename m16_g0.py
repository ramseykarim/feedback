"""
For exploring the stars and radiation field of M16 / NGC 6611


If you're looking for where I load in the Hillenbrand stars and make the lists and find G0,
while you'd be correct in looking here first given the title, you want the
queries.py (vizier_queries_m16_g0.py now) file. I use a vizier query to load in the catalog.

This is where I mess around with the Stoop catalog though.

Created: June 16, 2022
"""
__author__ = "Ramsey Karim"

# All imports dumped from the last file, m16_investigation
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 13}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import sys
import os
import glob

from math import ceil
from scipy import signal
from scipy.interpolate import UnivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.table import Table, QTable
from astropy import constants as const

import pandas as pd
from io import StringIO

from photutils import centroids

# Lord forgive me
from . import crosscut
pvdiagrams = crosscut.pvdiagrams
misc_utils = pvdiagrams.misc_utils
catalog = pvdiagrams.catalog
cube_utils = pvdiagrams.cube_utils
reproject_interp = pvdiagrams.reproject_interp
pvextractor = pvdiagrams.pvextractor
regions = pvdiagrams.regions
SpectralCube = pvdiagrams.SpectralCube
Cutout2D = pvdiagrams.Cutout2D

from . import cube_pixel_spectra as cps1
from . import cube_pixel_spectra_2 as cps2

from . import vizier_queries_m16_g0 as vizier_queries

from .g0_stars import print_val_err

scoby_import_path = "/home/ramsey/Documents/Research/scoby-all/scoby"
if os.path.isdir(scoby_import_path):
    sys.path.append(scoby_import_path)
    import scoby
else:
    scoby = None


mpl_cm = pvdiagrams.mpl_cm
mpl_colors = pvdiagrams.mpl_colors
mpl_transforms = pvdiagrams.mpl_transforms
mpatches = pvdiagrams.mpatches

make_vel_stub = lambda x : f"[{x[0].to_value():.1f}, {x[1].to_value():.1f}] {x[0].unit}"
kms = u.km/u.s
marcs_colors = ['#377eb8', '#ff7f00','#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


def get_pillar_image(w_to_project=None, shape_to_project=None, line='cii'):
    """
    June 16, 2022
    Get a quick moment image of the pillars to overlay (probably as contours)
    on something else. Optionally reproject the moment image to the argument
    WCS and shape.
    """
    if line == 'cii':
        line_kwargs = dict()
    else:
        line_kwargs = dict(data_filename=cube_utils.cubefilenames[line])
    cube = cps2.cutout_subcube(**line_kwargs, length_scale_mult=6)
    mom0 = cube.spectral_slab(19*kms, 27*kms).moment0()
    if w_to_project is not None:
        mom0_reproj = reproject_interp((mom0.to_value(), mom0.wcs), w_to_project, shape_out=shape_to_project, return_footprint=False)
        result = mom0_reproj
    else:
        result = mom0.to_value()
    return result




def compare_O5s_to_morestars():
    """
    June 16, 2022
    Compare the G0 map from just the two O5s to the map from all stars within
    like 4 arcmin. Goal here is to get an idea of how incorrect the assumption
    is that only the brightest stars matter

    Also compare >4.5 log10 G0 to entire cluster core
    """

    # g0_bright_map_name = f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/g0_hillenbrand_stars_O5s.fits"
    g0_bright_map_name = f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/g0_hillenbrand_stars_fuvgt5.0_ltxarcmin.fits"
    g0_4arcmin_map_name = f"/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/g0_hillenbrand_stars_ltxarcmin.fits"


    g0_maps, hdrs = list(zip(*[fits.getdata(x, header=True) for x in [g0_bright_map_name, g0_4arcmin_map_name]]))

    w = WCS(hdrs[0])
    map_ratio = g0_maps[1]/g0_maps[0]
    fig = plt.figure()
    ax = plt.subplot(111, projection=w)
    im = ax.imshow(map_ratio, vmin=1, vmax=3)
    fig.colorbar(im, ax=ax, label='ratio of $G_0$ values')
    ax.set_title("Ratio of entire cluster core to log10(G0)>5.0")

    pillar_mom0 = get_pillar_image(w, map_ratio.shape, line='hcop')
    ax.contour(pillar_mom0, colors='k', linewidths=0.9, levels=5, alpha=0.7)

    fig.savefig("/home/ramsey/Pictures/2022-06-16/g0_ratio_clustercore_to_fuvgt5.0.png",
        metadata=catalog.utils.create_png_metadata(title='hcop 19-27 overlaid for reference',
            file=__file__, func="compare_O5s_to_morestars"))


def estimate_los_distance_required_for_star_to_not_matter():
    """
    June 16, 2022
    There are a couple stars within an arcminute of the pillars, and I want to
    know what their l.o.s. distance would have to be for them to not matter so
    much for the total G0 on the pillars
    """
    los_distance = 2.*u.kpc
    radiation_field_1d = 1.6e-3 * u.erg / (u.cm**2 * u.s)

    approx_linear_distances = {401: 1*u.arcmin, 351: 0.6*u.arcmin, 367: 0.3*u.arcmin}
    stellar_types = {401: 'O8.5V', 351: 'B1V', 367: 'O9.5V'}
    log10_FUVsolLum_vals = {401: 4.57, 351: 4.06, 367: 4.35}

    def angular_to_physical_distance(angular_distance):
        return angular_distance*(los_distance/(1*u.rad)).decompose()

    def convert_log10fuv_to_g0(index, radial_physical):
        fuv_lum = (10**log10_FUVsolLum_vals[index])*u.solLum
        projected_physical = angular_to_physical_distance(approx_linear_distances[index])
        squared_physical_distance = projected_physical**2 + radial_physical**2
        result = (fuv_lum / (4*np.pi*squared_physical_distance)) / radiation_field_1d
        return result.decompose()

    approx_cluster_size = [2, 4, 6]*u.arcmin
    print("approximate cluster size: ", angular_to_physical_distance(approx_cluster_size).to(u.pc))

    r_array = np.arange(0, 4, 0.1)*u.pc

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    for idx in stellar_types:
        label = f"{stellar_types[idx]} ({idx})"
        ax.plot(r_array, convert_log10fuv_to_g0(idx, r_array), label=label)
    ax.legend()
    ax.set_ylim(0, 2000)
    ax.set_ylabel("$G_0$ (Habing units)")
    ax.set_xlabel("Line-of-sight distance component from pillars (pc)")

    fig.savefig("/home/ramsey/Pictures/2022-06-16/los_star_distance_g0_estimate.png",
        metadata=catalog.utils.create_png_metadata(title='assuming 401: 1\', 351: 0.6\', 367: 0.3\'',
            file=__file__, func='estimate_los_distance_required_for_star_to_not_matter'))



def estimate_pushing_around_gas_with_stars():
    """
    July 6, 2022
    Do we have the momentum to move the threads over the lifespan of the pillars?
    Depends on: momentum transfer from stars, pillar mass, pillar and star lifespans
    """
    # Make sure the right thing is returned from m16_stars (change the True/False flags to arguments)
    star_info_dict = vizier_queries.m16_stars()
    thread_masses = [7, 2] * u.solMass # East, West
    thread_collecting_area_fractions = np.array([2594., 11272.]) # 1/these. Assume threads are cylinders, did (pi r^2) / (4pi R^2)
    thread_velocity = 1*kms # order of magnitude
    mvflux = star_info_dict['mvflux'][0]
    timescale = (thread_masses*thread_velocity / mvflux).decompose() * thread_collecting_area_fractions
    print()
    print("TIMESCALE TO PUSH")
    print(timescale.to(u.Myr))


def convert_df_cols_from_str_to_coord_with_units(df, colname_ra, colname_dec, units=None, epoch="2000"):
    """
    August 3, 2023
    Convert an RA and Dec column pair to SkyCoord column using correct unit, if unit row given.
    Unit row is a Pandas row, so the units for the colname are obtained by indexing with the colnamed: units[colname]
    Unit not given should be interpreted as HHMMSS or something, but I'll cross that bridge only if I need to.
    Epoch defaults to 2000 and should be a string. It gets a "J" prepended to it and is passed to SkyCoord as obstime

    Returned value is a Series whose index matches df.index. Values are individual SkyCoord objects.
    """
    if units is None:
        raise NotImplementedError
    def get_coord_vals(colname):
        return df[colname].apply(lambda x: float(x)*u.Unit(units[colname]))
    # print(df[colname_ra].values.astype(float))
    ra_vals = get_coord_vals(colname_ra)
    dec_vals = get_coord_vals(colname_dec)

    epoch = "J"+epoch
    coord_vals = [SkyCoord(*radec, frame='icrs', obstime=epoch) for radec in zip(ra_vals, dec_vals)]

    coord_col = pd.Series(coord_vals, index=df.index)
    return coord_col


def convert_df_cols_to_skycoord_array(df, colname_ra, colname_dec, units=None, epoch="2000"):
    """
    August 3, 2023
    Very similar to convert_df_cols_from_str_to_coord_with_units but instead of
    creating individual SkyCoord objects, makes one SkyCoord with all the coords.
    This should facilitate quicker separation calculations.
    Same epoch and units rules as the other function.

    Returns a single SkyCoord which contains an array of coordinates
    """
    if units is None:
        raise NotImplementedError
    def get_coord_vals(colname):
        return df[colname].apply(lambda x: float(x)).values * u.Unit(units[colname])
    ra_vals = get_coord_vals(colname_ra)
    dec_vals = get_coord_vals(colname_dec)
    epoch = "J"+epoch
    c = SkyCoord(ra_vals, dec_vals, frame='icrs', obstime=epoch)
    return c


def poke_around_in_stoop_catalog(return_df_early=False, filter=None):
    """
    August 3, 2023
    Check out the Stoop catalog. catalogs/Stoop2023_tablec1.tsv
    """
    scoby.config.PRINT_WARNINGS = True
    df = pd.read_csv(catalog.utils.search_for_file("catalogs/Stoop2023_tablec1.tsv"), delimiter=';', comment='#', skiprows=[51]) # line 52 (or 51, for 0 indexed) is just hyphens
    # Get the units row; each column has units
    u_row = df.iloc[0]
    # Delete the unit row from the dataframe so that the rows are just stars
    df.drop(index=0, inplace=True)
    # print(df)
    # for i,x in enumerate(df.columns):
    #     print(i, x)

    """ Compare the three different coordinate columns """
    if False:
        # c1 = convert_df_cols_from_str_to_coord_with_units(df, ra1, dec1, units=u_row)
        c1 = convert_df_cols_to_skycoord_array(df, *(df.columns[x] for x in [0, 1]), units=u_row)
        c2 = convert_df_cols_to_skycoord_array(df, *(df.columns[x] for x in [4, 5]), units=u_row, epoch="2015.5")
        c3 = convert_df_cols_to_skycoord_array(df, *(df.columns[x] for x in [16, 17]), units=u_row)

        print(u.Quantity(c1.separation(c3).to(u.mas)))
        print(u.Quantity(c1.separation(c2).to(u.mas)))
        print(u.Quantity(c2.separation(c3).to(u.mas)))
        """
        C2 == C3, and C1 is ~20mas different which is attributable to its different rounding (see the table)
        Let's use C2 since it's the "official" author-provided table value and not a Simbad-provided value.
        """

    """
    Assign coordinates using C2 (author provided)
    """
    df['SkyCoord'] = convert_df_cols_from_str_to_coord_with_units(df, *(df.columns[x] for x in [4, 5]), units=u_row)


    sptype_col = df['SpType']

    """
    Adjustments to spectral types
    33: remove unknown binary
    146: remove question mark
    6: triple system, remove the parentheses around the binary system so I can parse it correctly
    8, 142: replace "f+" with "f" so that parsing binaries works properly
    52: Type Ae? Being parsed as A star, but not sure that is correct. For now, blank it
    """
    # Idx. 33 " + ?" and 146 "(?)" have question marks. 33 appears to be a binary with an unknown companion type, and 146 is just an uncertain type? (Check the paper)
    sptype_col[33] = sptype_col[33].replace('+ ?', '').strip()
    sptype_col[146] = sptype_col[146].replace('(?)', '').strip()
    sptype_col[6] = sptype_col[6].replace('(', '').replace(')', '')
    sptype_col[8] = sptype_col[8].replace('f+', 'f')
    sptype_col[142] = sptype_col[142].replace('f+', 'f')
    sptype_col[52] = sptype_col[52].replace('Ae', '')

    def parse_spectral_type(s):
        st_result = {}
        for st_binary_component in scoby.spectral.parse_sptype.st_parse_binary(s):
            # st_binary_component is string
            st_bc_possibilities = scoby.spectral.parse_sptype.st_parse_slashdash(st_binary_component)
            # st_bc_possibilities is list(string)
            st_bc_possibilities_t = [scoby.spectral.parse_sptype.st_parse_type(x) for x in st_bc_possibilities]
            # st_bc_possibilities_t is list(tuple(string))
            st_result[st_binary_component] = st_bc_possibilities_t
        return st_result


    """
    ####
    #### Next, try to just parse the spectral types. might need to load in a function from the parser, because I want to do it without initializing the CatR object which needs all the tables

    Put them back in the dataframe and get rid of unknown spectral type stars (more than half the catalog)
    """

    df['SpType'] = sptype_col


    if False:
        ### Counting stars and printing types
        # print(sptype_col)
        j = 0
        k = 0
        for i in df.index:
            s = df['SpType'][i]
            if s.strip():
                print(i, s)
                st = parse_spectral_type(s)
                print(st)
                k += len(st)
                print()
                j += 1
        print("TOTAL SYSTEMS", j)
        print("TOTAL MEMBERS", k)


    # print(len(df))
    ### Get rid of unknown spectral type stars
    # print(df._is_copy)
    df = df.loc[df['SpType'].apply(lambda x: bool(x.strip()))].copy() # copy() gets rid of a warning
    # print(df._is_copy)
    # print(len(df))
    # print(df)

    """
    At this point, after making the edits to the 6 stars, the spectral types appear
    to parse without issue. There are 59 systems containing 67 stars (unpacking binaries/triples)
    """

    """
    Now, run scoby on them
    """

    ### helper function make_cat_resolver defined below this
    catr = make_cat_resolver(df['SpType'])
    # print(catr)


    fuv_flux_array = u.Quantity([x[0] for x in catr.get_array_FUV_flux()])
    fuv_flux_unit = fuv_flux_array.unit
    df['log10FUV_flux_'+str(fuv_flux_unit).replace(' ', '')] = np.log10(fuv_flux_array.to_value())


    """
    Filter paragraph copied directly from vizier_queries_m16_g0

    filter 4 is the category for the paper, and filter 0 is no filter
    """
    # make some filters for the catalog so we can try different G0 maps
    if filter is None:
        filter = 4 # filter 4 (>4.5 and filter radius) is the pillar paper one
    filter_stub = ""
    if filter == 0:
        # no filter
        pass
    if filter == 1 or filter == 4:
        # filter log10FUV_flux_solLum > 4.5 (8 stars) or 5.0 (4 stars)
        df = df[df['log10FUV_flux_solLum'] > 4.5]
        filter_stub += "_fuvlt4.5"
    if filter == 2 or filter == 4:
        # filter < x arcmin from y
        center_coord = SkyCoord('18:18:35.9543 -13:45:20.364', unit=(u.hourangle, u.deg), frame='fk5')
        filter_radius = 3.90751*u.arcmin # my original number was like 3.90751*u.arcmin; I don't know why.. it's not even exactly 2 pc, though very close
        df = vizier_queries.filter_by_within_range(df, center_coord, radius_arcmin=filter_radius)
        df = df[df['is_within_3.9_arcmin']]
        filter_stub += "_ltxarcmin"
    # if filter == 3:
    #     # filter for just the 2 O5 stars
    #     catalog_df_OB = catalog_df_OB.loc[[175, 205]]
    #     filter_stub += "_O5s"
    # if filter == 5:
    #     # 175, 205, 222, 246
    #     catalog_df_OB = catalog_df_OB.loc[[246]]
    #     filter_stub += '_justone'

    if return_df_early:
        savename = catalog.utils.m16_data_path + f"catalogs/stoop_stars{filter_stub}.pkl"
        print(f"Returning early, saved Stoop DF {savename}")
        df.to_pickle(savename)
        return df

    ### remake catr now that we have filtered
    catr = make_cat_resolver(df['SpType'])
    print(catr)

    for s in catr.star_list:
        print(s.spectral_types)

    """
    Print quantities
    """
    if False:
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


    ke = u.Quantity([x[0] for x in catr.get_array_mechanical_luminosity()])
    print(ke)

    return df


def make_cat_resolver(spectral_types_column):
    """
    September 7, 2023
    """
    scoby.spectral.stresolver.UNCERTAINTY = False
    powr_tables = {x: scoby.spectral.powr.PoWRGrid(x) for x in ('OB',)}
    cal_tables = scoby.spectral.sttable.STTable(*catalog.spectral.martins.load_tables_df())
    ltables = scoby.spectral.leitherer.LeithererTable()
    catr = scoby.spectral.stresolver.CatalogResolver(spectral_types_column.values, calibration_table=cal_tables, leitherer_table=ltables, powr_dict=powr_tables)
    return catr


def crossmatch_stoop_and_hillenbrand():
    """
    October 31, 2023. Happy Halloween!
    Finally doing the crossmatching. I am taking notes in today's notes,
    and loosely following some crossmatching I did in catalogs/parse.py.
    Key is to use SkyCoord.match_to_catalog_sky
    """
    # Hillenbrand
    h_df_filename = "/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/hillenbrand_stars.pkl"
    if not h_df_filename:
        # Only need to run once, then saves
        vizier_queries.m16_stars(return_df_early=True, filter=0)
    else:
        h_df = pd.read_pickle(h_df_filename)
    print(h_df.columns)
    # Get a SkyCoord array from Hillenbrand
    h_sc = SkyCoord(h_df['SkyCoord'].values)
    print(f"H cat has {h_sc.size} items")

    # Stoop
    s_df_filename = "/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/stoop_stars.pkl"
    if not s_df_filename:
        s_df = poke_around_in_stoop_catalog(return_df_early=True, filter=0)
    else:
        s_df = pd.read_pickle(s_df_filename)
    print(s_df.columns)
    s_sc = SkyCoord(s_df['SkyCoord'].values)
    print(f"S cat has {s_sc.size} items")

    h_idx_xmatch, h_to_s_sep, _ = s_sc.match_to_catalog_sky(h_sc)
    s_idx_xmatch, s_to_h_sep, _ = h_sc.match_to_catalog_sky(s_sc)

    # Bunch of testing; change to True to see the tests
    if False:
        # Get a large-scale view that shows that any seps > 1 arcsec are
        # mostly also > 10 arcsec, clearly not matches
        plt.hist(h_to_s_sep.arcsec, range=(0, 64), bins=64*2)
        plt.hist(s_to_h_sep.arcsec, range=(0, 64), bins=64*2)
        plt.show()
    elif False:
        # Small scale view that shows that seps < 1 arcsec are all < 0.25,
        # so matches are good
        plt.hist(h_to_s_sep.arcsec, range=(0, 2), bins=32)
        plt.hist(s_to_h_sep.arcsec, range=(0, 2), bins=32)
        plt.show()
    elif False:
        # Show the above more definitely
        # min sep that is greater than 1 arcsec
        h2s_min_sep_gt1 = np.min(h_to_s_sep.arcsec[h_to_s_sep.arcsec > 1])
        s2h_min_sep_gt1 = np.min(s_to_h_sep.arcsec[s_to_h_sep.arcsec > 1])
        print(f"Min gt 1 {h2s_min_sep_gt1} and {s2h_min_sep_gt1} arcsec")
        # That shows that the min gt 1 is like 7 or 8 arcsec
        h2s_max_sep_lt1 = np.max(h_to_s_sep.arcsec[h_to_s_sep.arcsec < 1])
        s2h_max_sep_lt1 = np.max(s_to_h_sep.arcsec[s_to_h_sep.arcsec < 1])
        print(f"Max lt 1 {h2s_max_sep_lt1} and {s2h_max_sep_lt1} arcsec")
        # That shows that the max lt 1 is ~0.25

    ## Just like we did for Wd2
    def match_vetting(match1, sep1, match2, sep2,
        cutoff_arcsec=0.25):
        """
        Oct 31, 2023
        Not exactly like we did for Wd2 but pretty close.
        Check that matches are mutual
        """
        # Just loop, it's easier to think about
        count = 0
        for i in range(len(match1)):
            if sep1[i].arcsec > cutoff_arcsec:
                continue
            j = match1[i]
            assert (i == match2[j])
            count += 1
        print(count, "matches")
    if False:
        match_set_1 = (h_idx_xmatch, h_to_s_sep)
        match_set_2 = s_idx_xmatch, s_to_h_sep
        match_vetting(*match_set_1, *match_set_2)
        match_vetting(*match_set_2, *match_set_1)
        # Those work!! no assert errors
    """
    All crossmatches between the full H and S catalogs are successfull if they
    are < 0.25 arcsec. All "matches" above 0.25 arcsec are > 7 arcsec and are
    unambiguously not matches. All matches are unambiguous!
    41 matches between the 62 H stars and 59 S stars.
    """
    # Compile a super-catalog
    # start from the Stoop catalog and add Hillenbrand to the relevant ones.
    # reduce the columns in Stoop too, there are lots of unecessary ones.
    #### we might consider adding in the extinction measurements to make a point...

    if False:
        ### Radial velocities
        # print(s_df['RV'])
        valid_numbers = s_df['RV'].apply(lambda x: bool(x.strip()))
        o_stars = s_df['SpType'].apply(lambda x: 'O' in x)
        # print(o_stars)
        s_df_valid = s_df['RV'][valid_numbers].apply(float)
        s_df_valid_o = s_df['RV'][valid_numbers & o_stars].apply(float)
        # print(s_df_valid)
        _, bins, _ = plt.hist(s_df_valid, bins=32, histtype='stepfilled')
        plt.hist(s_df_valid_o, bins=bins, histtype='stepfilled')
        plt.show()
    s_cols_to_keep = ['SkyCoord', 'SpType', 'log10FUV_flux_solLum', 'RV', 'E(B-V)']
    h_cols_to_keep = ['SkyCoord', 'SpType', 'log10FUV_flux_solLum']
    super_cat_df = s_df[s_cols_to_keep]
    super_cat_df = super_cat_df.rename({x: "s_"+x for x in s_cols_to_keep}, axis='columns').reset_index(names="Stoop_tC1_idx")
    # Get the H stars now
    h_df_renamed = h_df[h_cols_to_keep].rename({x: "h_"+x for x in h_cols_to_keep}, axis='columns').reset_index(names="Hillenbrand_t3A_idx")
    # Get a True/False column for whether S stars matched an H star
    s_matched_successfully = h_to_s_sep.arcsec < 1
    # Use that to get the S-matched H indices (0-indexed out of 62, not the H table index)
    h_idx_matched_successfully = h_idx_xmatch[s_matched_successfully]
    # h_idx_matched_successfully is the list of 41 matched stars' indices out of 62 H stars. (not Hillenbrand index, just 0-61)
    h_matched = h_df_renamed.loc[h_idx_matched_successfully].set_index(super_cat_df[s_matched_successfully].index)
    super_cat_df = pd.concat([super_cat_df, h_matched], axis='columns')
    # print(super_cat_df[['Stoop_tC1_idx', 'Hillenbrand_t3A_idx']])
    # print(super_cat_df[super_cat_df['Hillenbrand_t3A_idx'].isin([161, 166, 175, 197, 205, 210, 222, 246])][['Stoop_tC1_idx', 'Hillenbrand_t3A_idx']])
    # print(super_cat_df.columns)

    # Grab the unique Hillenbrand items (not matched to S)
    # h_idx_unique is the complement out of range(62)
    h_idx_unique = [i for i in range(len(h_sc)) if i not in h_idx_matched_successfully]
    h_unique = h_df_renamed.loc[h_idx_unique]
    # print(h_unique[['h_SpType', 'h_log10FUV_flux_solLum']])
    # print(super_cat_df)
    super_cat_df = pd.concat([super_cat_df, h_unique], axis='index', ignore_index=True)
    idx_col_names = ['Hillenbrand_t3A_idx', 'Stoop_tC1_idx']
    # for colname in idx_col_names:
    #     super_cat_df[colname] = super_cat_df[colname].astype(int)
    # super_cat_df = super_cat_df.convert_dtypes()

    ### Sort by Hillenbrand index first and then by Stoop where no H
    super_cat_df = super_cat_df.sort_values(by=["Hillenbrand_t3A_idx", "Stoop_tC1_idx"])

    # super_cat_df['SkyCoord'] = super_cat_df[]
    super_cat_df['SkyCoord'] = super_cat_df['s_SkyCoord'].where(super_cat_df['s_SkyCoord'].notnull(), super_cat_df['h_SkyCoord']).apply(lambda x: x.icrs)
    super_cat_df['RA'] = super_cat_df['SkyCoord'].apply(lambda x: x.icrs.ra.deg)
    super_cat_df['DE'] = super_cat_df['SkyCoord'].apply(lambda x: x.icrs.dec.deg)
    super_cat_df = super_cat_df.drop(columns=['s_SkyCoord', 'h_SkyCoord'])

    """
    Filter by both 5 and 20 arcsec. Gives us a bit of a spread of stars we're using, to see how things vary.
    """
    # center_coord = SkyCoord('18:18:35.9543 -13:45:20.364', unit=(u.hourangle, u.deg), frame='fk5') # My Paper 1 center
    center_coord = SkyCoord(274.67, -13.78, unit=u.deg, frame='fk5') # Stoop center
    # 274.67, -13.78 is the Stoop center coord
    # filter_radius = 3.90751*u.arcmin # my original number was like 3.90751*u.arcmin; I don't know why.. it's not even exactly 2 pc, though very close
    filter_radius_small = 5*u.arcmin
    super_cat_df = vizier_queries.filter_by_within_range(super_cat_df, center_coord, radius_arcmin=filter_radius_small)
    filter_radius_col_name_small = super_cat_df.columns[-1]
    # df = df[df['is_within_3.9_arcmin']]
    # filter_stub += "_ltxarcmin"
    filter_radius_large = 20*u.arcmin
    super_cat_df = vizier_queries.filter_by_within_range(super_cat_df, center_coord, radius_arcmin=filter_radius_large)
    filter_radius_col_name_large = super_cat_df.columns[-1]
    # print(super_cat_df.columns)
    # print(filter_radius_col_name_small, filter_radius_col_name_large)

    """
    Throw out 24 stars which won't work with scoby in either H or S
    i.e. they're too late-type in both / late in one and not catalogged in other
    """
    super_cat_df = super_cat_df.loc[(super_cat_df["h_"+"log10FUV_flux_solLum"] > 0) | (super_cat_df["s_"+"log10FUV_flux_solLum"] > 0)]


    """
    Throw out 3 stars outside 20 arcmin from cluster core
    """
    super_cat_df = super_cat_df.loc[super_cat_df[filter_radius_col_name_large]]

    """
    Reset the DataFrame index now, and sort by RA
    """
    super_cat_df = super_cat_df.sort_values(by=["RA"])
    super_cat_df.index = list(range(1, len(super_cat_df)+1))

    """
    Throw out 17 stars outside 5 arcmin from cluster core
    """
    fuv_cutoff = 4.49
    if True:
        super_cat_df = super_cat_df.loc[super_cat_df[filter_radius_col_name_small]]

        """
        Under the 5 arcmin filter, filter also by 4.49 FUV.
        Do this separately for H and S
        """
        select = 0 # 0 is no FUV filtering. 1 is H, 2 is S
        if select == 1:
            super_cat_df = super_cat_df.loc[(super_cat_df["h_"+"log10FUV_flux_solLum"] > fuv_cutoff)]
        elif select == 2:
            super_cat_df = super_cat_df.loc[(super_cat_df["s_"+"log10FUV_flux_solLum"] > fuv_cutoff)]


    # print(super_cat_df[['Stoop_tC1_idx', 'Hillenbrand_t3A_idx']])
    #### 80 unique stars between the two OB catalogs. Some of these don't have good types, like "Be" or other later Bs

    # Checking how many O stars in each catalog and their crossover
    if False:
        s_type_O = super_cat_df['s_SpType'].apply(lambda x: 'O' in str(x))
        h_type_O = super_cat_df['h_SpType'].apply(lambda x: 'O' in str(x))
        super_cat_df = super_cat_df[s_type_O & ~h_type_O][['h_SpType', 's_SpType']]
        print(super_cat_df)

    fuv_cutoff_col_stub = f"FUVgt{fuv_cutoff:.2f}"
    if True:
        for prefix in ["h_", "s_"]:
            super_cat_df[prefix+fuv_cutoff_col_stub] = ""
            super_cat_df[prefix+fuv_cutoff_col_stub].mask(super_cat_df[prefix+"log10FUV_flux_solLum"] > fuv_cutoff, "X", inplace=True)
        super_cat_df["is_within_small"] = ""
        super_cat_df["is_within_small"].mask(super_cat_df[filter_radius_col_name_small], "X", inplace=True)
        # super_cat_df["is_within_large"] = ""
        # super_cat_df["is_within_large"].mask(super_cat_df[filter_radius_col_name_large], "X", inplace=True)

        if False:
            # Count the number of stars in each filter and each catalog
            for prefix in ["h_", "s_"]:
                for frcn in [filter_radius_col_name_small, filter_radius_col_name_large]:
                    for whether_fuv_cutoff in range(3):
                        whether_fuv_cutoff_str = ["", "has_fuv", fuv_cutoff_col_stub][whether_fuv_cutoff]
                        catalog_stub = prefix[0].upper()
                        if whether_fuv_cutoff == 0:
                            count = sum(super_cat_df[frcn] & super_cat_df[prefix+"log10FUV_flux_solLum"].notnull())
                        elif whether_fuv_cutoff == 1:
                            count = sum(super_cat_df[frcn] & (super_cat_df[prefix+"log10FUV_flux_solLum"] > 0))
                        else:
                            count = sum(super_cat_df[frcn] & (super_cat_df[prefix+"log10FUV_flux_solLum"] > fuv_cutoff))
                        stub = f"{catalog_stub} {frcn} {whether_fuv_cutoff_str}"
                        print(f"{stub:<40} {count}")
                print()

        super_cat_df["DEBUG"] = ""
        # condition = super_cat_df["Hillenbrand_t3A_idx"].notnull() & super_cat_df["Stoop_tC1_idx"].isnull() & super_cat_df["is_within_small"]
        # condition = (super_cat_df["h_"+"log10FUV_flux_solLum"] > 0) | (super_cat_df["s_"+"log10FUV_flux_solLum"] > 0)
        # condition = ~(super_cat_df["h_"+"log10FUV_flux_solLum"] > 0) & ~(super_cat_df["s_"+"log10FUV_flux_solLum"] > 0)
        condition = (super_cat_df["h_"+"log10FUV_flux_solLum"] > 0) & (super_cat_df["s_"+"log10FUV_flux_solLum"] > 0)
        print("cond", sum(condition))
        print()
        super_cat_df["DEBUG"].mask(condition, "X", inplace=True)

        super_cat_df['IN_BOTH'] = ""
        super_cat_df['IN_BOTH'].mask(super_cat_df["Hillenbrand_t3A_idx"].notnull() & super_cat_df["Stoop_tC1_idx"].notnull(), "X", inplace=True)
        print("BOTH (intersection)", sum(super_cat_df["Hillenbrand_t3A_idx"].notnull() & super_cat_df["Stoop_tC1_idx"].notnull()))
        print("ALL (union) (len(df))", len(super_cat_df))
        print("IN S", sum(super_cat_df["Stoop_tC1_idx"].notnull()))
        print("IN H", sum(super_cat_df["Hillenbrand_t3A_idx"].notnull()))

        if False:
            # Trim down to 5 arcmin and either S or H > fuv_cutoff
            super_cat_df = super_cat_df.loc[super_cat_df[filter_radius_col_name_small] & ((super_cat_df["h_log10FUV_flux_solLum"] > fuv_cutoff) | (super_cat_df["s_log10FUV_flux_solLum"] > fuv_cutoff))]
        elif False:
            # Trim down to 5 arcmin
            super_cat_df = super_cat_df.loc[super_cat_df[filter_radius_col_name_small]]

        super_cat_df = super_cat_df[["RA", "DE", "Hillenbrand_t3A_idx", "Stoop_tC1_idx", "DEBUG", "h_log10FUV_flux_solLum", "s_log10FUV_flux_solLum", "h_SpType", "s_SpType", f"h_{fuv_cutoff_col_stub}", f"s_{fuv_cutoff_col_stub}", "is_within_small"]]
        # super_cat_df
        super_cat_df.drop(columns=[x for x in super_cat_df.columns if "SkyCoord" in x]).to_html("/home/ramsey/Downloads/hillenbrand_stoop.html", na_rep="")
        # super_cat_df.to_csv(catalog.utils.m16_data_path + "catalogs/HS_super_catalog.csv", na_rep="")


if __name__ == "__main__":
    # df = poke_around_in_stoop_catalog()

    ## Crossmatching!
    crossmatch_stoop_and_hillenbrand()
