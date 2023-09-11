"""
For exploring the stars and radiation field of M16 / NGC 6611


If you're looking for where I load in the stars and make the lists and find G0,
while you'd be correct in looking here first given the title, you want the
queries.py file. I use a vizier query to load in the catalog.

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


def poke_around_in_stoop_catalog():
    """
    August 3, 2023
    Check out the Stoop catalog. catalogs/Stoop2023_tablec1.tsv
    """
    df = pd.read_csv(catalog.utils.search_for_file("catalogs/Stoop2023_tablec1.tsv"), delimiter=';', comment='#', skiprows=[51]) # line 52 (or 51, for 0 indexed) is just hyphens
    u_row = df.iloc[0]
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
        filter_radius = 3.90751*u.arcmin
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


if __name__ == "__main__":
    df = poke_around_in_stoop_catalog()
