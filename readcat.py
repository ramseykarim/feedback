"""
script to read/create star catalogs with pandas
created: October 21, 2019

Added some tables from different authors and standardized coordinates into
SkyCoords earlier in the process. Added a lot of documentation.
updated: April 14, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5, ICRS
import matplotlib.pyplot as plt
import glob
import sys
from misc_utils import flquantiles
import pandas as pd
import math

catalog_directory = "../ancillary_data/catalogs/"

def convert_hhmmss(hhmmss_hhmmss):
    """
    Split a single ra,dec string without spaces into separate RA and Dec strings
    Usage example:
    # tft_name = '102400.52-574444.6'
    # vphasra, vphasdec = 155.979910, -57.757480
    #
    # vphas_coord = SkyCoord(vphasra, vphasdec, unit=u.deg)
    # tft_coord = SkyCoord(*convert_hhmmss(tft_name), unit=(u.hourangle, u.deg))
    Doesn't seem like unique functionality, but I'll delete it when it's
    clear I won't use it
    """
    char_list = list(hhmmss_hhmmss)
    coord, second_coord = [], None
    count = 1
    while char_list:
        item = char_list.pop()
        if item.isnumeric():
            coord.append(item)
            if '.' in coord:
                if not count % 2:
                    coord.append(':')
                count += 1
        elif item == '.':
            coord.append(item)
        elif item in ['+', '-']:
            if coord[-1] == ':':
                coord.pop()
            coord.append(item)
            second_coord = coord
            coord = []
            count = 1
    ra, dec = coord, second_coord
    ra.reverse(), dec.reverse()
    if ra[0] == ':':
        ra.pop(0)
    return "".join(ra), "".join(dec)


def coords_from_hhmmss(df, frame=FK5):
    """
    Cobble together the HMS coordinates into RAdeg, DEdeg
    This is from the original catalog reduction, made for the Ascenso catalog,
        and I edited it to be more general and stop at the SkyCoord step
        (4/16/2020)
    """
    RAstr = df.apply(lambda row: ":".join(row[x] for x in ('RAh', 'RAm', 'RAs')), axis=1)
    DEstr = df.apply(lambda row: row['DE-'] + ":".join(row[x] for x in ('DEd', 'DEm', 'DEs')), axis=1)
    RADEstr = RAstr + " " + DEstr
    df['SkyCoord'] = RADEstr.apply(lambda radec_string: SkyCoord(radec_string, unit=(u.hourangle, u.deg), frame=frame))


def read_table_format(file_handle, n_cols):
    """
    Read one of those standard-format table descriptors
    Example of one of these:
        Byte-by-byte Description of file: table4.dat
        --------------------------------------------------------------------------------
           Bytes Format Units   Label     Explanations
        --------------------------------------------------------------------------------
           1-  2  I2    h       RAh       Right Ascension J2000 (hours)
           4-  5  I2    min     RAm       Right Ascension J2000 (minutes)
           7- 10  F4.1  s       RAs       Right Ascension J2000 (seconds)
              12  A1    ---     DE-       Declination J2000 (sign)
          13- 14  I2    deg     DEd       Declination J2000 (degrees)
          16- 17  I2    arcmin  DEm       Declination J2000 (minutes)
          19- 22  F4.1  arcsec  DEs       Declination J2000 (seconds)
          24- 29  F6.3  mag     Vmag      Mean observed Johnson V magnitude
          31- 35  F5.3  mag   e_Vmag      Uncertainty on the V magnitude
          37- 41  F5.3  mag     B-V       Mean observed Johnson B-V colour
          43- 47  F5.3  mag   e_B-V       Uncertainty on the B-V colour
        --------------------------------------------------------------------------------
    :param file_handle: should be open and currently at the first column line.
    :param n_cols: tells how many columns are present, and thus how many rows
        this program should read.
    :returns: the column byte intervals and the column names. These are ready
        to be passed directly to pandas.read_fwf as the colspecs and names
        keyword arguments.
     """
    col_intervals, col_labels = [], []
    # slices for the byte format beginning/end integers
    sl0, sl1 = slice(1, 4), slice(5, 8)
    # slice for the column name
    sln = slice(21, 32)
    for i in range(n_cols):
        line = file_handle.readline()
        if line[sl0].isspace() and line[sl1].isspace():
            continue
        start = int(line[sl0]) - 1 if not line[sl0].isspace() else int(line[sl1]) - 1
        end = int(line[sl1])
        col_intervals.append((start, end))
        label = max(line[sln].split(), key=len)
        col_labels.append(label)
    return col_intervals, col_labels


def skiplines(file_handle, n_lines):
    """
    skips ahead n_lines in the already-opened file_handle
    :param file_handle: file handle that is open
    :param n_lines: number of lines to skip
    """
    for i in range(n_lines):
        file_handle.readline()


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&% Tsujimoto %&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""

def openTFT_single(filenumber):
    """
    Open a Tsujimoto 2007 table (there are 3)
    Helper function for openTFT_complete
    :param filenumber: integer 1, 2 or 3 indicating table number
    """
    filename = catalog_directory+"Tsujimoto2007/tbl{:d}".format(filenumber)
    TFT_tblinfo = {1: (9, 22, 24), 2: (9, 20, 20), 3: (9, 14, 14)}
    with open(filename) as f:
        skip1, n_cols, skip2 = TFT_tblinfo[filenumber]
        skiplines(f, skip1)
        col_intervals, col_labels = read_table_format(f, n_cols)
        skiplines(f, skip2)
        df = pd.read_fwf(f, colspecs=col_intervals, names=col_labels, index_col='Num')
    return df


def openTFT_complete():
    """
    Open all 3 TFT tables and combine them
    Tables are from Tsujimoto et al. 2007 (ApJ 665:719-735)
    Go in here and print df.columns if you want to know what's in there
    TFT Specifics:
        Includes original SIRIUS NIR data from this project.
            Tsujimoto states that the SIRIUS data is a larger FOV than the
            similar Ascenso et al. 2007 data.
            Both this SIRIUS data and Ascenso are J,H,Ks bands.
        Cross-matches to NOMAD, 2MASS, (SIRIUS,) and GLIMPSE
        NOMAD: Naval Observatory Merged Astrometric Dataset, "simple merge of data from the Hipparcos, Tycho-2, UCAC-2 and USNO-B1 catalogues,
            supplemented by photometric information from the 2MASS final release point source catalogue"
        SIRIUS: IR instrument on the IRSF, a 55in reflector telescope at the
            SAAO, the South African Astronomical Observatory
        The flag in the "NIR" column is blank if no NIR data, T if data from 2MASS,
            and S if data from SIRIUS
        IDs in the NOMAD, 2MASS, and GLIMPSE catalogs are given.
        For the RAdeg, DEdeg given in table 1, separations compared to the IAU
        name (ICRS frame, presumably) suggest that these are FK5
    :returns: pandas dataframe
    """
    # The first is like RA,Dec and stuff (mostly unimportant)
    df_TFT = openTFT_single(1).filter(['RAdeg', 'DEdeg', 'PosErr'])
    # Second is X ray stuff
    df_TFT_X = openTFT_single(2)
    # Third is cross matching in IR
    df_TFT_cross = openTFT_single(3)
    for colname in df_TFT_cross:
        df_TFT[colname] = df_TFT_cross[colname]
    for colname in df_TFT_X:
        df_TFT[colname] = df_TFT_X[colname]
    del df_TFT_cross, df_TFT_X
    # Convert RAdeg and DEdeg to SkyCoords
    def make_skycoords(row):
        return SkyCoord(row['RAdeg'], row['DEdeg'], unit=u.deg, frame=FK5)
    df_TFT['SkyCoord'] = df_TFT.apply(make_skycoords, axis=1)
    return df_TFT


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&% VPHAS+ &%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""


def openVPHAS_single(tablenumber):
    """
    Open one of the two VPHAS+ tables
    Helper function for openVPHAS_complete
    :param tablenumber: integer, either 5 or 6, for Table 5 or 6
    :returns: pandas dataframe
    """
    with open(catalog_directory+f"VPHAS/table{tablenumber}.dat") as f:
        # VPHAS .dat tables
        header = f.readline().strip('#').split()
        df_VPHAS = pd.read_table(f, comment='#', names=header, delim_whitespace=True,
            dtype={**{x:int for x in ('ID', 'VPHAS_ID')}, **{x:str for x in ('VA_ID', 'MSP_ID', 'TFT_ID', 'SIMBAD_ID', 'notes')}},
            index_col=('ID' if tablenumber == 5 else 'VPHAS_ID'))
    return df_VPHAS


def openVPHAS_complete():
    """
    Open both VPHAS+ tables and return a combined table
    Both tables from Mohr-Smith et al. 2015 (MNRAS 450,3855–3873).
    This is an optical study done with the VST.
    The full table is 1073 items and is much more than we need
    The estimates for parameters Teff and DM are poorly constrained, according
        to the authors. A0 and Rv are well constrained and informative.
        Teff could still be useful if no other information is available for that
        source.
    They did some cross-matching with Tsujimoto et al. 2007 and
        Vargas Alvarez et al. 2013, though this notably will not catch cross-
        matches between VA13 and TFT if there was no VPHAS detection.
    They include JHKs NIR photometry from Ascenso et al. 2007 when possible,
        and 2MASS when not. There could still be cross-matches between Ascenso
        and Tsuimoto (though these would be NIR-redundant...)
    :returns: pandas dataframe
    """
    # General info, photometry, cross-matching with TFT and VA
    df_VPHAS_info = openVPHAS_single(5)
    df_VPHAS_info.index.name = 'VPHAS_ID'
    # Fitted parameters, including effective temperature and reddening
    df_VPHAS_params = openVPHAS_single(6)
    for colname in df_VPHAS_params:
        df_VPHAS_info[colname] = df_VPHAS_params[colname]
    del df_VPHAS_params
    # Make SkyCoords
    def make_skycoords(row):
        return SkyCoord(row['RA'], row['DEC'], unit=u.deg, frame=FK5)
    df_VPHAS_info['SkyCoord'] = df_VPHAS_info.apply(make_skycoords, axis=1)
    return df_VPHAS_info


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&% Vargas Alvarez &%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""


def openVA_simplecatalog():
    """
    Open the main catalog table, Table 2, from the HST survey
    :returns: pandas dataframe
    """
    with open(catalog_directory+"VargasAlvarez2013/tbl2") as f:
        skiplines(f, 10)
        col_intervals, col_labels = read_table_format(f, 18)
        skiplines(f, 6)
        df_VA = pd.read_fwf(f, colspecs=col_intervals, names=col_labels, na_values=("99.999", "0.000"),
            index_col='ID')
    return df_VA


def openVA_ET():
    """
    Open the spectral type table, Table 6, from the HST survey
    This table has spectroscopically derived types as well as some distance
        estimates (that probably shouldn't be trusted)
    :returns: pandas dataframe
    """
    with open(catalog_directory+"VargasAlvarez2013/tbl6") as f:
        skiplines(f, 3)
        header = f.readline().split()[:3]
        header.append("subtype")
        skiplines(f, 1)
        df_VA_ET = pd.read_table(f, delim_whitespace=True, names=header,
            skipfooter=2, engine='python', usecols=[0, 1, 2, 3], index_col='ID')
    return df_VA_ET


def openVA_complete():
    """
    Opens the full catalog from the HST survey and adds in spectral types
        from the spectroscopically analyzed subset of the catalog
    Catalog from Vargas Alvarez et al 2013 (AJ 145:125)
    :returns: pandas dataframe
    """
    # Get Table 5, the full catalog
    df_VA = openVA_simplecatalog()
    # Get Table 6, with spectroscopically determined types
    df_VA_ET = openVA_ET()
    for colname in ['Spectral', 'subtype']:
        df_VA[colname] = df_VA_ET[colname]
    del df_VA_ET
    # Make SkyCoords
    def make_skycoords(row):
        return SkyCoord(row['RAdeg'], row['DEdeg'], unit=u.deg, frame=FK5)
    df_VA['SkyCoord'] = df_VA.apply(make_skycoords, axis=1)
    return df_VA



"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&% Ascenso %&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""


# OPEN ASCENSO
def openAscenso_simplecatalog():
    """
    Open the Ascenso catalog, Table 2
    :returns: pandas dataframe
    """
    # Column names and data are in 2 different files
    with open(catalog_directory+"Ascenso2007/ReadMe") as f:
        skiplines(f, 32)
        col_intervals, col_labels = read_table_format(f, 14)
    with open(catalog_directory+"Ascenso2007/w2phot.dat") as f:
        df_Ascenso = pd.read_fwf(f, colspecs=col_intervals, names=col_labels,
            na_values=("99.99", "9.999"), index_col='Seq',
            dtype={x:str for x in ('RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs')})
    return df_Ascenso


def openAscenso_complete():
    """
    Open the NIR catalog from Ascenso et al. 2007 (A&A 466, 137–149)
    The catalog contains J,H,Ks band photometry from SOFI instrument on the
        ESO New Technology Telescope (NTT) in Chile.
    Vargas Alvarez et al. 2013 and Mohr-Smith et al. 2015 (VPHAS+) both use
        Ascenso's JHKs photometry. VPHAS+ gives JHKs from Ascenso where
        possible, and 2MASS otherwise, though they don't specify which (like
        Tsujimoto does with SIRIUS vs 2MASS).
        VPHAS+ including Ascenso photometry, again, does not preclude
        cross-matches between Ascenso and Tsujimoto, or Ascenso and VA (which
        where not given in the paper), or Ascenso and Rauw(?)
    :returns: pandas dataframe
    """
    df_Ascenso = openAscenso_simplecatalog()
    coords_from_hhmmss(df_Ascenso)
    return df_Ascenso


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&% Rauw &%&%&%&%&%&%&%&%&%&%&%%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%%&
"""


def openRauw():
    """
    Open the BV catalog from Rauw et al. 2007 (A&A 463, 981–991) made using
        ANDICAM on CTIO
    The catalog only gives BV photometry.
    Rauw et al. 2007 and 2011 both give some spectral types, but need to be
        put in manually since they never made good tables. There are Rauw 2011
        Objects A B C D .. etc whose coordinates are only given in text, not
        a table.
    """
    with open(catalog_directory+"/Rauw/ReadMe") as f:
        skiplines(f, 92)
        col_intervals, col_labels = read_table_format(f, 11)
    with open(catalog_directory+"/Rauw/table4.dat") as f:
        df_Rauw = pd.read_fwf(f, colspecs=col_intervals, names=col_labels,
            dtype={x:str for x in ('RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs')})
    # Re-index to 1-indexed
    df_Rauw.index = list(x+1 for x in range(len(df_Rauw)))
    coords_from_hhmmss(df_Rauw)
    return df_Rauw


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&% Main function %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
"""



def main():
    catalog_reduction_v2()


def catalog_reduction_v2():
    """
    Updated on 4/14/20
    """
    print("All is well so far")
    df = openRauw()
    df.to_html("~/Downloads/test.html", na_rep="")










"""
%&%&%&%&%&%&%&%&&%&%%&%&%&%&%&%&%&%&%&%&&%&%%&%&%
"""
def original_catalog_reduction():
    # OPEN TSUJIMOTO tables
    df_TFT = openTFT_single(1).filter(['RAdeg', 'DEdeg', 'PosErr'])
    df_TFT_cross = openTFT_single(3)
    df_TFT_X = openTFT_single(2)
    for colname in df_TFT_cross:
        df_TFT[colname] = df_TFT_cross[colname]
    for colname in df_TFT_X:
        df_TFT[colname] = df_TFT_X[colname]
    del df_TFT_cross, df_TFT_X

    # Get all ET flagged stars from TFT
    flagged_TFT = df_TFT.loc[df_TFT['F-ID'].notnull()]
    def check_ET(row):
        return 'ET' in row['F-ID']
    flagged_ET_TFT = flagged_TFT[flagged_TFT.apply(check_ET, axis=1)]
    del flagged_TFT


    # OPEN VARGAS-ALVAREZ table 2, full photometry
    df_VA = openVA_full()
    # OPEN VARGAS-ALVAREZ table 6, early type sample
    df_VA_ET = openVA_ET()

    # Get early type rows from VA photom catalog
    VA_ETsubset_photom = df_VA.loc[df_VA_ET.index]
    for colname in ['Spectral', 'subtype']:
        VA_ETsubset_photom[colname] = df_VA_ET[colname]
    del df_VA_ET

    # OPEN VPHAS
    df_VPHAS = openVPHAS()

    # Get VPHAS objects with ST
    VPHAS_ST = df_VPHAS.loc[df_VPHAS['ST'].notnull()]

    # New columns names for synthesized catalog
    VPHAS_columns = ["{:s}__VPHAS".format(s) for s in VPHAS_ST.columns]
    VA_columns = ["{:s}__VA".format(s) for s in VA_ETsubset_photom.columns]
    TFT_columns = ["{:s}__TFT".format(s) for s in flagged_ET_TFT.columns]
    # short name function for column names
    short_name = lambda colname: colname.split('__')[0]
    # initialize the big dataframe
    big_df = pd.DataFrame(columns=VPHAS_columns+VA_columns+TFT_columns)
    # Find out where VPHAS has TFT or VA crossmatches
    VPHAS_hasTFT = VPHAS_ST['TFT_ID'].notnull()
    VPHAS_hasVA = VPHAS_ST['VA_ID'].notnull()

    def xmatch_ids(cat_to_match, index):
        # Tries to find a row in the cat_to_match whose index_name value matches index
        # Returns either a Pandas Series object or None (if row not found)
        # Prints message if multiple rows match (which should not happen)
        if index in cat_to_match.index:
            return cat_to_match.loc[index]
        else:
            return None

    def fill_entry(new_entry, new_column_names, found_row):
        # Check to see if a row has been found
        # If it has, add it to the new_entry using new_column_names
        if found_row is not None:
            for colname in new_column_names:
                if short_name(colname) in found_row:
                    new_entry[colname] = found_row[short_name(colname)]


    TFT_ids = set(flagged_ET_TFT.index)
    VA_ids = set(VA_ETsubset_photom.index) # use set.discard
    entries = []
    for i, row in VPHAS_ST.iterrows():
        new_entry = {}
        fill_entry(new_entry, VPHAS_columns, row)
        if VPHAS_hasTFT[i]:
            TFT_ID = int(new_entry['TFT_ID__VPHAS'])
            fill_entry(new_entry, TFT_columns, xmatch_ids(df_TFT, TFT_ID))
            TFT_ids.discard(TFT_ID)
        if VPHAS_hasVA[i]:
            VA_ID = int(new_entry['VA_ID__VPHAS'])
            xmatched_row = xmatch_ids(VA_ETsubset_photom, VA_ID)
            if xmatched_row is None:
                xmatched_row = xmatch_ids(df_VA, VA_ID)
            fill_entry(new_entry, VA_columns, xmatched_row)
            # Try overwriting with the more limited catalog with spectral types
            VA_ids.discard(VA_ID)
        entries.append(new_entry)
    # Should eventually cross-match VA and TFT before adding
    # I'm sure some of these are redundant
    for VA_ID in sorted(VA_ids):
        new_entry = {}
        # I added columns to the VA table, so grab from that modified table
        fill_entry(new_entry, VA_columns, xmatch_ids(VA_ETsubset_photom, VA_ID))
        entries.append(new_entry)
    for TFT_ID in sorted(TFT_ids):
        new_entry = {}
        # Compiled all the TFT tables, so just grab from the original table
        fill_entry(new_entry, TFT_columns, xmatch_ids(df_TFT, TFT_ID))
        entries.append(new_entry)
    big_df = big_df.append(entries)

    # print(big_df.columns)
    # RADEC list ['RA__VPHAS', 'RAdeg__VA', 'RAdeg__TFT', 'DEC__VPHAS', 'DEdeg__VA', 'DEdeg__TFT']
    radec_df = big_df.filter(['RA__VPHAS', 'RAdeg__VA', 'RAdeg__TFT', 'DEC__VPHAS', 'DEdeg__VA', 'DEdeg__TFT'])
    def gen_make_coords(raname, decname):
        def make_coords(row):
            if math.isnan(row[raname]):
                return row[raname] # NaN
            else:
                return SkyCoord(row[raname], row[decname], unit=u.deg)
        return make_coords
    radec_df['coord__VPHAS'] = radec_df.apply(gen_make_coords('RA__VPHAS', 'DEC__VPHAS'), axis=1)
    radec_df['coord__VA'] = radec_df.apply(gen_make_coords('RAdeg__VA', 'DEdeg__VA'), axis=1)
    radec_df['coord__TFT'] = radec_df.apply(gen_make_coords('RAdeg__TFT', 'DEdeg__TFT'), axis=1)
    radec_df = radec_df.filter(['coord__VPHAS', 'coord__VA', 'coord__TFT'])

    # Get rows with only TFT and nothing else
    only_TFT_i = radec_df.coord__VA.isnull() & radec_df.coord__TFT.notnull() & radec_df.coord__VPHAS.isnull()
    only_TFT = radec_df.loc[only_TFT_i]
    # Get rows with only VA and nothing else
    only_VA_i = radec_df.coord__VA.notnull() & radec_df.coord__TFT.isnull() & radec_df.coord__VPHAS.isnull()
    only_VA = radec_df.loc[only_VA_i]

    # Get distances between all of these (n*m distances)
    # For only-TFTs, place min distance and VA index in two columns
    # THIS WOULD BE FAST AS A GRID (TFT vs VA)
    def minimize_TFT_dist(TFT_row):
        def comparewith(VA_row):
            return TFT_row.coord__TFT.separation(VA_row.coord__VA).arcsec
        seps = only_VA.apply(comparewith, axis=1)
        min_i = seps.idxmin()
        sepmin = seps[min_i]
        return min_i, sepmin
    def minimize_VA_dist(VA_row):
        def comparewith(TFT_row):
            return VA_row.coord__VA.separation(TFT_row.coord__TFT).arcsec
        seps = only_TFT.apply(comparewith, axis=1)
        min_i = seps.idxmin()
        sepmin = seps[min_i]
        return min_i, sepmin

    TFT_mins = only_TFT.apply(minimize_TFT_dist, axis=1)
    TFT_mins = pd.DataFrame(TFT_mins.tolist(), index=TFT_mins.index)
    TFT_mins.columns = ['minmatch_idx', 'minmatch_sep']
    VA_mins = only_VA.apply(minimize_VA_dist, axis=1)
    VA_mins = pd.DataFrame(VA_mins.tolist(), index=VA_mins.index)
    VA_mins.columns = ['minmatch_idx', 'minmatch_sep']
    # Check for mutual matches
    # Merge the TFT row into the VA row and delete the TFT row
    row_indices_to_drop = []
    for i, row in TFT_mins.iterrows():
        # i is the index of the TFT row that matches the VA row
        # j is the index of the VA row that matches the TFT row
        # In general, i > j (VA/j appears earlier)
        j = int(row['minmatch_idx'])
        # if separation is less than half an arcsec and the match is mutual
        if VA_mins.loc[j, 'minmatch_idx'] == i and row['minmatch_sep'] < 0.5:
            # update the VA row with the TFT info
            TFT_row_to_merge = big_df.loc[i]
            TFT_row_to_merge.name = j
            big_df.update(pd.DataFrame([TFT_row_to_merge]))
            big_df.loc[j, 'Notes'] = "crossmatched w/TFT"
            row_indices_to_drop.append(i)
    # Drop the TFT remnants of the merged rows
    big_df = big_df.drop(index=row_indices_to_drop)
    radec_df = radec_df.drop(index=row_indices_to_drop)
    # Reindex to avoid skipping indices, and start at 1 rather than 0
    big_df.index = list(x+1 for x in range(len(big_df)))
    radec_df.index = big_df.index

    # Priority is VA coords since HST has highest resolution
    radec_df['coords'] = radec_df.coord__TFT
    radec_df['coords'].update(radec_df.coord__VPHAS)
    radec_df['coords'].update(radec_df.coord__VA)
    radec_df['RAdeg'] = radec_df.coords.apply(lambda x: x.ra.deg)
    radec_df['DEdeg'] = radec_df.coords.apply(lambda x: x.dec.deg)

    # Priority is VPHAS spectral type since that is most recent work
    radec_df['SpectralType'] = big_df['F-ID__TFT'].where(big_df['F-ID__TFT'].isnull(), other='ET')
    radec_df['SpectralType'].update(big_df['Spectral__VA']+big_df['subtype__VA'])
    radec_df['SpectralType'].update(big_df['ST__VPHAS'])

    radec_df['MSP'] = big_df['F-ID__TFT'].where(big_df['F-ID__TFT'].apply(lambda x: 'MSP' in str(x)))
    radec_df['MSP'].update(big_df['MSP91__VA'])
    radec_df['MSP'].update(big_df['MSP_ID__VPHAS'])

    # radec_df.filter(['RAdeg', 'DEdeg', 'SpectralType', 'MSP']).to_html(catalog_directory+"OBradec.html", na_rep="")
    # radec_df.filter(['RAdeg', 'DEdeg', 'coords', 'SpectralType', 'MSP']).to_pickle(catalog_directory+"OBradec.pkl")


if __name__ == "__main__":
    main()
