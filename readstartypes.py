# script to read the Sternberg et al spectral types tables
# also will contain functions for parsing the catalog's spectral types
# created: November 11, 2019
__author__ = "Ramsey Karim"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import misc_utils
from scipy.interpolate import interp1d
import re

spectral_subtypes = ('V', 'III', 'I')

catalogs_directory = "../ancillary_data/catalogs/"
spectypes_directory = f"{catalogs_directory}SpecTypes/"
spectypes_table = f"{spectypes_directory}spectypes.txt"
column_name_table = f"{spectypes_directory}colnames.txt"

# search for: standard types, WR, Herbig Ae/Be (type of PMS), and pre main sequence
nonstandard_types_re = '(W(N|C))|(HAeBe)|(PMS)|(C)'
standard_types_re = '[OBAFGKM]'
letter_re = f'({standard_types_re}|{nonstandard_types_re})'
roman_num_re = '(I+|I?V)'
peculiar_re = '((\\+|(ha)|n|(\\(*(f|e)(\\+|\\*)?\\)*))+\\*?)'
number_re = '(\\d{1}(\\.\\d)?)'
slashdash_re = '(/|-)'

INVALID_STAR_FLAG = 999

def table_name(spectral_subtype):
    # Spectral subtypes V, III, and I are available
    return f"{spectypes_directory}class{spectral_subtype}.txt"

def generate_SpectralType_DataFrame():
    # Load column names and units
    with open(column_name_table, 'r') as f:
        colnames = f.readline().split()
        units = f.readline().split()
    # Load spectral types;;;; DO WE USE THESE EVER??
    with open(spectypes_table, 'r') as f:
        spectypes = f.readline().split()
    # Create units table
    col_units = pd.DataFrame(units, index=colnames, columns=['Units'])
    # Create star tables
    spectraltype_dfs = {spectral_subtype: pd.read_table(table_name(spectral_subtype), index_col=0, names=colnames) for spectral_subtype in spectral_subtypes}
    # Fix the Teff column comma issue
    for spectral_subtype in spectral_subtypes:
        spectraltype_dfs[spectral_subtype]['Teff'] = spectraltype_dfs[spectral_subtype].apply(lambda row: float(row['Teff'].replace(',', '')), axis=1)
    return spectraltype_dfs, col_units


def re_parse_helper(pattern, string):
    # Return match string if it exists, else empty string
    match = re.search(pattern, string)
    if match:
        return match.group()
    else:
        return ''


def st_parse_type(spectral_type_string):
    # Parse a single full class string (no slashes or dashes)
    # i.e. O5.5III((f))* => 'O', '5.5', 'III', '((f))*'
    pec = re_parse_helper(peculiar_re, spectral_type_string)
    lumclass = re_parse_helper(roman_num_re, spectral_type_string)
    subtype = re_parse_helper(number_re, spectral_type_string)
    lettertype = re_parse_helper(letter_re, spectral_type_string)
    return (lettertype, subtype, lumclass, pec)


def st_parse_slashdash(spectral_type_string):
    if not re.search(slashdash_re, spectral_type_string):
        # Check for / and return list(string) if not
        return [spectral_type_string]
    # Parse the slash notation from SINGLE star (not binary)
    # The O/WN stars are a confusing category: see Wikipedia Wolf-Rayet stars: Slash stars
    elif '/' in spectral_type_string and spectral_type_string[spectral_type_string.index('/') + 1] == 'W':
        # This is a "slash star", return both complete spectral types
        return spectral_type_string.split('/')
    # Add possible luminosity classes to this list
    possible_lumclasses = []
    pattern_lumclass = f'{roman_num_re}{peculiar_re}?({slashdash_re}{roman_num_re}{peculiar_re}?)+'
    match_lumclass = re.search(pattern_lumclass, spectral_type_string)
    if match_lumclass:
        # print("-----", spectral_type_string, '->', match_lumclass.group())
        possible_lumclasses.extend(match_lumclass.group().replace('-', '/').split('/'))
        for i in range(len(possible_lumclasses)):
            possible_lumclasses[i] = spectral_type_string.replace(match_lumclass.group(), possible_lumclasses[i])
    else:
        possible_lumclasses.append(spectral_type_string)
    possible_classes = []
    for possible_lumclass in possible_lumclasses:
        pattern_subclass = f'{number_re}({slashdash_re}{number_re})+'
        match_subclass = re.search(pattern_subclass, possible_lumclass)
        if match_subclass:
            # print("-----"*2, match_subclass, '->', match_subclass.group())
            possible_subclasses = match_subclass.group().replace('-', '/').split('/')
            for i in range(len(possible_subclasses)):
                possible_classes.append(possible_lumclass.replace(match_subclass.group(), possible_subclasses[i]))
        else:
            possible_classes.append(possible_lumclass)
    return possible_classes


def st_parse_binary(spectral_type_string):
    # Identify binaries (+)
    if spectral_type_string[-1] == '+':
        return [spectral_type_string]
    else:
        return spectral_type_string.split('+')



def st_tuple_to_string(s):
    return s[0] + s[1] + s[2]


def st_to_number(spectral_type):
    # Returns a float number based on the spectral type, increasing with later type
    # Just simple letter+number spectral_type, decminal optional
    # OR tuple ('letter', 'number', 'lumclass', 'peculiarity')
    # Like "O3" or "B1.5"
    # Wow ok this exact system was used in Vacca et al 1996... nice one dude
    type_key = 'OBAFGKM'
    if isinstance(spectral_type, str):
        if re.search(nonstandard_types_re, spectral_type):
            return INVALID_STAR_FLAG
        t = re.search(standard_types_re, spectral_type).group()
        subt = re.search(number_re, spectral_type).group()
    else:
        t, subt = spectral_type[0], spectral_type[1]
    if t not in type_key:
        return INVALID_STAR_FLAG
    else:
        return type_key.index(t)*10. + float(subt)


def lc_to_number(lumclass):
    if isinstance(lumclass, tuple):
        lumclass = lumclass[2]
    if not lumclass:
        return INVALID_STAR_FLAG
    return ['I', 'II', 'III', 'IV', 'V'].index(lumclass) + 1


def st_reduce_to_brightest_star(st):
    # st = string
    st = st_parse_binary(st)
    # st = list(string)
    st = [st_parse_slashdash(x) for x in st]
    # st = list(list(string))
    st = [[st_parse_type(y) for y in x] for x in st]
    # st = list(list(tuple(string)))
    st = [min(x, key=st_to_number) for x in st]
    # st = list(tuple(string))
    st = min(st, key=st_to_number)
    # st = tuple(string)
    return st


def reduce_catalog_spectral_types(cat):
    cat['SpectralType_Adopted'] = cat.SpectralType.where(cat.SpectralType != 'ET', other='O9V', inplace=False)
    cat['SpectralType_ReducedTuple'] = cat.SpectralType_Adopted.apply(st_reduce_to_brightest_star)
    cat['SpectralType_Reduced'] = cat.SpectralType_ReducedTuple.apply(st_tuple_to_string)
    cat['SpectralType_Number'] = cat.SpectralType_Reduced.apply(st_to_number)

"""
===============================================================================
================== Tests ======================================================
===============================================================================
"""


def test_st_parse_slashdash():
    cat = pd.read_pickle(f"{catalogs_directory}/Ramsey/OBradec.pkl")
    tests = [cat.SpectralType[19], cat.SpectralType[5], cat.SpectralType[7], 'B5/6.5III', 'O4II/III', cat.SpectralType[26], cat.SpectralType[27]]
    for t in tests:
        l = st_parse_slashdash(t)
        print(t, '\t', l)
        print('\t', [st_parse_type(x) for x in l])
        print()


def test_full_st_parse():
    cat = pd.read_pickle(f"{catalogs_directory}/Ramsey/OBradec.pkl")
    count = 0
    for st in cat.SpectralType:
        assert isinstance(st, str)
        if st == 'ET':
            st = 'O9.5V'
            count += 1
        stars = [[st_parse_type(x) for x in st_parse_slashdash(y)] for y in st_parse_binary(st)]
        types = [x[0][0:2] for x in stars]
        if types and all(types[0]):
            print([i[0]+i[1] for i in types])
        else:
            print(stars)
    print(f"There are {count} ET stars")




"""
===============================================================================
================== For Sternberg and Vacca ====================================
===============================================================================
"""

def fit_characteristic(df_subtype, characteristic):
    # Get characteristic interp (i.e. "Teff")
    # from df_subtype (i.e. spectral_type_df_dict["III"])
    independent, dependent = np.array([st_to_number(x) for x in df_subtype.index]), df_subtype[characteristic]
    interp_from_number = interp1d(independent, dependent, kind='linear')
    def interp_function(spectral_type):
        try:
            return interp_from_number(st_to_number(spectral_type))
        except:
            return 0
    return interp_function


class SpectralTypeTables:

    def __init__(self):
        self.star_tables, self.column_units = generate_SpectralType_DataFrame()
        self.memoized_interpolations = {}
        self.memoized_type_names = {}

    def lookup_characteristic(self, spectral_type_tuple, characteristic):
        # Spectral type tuple can be 3 or 4 elements (4th is peculiarity, ignored)
        # Characteristic is a valid column name, i.e. Teff or log_L
        if characteristic not in self.column_units.index:
            raise RuntimeError(f"{characteristic} is not a recognized column name.")
        # Discard peculiarity and sanitize input
        spectral_type_tuple = spectral_type_tuple[:3]
        spectral_type_tuple = self.sanitize_tuple(spectral_type_tuple)
        if not spectral_type_tuple:
            return 0
        lettertype, subtype, lumclass = spectral_type_tuple
        # Get subtype DataFrame
        df_lumclass = self.star_tables[lumclass]
        if lettertype+subtype in df_lumclass.index:
            # If the spectral type is in this table, return value for characteristic
            return df_lumclass[characteristic].loc[lettertype+subtype]
        else:
            # If type is not in table, interpolate between types using number approximation
            if spectral_type_tuple in self.memoized_interpolations:
                # Check if there is a saved interpolation for this type/subtype
                interp_function = self.memoized_interpolations[spectral_type_tuple]
            else:
                # If we don't, make one
                interp_function = fit_characteristic(df_lumclass, characteristic)
                self.memoized_interpolations[spectral_type_tuple] = interp_function
            # Use interpolation to return a value for the characteristic
            return interp_function(spectral_type_tuple)

    def sanitize_tuple(self, spectral_type_tuple):
        if re.search(nonstandard_types_re, spectral_type_tuple[0]):
            return False
        if not re.search(roman_num_re, spectral_type_tuple[2]):
            spectral_type_tuple = list(spectral_type_tuple)
            spectral_type_tuple[2] = 'V'
            return tuple(spectral_type_tuple)
        return spectral_type_tuple


    def lookup_units(self, characteristic):
        if characteristic not in self.column_units.index:
            raise RuntimeError(f"{characteristic} is not a recognized column name.")
        return self.column_units['Units'].loc[characteristic]


def get_catalog_properties_sternberg(cat, characteristic):
    sternberg_tables = SpectralTypeTables()
    cat[characteristic] = cat.SpectralType_ReducedTuple.apply(sternberg_tables.lookup_characteristic, args=(characteristic,))


def vacca_calibration(spectral_type_number, luminosity_class_number, characteristic):
    # Coding up Vacca 1996 Table 4
    if luminosity_class_number == 1 and spectral_type_number > 9.5:
        raise RuntimeWarning('LC I(a) stars above O9.5 are not supported by this calibration.')
    coefficients = {
        'Teff': (59.85, -3.10, -0.19, 0.11),
        'log_g': (4.429, -0.140, -0.039, 0.022), # evolved g
    }
    multiplier = {'Teff': 1e3, 'log_g': 1}
    S, L = spectral_type_number, luminosity_class_number
    A, B, C, D = coefficients[characteristic]
    return (A + (B*S) + (C*L) + (D*S*L)) * multiplier[characteristic]

def plot_sternberg_stuff():
    dfs, col_units = generate_SpectralType_DataFrame()
    colors = {'I':'red','III':'green','V':'blue'}
    plt.figure(figsize=(14, 9))
    for spectral_subtype in spectral_subtypes:
        Teff, log_g = 'Teff', 'log_g'
        charX, charY = "Spectral Type", Teff
        independent, dependent = np.array([st_to_number(i) for i in  dfs[spectral_subtype].index]), dfs[spectral_subtype][charY]
        plt.plot(independent, dependent, '.', color=colors[spectral_subtype])
        fit = interp1d(independent, dependent, kind='linear')
        x = np.linspace(independent.min(), independent.max(), 50)
        plt.plot(x, fit(x), '--', label='fit to Sternberg+2003 class {}'.format(spectral_subtype), color=colors[spectral_subtype])
        plt.plot(x, [vacca_calibration(i, {'I':1,'III':3,'V':5}[spectral_subtype], charY) for i in x], '-', color=colors[spectral_subtype], label='Vacca+1996 calib. class {}'.format(spectral_subtype))
    plt.legend()
    plt.ylabel(charY), plt.xlabel(charX)
    plt.show()



if __name__ == "__main__":
    cat = pd.read_pickle(f"{catalogs_directory}/Ramsey/OBradec.pkl")
    reduce_catalog_spectral_types(cat)

