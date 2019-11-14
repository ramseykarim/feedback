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

letter_re = '((W(N|C))|[OBAFGKM])'
roman_num_re = '(I+|I?V)'
peculiar_re = '(((ha)|n|(\\(*(f|e)(\\+|\\*)?\\)*))+\\*?)'
number_re = '(\\d{1}(\\.\\d)?)'
slashdash_re = '(/|-)'


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
    return spectral_type_string.split('+')


def st_to_number(spectral_type):
    # Returns a float number based on the spectral type, increasing with later type
    # Just simple letter+number spectral_type, decminal optional
    # Like "O3" or "B1.5"
    # Wow ok this exact system was used in Vacca et al 1996... nice one dude
    type_key = 'OBAFGKM'
    return type_key.index(spectral_type[0])*10. + float(spectral_type[1:])


def lc_to_number(lumclass):
    return ['I', 'II', 'III', 'IV', 'V'].index(lumclass) + 1


def fit_characteristic(df_subtype, characteristic):
    # Get characteristic interp (i.e. "Teff")
    # from df_subtype (i.e. spectral_type_df_dict["III"])
    independent, dependent = np.array([ST_to_number(x) for x in df_subtype.index]), df_subtype[characteristic]
    interp_from_number = interp1d(independent, dependent, kind='linear')
    def interp_function(spectral_type):
        return interp_from_number(ST_to_number(spectral_type))
    return interp_function


class SpectralTypeTables:

    def __init__(self):
        self.star_tables, self.column_units = generate_SpectralType_DataFrame()
        self.memoized_interpolations = {}
        self.memoized_type_names = {}

    def lookup_characteristic(spectral_type_tuple, characteristic):
        # Spectral type tuple can be 3 or 4 elements (4th is peculiarity, ignored)
        # Characteristic is a valid column name, i.e. Teff or log_L
        if characteristic not in self.column_units.index:
            raise RuntimeError(f"{characteristic} is not a recognized column name.")
        # Discard peculiarity
        spectral_type_tuple = spectral_type_tuple[:3]
        lettertype, subtype, lumclass = spectral_type_tuple
        # Get subtype DataFrame
        df_subtype = self.star_tables[lumclass]
        if spectral_type in df_subtype.index:
            # If the spectral type is in this table, return value for characteristic
            return df_subtype[characteristic].loc[spectral_type]
        else:
            # If type is not in table, interpolate between types using number approximation
            if spectral_type_tuple in self.memoized_interpolations:
                # Check if there is a saved interpolation for this type/subtype
                interp_function = self.memoized_interpolations[spectral_type_tuple]
            else:
                # If we don't, make one
                interp_function = fit_characteristic(df_subtype, characteristic)
                self.memoized_interpolations[spectral_type_tuple] = interp_function
            # Use interpolation to return a value for the characteristic
            return interp_function(spectral_type)


    def lookup_units(characteristic):
        if characteristic not in self.column_units.index:
            raise RuntimeError(f"{characteristic} is not a recognized column name.")
        return self.column_units['Units'].loc[characteristic]


def plot_sternberg_stuff():
    dfs, col_units = generate_SpectralType_DataFrame()
    plt.figure(figsize=(14, 9))
    for spectral_subtype in spectral_subtypes:
        Teff, log_g = 'Teff', 'log_g'
        independent, dependent = dfs[spectral_subtype][Teff], dfs[spectral_subtype][log_g]
        fit = interp1d(independent, dependent, kind='linear')
        plt.plot(independent, dependent, '.')
        x = np.linspace(independent.min(), independent.max(), 50)
        plt.plot(x, fit(x), '--', label=spectral_subtype)
    plt.legend()
    plt.ylabel(log_g), plt.xlabel(Teff)
    plt.show()


def test_ST_parse_slashdash():
    cat = pd.read_pickle(f"{catalogs_directory}/Ramsey/OBradec.pkl")
    tests = [cat.SpectralType[19], cat.SpectralType[5], cat.SpectralType[7], 'B5/6.5III', 'O4II/III', cat.SpectralType[26], cat.SpectralType[27]]
    for t in tests:
        l = ST_parse_slashdash(t)
        print(t, '\t', l)
        print('\t', [ST_parse_type(x) for x in l])
        print()



if __name__ == "__main__":
    test_ST_parse_slashdash()
