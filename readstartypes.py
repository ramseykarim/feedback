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


def ST_parse_type(spectral_type_string):
    # Get type and subtype from spectral type, i.e. O4.5III or B1 V
    subtype = None
    for st in spectral_subtypes:
        if st in spectral_type_string:
            subtype = st
            spectral_type_string = spectral_type_string.replace(st, '')
            break
    letter = spectral_type_string[0]
    number, i = [], 1
    while i < len(spectral_type_string) and spectral_type_string[i] in '1234567890.':
        number.append(spectral_type_string[i])
        i += 1
    return f"{letter}{''.join(number)}", subtype

    
# Then remove f's and *'s and stuff like that

# Then split up the luminosity class from the spectral type


def ST_parse_slash(spectral_type_string):
    if '/' not in spectral_type_string:
        # Check for / and return list(string) if not
        return [spectral_type_string]
    roman_num = '(I+|I?V)'
    peculiar = '(\\(*(f|e)(\\+|\\*)?\\)*\\*?)'
    number = '(\\d\\.?\\d?)' # HAVE NOT TRIED THIS YET
    # Parse the slash notation from SINGLE star (not binary)
    # The O/WN stars are a confusing category: see Wikipedia Wolf-Rayet stars: Slash stars
    if spectral_type_string[spectral_type_string.index('/') + 1] == 'W':
        # This is a "slash star", return both complete spectral types
        return spectral_type_string.split('/')
    # Add possible luminosity classes to this list
    possibile_classes = []
    pattern_lumclass = f'({roman_num}{peculiar}?)/({roman_num}{peculiar}?)'
    match_lumclass = re.search(pattern_lumclass, spectral_type_string)
    if match:
        lumclasses = match.group().split('/')
        for c in lumclasses:
            possibile_classes.append(spectral_type_string.replace(match.group(), c))
    else:
        possibile_classes.append(spectral_type_string)
    # NEED TO GO THRU POSSIBLE_CLASSES AND LOOK FOR POSSIBLE SUBTYPES




def ST_parse_full(spectral_type_string):
    # Work from broadest issue to most specific issues
    # Identify binaries (+)
    stars = spectral_type_string.split('+')



def ST_to_number(spectral_type):
    # Returns a float number based on the spectral type, increasing with later type
    # Just simple letter+number spectral_type, decminal optional
    # Like "O3" or "B1.5"
    # Wow ok this exact system was used in Vacca et al 1996... nice one dude
    type_key = 'OBAFGKM'
    return type_key.index(spectral_type[0])*10. + float(spectral_type[1:])


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

    def lookup_characteristic(spectral_class, characteristic):
        # Spectral class includes luminosity class, i.e. O9.5III or B4V
        # Characteristic is a valid column name, i.e. Teff or log_L
        if characteristic not in self.column_units.index:
            raise RuntimeError(f"{characteristic} is not a recognized column name.")
        # Check if this full spectral type has been searched before, attempt to skip parsing
        if spectral_class in self.memoized_type_names:
            spectral_type, subtype = self.memoized_type_names[spectral_class]
        else:
            spectral_type, subtype = ST_parse_type(spectral_class)
            self.memoized_type_names[spectral_class] = (spectral_type, subtype)
        # Get subtype DataFrame
        df_subtype = self.star_tables[subtype]
        if spectral_type in df_subtype.index:
            # If the spectral type is in this table, return value for characteristic
            return df_subtype[characteristic].loc[spectral_type]
        else:
            # If type is not in table, interpolate between types using number approximation
            if (spectral_type, subtype) in self.memoized_interpolations:
                # Check if there is a saved interpolation for this type/subtype
                interp_function = self.memoized_interpolations[(spectral_type, subtype)]
            else:
                # If we don't, make one
                interp_function = fit_characteristic(df_subtype, characteristic)
                self.memoized_interpolations[(spectral_type, subtype)] = interp_function
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



if __name__ == "__main__":
    roman_num = '(I+|I?V)'
    peculiar = '(\\(*(f|e)(\\+|\\*)?\\)*\\*?)?'
    cat = pd.read_pickle(f"{catalogs_directory}/Ramsey/OBradec.pkl")


