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
from astropy import units as u

spectral_subtypes = ('V', 'III', 'I')

catalogs_directory = "../ancillary_data/catalogs/"
spectypes_directory = f"{catalogs_directory}SpecTypes/"
spectypes_table = f"{spectypes_directory}spectypes.txt"
column_name_table = f"{spectypes_directory}colnames.txt"
powr_directory = f"{catalogs_directory}SpecTypes/PoWR/"

# search for: standard types, WR, Herbig Ae/Be (type of PMS), and pre main sequence
nonstandard_types_re = '(W(N|C))|(HAeBe)|(PMS)|(C)'
standard_types_re = '[OBAFGKM]'
letter_re = f'({standard_types_re}|{nonstandard_types_re})'
roman_num_re = '(I+|I?V)'
peculiar_re = '((\\+|(ha)|n|(\\(*(f|e)(\\+|\\*)?\\)*))+\\*?)'
number_re = '(\\d{1}(\\.\\d)?)'
slashdash_re = '(/|-)'

INVALID_STAR_FLAG = 999


"""
===============================================================================
================== Spectral Type parsing via regex ============================
===============================================================================
"""


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
================== For Sternberg and Vacca ====================================
===============================================================================
"""

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


"""
===============================================================================
================== For PoWR  ==================================================
===============================================================================
"""

def skiplines(file_handle, n_lines):
    # skips ahead n_lines in the already-opened file_handle
    for i in range(n_lines):
        file_handle.readline()

def load_powr_grid_info(grid_name):
    fn = f"{powr_directory}{grid_name}/modelparameters.txt"
    with open(fn) as f:
        skiplines(f, 5)
        colnames = [s.replace(' ', '_') for s in re.split('\s{2,}', f.readline().strip())]
        skiplines(f, 2)
        tbl = pd.read_table(f, names=colnames, sep='\s+')
    return tbl

# These four trim functions round the grid parameters to the nearest gridpoint

def trim_Teff(Teff):
    # Rounds Teff to nearest kK
    return round(Teff, ndigits=-3)

def trim_logTeff(Teff):
    # Rounds log Teff to nearest 20th (e.g. 4.65, 4.7, 4.75, ...)
    return round(Teff*20)/20

def trim_logg(log_g):
    # Rounds log_g to nearest even-decimal
    return round(log_g*5)/5

def trim_logRt(Rt):
    # Rounds log Rt to nearest 10th
    return round(Rt, ndigits=1)

def FUV_nonionizing_mask(wl_A):
    # wl_A should be a scipy.units quantity
    energy_eV = wl_A.to(u.eV, equivalencies=u.spectral()).to_value()
    return (energy_eV > 6) & (energy_eV < 13.6)

class PoWRGrid:

    def __init__(self, grid_name):
        if grid_name not in ['OB', 'WNE', 'WNL']:
            raise RuntimeError(f"{grid_name} not available")
        self.grid_name = grid_name
        self.grid_info = load_powr_grid_info(self.grid_name)
        self.paramx, self.paramy = None, None
        self.paramx_name, self.paramy_name = None, None
        self.get_params()

    def get_model_info(self, *args):
        qparamx, qparamy = self.parse_query_params(*args)
        # If this EXACT combo is NOT present in the grid, return an error!
        # close enough is NOT close enough!!!!
        model = self.grid_info.loc[(self.paramx == qparamx) & (self.paramy == qparamy)]
        if model.empty:
            raise RuntimeError(f"Could not find x: {qparamx} / y: {qparamy} model in the {self.grid_name} grid.")
        else:
            return model.loc[model.index[0]]

    def get_model_filename(self, *args):
        # args are either (Teff, log_g) or the pandas.Series result from above
        if len(args) == 2:
            model = self.get_model_info(*args)
        else:
            model = args[0]
        suffix = '-i' if self.grid_name == 'OB' else ''
        return f'{powr_directory}{self.grid_name}/{self.grid_name.lower()}{suffix}_{model.MODEL}_sed.txt'

    def get_model(self, *args):
        data = np.genfromtxt(self.get_model_filename(*args))
        wl = (10**data[:, 0]) * u.Angstrom
        # flux comes in erg/ (cm2 s A) at 10pc, so convert to area-integrated flux
        flux_units = (u.erg / (u.s * u.Angstrom))
        total_flux_units = flux_units * (4*np.pi * (10*u.pc.to('cm'))**2)
        flux = (10**data[:, 1]) * total_flux_units
        return wl, flux

    def get_params(self):
        # Identify the grid parameters based on the type of grid
        self.paramx = self.grid_info.T_EFF
        self.paramx_name = "T_EFF"
        if self.grid_name == 'OB':
            self.paramx = self.paramx.apply(trim_Teff)
            self.paramy = self.grid_info.LOG_G.apply(trim_logg)
            self.paramy_name = "LOG_G"
        else:
            self.paramx = np.log10(self.paramx).apply(trim_logTeff)
            self.paramy = np.log10(self.grid_info.R_TRANS).apply(trim_logRt)
            self.paramx_name = "LOG_" + self.paramx_name
            self.paramy_name = "LOG_R_TRANS"

    def parse_query_params(self, *args):
        # takes input from self.get_model_info
        # figures out how to turn it into grid parameters
        if self.grid_name == 'OB':
            # input is Teff, log_g
            Teff, log_g = args
            # Give Teff in K, log_g in dex
            return trim_Teff(Teff), trim_logg(log_g)
        else:
            # if input is 2 elements, Teff and Rt
            # input is Teff, Rstar, Mdot (LINEAR not log)
            # vinf and D are assumed based on the grid type
            if len(args) > 2:
                Teff, Rstar, Mdot = args
                if self.grid_name == 'WNE':
                    vinf, D = 1600, 4
                elif self.grid_name == 'WNL':
                    vinf, D = 1000, 4
                else:
                    raise RuntimeError("grid name not valid")
                Rt = self.calculate_Rt(Rstar, Mdot, vinf, D)
            else:
                Teff, Rt = args
            if Teff > 10:
                Teff = np.log10(Teff)
            if Rt > 1.7:
                # logRt and Rt overlap between 1.2 and 1.7. Assume log.
                Rt = np.log10(Teff)
            Teff = trim_logTeff(Teff)
            Rt = trim_logRt(Rt)
            return Teff, Rt

    def plot_grid_space(self, c=None, clabel=None):
        plt.figure(figsize=(13, 9))
        plt.scatter(self.paramx, self.paramy, c=c)
        plt.xlabel(self.paramx_name)
        plt.ylabel(self.paramy_name)
        if c is not None:
            plt.colorbar(label=clabel)
        plt.show()

    def iter_models(self):
        return self.grid_info.itertuples()

    @staticmethod
    def plot_spectrum(wl, flux, setup=True, show=True, fuv=False,
            xlim=None, ylim=None, xunit=None,
            xlog=True, ylog=True):
        if setup:
            plt.figure(figsize=(13, 9))
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
        if fuv:
            mask = FUV_nonionizing_mask(wl)
            wl, flux = wl[mask], flux[mask]
        if xunit:
            wl = wl.to(xunit, equivalencies=u.spectral())
        plt.plot(wl, flux)
        if setup:
            plt.xlabel(f'wavelength ({wl.unit.to_string()})')
            plt.ylabel(f'flux ({flux.unit.to_string()})')
            if xlog:
                plt.xscale('log')
            if ylog:
                plt.yscale('log')
        if show:
            plt.show()

    @staticmethod
    def integrate_flux(wl, flux):
        # integrates flux from 6 to 13.6 eV
        # wl is in Angstroms
        mask = FUV_nonionizing_mask(wl)
        lum = np.trapz(flux[mask], x=wl[mask]).to('solLum')
        return lum

    @staticmethod
    def calculate_Rt(Rstar, Mdot, vinf, D):
        v = vinf/2500.
        M = Mdot*1e4 * np.sqrt(D)
        return Rstar * (v/M)**(2./3)


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


def test_powr_totalL_accuracy():
    tbl = PoWRGrid()
    for model_info in tbl.iter_models():
        wl, flux = tbl.get_model(model_info)
        lum = np.trapz(flux, x=wl).to('solLum')
        print(f"Model {model_info.MODEL} (Teff{model_info.T_EFF}/log_g{model_info.LOG_G}:")
        print(f" -> Integrated luminosity (numpy): {lum.to_string()}")
        print(f" -> Accuracy: {lum.to_value() / (10**model_info.LOG_L)}")
        print()


def test_powr_totalL_plot():
    tbl = PoWRGrid()
    tbl.plot_grid_space(tbl.grid_info.LOG_L, "log L")


if __name__ == "__main__":
    tbl = PoWRGrid('WNL')
    wf1 = tbl.get_model(4.4, 1.7)
    wf2 = tbl.get_model(5.0, 0.0)
    tbl.plot_spectrum(*wf1, show=False, xunit=u.eV, fuv=True, ylog=False)
    tblob = PoWRGrid('OB')
    wf3 = tblob.get_model(43000, 3.8)
    tbl.plot_spectrum(*wf2, show=False, setup=False, xunit=u.eV, fuv=True, ylog=False)
    wf4 = tblob.get_model(16000, 3.0)
    tbl.plot_spectrum(*wf3, show=False, setup=False, xunit=u.eV, fuv=True, ylog=False)
    tbl.plot_spectrum(*wf4, setup=False, xunit=u.eV, fuv=True, ylog=False)
