"""
script to read the Sternberg et al spectral types tables
also will contain functions for parsing the catalog's spectral types
file used to be called readstartypes.py
created: November 11, 2019

I am returning to this code to treat uncertainties and binaries properly
Wow there's a lot of regex in here, good job past Ramsey
revisited: April 29-30, 2020

Returning once again to update mass loss rates with models from
Leitherer et al 2010, who uses WM-basic (Pauldrach et al 2001).
The Sternberg models don't take clumping into account, so their mass loss rates
are likely overestimated (Puls et al 2008).
Leitherer et al are responsible for Starburst 99.
revisited: June 1, 2020
"""
__author__ = "Ramsey Karim"


import sys
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from astropy import units as u

import misc_utils
import catalog_utils


luminosity_classes = ('V', 'III', 'I')

catalogs_directory = f"{catalog_utils.ancillary_data_path}catalogs/"
spectypes_directory = f"{catalog_utils.misc_data_path}SpectralTypes/"

powr_directory = f"{spectypes_directory}PoWR/"

sternberg_path = f"{spectypes_directory}Sternberg/"
spectypes_table = f"{sternberg_path}spectypes.txt"
column_name_table = f"{sternberg_path}colnames.txt"

# search for: standard types, WR, Herbig Ae/Be (type of PMS), and pre main sequence
nonstandard_types_re = '(W(N|C))|(HAeBe)|(PMS)|(C)'
standard_types_re = '[OBAFGKM]'
letter_re = f'({standard_types_re}|{nonstandard_types_re})'
roman_num_re = '(I+|I?V)'
peculiar_re = '((\\+|(ha)|n|(\\(*(f|e)(\\+|\\*)?\\)*))+\\*?)'
number_re = '(\\d{1}(\\.\\d)?)'
slashdash_re = '(/|-)'

INVALID_STAR_FLAG = float('NaN')


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
    """
    Parse a single full class string (no slashes or dashes)
    i.e. O5.5III((f))* => 'O', '5.5', 'III', '((f))*'
    :param spectral_type_string: string descrbing spectral type
        No uncertainty ('/' or '-') and no binaries ('+')
        Peculiarities are ok
    :returns: tuple(string), set up like:
        (letter, number, luminosity_class, peculiarity)
    """
    pec = re_parse_helper(peculiar_re, spectral_type_string)
    lumclass = re_parse_helper(roman_num_re, spectral_type_string)
    subtype = re_parse_helper(number_re, spectral_type_string)
    lettertype = re_parse_helper(letter_re, spectral_type_string)
    return (lettertype, subtype, lumclass, pec)


def st_parse_slashdash(spectral_type_string):
    """
    Take a spectral type string and return a list of all uncertain
        possibilities based on that string's use of dashes and slashes
    :param spectral_type_string: string descrbing spectral type
        No '+' signs, cannot be binary
        Peculiarities are ok
    :returns: list(string) where strings are possible spectral types
    """
    # Editorial note (April 29, 2020): seems we replace '-' with '/', which for uncertainty
    #   purposes seems fair.
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
        elif '/' in possible_lumclass and possible_lumclass[possible_lumclass.index('/') + 1] in 'OBAFGKM':
            # This could be two different letter types, just split along slash and proceed with possiblities
            # I added this on April 30, 2020, and I did not extensively test it.
            possible_classes.extend(possible_lumclass.split('/'))
        else:
            possible_classes.append(possible_lumclass)
    return possible_classes


def st_parse_binary(spectral_type_string):
    """
    Identify binaries (+)
    :param spectral_type_string: string describing spectral type
    :returns: list(string) where strings are binary components
    """
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


def number_to_st(spectral_type_number):
    """
    Written: April 29, 2020
    Reverse of st_to_number
    Returns the tuple expression of spectral type; type and subtype only
        e.g. 11.5 -> ('B', '1.5')
    """
    t = int(spectral_type_number//10)
    subt = spectral_type_number % 10
    return ('OBAFGKM'[t], f"{subt:.1f}")


def lc_to_number(lumclass):
    if isinstance(lumclass, tuple):
        lumclass = lumclass[2]
    if not lumclass:
        # Edited April 29, 2020: I used to return INVALID_STAR_FLAG here,
        #   but now I'm going to assign 'V' because that makes more sense
        lumclass = 'V'
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
    """
    Created: Unsure, probably November 2019
    Reviewed April 29, 2020
    This seems to be where I make a lot of my assumptions and boil down the
        spectral types, getting rid of binaries and glossing over uncertainties
        (ranges of possible types)
    """
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
    """
    Filenames of Sternberg tables
    """
    # Spectral subtypes V, III, and I are available
    return f"{spectypes_directory}class{spectral_subtype}.txt"


def generate_SpectralType_DataFrame():
    """
    Load Sternberg tables as DataFrames into a dictionary
    """
    # Load column names and units
    with open(column_name_table, 'r') as f:
        colnames = f.readline().split()
        units = f.readline().split()
    # Load spectral types;;;; DO WE USE THESE EVER??
    # Editorial note (April 29, 2020) it seems these are just a list of spectral types, which doesn't seem all that special
    with open(spectypes_table, 'r') as f:
        spectypes = f.readline().split()
    # Create units table
    col_units = pd.DataFrame(units, index=colnames, columns=['Units'])
    # Create star tables
    lc_dfs = {lc: pd.read_table(table_name(lc), index_col=0, names=colnames) for lc in luminosity_classes}
    # Fix the Teff column comma issue
    for lc in luminosity_classes:
        lc_dfs[lc]['Teff'] = lc_dfs[lc].apply(lambda row: float(row['Teff'].replace(',', '')), axis=1)
    return lc_dfs, col_units


def fit_characteristic(df_subtype, characteristic):
    # Get characteristic interp (i.e. "Teff")
    # from df_subtype (i.e. spectral_type_df_dict["III"])
    independent, dependent = np.array([st_to_number(x) for x in df_subtype.index]), df_subtype[characteristic]
    interp_from_number = interp1d(independent, dependent, kind='linear')
    def interp_function(spectral_type):
        try:
            return interp_from_number(st_to_number(spectral_type))
        except:
            return np.nan
    return interp_function


class S03_OBTables:
    """
    This is a wrapper class for the Sternberg tables (Sternberg et al. 2003)
    You should be able to give the "spectral type tuple", which is
        something like ("O", "7.5", "V"), and you can call for characteristics
        in the Sternberg tables.
    The characteristic lookup will first try to match your spectral type
        exactly, and next will interpolate using the "spectral type to number"
        map that Vacca used. This is NOT Vacca's Teff, log_g calibration (though
        Sternberg says they use Vacca's calibrations for some things.)
    """
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
    """
    DataFrame-friendly version of S03_OBTables.lookup_characteristic
    These are the Sternberg tables
    cat is a pandas DataFrame with a SpectralType_ReducedTuple column
    """
    sternberg_tables = S03_OBTables()
    cat[characteristic+'_S03'] = cat.SpectralType_ReducedTuple.apply(sternberg_tables.lookup_characteristic, args=(characteristic,))


def plot_sternberg_stuff():
    dfs, col_units = generate_SpectralType_DataFrame()
    colors = {'I':'red','III':'green','V':'blue'}
    plt.figure(figsize=(14, 9))
    for lc in luminosity_classes:
        Teff, log_g = 'Teff', 'log_g'
        charX, charY = "Spectral Type", Teff
        independent, dependent = np.array([st_to_number(i) for i in  dfs[lc].index]), dfs[lc][charY]
        plt.plot(independent, dependent, '.', color=colors[lc])
        fit = interp1d(independent, dependent, kind='linear')
        x = np.linspace(independent.min(), independent.max(), 50)
        plt.plot(x, fit(x), '--', label='fit to Sternberg+2003 class {}'.format(lc), color=colors[lc])
        plt.plot(x, [vacca_calibration((*number_to_st(i), lc), charY) for i in x], '-', color=colors[lc], label='Vacca+1996 calib. class {}'.format(lc))
        # plt.plot(x, [vacca_calibration(i, {'I':1,'III':3,'V':5}[lc], charY) for i in x], '-', color=colors[lc], label='Vacca+1996 calib. class {}'.format(lc))
    plt.legend()
    plt.ylabel(charY), plt.xlabel(charX)
    plt.show()


def vacca_calibration(spectral_type_tuple, characteristic):
    """
    Coding up Vacca 1996 Table 4
    Created: November 2019
    Rewritten: April 29, 2020
    Replacing the old "vacca_calibration" function that has a very different
        call signature than S03_OBTables.lookup_characteristic.
    :param spectral_type_tuple: spectral type expressed as a tuple
    :param characteristic: string characteristic descriptor. For Vacca,
        this would be "Teff" or "log_g"
    """
    spectral_type_number = st_to_number(spectral_type_tuple)
    luminosity_class_number = lc_to_number(spectral_type_tuple[2])
    # Original vacca_calibration code below here
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


def get_catalog_properties_vacca(cat, characteristic):
    """
    DataFrame-friendly version of vacca_calibration
    cat is a pandas DataFrame with a SpectralType_ReducedTuple column
    """
    cat[characteristic+"_V96"] = cat.SpectralType_ReducedTuple.apply(vacca_calibration)


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
        """
        What does this function do? See below: (April 29, 2020)
        This function takes in the parameter combo, whatever it may be,
            and returns the model info as a pandas dataframe. The model info
            is a single row from the "modelparameters.txt" file.
        """
        qparamx, qparamy = self.parse_query_params(*args)
        # If this EXACT combo is NOT present in the grid, return an error!
        # close enough is NOT close enough!!!!
        # Editorial note (April 2020): it seems we are approximating, and that's probably fine
        model = self.grid_info.loc[(self.paramx == qparamx) & (self.paramy == qparamy)]
        if model.empty:
            return self.grid_info.loc[((self.paramx - qparamx)**2 + (self.paramy - qparamy)**2).idxmin()]
        else:
            return model.loc[model.index[0]]
            # if model.empty:
            #     raise RuntimeError(f"Could not find x: {qparamx} / y: {qparamy} model in the {self.grid_name} grid.")
        return model

    def get_model_filename(self, *args):
        # args are either (Teff, log_g) or the pandas.Series result from above (or dictionary made from it)
        if len(args) == 2:
            model = self.get_model_info(*args)
        else:
            model = args[0]
        suffix = '-i' if self.grid_name == 'OB' else ''
        model_id = model['MODEL']
        return f'{powr_directory}{self.grid_name}/{self.grid_name.lower()}{suffix}_{model_id}_sed.txt'

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

    def plot_grid_space(self, c=None, clabel=None, setup=True, show=True,
        **plot_kwargs):
        if setup:
            plt.figure(figsize=(13, 9))
        plt.scatter(self.paramx, self.paramy, c=c, **plot_kwargs)
        plt.xlabel(self.paramx_name)
        plt.ylabel(self.paramy_name)
        if (c is not None) and not (isinstance(c, str)):
            plt.colorbar(label=clabel)
        if show:
            plt.show()

    def iter_models(self):
        return self.grid_info.itertuples()

    @staticmethod
    def plot_spectrum(*args, setup=True, show=True, fuv=False,
            xlim=None, ylim=None, xunit=None,
            xlog=True, ylog=True, label=None):
        if len(args) == 2:
            wl, flux = args
        else:
            wl, flux = args[0]
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
        plt.plot(wl, flux, label=label)
        if setup:
            plt.xlabel(f'wavelength ({wl.unit.to_string()})')
            plt.ylabel(f'flux ({flux.unit.to_string()})')
            if xlog:
                plt.xscale('log')
            if ylog:
                plt.yscale('log')
        if show:
            plt.legend()
            plt.show()

    @staticmethod
    def integrate_flux(*args):
        if len(args) == 2:
            wl, flux = args
        else:
            wl, flux = args[0]
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
================== All-in-one reading object ==================================
===============================================================================
April 29, 2020
The idea here is to wrap all the spectral type reading into one object
    that handles binaries and uncertainties well.
Each spectral type will be wrapped in an instance of this object. The object
    will have common methods for fluxes/stellar winds and their
    associated uncertainties, but will handle binaries and spectral types with
    ranges differently under the hood.
"""

class STResolver:
    """
    Spectral Type Resolver
    Read a single spectral type string and figure it all out
    The string can contain binaries and uncertainties
    This works best with OB types, but can handle WR types if we hardcode the
        parameters (which seems necessary....)
    Written: April 29-30, 2020
    This class is complete. Usage should be something like this:
    >>> powr_grids = {x: PoWRGrid(x) for x in ('OB', 'WNL', 'WNE')}
    >>> sternberg_tables = S03_OBTables()
    >>> s = STResolver("O7.5V")
    >>> s.link_powr_grids(powr_grids)
    >>> fuv_flux = s.get_FUV_flux()
    >>> m_dot = s.get_mass_loss_rate(sternberg_tables)
    >>> v_inf = s.get_terminal_wind_velocity(sternberg_tables)
    """

    """
    Hardcoded parameters
    """

    wr_params = {
        ("WN", "6"): (43000, PoWRGrid.calculate_Rt(19.7, 8.5e-6, 1600., 4.)),
    }

    """
    Setup
    """

    def __init__(self, st):
        """
        Taking a lot of cues from st_reduce_to_brightest_star
        :param st: string spectral type, like "O3-5I/III(f)"
        """
        # Dictionary holding the spectral types of binary components,
        #   decomposed as lists into all their possibilities
        self.spectral_types = {}
        # st is a string
        for st_binary_component in st_parse_binary(st):
            # st_binary_component is a string
            st_bc_possibilities = st_parse_slashdash(st_binary_component)
            # st_bc_possibilities is a list(string)
            st_bc_possibilities_t = [st_parse_type(x) for x in st_bc_possibilities]
            # st_bc_possibilities_t is list(tuple(string))
            # Compose the components dictionary
            """
            self.spectral_types is a map between a single component's spectral
              type string and the list of possibilities, with possibilities
              represented in tuple format:
                  (letter, number, luminosity_class, peculiarity)
            """
            self.spectral_types[st_binary_component] = st_bc_possibilities_t
        self.get_search_parameters()

    def isbinary(self):
        """
        :returns: boolean, True if this is a binary system, False if singular
        """
        return len(self.spectral_types) > 1


    def get_search_parameters(self):
        """
        Get the inputs to the PoWR grid using either the Vacca calibration
            or a hardcoded list of WR parameters
        These are paramx, paramy of the grid
        For OB stars, that's Teff and log_g
        For WR stars, that's Teff and R_trans
        This is run as part of __init__
        """
        # Make a function to get params from a spectral type tuple
        def get_params(st_tuple):
            # st_tuple is a tuple representing spectral type of a single
            #   component possibility
            if STResolver.isWR(st_tuple):
                # This is a WR; use hardcoded parameters
                params = self.get_WR_params(st_tuple)
            else:
                # This is an OB; use Vacca calibration
                paramx = vacca_calibration(st_tuple, 'Teff')
                paramy = vacca_calibration(st_tuple, 'log_g')
                params = (paramx, paramy)
            return params
        # Iterate over the self.spectral_types dictionary
        # Use that nifty function I wrote; "dictionaries" is left default
        self.powr_parameters = STResolver.map_to_components(get_params, (self.spectral_types,))

    def link_powr_grids(self, powr_dict):
        """
        Get PoWR model names for each eligible star/possibility
        This does not collect the full UV spectra, just the parameters.
        ****
        This needs to be called by the user since it requires PoWR grids as input.
        ****
        :param powr_dict: dictionary mapping grid_name to the grid object,
            represented by PoWRGrid instance. Grid name is PoWRGrid.grid_name
        """
        # Make a function to get a PoWR model for a spectral type tuple
        def find_model(st_tuple, params):
            # Get the name of the grid (WNE, OB, etc)
            selected_grid_name = STResolver.select_powr_grid(st_tuple)
            # If there is no grid, return None
            if selected_grid_name is None:
                return None
            # Get the grid
            selected_grid = powr_dict[selected_grid_name]
            # If the parameters are NaN, return None
            if np.any(np.isnan(params)):
                return None
            # Get the model (pandas df), cast as dict (this works, I checked)
            model_info = dict(selected_grid.get_model_info(*params))
            # Attach the PoWR grid object so we can look up the flux
            model_info['grid'] = selected_grid
            return model_info
        self.powr_models = STResolver.map_to_components(find_model, (self.spectral_types, self.powr_parameters))

    """
    ============================================================================
    ==================== Property-finding functions ============================
    ============================================================================
    """

    def get_mass_loss_rate(self, sternberg_tables):
        """
        Get the stellar wind mass loss rate in Msun / year
        The source of this information is different for OB vs WR;
            OB uses Sternberg tables (so they are needed as arg here)
            and WR uses PoWR simulations
        :param sternberg_tables: a S03_OBTables instance
        """
        # Make a mass loss rate finding function
        def find_mass_loss_rate(st_tuple, model_info):
            # Takes st_tuple (self.spectral_types) and PoWR model_info (self.powr_models)
            # Set up return unit:
            mdot_unit = u.solMass / u.yr
            # First, check for model_info at all. All OBs and valid WRs should have it
            if model_info is None:
                # This star won't be in PoWR or Sternberg (invalid WR or odd type)
                mdot = np.nan
            elif STResolver.isWR(st_tuple):
                # This is a WR with a model; use PoWR
                mdot = 10.**(model_info['LOG_MDOT'])
            else:
                # This must be an OB star, use Sternberg
                mdot = sternberg_tables.lookup_characteristic(st_tuple, 'Mdot')
                if np.isnan(mdot):
                    # Not found in Sternberg tables; default to PoWR
                    mdot = 10.**(model_info['LOG_MDOT'])
            return mdot * mdot_unit
        component_mdots = STResolver.map_to_components(find_mass_loss_rate, (self.spectral_types, self.powr_models))
        return STResolver.resolve_uncertainty(component_mdots)

    def get_terminal_wind_velocity(self, sternberg_tables):
        """
        Get the stellar wind terminal velocity in km / s
        The source of this information is different for OB vs WR;
            OB uses Sternberg tables (so they are needed as arg here)
            and WR uses PoWR simulations
        Most of this code is copied from STResolver.get_mass_loss_rate
        :param sternberg_tables: a S03_OBTables instance
        """
        # Make a terminal velocity finding function
        def find_vinf(st_tuple, model_info):
            # Takes st_tuple (self.spectral_types) and PoWR model_info (self.powr_models)
            # Set up return unit:
            vinf_unit = u.km / u.s
            # First, check for model_info at all. All OBs and valid WRs should have it
            if model_info is None:
                # This star won't be in PoWR or Sternberg (invalid WR or odd type)
                vinf = np.nan
            elif STResolver.isWR(st_tuple):
                # This is a WR with a model; use PoWR
                vinf = model_info['V_INF']
            else:
                # This must be an OB star, use Sternberg
                vinf = sternberg_tables.lookup_characteristic(st_tuple, 'v_terminal')
                if np.isnan(vinf):
                    # Not found in Sternberg tables; default to PoWR
                    vinf = model_info['V_INF']
            return vinf * vinf_unit
        component_mdots = STResolver.map_to_components(find_vinf, (self.spectral_types, self.powr_models))
        # Terminal velocity shouldn't be summed over binary components; we'll average it to approximate
        return STResolver.resolve_uncertainty(component_mdots, dont_add=True)

    def get_FUV_flux(self):
        """
        Get the FUV flux (6 to 13 eV) of the star/binary.
        If one of the possible spectral types cannot be looked up in PoWR,
            ignore it and only use the other(s).
        If one of the binary components cannot be looked up at all, ignore it

        :param powr_dict: dictionary mapping grid_name to the grid object,
            represented by PoWRGrid instance. Grid name is PoWRGrid.grid_name
        :returns: value, (lower limit, upper limit), as astropy Quantities
        """
        # Make a FUV flux-finding function
        def find_FUV_flux(model_info):
            # model_info is a dictionary containing all the columns in modelparameters.txt
            # as well as 'grid' which contains the PoWR grid object
            # Isn't that nifty ;)
            if model_info is None:
                return np.nan * u.solLum
            wlflux = model_info['grid'].get_model(model_info)
            return PoWRGrid.integrate_flux(wlflux)
        component_fluxes = STResolver.map_to_components(find_FUV_flux, (self.powr_models,))
        return STResolver.resolve_uncertainty(component_fluxes)

    """
    Static methods
    """

    @staticmethod
    def isWR(st_tuple):
        """
        Check if this is a WR star
        :param st_tuple: standard tuple format of spectral type
        :returns: boolean, True if WR
        """
        return 'W' in st_tuple[0]

    @staticmethod
    def get_WR_params(st_tuple):
        """
        Retrieve the hardcoded WR parameters, or NaNs if not present
        :param st_tuple: standard tuple format of spectral type
        :returns: tuple(paramx, paramy), with float params
        """
        return STResolver.wr_params.get(st_tuple[:2], (np.nan, np.nan))

    @staticmethod
    def select_powr_grid(st_tuple):
        """
        This is, by necessity, a big, nested if-else block
        We have to exhaust realistic possibilities of WN subtypes, other WR
            types, and finally OB stars
        """
        if STResolver.isWR(st_tuple):
            # This is a WR; check the type and number
            if 'N' in st_tuple[0]:
                # This is a WN star
                if not st_tuple[1]:
                    # Just WN, nothing else
                    return None
                if int(st_tuple[1]) <= 6:
                    # WN6 or earlier
                    return 'WNE'
                else:
                    # WN7 or later
                    return 'WNL'
            else:
                # This is WC or WO or something
                return None
        elif (len(st_tuple[0]) == 1) and (st_tuple[0] in 'OBAFGKM'):
            # This is an OB star
            return 'OB'
        else:
            # Not a supported spectral type
            return None

    @staticmethod
    def map_to_components(f, dictionaries):
        """
        Iterate through all possibilities of all components, operate callable
            f on them, and return a dictionary of the results
        :param f: callable, takes whatever is at the bottom level of the
            argument dictionaries. If multiple dictionaries are given, f must
            take multiple arguments, and should take each dictionary's contents
            in the order the dictionary is listed here. Yeah, this is confusing.
            Just read the code, it'll make sense.
            This is kind of a big "zip+map"
        :param dictionaries: sequence of dictionaries to iterate through
            The dictionaries should all be structured the exact same way
                as self.spectral_types
            If only one dictionary, then use a 1-element tuple: (x,)
        :returns: dictionary structured the same as self.spectral_types
        """
        # Set up return dictionary
        return_dict = {}
        for component in dictionaries[0]:
            # Set up list of results of f calls on possibilities
            f_of_possibilities = []
            for possibility_args in zip(*(d[component] for d in dictionaries)):
                # Cycle through the component possibilities and call f
                # possibility_args is a tuple of everything associated with this
                #   spectral type possibility
                f_of_possibilities.append(f(*possibility_args))
            return_dict[component] = f_of_possibilities
        return return_dict

    @staticmethod
    def resolve_uncertainty(value_dictionary, dont_add=False):
        """
        A function to deal with variation of values of some physical property
            across the uncertainty in the star's spectral type.
        By default, sums over binary components; the properties are assumed to
            be things like mass loss and FUV flux, where 2 stars is twice the
            physical output.
            This behavior can be altered to be the average if dont_add is set
            to True.
        This function may produce some Numpy RuntimeWarnings from all-NaN
            slices, but it will produce the correct results.
            Edit (April 30, 2020): it won't produce Warnings anymore, though
            that way of writing it is probably cleaner. I am taking out all the
            NaNs now, and I did that because I thought it would fix a bug
            about calling np.array on astropy Quantities, but it didn't.
        :param value_dictionary: dictionary containing values associated with
            each possibility, which are in turn associated with binary components.
            Value dictionary should be structured like self.spectral_types.
        :param dont_add: alters the binary reduction behavior to be average if
            True and sum if False (default)
        :returns: value, (lower bound, upper bound)
            These will be NaN if the object couldn't be looked up at all
                (all the values were NaN)
        """
        # I have to do this loop thing because the NaNs really mess things up
        # Apparently when I do np.nanmedian(NaN-only-quantity-array), the result
        # is a DIMENSIONLESS NaN, which just totally doesn't make sense
        # The final values from each component
        component_values = []
        # The final lower, upper bounds for each component
        component_lo_bounds = []
        component_hi_bounds = []
        for component in value_dictionary:
            # Convert to Quantity arrays and get rid of NaNs
            # If there aren't units, it's dimensionless. This works, np.array() doesn't
            values = u.Quantity(value_dictionary[component])
            value_unit = values.unit
            values_finite = values[np.isfinite(values)]
            # Append unit-adjusted NaNs if there aren't any values to use
            if values_finite.size == 0:
                # This helps the units feel better about themselves
                component_values.append(np.nan * value_unit)
                component_lo_bounds.append(np.nan * value_unit)
                component_hi_bounds.append(np.nan * value_unit)
            else:
                # Use the median to get the value, min and max to get bounds
                component_values.append(np.median(values_finite))
                component_lo_bounds.append(np.min(values_finite))
                component_hi_bounds.append(np.max(values_finite))
        # Combine the upper and lower bounds; there should be no NaNs now
        # Empty arrays will sum to 0... which is probably fine TBH, it'll be obvious
        # This was also updated to keep the right units
        reduce_func = mean_or_0 if dont_add else np.nansum
        component_values = u.Quantity(component_values)
        component_lo_bounds = u.Quantity(component_lo_bounds)
        component_hi_bounds = u.Quantity(component_hi_bounds)
        final_value = reduce_func(component_values)
        final_lo_bound = reduce_func(component_lo_bounds)
        final_hi_bound = reduce_func(component_hi_bounds)
        # Adjust hi_bound to be + 1x final_value if binary and one component is unknown
        # Only do this if we're adding binary properties
        if not dont_add and (len(value_dictionary) == 2) and np.any(np.isnan(component_values)):
            final_hi_bound += final_value
        return final_value, (final_lo_bound, final_hi_bound)

    """
    Stuff for printing
    """

    def __repr__(self):
        if self.isbinary():
            text="<Binary:"
        else:
            text="<Star:"
        for st in self.spectral_types:
            stub = "/".join([st_tuple_to_string(x) for x in self.spectral_types[st]])
            text += f"({stub})"
        return text + ">"

    def __str__(self):
        text = "+".join(self.spectral_types.keys())
        return f"<{text}>"

    def rollcall(self, dictionary=None, f=None):
        """
        Prints every star and every possibility, longest-form
        Can add a function f to operate on each possibility
        """
        if f is None:
            f = lambda x: x
        # Default to the component dictionary of spectral types
        if dictionary is None:
            dictionary = self.spectral_types
        # Print a bunch of information out
        print(str(self))
        for component in dictionary:
            print("|*\t " + component)
            for possibility in dictionary[component]:
                print("|-\t|p\t", f(possibility))


def mean_or_0(arg):
    """
    Applies np.mean to the arg and if the result is np.nan, returns 0
    Since I don't want to handle multiple arguments or array answers,
        this can only have one argument and should return a scalar
    """
    result = np.mean(arg)
    if np.isnan(result):
        # Keeps correct units and returns 0.
        return np.nansum(arg)
    else:
        return result


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
    tbl = PoWRGrid('OB')
    count = 0
    for model_info in tbl.iter_models():
        count += 1
        print(type(model_info))
        # wl, flux = tbl.get_model(model_info)
        # lum = np.trapz(flux, x=wl).to('solLum')
        # print(f"Model {model_info.MODEL} (Teff{model_info.T_EFF}/log_g{model_info.LOG_G}:")
        # print(f" -> Integrated luminosity (numpy): {lum.to_string()}")
        # print(f" -> Accuracy: {lum.to_value() / (10**model_info.LOG_L)}")
        print()
        if count > 20:
            break


def test_powr_totalL_plot():
    # tbl.plot_grid_space(tbl.grid_info.LOG_L, "log L")
    tbl = PoWRGrid('WNE')
    tbl.plot_grid_space(c='blue', show=False, alpha=0.3)
    tbl = PoWRGrid('WNL')
    tbl.plot_grid_space(c='red', setup=False, alpha=0.3)


def testSuite_PoWR():
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


def test_STResolver():
    def f(x):
        # function to pass NaNs and print MODEL from DataFrames
        try:
            return x['MODEL']
        except:
            return "[NO MODEL]"
    tbls = {x: PoWRGrid(x) for x in ('OB', "WNE", "WNL")}
    sb_tables = S03_OBTables()
    # cat = pd.read_pickle(f"{catalogs_directory}/Ramsey/OBradec.pkl")
    tests = ['B5/6.5III', 'O4I/III', 'WN', 'C*']
    count = 0
    for t in tests:
        if t == "ET":
            t = "O7.5/B1"
        s = STResolver(t)
        # print(t, '\n\t', s)
        # print('\t', s.__repr__())
        s.link_powr_grids(tbls)
        v, lu = s.get_mass_loss_rate(sb_tables)
        l, u = lu
        dl, du = v-l, u-v
        # vtxt = "---N/A---" if ((v == 0.) or np.isnan(v)) else f"{v:.2E}"
        # etxt = "" if ((dl == 0.) and (du == 0.)) else f" (-{dl:.2E}, +{du:.2E})"
        vtxt = str(v) + " +/- "
        etxt = str(dl) + "," + str(du)
        print(f"{s.__repr__():.<30s}: {vtxt}{etxt}")
        count += 1
        if count > 95:
            break
    return s



if __name__ == "__main__":
    args = test_STResolver()
