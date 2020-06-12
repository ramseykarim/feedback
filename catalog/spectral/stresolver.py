"""
===============================================================================
================== All-in-one reading object ==================================
===============================================================================
The idea here is to wrap all the spectral type reading into one object
    that handles binaries and uncertainties well.
Each spectral type will be wrapped in an instance of this object. The object
    will have common methods for fluxes/stellar winds and their
    associated uncertainties, but will handle binaries and spectral types with
    ranges differently under the hood.

Created: April 29, 2020
Split from catalog_spectral.py (previously readstartypes.py) on June 2, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np
from astropy import units as u

from . import powr
from . import parse_sptype
from . import vacca


class STResolver:
    """
    Spectral Type Resolver
    Reads a single spectral type string and figures it all out.
    The string can contain binaries and uncertainties
    This works best with OB types, but can handle WR types if we hardcode the
        parameters (which seems necessary....)
    Written: April 29-30, 2020
    Revised: June 12, 2020
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
        # Unsure where these parameters are from exactly, but I should be using
        # the Rauw 2005 parameters.
        # T, Rstar, Mdot, vinf, D (1/f)
        ("WN", "6"): (43000, 19.7, 8.5e-6, 1600., 4.),
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
        for st_binary_component in parse_sptype.st_parse_binary(st):
            # st_binary_component is a string
            st_bc_possibilities = parse_sptype.st_parse_slashdash(st_binary_component)
            # st_bc_possibilities is a list(string)
            st_bc_possibilities_t = [parse_sptype.st_parse_type(x) for x in st_bc_possibilities]
            # st_bc_possibilities_t is list(tuple(string))
            # Compose the components dictionary
            """
            self.spectral_types is a map between a single component's spectral
              type string and the list of possibilities, with possibilities
              represented in tuple format:
                  (letter, number, luminosity_class, peculiarity)
            """
            self.spectral_types[st_binary_component] = st_bc_possibilities_t

    def isbinary(self):
        """
        :returns: boolean, True if this is a binary system, False if singular
        """
        return len(self.spectral_types) > 1

    def link_calibration_table(self, table):
        """
        Link either the Sternberg or Martins tables for calibrating spectral
        types to physical properties Teff and log_g. Vacca also serves this
        purpose without a table. Note that Vacca and Sternberg agree with each
        other but not Martins.
        Right now, this class is set up to handle the Martins tables. I don't
        see a reason to support both simultaneously, since Martins is more
        recent. They should be standardized anyway.
        ****
        This needs to be called by the user since it requires STTable as input.
        ****
        :param table: the STTable object wrapper for the calibration table.
        """
        self.calibration_table = table

    def link_leitherer_table(self, table):
        """
        Link the Leitherer table, for mass loss rates of O stars.
        ****
        This needs to be called by the user since it requires LeithererTable as input.
        ****
        :param table: the LeithererTable wrapper object.
        """
        self.leitherer_table = table

    def link_powr_grids(self, powr_dict):
        """
        First, get the inputs to the PoWR grid using either the Vacca
            calibration or a hardcoded list of WR parameters
        These are paramx, paramy of the grid
        For OB stars, that's Teff and log_g
        For WR stars, that's Teff and R_trans
        Must run self.link_calibration_table(table) before this method can run.
        Then, get PoWR model names for each eligible star/possibility
        This does not collect the full UV spectra, just the parameters.
        ****
        This needs to be called by the user since it requires PoWR grids as input.
        ****
        :param powr_dict: dictionary mapping grid_name to the grid object,
            represented by PoWRGrid instance. Grid name is PoWRGrid.grid_name
        """
        # Make a function to get params from a spectral type tuple and then
        # get a PoWR model for a spectral type tuple
        def find_model(st_tuple):
            # st_tuple is a tuple representing spectral type of a single
            #   component possibility
            # Get the name of the grid (WNE, OB, etc)
            selected_grid_name = STResolver.select_powr_grid(st_tuple)
            # If there is no grid, return None
            if selected_grid_name is None:
                return None
            # Get the grid
            selected_grid = powr_dict[selected_grid_name]
            # Get the parameters
            if STResolver.isWR(st_tuple):
                # This is a WR; use hardcoded parameters
                params = self.get_WR_params(st_tuple)
            else:
                # This is an OB; use Martins calibration
                paramx = self.calibration_table.lookup_characteristic('Teff', st_tuple)
                paramy = self.calibration_table.lookup_characteristic('log_g', st_tuple)
                params = (paramx, paramy)
            # If the parameters are NaN, return None
            if np.any(np.isnan(params)):
                return None
            # Get the model (pandas df), cast as dict (this works, I checked)
            model_info = dict(selected_grid.get_model_info(*params))
            # Attach the PoWR grid object so we can look up the flux
            model_info['grid'] = selected_grid
            return model_info
        # Iterate over the self.spectral_types dictionary using that nifty
        # function I wrote
        self.powr_models = STResolver.map_to_components(find_model, (self.spectral_types,))

    """
    ============================================================================
    ==================== Property-finding functions ============================
    ============================================================================
    These used to be "getter" functions that returned the median value across
    possibilities as well as the min and max bounds. (circa April 30, 2020)
    I altered the behavior to make these functions "populate" dictionaries
    (same shape/structure as self.spectral_types) with the values for each
    component/possibility. This way, these can be quickly resampled without
    needing to be recalculated (expensive for FUV flux, etc).
    """

    def populate_mass_loss_rate(self):
        """
        Get the stellar wind mass loss rate in Msun / year
        Populate self.mdot with possibilities
        The source of this information is different for OB vs WR;
            OB uses the LeithererTable (which needs to have been linked with
            self.link_leitherer_table(table)) and WR uses hardcoded values
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
                # This is a WR; we have this hardcoded
                mdot = 10.**(STResolver.get_WR_mdot(st_tuple))
            else:
                # This must be an OB star, use Leitherer
                mdot = self.leitherer_table.lookup_characteristic('log_Mdot', None) # FINISH THIS
                mdot = sternberg_tables.lookup_characteristic(st_tuple, 'Mdot')
                if np.isnan(mdot):
                    # Not found in Sternberg tables; default to PoWR
                    mdot = 10.**(model_info['LOG_MDOT'])
            return mdot * mdot_unit
        self.mdot = STResolver.map_to_components(find_mass_loss_rate, (self.spectral_types, self.powr_models))

    def populate_terminal_wind_velocity(self, sternberg_tables):
        """
        Get the stellar wind terminal velocity in km / s
        Populate self.vinf with possibilities
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
                # This is a WR with a model; we have this hardcoded
                winf = STResolver.get_WR_vinf(st_tuple)
            else:
                # This must be an OB star, use Sternberg
                vinf = sternberg_tables.lookup_characteristic(st_tuple, 'v_terminal')
                if np.isnan(vinf):
                    # Not found in Sternberg tables; default to PoWR
                    vinf = model_info['V_INF']
            return vinf * vinf_unit
        self.vinf = STResolver.map_to_components(find_vinf, (self.spectral_types, self.powr_models))

    def populate_FUV_flux(self):
        """
        Get the FUV flux (6 to 13 eV) of the star/binary.
        Populate self.fuv with possibilities
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
            return powr.PoWRGrid.integrate_flux(wlflux)
        self.fuv = STResolver.map_to_components(find_FUV_flux, (self.powr_models,))

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
        all_params = STResolver.wr_params.get(st_tuple[:2], (np.nan,)*5)
        paramx = all_params[0]
        paramy = powr.PoWRGrid.calculate_Rt(all_params[1:])
        return paramx, paramy

    @staticmethod
    def get_WR_mdot(st_tuple):
        """
        Quick way to get the mass loss rate for the WR stars supported in this
        class.
        :param st_tuple: standard tuple format of spectral type
        :returns: float mass loss rate (solMass / yr)
        """
        all_params = STResolver.wr_params.get(st_tuple[:2], (np.nan,)*5)
        return all_params[2]

    @staticmethod
    def get_WR_vinf(st_tuple):
        """
        Quick way to get the terminal velocity for the WR stars supported in
        this class.
        :param st_tuple: standard tuple format of spectral type
        :returns: float terminal velocity (km /s)
        """
        all_params = STResolver.wr_params.get(st_tuple[:2], (np.nan,)*5)
        return all_params[3]

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
    def random_possibility(value_dictionary):
        """
        Pick a random possibility for each component from the value_dictionary
        and return a dictionary of these values (same keys as value_dictionary)
        Written June 12, 2020
        :param value_dictionary: dictionary containing values associated with
            each possibility, which are in turn associated with binary components.
            Value dictionary should be structured like self.spectral_types.
        Still under construction
        """
        pass

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
        TODO: delete (or heavily repurpose) this function, since it doesn't
            accurately handle errors (June 12, 2020)
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
            stub = "/".join([parse_sptype.st_tuple_to_string(x) for x in self.spectral_types[st]])
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
