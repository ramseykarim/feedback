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
import random

from astropy import units as u

from ... import misc_utils

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
        # Rauw 2005 parameters.
        # T, Rstar, Mdot, vinf, D (1/f)
        ("WN", "6"): (43000., 19.3, 8.5e-6, 2800., 10.),
    }

    wr_uncertainties = {
        # Floats are 1-sigma uncertainties, tuples are bounds between which
        # the distribution will be assumed ot be uniform
        ("WN", "6"): (2000., 0.5, 8.5e-7, (1000., 3000.), (4., 10.)),
    }

    """
    Supported property names
    """
    property_names = {
        'mass_loss_rate': 'mdot',
        'terminal_wind_velocity': 'vinf',
        'momentum_flux': 'mv_flux',
        'mechanical_luminosity': 'lmech',
        'FUV_flux': 'fuv',
    }

    """
    Class-wide dictionaries
    """
    # Set up FUV memoization dictionary to avoid too much computation
    fuv_memoization = {}
    # Set up WR uncertainty sampling dictionary; keys are the same as above
    wr_samples = {}


    """
    Setup
    """

    def __init__(self, st, container=None):
        """
        Taking a lot of cues from st_reduce_to_brightest_star
        :param st: string spectral type, like "O3-5I/III(f)"
        :param container: CatalogResolver object containing this
            and other instances. If this is given, you don't need to
            explicitly link any of the tables or grids.
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
              The list will contain both the possibilities explicitly enumerated
              in the string type as well as a half spectral type above and below
              each possibility to capture some intrinsic scatter in spectral
              type calibration. This also means that "well defined" spectral
              types with no explicit uncertainty are given some uncertainty,
              which is more realistic.
            If the spectral type refers to a WR star, the tuple format is
              extended to include the 5 defining WR parameters, since there is
              not a simple map between WR spectral type and physical properties.
              This means that WR stars will have a 9-element tuple;
                  (letters, number, blank string, peculiarity,
                    Teff, Rstar, Mdot, vinf, D)
              This allows for sampling of the WR parameter space, since we
              simply cannot sample the map like we do for OB stars.
            """
            full_possibilities_list = []
            for st_tuple in st_bc_possibilities_t:
                if STResolver.isWR(st_tuple):
                    # WR star case; add possibilites from wr_samples
                    param_samples = STResolver.sample_WR(st_tuple)
                    # Prepend the spectral type tuple to the parameter tuples
                    full_possibilities_list.extend([st_tuple + ps for ps in param_samples])
                elif STResolver.isMS(st_tuple):
                    # OB star; add the original and adjacent possibilities
                    # Adjacent list contains original, so we will have 2 copies of original
                    # This is intentional, to weight towards original type
                    full_possibilities_list.extend(parse_sptype.st_adjacent(st_tuple))
                    full_possibilities_list.append(st_tuple)
                else:
                    # Nonstandard type; just throw it in, it won't matter
                    full_possibilities_list.append(st_tuple)
            # Assign the full possibilities list to the spectral_types dictionary
            self.spectral_types[st_binary_component] = full_possibilities_list
        # Check if this instance is part of a CatalogResolver container
        if container is not None:
            self.container = container
            self.link_calibration_table(self.container.calibration_table)
            self.link_leitherer_table(self.container.leitherer_table)
            self.link_powr_grids(self.container.powr_dict)

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
        This needs to be called by the user since it requires STTable as input,
            unless this instance is part of a CatalogResolver.
        ****
        :param table: the STTable object wrapper for the calibration table.
        """
        self.calibration_table = table

    def link_leitherer_table(self, table):
        """
        Link the Leitherer table, for mass loss rates of O stars.
        ****
        This needs to be called by the user since it requires LeithererTable as input,
            unless this instance is part of a CatalogResolver.
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
        This needs to be called by the user since it requires PoWR grids as input,
            unless this instance is part of a CatalogResolver.
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
                params = STResolver.get_WR_params(st_tuple)
            elif STResolver.isMS(st_tuple):
                # This is an OB; use Martins calibration
                paramx = self.calibration_table.lookup_characteristic('Teff', st_tuple)
                paramy = self.calibration_table.lookup_characteristic('log_g', st_tuple)
                params = (paramx, paramy)
            else:
                # Nonstandard star; nothing we can do
                return None
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

    def populate_all(self):
        """
        Do I need this function?
        """
        self.populate_mass_loss_rate()
        self.populate_terminal_wind_velocity()
        self.populate_momentum_flux()
        self.populate_mechanical_luminosity()
        self.populate_FUV_flux()

    def __getattr__(self, name):
        """
        Overloading __getattr__ to try to find a property first.
        General function for pulling the final sampled property.
        The property_name dictionary defined in this class is used to
            find the attribute names for property dictionaries.
        This function accepts attribute queries starting with "get_"
            as well as "get_array_".
            "get_" will get a single, uncertainty-resolved value for this
            instance. "get_array_" will get a range of possibilities selected
            at random.
        """
        if "array" in name:
            # Return samples array case
            name_stub = name.replace('get_array_', '')
            return_samples = True
        else:
            # Return single uncertainty-resolved value case
            name_stub = name.replace('get_', '')
            return_samples = False
        if name_stub in STResolver.property_names:
            # Get the short property name ('mdot', etc)
            property_name = STResolver.property_names[name_stub]
            # Average the velocities, don't add
            dont_add = ('velocity' in name_stub)
            def property_getter_method(nsamples=200):
                if not hasattr(self, property_name):
                    getattr(self, "populate_" + name_stub)()
                return STResolver.resolve_uncertainty(getattr(self, property_name),
                    dont_add=dont_add, return_samples=return_samples, nsamples=nsamples)
            return property_getter_method
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

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
                # This is a WR; use the function
                mdot = STResolver.get_WR_mdot(st_tuple)
            else:
                # This must be an OB star, since we already checked model_info
                # Use Leitherer
                paramx = self.calibration_table.lookup_characteristic('Teff', st_tuple)
                paramy = self.calibration_table.lookup_characteristic('log_L', st_tuple)
                mdot = 10.**self.leitherer_table.lookup_characteristic('log_Mdot', paramx, paramy)
                if np.isnan(mdot):
                    # Not found in Sternberg tables; default to PoWR
                    mdot = 10.**(model_info['LOG_MDOT'])
            return mdot * mdot_unit
        self.mdot = STResolver.map_to_components(find_mass_loss_rate, (self.spectral_types, self.powr_models), f_list=u.Quantity)

    def populate_terminal_wind_velocity(self):
        """
        Get the stellar wind terminal velocity in km / s
        Populate self.vinf with possibilities
        The source of this information is different for OB vs WR;
            OB uses Leitherer tables (need to run self.link_leitherer_table)
            and WR uses PoWR simulations
        Most of this code is copied from STResolver.get_mass_loss_rate
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
                vinf = STResolver.get_WR_vinf(st_tuple)
            else:
                # This must be an OB star, use Leitherer
                paramx = self.calibration_table.lookup_characteristic('Teff', st_tuple)
                paramy = self.calibration_table.lookup_characteristic('log_L', st_tuple)
                vinf = self.leitherer_table.lookup_characteristic('v_inf', paramx, paramy)
                if np.isnan(vinf):
                    # Not found in Sternberg tables; default to PoWR
                    vinf = model_info['V_INF']
            return vinf * vinf_unit
        self.vinf = STResolver.map_to_components(find_vinf, (self.spectral_types, self.powr_models), f_list=u.Quantity)

    def populate_momentum_flux(self):
        """
        Get the momentum flux in dynes by multipying mass loss rate by terminal
            velocity.
        """
        def find_mv_flux(mdot, vinf):
            # multiply these together and convert to dynes
            return (mdot * vinf).to(u.dyne)
        self.mv_flux = STResolver.map_to_components(find_mv_flux, (self.mdot, self.vinf), f_list=u.Quantity)

    def populate_mechanical_luminosity(self):
        """
        Get mechanical luminosity in erg/s by using mass loss rate and terminal
            velocity in the kinetic energy equation, 1/2 m v^2
        """
        def find_KE_rate(mdot, vinf):
            # multipy and convert to erg/s
            return (mdot * vinf * vinf / 2.).to(u.erg/u.s)
        self.lmech = STResolver.map_to_components(find_KE_rate, (self.mdot, self.vinf), f_list=u.Quantity)

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
            # Get a unique model ID for memoization
            model_identifier = model_info['grid'].grid_name + model_info['MODEL']
            if model_identifier in STResolver.fuv_memoization:
                # Check if we have it memoized
                return STResolver.fuv_memoization[model_identifier]
            else:
                # Calculate it
                wlflux = model_info['grid'].get_model_spectrum(model_info)
                integrated_flux = powr.PoWRGrid.integrate_flux(wlflux)
                # Memoize it
                STResolver.fuv_memoization[model_identifier] = integrated_flux
                return integrated_flux
        self.fuv = STResolver.map_to_components(find_FUV_flux, (self.powr_models,), f_list=u.Quantity)


    """
    WR instance methods
    """

    """
    TODO:::::::::::::::::::::::: I LEFT OFF HERE
    My plan is:
    if you look at STResolver.link_powr_grids and stuff like that, it's already
    pretty well set up to handle a bunch of random WR stars. I don't need
    to duplicate that work.
    There's nothing sacred about the possibility lists; they don't have to
    match up to their keys (component spectral type strings) exactly. These
    lists would be a good place to dump uncertainty

    Ideally, I can use this function (sample_WR) to populate the class-wide
    list exactly once, and just check to see if it's there every other time.
    Then when I come across a WR star in __init__, instead of one possibility
    in the list, I just link the entire class-wide list for that WR type.
    self.spectral_types[component] = STResolver.wr_samples[component?]
        I should go in and check if it's worth it to "memoize" the grids
        associated with each WR sample. I suspect not; only a factor of 2-3.
    The FUV integrated fluxes will already be memoized in find_FUV_flux,
        so I'll save time there, especially for the WRs.
    resolve_uncertainty shouldn't need to be changed much at all, either.
    It just runs across the wider set of possibilities. Doesn't need to know
        the difference between WR and OB.
    """

    """
    Static methods
    """

    @staticmethod
    def sample_WR(st_tuple, nsamples=100):
        """
        Get a list of samples of parameters for a given WR spectral type.
        This function handles some memoization since these samples
            don't need to be recalculated.
        This function will return the sampled list.

        If it is not already memoized:
        Populate a class-wide list of samples of all WR parameters.
        The list will be contained in a dictionary keyed the same as
            STResolver.wr_params and wr_uncertainties.
        The list will be shared among all classes since the work is the same.
        :param st_tuple: WR spectral type tuple, at least 2 elements
        :param nsamples: number of parameter combinations to draw.
            Not used if this spectral type has already been memoized.
        """
        # First, check if we already have it and return it if so
        if st_tuple[:2] in STResolver.wr_samples:
            return STResolver.wr_samples[st_tuple[:2]]
        # If not, calculate it
        base_parameters = STResolver.wr_params[st_tuple[:2]]
        uncertainties = STResolver.wr_uncertainties[st_tuple[:2]]
        # Treat Gaussian and uniform uncertainties differently
        samples_list = []
        for p, p_e in zip(base_parameters, uncertainties):
            if isinstance(p_e, tuple):
                # Uniform uncertainty
                # Half of samples from uniform distribution, other half are
                # base parameter value. This weights toward the value.
                param_samples = list(np.random.uniform(low=p_e[0], high=p_e[1], size=(nsamples//2)))
                param_samples += [p]*(nsamples - len(param_samples))
                # Shuffle values
                random.shuffle(param_samples)
                # This list should contain no negative values
                samples_list.append(param_samples)
            else:
                # Gaussian uncertainty
                param_samples = np.random.normal(p, p_e, size=nsamples)
                # Resample any negatives
                n_negatives = param_samples[param_samples <= 0].size
                if n_negatives > 0:
                    print('had to fix negatives')
                    param_samples[param_samples <= 0] = np.random.normal(p, p_e, size=nsamples)
                    # If there are any remaining negatives, set them to the base value
                    param_samples[param_samples <= 0] = p
                samples_list.append(list(param_samples))
        # Reshape to be nsamples-len list of tuples of the ~5 parameters
        samples_list = list(zip(*samples_list))
        # Memoize
        STResolver.wr_samples[st_tuple[:2]] = samples_list
        # Return it
        return samples_list

    @staticmethod
    def isMS(st_tuple):
        """
        Check if this is a main sequence star
        :param st_tuple: standard tuple format of spectral type
        :returns: boolean, True if main sequence
        """
        return st_tuple[0] in parse_sptype.standard_types

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
        Retrieve the WR PoWR parameters, or NaNs if not present, from the
            spectral type tuple. There are parameters embedded in the tuple
            for WR stars (not for OB or other stars)
        :param st_tuple: standard tuple format of spectral type, extended
            for WR stars. 9 elements total.
        :returns: tuple(paramx, paramy), with float params
        """
        paramx = st_tuple[4]
        paramy = powr.PoWRGrid.calculate_Rt(*st_tuple[5:])
        return paramx, paramy

    @staticmethod
    def get_WR_mdot(st_tuple):
        """
        Quick way to get the mass loss rate for the WR stars supported in this
        class.
        :param st_tuple: standard tuple format of spectral type, extended
            for WR stars.
        :returns: float mass loss rate (solMass / yr)
        """
        return st_tuple[6]

    @staticmethod
    def get_WR_vinf(st_tuple):
        """
        Quick way to get the terminal velocity for the WR stars supported in
        this class.
        :param st_tuple: standard tuple format of spectral type, extended
            for WR stars.
        :returns: float terminal velocity (km /s)
        """
        return st_tuple[7]

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
    def map_to_components(f, dictionaries, f_list=None):
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
        :param f_list: optional function to operate on each list, the level
            between the bottom level (operated on by f) and the top level
            dictionary. This could be used, for example, to change the type
            from list to another sequence (like Quantity)
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
            if f_list is not None:
                f_of_possibilities = f_list(f_of_possibilities)
            return_dict[component] = f_of_possibilities
        return return_dict

    @staticmethod
    def random_possibility(value_dictionary, dont_add=False):
        """
        Pick a random possibility for each component from the value_dictionary
        and return the combination (mean if dont_add else sum) of the values
        Written June 12, 2020, edited June 17, 2020
        :param value_dictionary: dictionary containing values associated with
            each possibility, which are in turn associated with binary components.
            Value dictionary should be structured like self.spectral_types.
        :returns: scalar Quantity value, combination of random possibilities
            for each component
        """
        selected_values = []
        for component in value_dictionary:
            # Get a possible value for each component
            value = random.choice(value_dictionary[component])
            # Get the unit (should still be there even if NaN)
            value_unit = value.unit
            if np.isfinite(value):
                # Append if value is finite
                selected_values.append(value)
        if not selected_values:
            # If no valid values, return 0
            return 0. * value_unit
        else:
            # Cast to Quantity to avoid numpy errors
            selected_values = u.Quantity(selected_values)
            # Return the mean or sum of the component values
            reduce_func = np.mean if dont_add else np.sum
            return reduce_func(selected_values)


    @staticmethod
    def resolve_uncertainty(value_dictionary, nsamples=200, dont_add=False,
        return_samples=False):
        """
        Need to rewrite description because this isn't accurate anymore.



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
        # Value dictionary should already have Quantity arrays
        # Build list of sample arrays
        component_samples = []
        # Cycle through components and make sample arrays for those that have
        # some valid possibilities
        for component in value_dictionary:
            values = u.Quantity(value_dictionary[component])
            value_unit = values.unit
            values_finite = values[np.isfinite(values)]
            if values_finite.size > 0:
                samples_array = u.Quantity(random.choices(values_finite, k=nsamples))
                component_samples.append(samples_array)
        # Just walk case-by-case in how many stars have sample sequences,
        # combine them based on dont_add flag
        # Remember if it was the single option.
        one_star_with_values = False
        # Flag if there are no samples to return (for return_samples)
        no_samples = False
        if len(component_samples) == 0:
            # No samples
            no_samples = True
        elif len(component_samples) == 1:
            # One star with samples; prepare those samples
            samples_array = component_samples.pop()
            one_star_with_values = True
        else:
            # More than one star with samples
            samples_array = u.Quantity(component_samples)
            if not dont_add:
                samples_array = np.sum(samples_array, axis=0)
            else:
                samples_array = np.mean(samples_array, axis=0)

        if return_samples:
            # Return the full sample arrays (if possible)
            if no_samples:
                return None
            else:
                return samples_array
        else:
            if no_samples:
                # No samples; return zero (with units)
                zero = 0. * value_unit
                return zero, (zero, zero)
            else:
                # Calculate median, lo, hi, and return those
                final_value = np.median(samples_array)
                lo, hi = misc_utils.flquantiles(samples_array, 4)
                if not dont_add and (len(value_dictionary) == 2) and one_star_with_values:
                    hi += final_value
                return final_value, (lo, hi)


        """
        TODO: LEFT OFF HERE.
        June 15, 2020:
        I edited this function a little bit to allow it to return the samples
        array. I need to implement the CatalogResolver class below to use this
        behavior to sample across the entire cluster.

        I should also look into sampling more uncertainty in spectral types:
            for each possibility, go a half step up and down and add those to
            possibilities. re-add the original possibility as weight towards
            the center. (3 -> 2 3 3 4)
            For WR: hardcode uncertainty in input parameters, sample from
            normal distributions. For FUV flux, sample from T and Rtrans and
            get a number of different models? Need to think more on this

        Earlier:
        I need to continue reviewing this to make sure it looks legal.
        I have tested it, and it looks like it works.

        Couple things to consider:
        1) should I return final, (lo, hi) or final, (d_lo, d_hi) ?
        2) this doesn't take into account uncertainty in the characteristics
            themselves. Could I have dicts like self.mdot_err and use those
            to not just sample from the component possibilities (in case there's
            just one), but also the (Gaussian) uncertainty distribution of each
            characteristic?
            Where do I even find errors for the Martins/Leitherer stuff?
            Something with curve fitting? Assert 5 or 10%?
            This would help with WR characteristics which (should) have errors
            in Rauw 2005
                On this topic, mdot from Rauw doesn't have error, but they
                discuss scaling with sqrt(f), so I could invent my own.
            Also, geometric mean with mdot? Very logarithmic

        Also, I should use a quantile-like function to see what the 1-std error
        on an asymmetrical distribution is. Like, find the fractional coverage
        (34%?) of one standard deviation (on one side of the median) and then
        apply that to the distribution found here, on each side separately so
        I capture asymmetry
        """

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
        space4 = " "*4
        for component in dictionary:
            print(f"|*{space4}{component}")
            for possibility, name in zip(dictionary[component], self.spectral_types[component]):
                print(f"|-{space4} - p({parse_sptype.st_tuple_to_string(name)}){space4}\t{f(possibility)}")


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



class CatalogResolver:
    """
    A wrapper for STResolver when applied to a number of stars.
    """

    def __init__(self, sptype_list,
        calibration_table=None, leitherer_table=None, powr_dict=None):
        """
        :param sptype_list: some iterable sequence of string spectral types
             that could individually be read by STResolver
        :param calibration_table: Martins or Sternberg table, STTable object
        :param leitherer_table: LeithererTable object
        :param powr_dict: dictionary of PoWRGrid objects
        """
        # Link all tables/grids
        self.calibration_table = calibration_table
        self.leitherer_table = leitherer_table
        self.powr_dict = powr_dict
        # Make the STResolver instances
        self.star_list = [STResolver(s, container=self) for s in sptype_list]
        # Populate the instances with possible values for every property
        # based on spectral type uncertainty
        for s in self.star_list:
            s.populate_all()

    def __getattr__(self, name):
        try:
            # See if the method exists
            getattr(self.star_list[0], name)(nsamples=1)
        except Exception as e:
            attrib_e = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            # Use the "from" statement to assign blame (cause)
            raise attrib_e from e
        # Make a property getter function to return
        def property_getter_method(**kwargs):
            # kwargs can have "nsamples" argument
            # Prepare to collect quantity arrays
            all_star_samples = []
            for s in self.star_list:
                # Pass off the bulk of the work to STResolver, using
                # the "get_array_" prefix to signal returning sample arrays
                samples_array = getattr(s, name.replace('get_', 'get_array_'))(**kwargs)
                if samples_array is not None:
                    all_star_samples.append(samples_array)
            # Make 2D Quantity
            all_star_samples = u.Quantity(all_star_samples)
            # Don't add if velocity; average instead
            reduce_func = np.mean if ('velocity' in name) else np.sum
            # Reduce all star stamples into full cluster samples
            cluster_samples = reduce_func(all_star_samples, axis=0)
            # Return median of cluster samples
            print('remove this')
            return cluster_samples
            return np.median(cluster_samples)
        return property_getter_method

    def __str__(self):
        text = f"<Catalog of {len(self.star_list)} stars>"
        return text

    def __repr__(self):
        text = f"<CatalogResolver({len(self.star_list)})>"
        return text

    """
    Fill the rest of this in. Use random_possibility
    """
