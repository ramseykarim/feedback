"""
PoWR model grid wrapper class & functions

Split from catalog_spectral.py (previously readstartypes.py) on June 2, 2020

Created: June 2, 2020
Reviewed June 11, 2020, looks like it doesn't need any edits.
"""
__author__ = "Ramsey Karim"

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u

from .. import utils


powr_directory = f"{utils.misc_data_path}SpectralTypes/PoWR/"


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
        if grid_name not in ['OB', 'WNE', 'WNL', 'WNL-H50']:
            raise RuntimeError(f"{grid_name} not available")
        self.grid_name = grid_name
        self.grid_info = load_powr_grid_info(self.grid_name)
        self.paramx, self.paramy = None, None
        self.paramx_name, self.paramy_name = None, None
        self.get_params()

    def __str__(self):
        return f"<PoWR.{self.grid_name}>"

    def __repr__(self):
        return f"<PoWR.{self.grid_name}grid({len(self.grid_info)} models)>"

    def get_model_info(self, *args):
        """
        What does this function do? See below: (April 29, 2020)
        This function takes in the parameter combo, whatever it may be,
            and returns the model info as a pandas dataframe (Series?). The model info
            is a single row from the "modelparameters.txt" file.
        """
        qparamx, qparamy = self.parse_query_params(*args)
        # <Original comment>
        # If this EXACT combo is NOT present in the grid, return an error!
        # close enough is NOT close enough!!!! </Original comment>
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
        """
        args are either (Teff, log_g) or the pandas.Series result from above
            (or dictionary made from it)
        """
        if len(args) == 2:
            model = self.get_model_info(*args)
        else:
            model = args[0]
        suffix = '-i' if self.grid_name == 'OB' else ''
        # command getattr(obj, "attr", default) returns obj.attr if possible, or default
        model_id = getattr(model, "MODEL", None)
        # "default" gets evaluated so I cannot put model['MODEL'] there.
        if model_id is None:
            model_id = model["MODEL"]
        return f'{powr_directory}{self.grid_name}/{self.grid_name.lower()}{suffix}_{model_id}_sed.txt'

    def get_model_spectrum(self, *args):
        """
        Passes args to self.get_model_filename, which either passes them to
        self.get_model_info (which passes to self.parse_query_params) or
        accepts the model output from self.get_model_info
        (see self.get_model_filename)
        """
        data = np.genfromtxt(self.get_model_filename(*args))
        wl = (10**data[:, 0]) * u.Angstrom
        # flux comes in (log) erg/ (cm2 s A) at 10pc, so convert to area-integrated flux
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
            try:
                return trim_Teff(Teff), trim_logg(log_g)
            except:
                raise RuntimeError(Teff.ndim, log_g.ndim, Teff)
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
                Rt = np.log10(Rt)
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
