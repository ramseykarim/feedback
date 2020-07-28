"""
Utility functions and classes for use with radio/mm data cubes
Created: July 22, 2020
"""
__author__ = "Ramsey Karim"

import os
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from spectral_cube import SpectralCube

from . import catalog

def beam_area(theta_a, theta_b):
    """
    Return beam area in steradians given the half-power full widths
    at the major and minor axes. Inputs should be Quantites.
    I am using the equation given in:
    https://science.nrao.edu/facilities/vla/proposing/TBconv
    :param theta_a: half-power full width along major axis, though order of
        major/minor doesn't actually matter. Quantity, angular
    :param theta_b: same as above for minor axis, though order doesn't matter.
    :returns: Quantity, solid angle, beam area
    """
    return (np.pi * theta_a * theta_b / (4 * np.log(2))).to(u.sr)


class CubeData:
    """
    Wrapper class for some of the cube data we have.
    Pretty much just wraps up SpectralCube and hides some peculiarities of
    data (like the BIMA 4-D cube...)
    Written July 21-22, 2020
    """
    def __init__(self, filename):
        self.full_path = catalog.utils.search_for_file(filename)
        self.telescope = self.full_path.split('/')[-2]
        self.basename = os.path.basename(self.full_path)
        with fits.open(self.full_path) as hdul:
            self.header = hdul[0].header
            self.wcs = WCS(self.header, naxis=3)
            self.wcs_flat = WCS(self.header, naxis=2)
            self.data = hdul[0].data
        if self.header['NAXIS'] == 4:
            self.data = self.data[0]
        brightness_unit = self.header['BUNIT']
        # Building this mostly with assert statements because this is
        # necessarily something I need to hardcode
        if 'TELESCOP' in self.header and self.header['TELESCOP'] == "HATCREEK":
            assert ('jy' in brightness_unit.lower()) and ('beam' in brightness_unit.lower())
            brightness_unit = u.Jy / u.beam
            self.data *= brightness_unit
            self.data = self.data.to(u.K, equivalencies=u.brightness_temperature(self.header['RESTFREQ']*u.Hz, beam_area=beam_area(7*u.arcsec, 4*u.arcsec)))
        else:
            assert 'K' in brightness_unit
            brightness_unit = u.K
            self.data *= brightness_unit
        # I should check if I need to use any beam efficiency corrections
        # Ask Marc via powerpoint slide
        self.data = SpectralCube(data=self.data, wcs=self.wcs)

    def help_plot_pv(self, axis):
        """
        If anything specific needs to be done to a PV diagram plot, do it here
        :param axis: the Axis object of the PV plot
        """
        if 'TELESCOP' in self.header and self.header['TELESCOP'] == "HATCREEK":
            axis.invert_yaxis()

    def name(self):
        """
        Return a short, descriptive of the data.
        Not like str or repr, because those describe the type of object as
            well. This only describes the data.
        For example, the BIMA data may be described as "BIMA 12CO(1-0)"
        """
        if self.telescope.lower() == 'sofia':
            line_description = "[12CII]"
        else:
            filename_components = self.basename.replace('.fits', '').split('_')
            line_description = [x for x in filename_components if 'CO' in x].pop()
        return f"{self.telescope.upper()} {line_description}"


    def __str__(self):
        return f"CubeData({self.basename})"

    def __repr__(self):
        return f"<CubeData wrapper around {self.data.__repr__()}>"
