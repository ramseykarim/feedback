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
from astropy import modeling

from spectral_cube import SpectralCube
from spectral_cube.spectral_cube import Beam
import MontagePy.main as montage

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
        self.directory = os.path.dirname(self.full_path)
        with fits.open(self.full_path) as hdul:
            self.header = hdul[0].header
            self.wcs = WCS(self.header, naxis=3)
            self.wcs_flat = WCS(self.header, naxis=2)
        self.data = SpectralCube.read(self.full_path)
        if self.data.unit == u.one:
            self.data._unit = u.K

        ### Is there any way I can save memory??
        # tmp = self.data.spectral_slab(-15*u.km/u.s, +75*u.km/u.s)
        # del self.data
        # self.data = tmp
        """
        I should check if I need to use any beam efficiency corrections
        Ask Marc via powerpoint slide
        Answer: I don't
        """

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
        This name has spaces in it
        """
        if self.telescope.lower() == 'sofia':
            line_description = "[12CII]"
        else:
            filename_components = self.basename.replace('.fits', '').split('_')
            line_description = [x for x in filename_components if 'CO' in x].pop()
        return f"{self.telescope.upper()} {line_description}"

    def filename_stub(self):
        """
        Return a short name appropriate for placement in a filename
        Similar to self.name(), but will not contain spaces
        """
        name = self.name().replace(self.telescope.upper(), self.telescope.lower()).replace(' ', '_')
        return name.replace('[', '').replace(']', '')

    def __str__(self):
        return f"CubeData({self.basename})"

    def __repr__(self):
        return f"<CubeData wrapper around {self.data.__repr__()}>"


def montage_ProjectCube(filename_cube, filename_target):
    """
    Code bits found here:
    https://github.com/Caltech-IPAC/MontageNotebooks/blob/master/mGetHdr.ipynb
    https://github.com/Caltech-IPAC/MontageNotebooks/blob/master/mProjectCube.ipynb
    """
    # Get the naxis values
    with fits.open(filename_cube) as hdul:
        hdr_cube = hdul[0].header
        naxis_cube = hdul[0].header['NAXIS']
    with fits.open(filename_target) as hdul:
        hdr_target = hdul[0].header
        naxis_target = hdul[0].header['NAXIS']
    # Print out some useful diagnostic information
    print("USING MONTAGE: projecting the cube")
    padding = " "*2
    print(padding, filename_cube)
    print(padding, f" (with {naxis_cube} axes)")
    print(padding+"to the spatial grid from")
    print(padding, filename_target)
    print(padding, f" (with {naxis_target} axes)")

    # Extract header file to a temporary text file
    filename_target_hdr_file = os.path.join(os.path.dirname(filename_target), os.path.basename(filename_target).replace('.fits', '_zzzHEADERzzz.hdr'))
    print("Extracting header to temporary file ")
    print(padding, filename_target_hdr_file)
    if False:
        rtn = montage.mGetHdr(filename_target, filename_target_hdr_file)
        print(padding, "mGetHdr: ", rtn)
        if int(rtn['status']) != 0:
            print("mGetHdr FAILED, exiting...")
            return
    else:
        spatial_cards_target = [x for x in hdr_target.keys() if x[-1] in ('1', '2')]
        spatial_cards_cube = [x for x in hdr_cube.keys() if x[-1] in ('1', '2')]
        history_cards_cube = [x for x in hdr_cube.keys() if 'HISTORY' in x]
        ax4_cards = [x for x in hdr_cube.keys() if x[-1] == '4']
        for card in spatial_cards_cube:
            hdr_cube.remove(card)
        for card in spatial_cards_target:
            hdr_cube[card] = hdr_target[card]
        print(padding+f"Removing {len(history_cards_cube)} HISTORY cards.")
        for card in history_cards_cube:
            hdr_cube.remove(card)
        for card in ax4_cards:
            hdr_cube.remove(card)
        hdr_cube['NAXIS'] = 3
        hdr_cube.totextfile(filename_target_hdr_file, overwrite=True)
    # Use mProjectCube to execute the projection
    target_name_stub = os.path.basename(filename_target).replace('.fits', '').lower()
    filename_cube_projected = os.path.join(os.path.dirname(filename_cube), os.path.basename(filename_cube).replace('.fits', f'_REPROJ_{target_name_stub}_GRID.fits'))
    rtn = montage.mProjectCube(filename_cube, filename_cube_projected, filename_target_hdr_file)
    print(padding, "mProjectCube:", rtn)
    if int(rtn['status']) != 0:
        print("mProjectCube FAILED, exiting...")
        return

    if os.path.exists(filename_target_hdr_file) and (filename_target_hdr_file != filename_target):
        print("Removing temporary header file ")
        print(padding, filename_target_hdr_file)
        # os.remove(filename_target_hdr_file)
    else:
        print("TARGET HEADER FILE NOT FOUND!", filename_target_hdr_file)

    print("SUCCESS! Projected to")
    print(padding, filename_cube_projected)
    print()


# class TripleGaussian1D(modeling.Fittable1DModel):
#
#     # Triplet of parameters for each Gaussian
#     a1 = modeling.Parameter()
#     mu1 = modeling.Parameter()
#     std1 = modeling.Parameter()
#
#     a2 = modeling.Parameter()
#     mu2 = modeling.Parameter()
#     std2 = modeling.Parameter()
#
#     a3 = modeling.Parameter()
#     mu3 = modeling.Parameter()
#     std3 = modeling.Parameter()
#
#     @staticmethod
#     def evaluate(x, a1, mu1, std1, a2, mu2, std2, a3, mu3, std3):
#         return modeling.models.Gaussian1D.evaluate(x, a1, mu1, std1) + modeling.models.Gaussian1D.evaluate(x, a2, mu2, std2) + modeling.models.Gaussian1D.evaluate(x, a3, mu3, std3)
#
#     @staticmethod
#     def fit_deriv(x, a1, mu1, std1, a2, mu2, std2, a3, mu3, std3):
#         return modeling.models.Gaussian1D.fit_deriv(x, a1, mu1, std1) + modeling.models.Gaussian1D.fit_deriv(x, a2, mu2, std2) + modeling.models.Gaussian1D.fit_deriv(x, a3, mu3, std3)


if __name__ == "__main__":
    montage_ProjectCube(catalog.utils.search_for_file("bima/M16_12CO1-0_7x4.fits"), catalog.utils.search_for_file("sofia/M16_CII_U.fits"))
    # montage_ProjectCube(catalog.utils.search_for_file("apex/M16_12CO3-2.fits"), catalog.utils.search_for_file("apex/M16_13CO3-2.fits"))
