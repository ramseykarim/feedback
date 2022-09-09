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
try:
    import MontagePy.main as montage
except ModuleNotFoundError:
    montage = None

from . import catalog


"""
All this uncertainty stuff is copied from m16_threads.channel_maps_again, where I first ironed out this technique
The contour levels can now be tied to the noise (which I also figured out for each map pretty well)
On feb 22 2022 I moved this from m16_investigation.overlaid_contours_for_offset
to here so that I can have just one main copy and make sure I'm always using
the same noise values across all functions.
I will always use "CONV" to mean CII beam. If I mean APEX CO 3-2, I'll just say APEX and it will always mean the 20'' beam
"""
onesigmas = { # all values in K. These are the 1sigma noise levels, which contours will be based on
    'cii': 1.0, # CII (might be 1.2 but sampling that really dark area shows 1 and that seems good to me)
    'ciiAPEX': 0.8, # August 18, 2022 (mostly approx'd from DS9)
    'hcn': 0.55, 'hcnCONV': 0.26, # HCN: I finally checked it, it is very close to HCO+
    'hcop': 0.57, 'hcopCONV': 0.27, # HCO+/conv
    '12co10': 6.2, '12co10CONV': 2.0, # 12co10/conv
    '12co10APEX': 1.5, # August 18, 2022; hard to estimate
    '13co10': 2.6, '13co10CONV': 0.7, # 13co10/conv
    'c18o10': 0.66, 'c18o10CONV': 0.40, # c18o10/conv
    'co65': 1.65, 'co65CONV': 0.57, # checked August 11 and 18, 2022
    '12co32': 0.69, '13co32': 0.62, # checked August 18, 2022; I wonder if these should be the same
    'n2hp': 0.56, 'n2hpCONV': 0.24, # Finally checked these on April 25 and 26, 2022
    'cs': 0.60, 'csCONV': 0.24, # Both cs and n2hp CONV are 0.24, it's not just copy-paste
}

cubenames = {
    'cii': '[C II]', 'ciiAPEX': '[C II] (CO(3-2) beam)',
    'hcn': 'HCN(1-0)', 'hcnCONV': 'HCN(1-0) (CII beam)',
    'hcop': 'HCO+(1-0)', 'hcopCONV': 'HCO+(1-0) (CII beam)',
    '12co10': '$^{12}$CO(1-0)', '12co10CONV': '$^{12}$CO(1-0) (CII beam)',
    '12co10APEX': '$^{12}$CO(1-0) (CO(3-2) beam)',
    '13co10': "$^{13}$CO(1-0)", '13co10CONV': "$^{13}$CO(1-0) (CII beam)",
    'c18o10': "C$^{18}$O(1-0) (smooth)", 'c18o10CONV': "C$^{18}$O(1-0) (CII beam, smooth)",
    'co65': "$^{12}$CO(6-5)", 'co65CONV': "$^{12}$CO(6-5)  (CII beam)",
    '12co32': "$^{12}$CO(3-2)", '13co32': "$^{13}$CO(3-2)",
    'n2hp': 'N$_2$H+(1-0)', 'n2hpCONV': 'N$_2$H+(1-0) (CII beam)',
    'cs': 'CS(2-1)', 'csCONV': 'CS(2-1) (CII beam)',
}


carma_template = lambda mol, conv : f"carma/M16.ALL.{mol}.sdi.cm.subpv" + ('.SOFIAbeam' if conv else '') + ".fits"
cubefilenames = {
    'cii': "sofia/M16_CII_U.fits", 'ciiAPEX': "sofia/M16_CII_U_APEXbeam.fits",
    'hcn': carma_template('hcn', False), 'hcnCONV': carma_template('hcn', True),
    'hcop': carma_template('hcop', False), 'hcopCONV': carma_template('hcop', True),
    'n2hp': "carma/n2hp_fullres_j_10_f1_01_f_12.fits", 'n2hpCONV': "carma/n2hp_fullres_j_10_f1_01_f_12.SOFIAbeam.fits",
    'cs': carma_template('cs', False), 'csCONV': carma_template('cs', True),
    '12co10': "bima/M16_12CO1-0_7x4.fits", '12co10CONV': "bima/M16_12CO1-0_14x14.fits",
    '12co10APEX': "bima/M16_12CO1-0_APEXbeam.fits",
    '13co10': "bima/M16.BIMA.13co1-0.fits", '13co10CONV': "bima/M16.BIMA.13co1-0.SOFIAbeam.fits",
    'c18o10': "bima/M16.BIMA.c18o.cm.SMOOTH.fits", 'c18o10CONV': "bima/M16.BIMA.c18o.cm.SOFIAbeam.SMOOTH.fits",
    'co65': "apex/M16_CO6-5.fits", 'co65CONV': "apex/M16_CO6-5.SOFIAbeam.fits",
    '12co32': "apex/M16_12CO3-2_truncated.fits", '13co32': "apex/M16_13CO3-2_truncated.fits", # spectra trimmed to approx. CII limits

}


def beam_area(theta_a, theta_b):
    """
    Return beam area in steradians given the half-power full widths
    at the major and minor axes. Inputs should be Quantites.
    I am using the equation given in:
    https://science.nrao.edu/facilities/vla/proposing/TBconv
    EDIT: there's a whole package for radio beams, so I probably shouldn't
        try to reinvent any wheels and risk messing up
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
    Updated November 16, 2020 to take CARMA data
    """
    def __init__(self, filename):
        if not os.path.exists(filename):
            self.full_path = catalog.utils.search_for_file(filename)
        else:
            self.full_path = os.path.abspath(filename)
        self.telescope = self.full_path.split('/')[-2]
        self.basename = os.path.basename(self.full_path)
        self.directory = os.path.dirname(self.full_path)
        with fits.open(self.full_path) as hdul:
            self.header = hdul[0].header
            self.wcs = WCS(self.header, naxis=3)
        self.data = SpectralCube.read(self.full_path)
        if self.data.unit == u.one:
            self.data._unit = u.K
        self.wcs_flat = self.data[0, :, :].wcs
        self.equivalencies = None
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
        elif self.telescope.lower() == 'bima':
            filename_components = self.basename.replace('.fits', '').split('_')
            line_description = [x for x in filename_components if 'CO' in x].pop()
        elif self.telescope.lower() == 'carma':
            line_description = self.basename.split('.')[2].upper().replace('P', '+')
        else:
            raise NotImplementedError(self.telescope)
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

    def equivalency(self):
        if self.equivalencies is None:
            if 'beam' in str(self.data.unit).lower():
                # BIMA
                try:
                    restfrq = self.header['RESTFREQ'] * u.Hz
                except:
                    print(self.header)
                    raise NotImplementedError
                beam = self.data.beam
                beam_area = 2.*np.pi*beam.major*beam.minor/2.355**2
                self.equivalencies = u.brightness_temperature(restfrq, beam_area)
            else:
                print(f"\'beam\' not in unit: {str(self.data.unit).lower()}")
                raise NotImplementedError("Why's this happening?")
        return self.equivalencies

    def convert_to_K(self):
        if self.data.unit != u.K:
            self.data = self.data.to(u.K, equivalencies=self.equivalency())
        return self

    def refresh_wcs(self):
        """
        Reassign the WCS attributes

        Musings on May 25, 2021: Why did I make this function? Was it important?
        Should I be using it more often?
        Answer: no, it's apparently useful if I want to convert a *cutout* of
        a cube from Jy/beam to K. I guess the conversion doesn't use the wcs,
        but it's good practice to make sure they match
        Future: Should wrap this into a setter function for self.data !
        """
        self.wcs = self.data.wcs
        self.wcs_flat = self.data[0, :, :].wcs


def make_moment_series(cube, velocity_range, velocity_spacing, return_nchannels=False):
    """
    :param cube: SpectralCube with at least good velocity units
    :param velocity_range: two-element sequence of (low, high) velocity
        limits. The high limit is not inclusive, the low limit is.
        The limits should be Quantities.
        These are the first two arguments for a "range" function
    :param velocity_spacing: Spacing for the channel maps. Should be a Quantity.
        This is the third argument to a "range" function.
    :param return_nchannels: bool: also return a list, same length as the regular return value,
        of the number of channels averaged into each moment0.
        This return value is optional for backwards compatibility
    :returns: a list whose length is the number of moment images in the range.
        Each element is a 3-item tuple: low velocity limit, high velocity limit,
        and the moment image in units of K*velocity_unit
        The low and high velocity limits are for each channel
        If return_nchannels is True, also returns a list of ints for the number
        of original channels that were integrated into each moment (this is
        useful for noise estimation purposes)
    2021-10-11: Moved from m16_pictures.py to cube_utils.py
    """
    v_unit = velocity_spacing.unit
    # This is the "left edge" of each channel map. The right edge will be this
    # plus velocity_spacing
    v0_range = np.arange(*(v.to(v_unit).to_value() for v in velocity_range), velocity_spacing.to_value()) * v_unit
    # Gather moments into list
    moments = []
    nchannels = []
    for v0 in v0_range:
        v1 = v0 + velocity_spacing
        spectral_slab = cube.spectral_slab(v0, v1)
        nchannels.append(spectral_slab.shape[0])
        moments.append((v0, v1, spectral_slab.moment0().to(u.K*v_unit)))
    if return_nchannels:
        return moments, nchannels
    else:
        return moments


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
