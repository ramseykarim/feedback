"""
Heads up, this file is a mess, but did its job. It probably shouldn't be used
as example code unless you know what you're doing.

Preprocess the 70 and 160 micron Herschel PACS photometry so that they are on
the same grid and at the same resolution.
This serves the same purpose as Tracy's IDL procedures as well as my
preprocess.ipynb notebook in the helpss repository.

This isn't (yet) meant to be a formal writeup of this process. It's just a patch
(like preprocess.ipynb) used a couple times wherever running my wrapper over
Tracy's routine doesn't make sense.

This code is approximately copied from preprocess.ipynb
(https://github.com/ramseykarim/helpss/blob/master/preprocess.ipynb)

The first use case here is RCW 120.
The files were saved in the worst way possible for WCS, so this file contains
a lot of unprecedented surgery and special handling and is not as generalizable
as I'd hoped.

Created: November 30, 2022
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import datetime

from astropy.io import fits
from reproject import reproject_interp

from radio_beam import Beam
from astropy.convolution import convolve
from astropy import units as u
from astropy.wcs import WCS
from astropy.modeling import models

from scipy.interpolate import UnivariateSpline

from math import ceil


class Preprocess():

    band_stubs = {70: "PACS70um", }
    wavelengths = {"PACS70um": 70, "PACS160um": 160}
    working_dir = "/home/ramsey/Documents/Research/Feedback/rcw120_data"

    def __init__(self):
        # No arguments, just vibes
        # Just loading in some instance attributes which could theoretically change
        self.ref_band = "PACS160um"
        self.raw_band = "PACS70um"

        self.filename_dict = {
            "PACS70um": os.path.join(self.working_dir, "rcw120_070.fits"),
            "PACS160um": os.path.join(self.working_dir, "rcw120_160.fits")
        }


    @staticmethod
    def modify_header(header):
        # Remove a bunch of troublesome keywords from the old headers
        del_list = []
        for k in header:
            if ('PLANE' in k) or (k[-1] == '3'):
                del_list.append(k)
        for k in del_list:
            del header[k]
        header['NAXIS'] = 2
        return header

    @staticmethod
    def make_new_header(filename_for_model_header, ref_wcs):
        # Pull an old header, mess with it, and return a better one
        header = Preprocess.modify_header(fits.getheader(filename_for_model_header))
        header.update(ref_wcs.to_header())
        header['BUNIT'] = str(u.MJy/u.sr)
        return header

    """
    Herschel BEAM AREAS
    From preprocess.ipynb:
    For the convolution, I need the beam sizes of the maps. These are not listed in
    the FITS headers of these particular files, but we can get them from:

    PACS Observer's Manual (Section 3.1)
    (https://www.cosmos.esa.int/documents/12133/996891/PACS+Observers'+Manual)

    SPIRE Handbook (Section 5, Table 5.2 in version 3.2)
    (https://www.cosmos.esa.int/documents/12133/1035800/The+Herschel+Explanatory+Supplement%2C%20Volume+IV+-+THE+SPECTRAL+AND+PHOTOMETRIC+IMAGING+RECEIVER+%28SPIRE%29/c36d074d-32b4-48ec-b13f-4ca320788df3)

    Note: the RCW 120 photometry was taken in SCAN mode at 20 arcsec/s (different than the usual Parallel 60)
    """
    beam_params_dict = {
        # major (as), minor (as), PA (deg)
        # 'PACS70um': (12.16, 5.86, 63.0), # 60 arcsec/s (Parallel)
        # 'PACS160um': (15.65, 11.64, 53.4), # 60 arcsec/s (Parallel)
        'PACS70um': (5.76, 5.46, 0), # 20 arcsec/s (Scan)
        'PACS160um': (12.13, 10.65, 9.3), # 20 arcsec/s (Scan)

        'SPIRE250um': (18.1,),
        'SPIRE350um': (24.9,),
        'SPIRE500um': (36.4,),
    }
    @staticmethod
    def get_beam(band_stub):
        """
        :param band_stub: string like "PACS70um" which identifies the band
        :returns: Beam
        """
        beam_params = __class__.beam_params_dict[band_stub] # neat solution! https://stackoverflow.com/questions/18431313/how-can-static-method-access-class-variable-in-python
        if len(beam_params) > 1:
            # PACS, elliptical
            return Beam(major=beam_params[0]*u.arcsec, minor=beam_params[1]*u.arcsec, pa=beam_params[2]*u.deg)
        else:
            # SPIRE, circular
            return Beam(beam_params[0]*u.arcsec)

    def processingstep_regrid(self):
        """
        Regrid the maps to the largest-pixel grid

        Keep this nice and tidy so I could switch to a longer wavelength band if I want
        to later. Don't assume there will only be these two bands
        """
        # Get reference image info
        ref_header = self.modify_header(fits.getheader(self.filename_dict[self.ref_band]))
        self.ref_wcs = WCS(ref_header)
        ref_shape = self.ref_wcs.array_shape

        # The band to be converted right now (run this file multiple times and change this)
        raw_data, raw_header = fits.getdata(self.filename_dict[self.raw_band], header=True)
        raw_header = self.modify_header(raw_header)
        raw_img = raw_data[0, :, :]
        # Convert units from Jy/pixel to MJy/sr
        self.pixel_scale = (np.abs(raw_header['CDELT1'])*u.deg).to(u.arcsec)
        raw_img = raw_img * u.Jy / self.pixel_scale**2  # Jy/pixel / (area/pixel) = Jy/area
        raw_img = raw_img.to(u.MJy/u.sr).to_value()

        regridded_img = reproject_interp((raw_img, raw_header), self.ref_wcs, shape_out=ref_shape, return_footprint=False)
        # print("Regridded shape:", regridded_img.shape)

        # Save this intermediate image
        savename = self.filename_dict[self.raw_band].replace(".fits", "-remapped.fits")

        header = self.make_new_header(self.filename_dict[self.raw_band], self.ref_wcs)
        header['HISTORY'] = f"Units converted to MJy/sr by Ramsey Karim, 2022-11-30"
        header['HISTORY'] = f"Regridded to {self.ref_band} grid by Ramsey Karim, 2022-11-30"
        new_hdu = fits.PrimaryHDU(data=regridded_img, header=header)
        new_hdu.writeto(savename, overwrite=True)

        # Return the regridded image for the convolution
        return regridded_img

    def processingstep_convolve(self, regridded_img):
        """
        Convolve the maps to the lowest resolution
        This needs beam areas
        """
        beam_raw = self.get_beam(self.raw_band)
        beam_ref = self.get_beam(self.ref_band)

        # Find the beam that will convolve the raw beam to the ref beam
        conv_beam = beam_ref.deconvolve(beam_raw)
        conv_kernel = conv_beam.as_kernel(self.pixel_scale)
        # Convolve
        conv_img = convolve(regridded_img, conv_kernel, preserve_nan=True)

        # Save this final image
        savename = self.filename_dict[self.raw_band].replace(".fits", "-remapped-conv.fits")

        header = self.make_new_header(self.filename_dict[self.raw_band], self.ref_wcs)
        header['HISTORY'] = f"Units converted to MJy/sr by Ramsey Karim, 2022-11-30"
        header['HISTORY'] = f"Regridded to {self.ref_band} grid by Ramsey Karim, 2022-11-30"
        header['HISTORY'] = f"Convolved to {self.ref_band} resolution by Ramsey Karim, 2022-11-30"
        new_hdu = fits.PrimaryHDU(data=conv_img, header=header)
        new_hdu.writeto(savename, overwrite=True)

    def processingstep_rewrite_reference_with_new_units(self):
        """
        Just rewrite the reference band (which otherwise is unedited) with new
        units. Herschel PACS photometry is in Jy/pixel off the archive, so we
        have to divide by the pixel area in angular units and convert to MJy/sr
        """
        savename = self.filename_dict[self.ref_band].replace(".fits", "-remapped-conv.fits")

        ref_data, ref_header = fits.getdata(self.filename_dict[self.ref_band], header=True)
        ref_header = self.modify_header(ref_header)
        ref_img = ref_data[0, :, :]
        # Convert units
        ref_img = ref_img * u.Jy / self.pixel_scale**2  # Jy/pixel / (area/pixel) = Jy/area
        ref_img = ref_img.to(u.MJy/u.sr).to_value()

        header = self.make_new_header(self.filename_dict[self.ref_band], self.ref_wcs)
        header['HISTORY'] = f"Units converted to MJy/sr by Ramsey Karim, 2022-11-30"
        new_hdu = fits.PrimaryHDU(data=ref_img, header=header)
        new_hdu.writeto(savename, overwrite=True)

    def run(self):
        """
        Run all preprocessing steps
        """
        regridded_img = self.processingstep_regrid()
        self.processingstep_convolve(regridded_img)
        self.processingstep_rewrite_reference_with_new_units()
        print("Preprocessing done.")


# Import pacs_calibrate
import sys
sys.path.append("/home/ramsey/Documents/Research/Filaments/pacs_calibrate")
import pacs_calibrate
# Adjust pacs_calibrate path configuration
planck_herschel_data_path = "/home/ramsey/Documents/Research/Filaments/filterInfo_PlanckHerschel/"
pacs_calibrate.path_config.gnilc_directory = planck_herschel_data_path
pacs_calibrate.path_config.planck_directory = planck_herschel_data_path
pacs_calibrate.path_config.herschel_bandpass_directory = planck_herschel_data_path
# Adjust Herschel beams in pacs_calibrate (this shouldn't have been set like this, but oh well)
# This is because the RCW 120 photometry was done in SCAN mode at 20 arcsec/s
pacs_calibrate.path_config.herschel_beam_sizes = (
    np.sqrt(np.prod(np.array(Preprocess.beam_params_dict['PACS70um'][:2]))),
    pacs_calibrate.path_config.herschel_beam_sizes[1],
    np.sqrt(np.prod(np.array(Preprocess.beam_params_dict['PACS160um'][:2]))),
    *pacs_calibrate.path_config.herschel_beam_sizes[3:],
)

class Calibrate():
    """
    Calibrate the PACS observations
    Copying/repurposing code from pacs_calibrate/calibrate.py, which is too
    specific to the HELPSS to be useful here
    I had to make some edits to pacs_calibrate, but it was well worth it and I
    got this to work as an imported package for the first time!
    """
    def __init__(self):
        print(f"Successfully loaded in pacs_calibrate v{pacs_calibrate.__version__}")
        self.bands = ["PACS70um", "PACS160um"]
        # Use the NATIVE RESOLUTION but regridded versions
        self.filename_dict = {
            "PACS70um": os.path.join(Preprocess.working_dir, f"rcw120_070-remapped.fits"),
            "PACS160um": os.path.join(Preprocess.working_dir, f"rcw120_160-remapped-conv.fits") # 160 micron wasn't actually convolved, but this has correct units
        }
        self.offset_dict = {}

    def find_offset(self, band):
        """
        Do all the calibration here (it's easy since it's all in the package)
        """
        pacs_flux_filename = self.filename_dict[band]
        model = pacs_calibrate.calc_offset.GNILCModel(pacs_flux_filename, target_bandpass=band, save_masks=False, default_savedir=Preprocess.working_dir)
        derived_offset = model.get_offset(savedir=Preprocess.working_dir, full_diagnostic=True)
        print(f"{band} offset {derived_offset}")
        self.offset_dict[band] = int(round(derived_offset))

    def run(self):
        """
        Calibrate all bands
        """
        for band in self.bands:
            self.find_offset(band)
        return self.offset_dict


# Import mantipython
sys.path.append("/home/ramsey/Documents/Research")
from mantipython.physics import greybody, dust, instrument


class DeriveDustProperties:
    """
    Follow g0_dust.fir_intensity_2 direct T,tau calculation for 70 and 160 maps
    """
    def __init__(self):
        self.filename_dict = {
            "PACS70um": os.path.join(Preprocess.working_dir, f"rcw120_070-remapped-conv.fits"),
            "PACS160um": os.path.join(Preprocess.working_dir, f"rcw120_160-remapped-conv.fits") # 160 micron wasn't actually convolved, but this has correct units
        }
        self.offset_dict = {"PACS70um": 146, "PACS160um": 799} # Calculated Dec 2, 2022 (I got 969 before for 160um, not sure what changed...)
        detectors = instrument.get_instrument([70, 160])
        self.detectors = {"PACS70um": detectors[0], "PACS160um": detectors[1]}
        self.data_dict = {}
        self.header_dict = {}

    def get_image(self, band):
        if band not in self.data_dict:
            # load one band and add offset
            img, hdr = fits.getdata(self.filename_dict[band], header=True)
            img += self.offset_dict[band]
            # hdr['HISTORY'] = f"Added zero-point offset {self.offset_dict[band]} MJy/sr, rkarim 2022-12-02"
            self.data_dict[band] = img
            self.header_dict[band] = hdr
        return self.data_dict[band]

    def make_spline_models(self):
        """
        Two models:
        1)
        Create a spline interpolation from the 70/160 ratio to the temperature
        This uses bandpass-integrated intensities via mantipython
        2)
        Create a spline interpolation from the temperature to the 160 intensity
        if tau were 0
        The ratio of the actual 160 to the zerotau-160 gives tau

        Easier to do these in the same loop over the model temperature array
        """
        model_T_arr = np.arange(1, 200, 0.1)
        # Stuff for the first model
        model_bandpass_br_ratio = np.zeros_like(model_T_arr)
        args_1 = (-8., dust.TauOpacity(2.))
        # For the second model
        zerotau_160intensity = np.zeros_like(model_T_arr)
        args_2 = (0, dust.TauOpacity(2.))
        # Loop thru T
        for i, t in enumerate(model_T_arr):
            gb = greybody.Greybody(t, *args_1)
            p70_I = self.detectors['PACS70um'].detect(gb)
            p160_I = self.detectors['PACS160um'].detect(gb)
            model_bandpass_br_ratio[i] = p70_I / p160_I
            zerotau_160intensity[i] = self.detectors['PACS160um'].detect(greybody.ThinGreybody(t, *args_2))
        self.model_bandpass_br_spline = UnivariateSpline(model_bandpass_br_ratio, model_T_arr, s=0)
        self.zerotau_I_spline = UnivariateSpline(model_T_arr, zerotau_160intensity, s=0)

    def solve_T(self):
        # T = model_bandpass_br_spline(obs70/obs160)
        observed_br_ratio = self.get_image("PACS70um") / self.get_image("PACS160um")
        self.T = self.model_bandpass_br_spline(observed_br_ratio)

    def solve_tau(self):
        # tau = obs160 / zerotau_I_spline(T)
        self.tau = self.get_image("PACS160um") / self.zerotau_I_spline(self.T)

    def save_solution(self):
        new_hdr = WCS(self.header_dict['PACS160um']).to_header()
        new_hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        new_hdr['CREATOR'] = f"rkarim, via {__file__}.{__class__}"
        new_hdr['AUTHOR'] = "Ramsey Karim"
        new_hdr['OBJECT'] = "RCW120"
        obsIDs = "1342216585,1342216586(both), 1342185553,1342185554(160only)"
        new_hdr['HISTORY'] = "Herschel PACS 70,160. 160 grid and beam (20arcsec/s SCAN)"
        new_hdr['HISTORY'] = "obsIDs " + obsIDs
        p70_correction, p160_correction = (self.offset_dict[b] for b in ["PACS70um", "PACS160um"])
        new_hdr['HISTORY'] = f"Zero-point offsets: {p70_correction} (70), {p160_correction} (160)"
        new_hdr['HISTORY'] = f"Zero-point offsets from {__file__}.Calibrate, calculated today"
        new_hdr['COMMENT'] = "T,tau calc'd using bandpasses; see color_temperature_comparison.ipynb"
        hdul = fits.HDUList([fits.PrimaryHDU(),
            fits.ImageHDU(data=self.T, header=new_hdr.copy()),
            fits.ImageHDU(data=self.tau, header=new_hdr.copy())])
        hdul[1].header['EXTNAME'] = 'T'
        hdul[1].header['BUNIT'] = 'K'
        hdul[2].header['EXTNAME'] = 'tau'
        hdul[2].header['BUNIT'] = 'optical depth at 160 micron'
        savepath = os.path.join(Preprocess.working_dir, "rcw120_T-tau_colorsolution.fits")
        print("SAVING TO", savepath)
        hdul.writeto(savepath)

    def run(self):
        self.make_spline_models()
        self.solve_T()
        self.solve_tau()
        self.save_solution()


class CalculateFIR:

    def __init__(self):
        self.dust_solution_path = os.path.join(Preprocess.working_dir, "rcw120_T-tau_colorsolution.fits")

    def load_dust(self):
        with fits.open(self.dust_solution_path) as hdul:
            self.T = hdul['T'].data
            self.tau = hdul['tau'].data
            self.header = hdul['T'].header

    def make_fir_array(self):
        """
        Copying code from g0_dust.fir_intensity_2
        """
        nanmask = ~(np.isfinite(self.T) & np.isfinite(self.tau)) | (self.T < 0)
        self.T[nanmask] = np.nan
        self.tau[nanmask] = np.nan
        print("array 2D shape", self.T.shape)
        wl_lims = np.array([40., 500.])*u.micron
        nu_lims = wl_lims.to(u.Hz, equivalencies=u.spectral())
        nu_array = np.linspace(nu_lims[1].to_value(), nu_lims[0].to_value(), 1000) * u.Hz

        step_size = 60
        n_step = int(ceil(self.T.shape[0]/step_size))
        print(f"Calculating in {n_step} blocks of {step_size} rows")
        self.FIR = np.zeros_like(self.T)

        for idx in range(n_step):
            i0, i1 = step_size*idx, step_size*(idx+1)
            bb = models.BlackBody(self.T[i0:i1, :, np.newaxis]*u.K)
            S_array = bb(nu_array[np.newaxis, np.newaxis, :])
            print(f"Step {idx} with chunk size {S_array.size} {S_array.shape}")
            # beta=2 law means tau(nu) = tau_160 * (nu/nu160)**2
            tau_array = self.tau[i0:i1, :, np.newaxis] * (nu_array[np.newaxis, np.newaxis, :] / (160*u.micron).to(u.Hz, equivalencies=u.spectral()))**2
            print("Made it 1")
            I_array = S_array * (1. - np.exp(-tau_array))
            print("Made it 2")
            F_array = np.trapz(x=nu_array, y=I_array).to('erg s-1 cm-2 sr-1').to_value()
            print("Made it 3")
            self.FIR[i0:i1, :] = F_array

        self.FIR[nanmask] = np.nan

    def save_solution(self):
        del self.header['EXTNAME']
        self.header['BUNIT'] = 'erg s-1 cm-2 sr-1'
        self.header['COMMENT'] = "FIR integrated between 40-500 micron, beta=2"
        self.header['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        self.header['CREATOR'] = f"rkarim, via {__file__}.{__class__}"
        savename = os.path.join(Preprocess.working_dir, "rcw120_FIR.fits")
        hdu = fits.PrimaryHDU(data=self.FIR, header=self.header)
        hdu.writeto(savename, overwrite=True)

    def run(self):
        self.load_dust()
        self.make_fir_array()
        self.save_solution()


"""
Plan for later:
Reorganize this file, make the above part only run when we want it (done)
Copy from calibrate.py to do PACS calibration (done)
Then copy from g0_dust.fir_intensity_2 (scroll down more) to get T, tau
and copy from that same function to get FIR intensity
"""

if __name__ == "__main__":
    # Preprocess().run()
    # offsets = Calibrate().run()
    # for b in offsets:
    #     print(f"{b} -> {offsets[b]} MJy/sr")
    # DeriveDustProperties().run()
    CalculateFIR().run()

    # nu, transmission = pacs_calibrate.path_config.HerschelConfig.bandpass_profile("PACS70um")
    # plt.plot(nu, transmission)
    # plt.show()
