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

from astropy.io import fits
from reproject import reproject_interp

from radio_beam import Beam
from astropy.convolution import convolve
from astropy import units as u
from astropy.wcs import WCS


band_stubs = {70: "PACS70um", 160: "PACS160um"}
wavelengths = {band_stubs[k]: k for k in band_stubs}

working_dir = "/home/ramsey/Documents/Research/Feedback/rcw120_data"
filename_dict = {
    "PACS70um": os.path.join(working_dir, "rcw120_070.fits"),
    "PACS160um": os.path.join(working_dir, "rcw120_160.fits")
}

"""
Regrid the maps to the largest-pixel grid

Keep this nice and tidy so I could switch to a longer wavelength band if I want
to later. Don't assume there will only be these two bands
"""

def modify_header(header):
    del_list = []
    for k in header:
        if ('PLANE' in k) or (k[-1] == '3'):
            del_list.append(k)
    for k in del_list:
        del header[k]
    header['NAXIS'] = 2
    return header

ref_band = "PACS160um"

ref_header = modify_header(fits.getheader(filename_dict[ref_band]))
ref_wcs = WCS(ref_header)
ref_shape = ref_wcs.array_shape


# The band to be converted right now (run this file multiple times and change this)
raw_band = "PACS70um"
raw_data, raw_header = fits.getdata(filename_dict[raw_band], header=True)
raw_header = modify_header(raw_header)
raw_img = raw_data[0, :, :]
# Convert units from Jy/pixel to MJy/sr
pixel_scale = (np.abs(raw_header['CDELT1'])*u.deg).to(u.arcsec)
raw_img = raw_img * u.Jy / pixel_scale**2  # Jy/pixel / (area/pixel) = Jy/area
raw_img = raw_img.to(u.MJy/u.sr).to_value()

regridded_img = reproject_interp((raw_img, raw_header), ref_wcs, shape_out=ref_shape, return_footprint=False)

print("Regridded shape:", regridded_img.shape)

# Save this intermediate image
savename = filename_dict[raw_band].replace(".fits", "-remapped.fits")


header = modify_header(fits.getheader(filename_dict[raw_band]))
header.update(ref_wcs.to_header())
header['BUNIT'] = str(u.MJy/u.sr)
header['HISTORY'] = f"Units converted to MJy/sr by Ramsey Karim, 2022-11-30"
header['HISTORY'] = f"Regridded to {ref_band} grid by Ramsey Karim, 2022-11-30"
new_hdu = fits.PrimaryHDU(data=regridded_img, header=header)
new_hdu.writeto(savename)


"""
Convolve the maps to the lowest resolution


This needs beam areas

From preprocess.ipynb:
For the convolution, I need the beam sizes of the maps. These are not listed in
the FITS headers of these particular files, but we can get them from:

PACS Observer's Manual (Section 3.1)
(https://www.cosmos.esa.int/documents/12133/996891/PACS+Observers'+Manual)

SPIRE Handbook (Section 5, Table 5.2 in version 3.2)
(https://www.cosmos.esa.int/documents/12133/1035800/The+Herschel+Explanatory+Supplement%2C%20Volume+IV+-+THE+SPECTRAL+AND+PHOTOMETRIC+IMAGING+RECEIVER+%28SPIRE%29/c36d074d-32b4-48ec-b13f-4ca320788df3)

"""
beam_params_dict = {
    # major (as), minor (as), PA (deg)
    'PACS70um': (12.16, 5.86, 63.0),
    'PACS160um': (15.65, 11.64, 53.4),

    'SPIRE250um': (18.1,),
    'SPIRE350um': (24.9,),
    'SPIRE500um': (36.4,),
}
def get_beam(band_stub):
    """
    :param band_stub: string like "PACS70um" which identifies the band
    :returns: Beam
    """
    beam_params = beam_params_dict[band_stub]
    if len(beam_params) > 1:
        # PACS, elliptical
        return Beam(major=beam_params[0]*u.arcsec, minor=beam_params[1]*u.arcsec, pa=beam_params[2]*u.deg)
    else:
        # SPIRE, circular
        return Beam(beam_params[0]*u.arcsec)



beam_raw = get_beam(raw_band)
beam_ref = get_beam(ref_band)


# Find the beam that will convolve the raw beam to the ref beam
conv_beam = beam_ref.deconvolve(beam_raw)
pixel_scale = (np.abs(ref_header['CDELT1'])*u.deg).to(u.arcsec)
conv_kernel = conv_beam.as_kernel(pixel_scale)

conv_img = convolve(regridded_img, conv_kernel, preserve_nan=True)

# Save this final image
savename = filename_dict[raw_band].replace(".fits", "-remapped-conv.fits")
header = modify_header(fits.getheader(filename_dict[raw_band]))
header.update(ref_wcs.to_header())
header['BUNIT'] = str(u.MJy/u.sr)
header['HISTORY'] = f"Units converted to MJy/sr by Ramsey Karim, 2022-11-30"
header['HISTORY'] = f"Regridded to {ref_band} grid by Ramsey Karim, 2022-11-30"
header['HISTORY'] = f"Convolved to {ref_band} resolution by Ramsey Karim, 2022-11-30"
new_hdu = fits.PrimaryHDU(data=conv_img, header=header)
new_hdu.writeto(savename)









# Rewrite the reference band with correct units
savename = filename_dict[ref_band].replace(".fits", "-remapped-conv.fits")

ref_data, ref_header = fits.getdata(filename_dict[ref_band], header=True)
ref_header = modify_header(ref_header)
ref_img = ref_data[0, :, :]
# Convert units
pixel_scale = (np.abs(ref_header['CDELT1'])*u.deg).to(u.arcsec)
ref_img = ref_img * u.Jy / pixel_scale**2  # Jy/pixel / (area/pixel) = Jy/area
ref_img = ref_img.to(u.MJy/u.sr).to_value()

header = modify_header(fits.getheader(filename_dict[ref_band]))
header.update(ref_wcs.to_header())
header['BUNIT'] = str(u.MJy/u.sr)
header['HISTORY'] = f"Units converted to MJy/sr by Ramsey Karim, 2022-11-30"
new_hdu = fits.PrimaryHDU(data=ref_img, header=header)
new_hdu.writeto(savename)





"""
Plan for later:
Reorganize this file, make the above part only run when we want it
Copy from calibrate.py to do PACS calibration
Then copy from g0_dust.fir_intensity_2 (scroll down more) to get T, tau
and copy from that same function to get FIR intensity
"""
