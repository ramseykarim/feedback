"""
Sole purpose of this script is to regrid the X-ray data for RCW 49 into
FK5 (J2000) coordinates for Maitraiyee.
Can grab the footprit from SOFIA CII and use the pixel scale from X-ray
There are functions in mosaic_vlt.py for this

Created: 8/11/2020
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from reproject import reproject_interp

from . import mosaic_vlt
from .catalog import utils as catalog_utils


xray_filename = f"{catalog_utils.ancillary_data_path}chandra/full_band.fits"
new_xray_filename = f"{catalog_utils.ancillary_data_path}chandra/chandraACIS_fullband_regrid_4.fits"

phase = 'do work'

if phase == 'do work':
    # Get the full velocity range moment 0 image; just need the footprint
    cii_img, cii_w = catalog_utils.load_cii()

    xray_img, xray_hdr = fits.getdata(xray_filename, header=True)
    xray_w = mosaic_vlt.WCS(xray_hdr)

    new_xray_w = mosaic_vlt.make_wcs_like(cii_w, xray_w)
    new_xray_img, fp = reproject_interp((xray_img, xray_w), new_xray_w, shape_out=new_xray_w.array_shape)

    new_xray_hdr = new_xray_w.to_header()
    kwargs_to_copy = [
        'BUNIT','BSCALE', 'HDUNAME', 'ORIGIN', 'MJD_OBS', 'CONTENT',
    ]
    for k in kwargs_to_copy:
        new_xray_hdr[k] = xray_hdr[k]
    new_xray_hdr['HISTORY'] = "rkarim regridded this to the SOFIA CII footprint on Aug 11, 2020"
    del new_xray_hdr['LATPOLE'], new_xray_hdr['LONPOLE']
    # new_xray_hdr['LATPOLE'] = 90.0
    # new_xray_hdr['LONPOLE'] = 180.0


    print(new_xray_hdr)

    # plt.subplot(111, projection=xray_w)
    # plt.imshow(fp, origin='lower')
    # plt.show()


    fits.writeto(new_xray_filename, new_xray_img, new_xray_hdr, overwrite=True)

    # plt.figure()
    # plt.subplot(121, projection=cii_w)
    # plt.imshow(cii_img, origin='lower')
    # plt.subplot(122, projection=xray_w)
    # plt.imshow(xray_img, origin='lower')
    # plt.show()
elif phase == 'check work':
    new_xray_img, new_xray_hdr = fits.getdata(new_xray_filename, header=True)
elif phase == 'do the bad version':
    cii_img, cii_w = catalog_utils.load_cii()

    xray_img, xray_hdr = fits.getdata(xray_filename, header=True)
    # print(xray_hdr)
    xray_w = mosaic_vlt.WCS(xray_hdr)

    new_xray_img, fp = reproject_interp((xray_img, xray_w), cii_w, shape_out=cii_w.array_shape)

    new_xray_hdr = cii_w.to_header()
    print(new_xray_hdr)

    kwargs_to_copy = [
        'BUNIT','BSCALE', 'HDUNAME', 'ORIGIN', 'MJD_OBS', 'CONTENT',
    ]
    for k in kwargs_to_copy:
        new_xray_hdr[k] = xray_hdr[k]
    new_xray_hdr['HISTORY'] = "rkarim regridded this to the SOFIA CII footprint on Aug 11, 2020"

    # fits.writeto(new_xray_filename, new_xray_img, new_xray_hdr, overwrite=True)
