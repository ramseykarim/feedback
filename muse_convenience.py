import numpy as np
from astropy.io import fits
import spectr
import extinction

"""
Convenience functions for MUSE cubes
Created: October 3, 2019
"""
__author__ = "Ramsey Karim"


cube_filename_raw = "./muse/ADP.2018-02-01T11_53_14.286.fits"
mask_filename_star = "./muse/ADP.2018-02-01T11_53_14.286.smsk.fits"
mask_filename_absorption = "./muse/ADP.2018-02-01T11_53_14.286.amsk.fits"
cube_filename_conv = "./muse/ADP.2018-02-01T11_53_14.286.conv.fits"
cube_filename_resamp = "./muse/ADP.2018-02-01T11_53_14.286.resamp.fits"
cube_filename_rsmperr = "./muse/ADP.2018-02-01T11_53_14.286.rsmperr.fits"
NH_filename = "./muse/ADP.2018-02-01T11_53_14.286.NH.fits"
Av_filename = "./muse/ADP.2018-02-01T11_53_14.286.Av.fits"

def unpack_header(header_g, header_d):
    """
    header_g is the global header (probably doesn't have WCS)
    header_d is the data frame header (has WCS)
    """
    wlo, whi = header_g['WAVELMIN'], header_g['WAVELMAX']
    naxis3 = header_d['NAXIS3']
    cdelt3 = header_d['CD3_3']
    crval3 = header_d['CRVAL3']
    cunit3 = header_d['CUNIT3']
    bunit = header_d['BUNIT']
    wlrange_A = np.linspace(wlo, whi, naxis3)*10
    result_dict = {
        "wavl": wlrange_A,
        "wavl_unit": cunit3,
        "flux_unit": bunit,
    }
    return result_dict


def generate_masks(data_cube, wavl_array):
    if False:
        # Get rudimentary star mask (True if star)
        flatimg_stars = np.nanmean(data_cube[(wavl_array > 9090) & (wavl_array < 9200)], axis=0)
        median_stars = np.nanmedian(flatimg_stars[flatimg_stars < 10])
        flatimg_stars -= median_stars
        star_mask = (flatimg_stars>25)
        absorption_mask = (flatimg < 0)
    else:
        # Load them in from saved FITS (already ran/saved above code)
        star_mask = fits.getdata(mask_filename_star)
        absorption_mask = fits.getdata(mask_filename_absorption)
    return star_mask, absorption_mask


def trim_MUSE(array, wavl_array):
    return array[wavl_array < 8200]


def convolve_MUSE(data_cube, pix_fwhm=7):
    iarr, jarr = (np.arange(data_cube.shape[1]) - data_cube.shape[1]//2), (np.arange(data_cube.shape[2]) - data_cube.shape[2]//2)
    conv_kernel = misc_utils.gaussian(iarr, 0, pix_fwhm/2.35, 1)[:, np.newaxis] * misc_utils.gaussian(jarr, 0, pix_fwhm/2.35, 1)[np.newaxis, :]
    conv_kernel /= np.sum(conv_kernel)
    result_cube = np.empty(data_cube.shape)
    for i in range(result_cube.shape[0]):
        result_cube[i, :] = convolve_fft(data_cube[i, :], conv_kernel, nan_treatment='fill', preserve_nan=True)
    return result_cube


def get_Hab(wavl_array, spectrum):
    # Integrate Halpha and Hbeta lines
    # Hbeta: 4861 A; found between 4854-4868 (& subtract 1)
    # Halpha 6563 A; found between 6554-6572 (& subtract 8)
    background_f, background_rms = spectr.extract_background(wavl_array, spectrum)
    spectrum -= background_f(wavl_array)
    line_slices = [spectr.acquire_line(wl, wavl_array, spectrum, 2*background_rms) for wl in (6563, 4861)]
    jalpha, jbeta = (np.sum(spectrum[sl]) for sl in line_slices)
    # N_H = extinction.Hab_reddening(jalpha/jbeta, Rv=5.5, Hab_ratio=2.85, method="Draine")
    # print("line ratio: {:.3f}".format(jalpha/jbeta))
    # print("N(H): {:.2E}".format(N_H))
    return jalpha/jbeta
    # Av = extinction.extinction(1e4/extinction.VBAND_WAVELENGTH_A, method="Draine", Rv=5.5) * 1.086 * N_H
    # print("Av: {:.2f}".format(Av))


def resample_img(data_cube, header, box_size=7):
    """
    Regrid ALREADY CONVOLVED data using "box size"
    Regrid the data_cube to a much coarser pixel grid
    Box size is an odd number to be used as the side length of a square box
    The central pixel in the box is assigned to the resampled map
    """
    result = np.empty((data_cube.shape[0], data_cube.shape[1]//box_size, data_cube.shape[2]//box_size))
    count = np.empty((data_cube.shape[1]//box_size, data_cube.shape[2]//box_size))
    for i in range(data_cube.shape[1]//box_size):
        ic = i*box_size
        for j in range(data_cube.shape[2]//box_size):
            jc = j*box_size
            # Counts approx how much valid nearby data there is
            count[i, j] = np.sum(~np.isnan(data_cube[:, ic:ic+box_size, jc:jc+box_size]))/(data_cube.shape[0] * box_size**2)
            # Try to grab center pixel
            c_val = data_cube[:, ic+(box_size//2), jc+(box_size//2)]
            # See how much of the center pixel's spectrum is valid
            notnan_score = np.sum(~np.isnan(c_val))/data_cube.shape[0]
            if (notnan_score < 0.9) and (count[i, j] > 0.7):
                c_val = np.nanmean(data_cube[:, ic:ic+box_size, jc:jc+box_size], axis=(-2, -1))
            result[:, i, j] = c_val
    fits.writeto(cube_filename_resamp, result, header, overwrite=True)
    fits.writeto(cube_filename_rsmperr, count, header, overwrite=True)


H_line_series = [
    ["Paschen", "4-3 ", 1.87561*1e4],
    ["Paschen", "5-3 ", 1.28216*1e4],
    ["Paschen", "6-3 ", 1.09411*1e4],
    ["Paschen", "7-3 ", 1.00521*1e4],
    ["Paschen", "8-3 ", 0.95486*1e4],
    ["Paschen", "9-3 ", 0.92315*1e4],
    ["Paschen", "10-3", 0.90174*1e4],
    ["Paschen", "11-3", 0.88652*1e4],
    ["Paschen", "12-3", 0.87529*1e4],
    ["Paschen", "13-3", 0.86674*1e4],
    ["Paschen", "14-3", 0.86008*1e4],
    ["Paschen", "15-3", 0.85477*1e4],
    ["Paschen", "16-3", 0.85048*1e4],
    ["Paschen", "17-3", 0.84696*1e4],
    ["Paschen", "18-3", 0.84403*1e4],
    ["Balmer", "3-2", 656.3*10],
    ["Balmer", "4-2", 486.1*10],
]

