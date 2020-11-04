"""
I want to quickly make some IRAC ratio maps
Created: May 22, 2020
Edited: October 22, 2020
Name changed to irac_stuff.py
I'm going to mosaic the M16 IRAC images here
"""
__author__ = "Ramsey Karim"

import numpy as np
import os
import glob
import datetime

from astropy.io import fits
from astropy.wcs import WCS

from . import catalog
from . import mosaic_vlt

norm_functions = {
    'sqrt': np.sqrt, 'log10': np.log10, 'arcsinh': np.arcsinh,
}


def make_ratio_between(n, m, nfunc=None, mfunc=None, write=True, return_all_info=False):
    """
    Make ratio of band n to band m
    n and m must both be 1 to 4, and it's best if they are not equal
    nfunc and mfunc, if not None, are applied to respective data before taking
        the ratio. Should be string key in norm_functions
    """
    data_n, head_n = catalog.utils.load_irac(n=n, header=True)
    if return_all_info:
        data_n_original = data_n.copy()
    if nfunc is not None:
        data_n = norm_functions[nfunc](data_n)
        nfunc = nfunc + " "
    else:
        nfunc = ""

    data_m, head_m = catalog.utils.load_irac(n=m, header=True)
    if return_all_info:
        data_m_original = data_m.copy()
    if mfunc is not None:
        data_m = norm_functions[mfunc](data_m)
        mfunc = mfunc + " "
    else:
        mfunc = ""

    head_ratio = fits.Header({"CREATOR": "rkarim, May 22, 2020", "HISTORY": f"IRAC color: band {nfunc}{n} to {mfunc}{m}", "BUNIT": f"IRAC band ratio"})
    for k in ['TELESCOP', 'INSTRUME']:
        head_ratio[k] = head_n[k]
    head_ratio.update(WCS(head_n).to_header())
    filename = f"{catalog.utils.irac_path}irac{nfunc.strip()}{n}-to-{mfunc.strip()}{m}ratio.fits"
    ratio = data_n/data_m
    if write:
        fits.writeto(filename, ratio, header=head_ratio)
        print(f"wrote IRAC {n} to {m} ratio")
    if return_all_info:
        return data_n_original, data_m_original, ratio, head_ratio, filename
    else:
        return ratio


def make_irac_mask(*args, **kwargs):
    """
    Edit this function to make masks based on ratios
    Passes all args and kwargs to make_ratio_between
    Example:
    make_irac_mask(2, 1, nfunc='arcsinh', mfunc='log10')
    """
    data_n, data_m, ratio, header, filename = make_ratio_between(*args, **kwargs, write=False, return_all_info=True)
    mask1 = ratio < 2.8
    mask2 = data_m > 30
    filename = filename.split('.')[0] + "-mask.fits"
    header['HISTORY'] = 'this is a mask where the 2-1 ratio is < 2.8 and IRAC 1 > 30'
    fits.writeto(filename, (mask1 & mask2).astype(float), header=header)
    print("wrote the mask: ", header['HISTORY'])


def mosaic_irac(n):
    """
    Mosaic the IRAC maps. Put them all onto the same grid, SPITZER_I4
    """
    ref_filename = catalog.utils.search_for_file("spitzer/SPITZER_I3_6049792_0000_5_E8698528_maic.fits")
    spitzer_dir = os.path.dirname(ref_filename)
    ref_filename = glob.glob(os.path.join(spitzer_dir, f"SPITZER_I{n}*.fits")).pop()
    # Get the larger IRAC filenames for this band
    tile_filenames = glob.glob(os.path.join(spitzer_dir, f"*I{n}.fits"))
    # Get the header from the reference image
    with fits.open(ref_filename) as hdul:
        assert len(hdul) == 1
        ref_hdr = hdul[0].header
    result, result_fp = mosaic_vlt.reproject_and_coadd(tile_filenames,
        ref_hdr, reproject_function=mosaic_vlt.reproject_interp)
    new_hdr = WCS(ref_hdr).to_header()
    new_hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    new_hdr['ORIGIN'] = "UMD Astronomy"
    new_hdr['CREATOR'] = f"Ramsey Karim, {__file__}"
    for kw in ['INSTRUME', 'CHNLNUM', 'AOT_TYPE', 'AORLABEL', 'OBJECT', 'BUNIT']:
        new_hdr[kw] = ref_hdr[kw]
    new_hdr['LATPOLE'] = 90.
    new_hdr['LONPOLE'] = 180.
    hdu = fits.PrimaryHDU(data=result, header=new_hdr)
    hdu.writeto(os.path.join(spitzer_dir, f"SPITZER_I{n}_mosaic.fits"))

if __name__ == "__main__":
    mosaic_irac(4)
