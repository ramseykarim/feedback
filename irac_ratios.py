"""
I want to quickly make some IRAC ratio maps
Created: May 22, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

from . import catalog

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
    """
    data_n, data_m, ratio, header, filename = make_ratio_between(*args, **kwargs, write=False, return_all_info=True)
    mask1 = ratio < 2.8
    mask2 = data_m > 30
    filename = filename.split('.')[0] + "-mask.fits"
    header['HISTORY'] = 'this is a mask where the 2-1 ratio is < 2.8 and IRAC 1 > 30'
    fits.writeto(filename, (mask1 & mask2).astype(float), header=header)
    print("wrote the mask: ", header['HISTORY'])


if __name__ == "__main__":
    pass
    make_irac_mask(2, 1, nfunc='arcsinh', mfunc='log10')
