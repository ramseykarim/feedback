"""
I want to quickly make some IRAC ratio maps
Created: May 22, 2020
"""
__author__ = "Ramsey Karim"

from astropy.io import fits
from astropy.wcs import WCS

import catalog_utils

def make_ratio_between(n, m):
    """
    Make ratio of band n to band m
    n and m must both be 1 to 4, and it's best if they are not equal
    """
    data_n, head_n = catalog_utils.load_irac(n=n, header=True)
    data_m, head_m = catalog_utils.load_irac(n=m, header=True)
    head_ratio = fits.Header({"CREATOR": "rkarim, May 22, 2020", "HISTORY": f"IRAC color: band {n} to {m}", "BUNIT": f"IRAC band {n} to {m} ratio"})
    for k in ['TELESCOP', 'INSTRUME']:
        head_ratio[k] = head_n[k]
    head_ratio.update(WCS(head_n).to_header())
    fits.writeto(f"{catalog_utils.irac_path}irac{n}-to-{m}ratio.fits", data_n/data_m, header=head_ratio)
    print(f"wrote IRAC {n} to {m} ratio")

if __name__ == "__main__":
    pass
    # make_ratio_between(1, 3)
