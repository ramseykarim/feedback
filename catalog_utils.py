"""
Helper functions for the catalog creation / reduction routines
A lot of stuff that was piled into g0_stars.py should go into here
Created: May 5, 2020 (deeper into quarantine)
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord


wd2_center_coord = SkyCoord("10 23 58.1 -57 45 49", unit=(u.hourangle, u.deg))


def irac_data():
    """
    Loads whatever IRAC band I have on my laptop
    """
    p = "/home/ramsey/Documents/Research/Feedback/ancillary_data/spitzer/irac/"
    fn = "30002561.30002561-28687.IRAC.1.median_mosaic.fits"
    img, hdr = fits.getdata(p+fn, header=True)
    return img, WCS(hdr)


def plot_coordinates(data_gen_f, coords, setup=True, show=True, subplot=(111,)):
    """
    Quickly throw up some coordinates onto an image
    :param data_gen_f: some function that returns an image and a WCS object
        Can be None if setup is False; in that case, plt.gca() should be a
        WCS projection
    :param coords: SkyCoord array
    :param setup: open the data? make a suplot?
    :param show: plt.show()?
    :param subplot: subplot number to pass to plt.subplot if setup==True.
        will be unpacked if it's a tuple
    """
    if setup:
        img, w = data_gen_f()
        if isinstance(subplot, tuple):
            plt.subplot(*subplot, projection=w)
        else:
            plt.subplot(subplot, projection=w)
        plt.imshow(np.arcsinh(img), origin='lower', vmin=np.arcsinh(1), vmax=np.arcsinh(80))
    plt.scatter(coords.ra.deg, coords.dec.deg, transform=get_transform(),
        color='red', s=2)
    if show:
        plt.show()


def get_transform():
    """
    Shortcut for plotting
    """
    return plt.gca().get_transform('world')


def save_df_html(df, na_rep='--'):
    """
    Quickly save the argument dataframe to a test.html file, and print its length
    """
    print(len(df))
    df.to_html("~/Downloads/test.html", na_rep=na_rep)
