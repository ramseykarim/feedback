import numpy as np
import matplotlib.pyplot as plt

from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from astropy.io import fits

import datetime

hst_path = "/home/rkarim/Research/Feedback/ancillary_data/hst/"

def process_lims(*ijlims):
    # should be ((ilo, ihi), (jlo, jhi))
    center = []
    width = []
    for lo, hi in ijlims:
        center.append(int(round((lo+hi)/2.)))
        width.append(hi-lo)
    # returns center as j, i and width as i, j
    return center[::-1], width

def plot_original(original, vmin, vmax, name):
    plt.subplot(121)
    plt.imshow(original, origin='lower', vmin=vmin, vmax=vmax)
    plt.title(name)


def plot_new(cutout, vmin, vmax):
    plt.subplot(122, projection=cutout.wcs)
    plt.imshow(cutout.data, origin='lower', vmin=vmin, vmax=vmax)

def make_cutout(original, ilims, jlims, hdr):
    return Cutout2D(original, *process_lims(ilims, jlims), wcs=WCS(hdr))

def write_cutout(cutout, original_hdr, name):
    original_hdr.update(cutout.wcs.to_header())
    original_hdr["HISTORY"] = f"trimmed -rkarim, {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    fits.writeto(hst_path+name+".fits", cutout.data, header=original_hdr)
    print(name+" written")


def do_trim(name, filename, vlims, jlims, ilims):
    img, hdr = fits.getdata(hst_path+filename, header=True)
    plot_original(img, *vlims, name)
    cutout = make_cutout(img, ilims, jlims, hdr)
    plot_new(cutout, *vlims)
    plt.show()
    if input("is this ok? (y/n)> ").lower() == "y":
        write_cutout(cutout, hdr, name)

F555W = True
F658N = False
F814W = True

if F658N:
    name = "F658N"
    fn = "hst_mos_1041812_acs_wfc_f658n_drz.fits"
    vlims = (0, 0.55)
    jlims = (5956, 11658) # X
    ilims = (5280, 11525) # Y
    do_trim(name, fn, vlims, jlims, ilims)
if F555W:
    name = "F555W"
    fn = "hst_mos_1041812_acs_wfc_f555w_long_drz.fits"
    vlims = (0, 0.55)
    jlims = (865, 14850) # X
    ilims = (870, 14324) # Y
    do_trim(name, fn, vlims, jlims, ilims)
if F814W:
    name = "F814W"
    fn = "hst_mos_1041812_acs_wfc_f814w_long_drz.fits"
    vlims = (0, 0.55)
    jlims = (865, 14850) # X
    ilims = (870, 14324) # Y
    do_trim(name, fn, vlims, jlims, ilims)

