import numpy as np
import matplotlib.pyplot as plt
import glob
import os, sys

from astropy.io import fits
from astropy.wcs import WCS

import readcat


def herschel_data(wl):
    i = [70, 160, 250, 350, 500].index(wl)
    band_stub = (["PACS"]*2 + ["SPIRE"]*3)[i] + str(wl) + "um"
    data_directory = "/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009/"
    if not os.path.isdir(data_directory):
        # Laptop path
        data_directory = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/processed/1342255009/"
    fn = glob.glob(f"{data_directory}{band_stub}-image-remapped-conv*.fits").pop()
    data, header = fits.getdata(fn, header=True)
    wavelen = header['WAVELNTH']
    return data, header

d, h = herschel_data(70)
d -= np.nanmin(d) - 1

ras, decs = [], []
names = ["VA", 'VPHAS+', 'TFT', 'A07', 'R07']
for dff in (readcat.openVA_complete, readcat.openVPHAS_complete, readcat.openTFT_complete, readcat.openAscenso_complete, readcat.openRauw):
    df = dff()
    ra = df.apply(lambda row: row['SkyCoord'].ra.deg, axis=1).values
    ras.append(ra)
    dec = df.apply(lambda row: row['SkyCoord'].dec.deg, axis=1).values
    decs.append(dec)
del dff

plt.subplot(111, projection=WCS(h))
plt.imshow(np.arcsinh(d), origin='lower', cmap='Greys_r')
colors = ['red', 'orange', 'brown', 'blue', 'green']
markers = ['|', '_', 'x', '4', '3']
for i in range(5):
    plt.scatter(ras[i], decs[i], marker=markers[i], color=colors[i], transform=plt.gca().get_transform('world'), label=names[i], alpha=0.3)
plt.legend()
plt.show()
