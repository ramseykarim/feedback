import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits


fn = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/RCW49large_2p_2BAND_beta2.0.fits"

t160, header = fits.getdata(fn, 2, header=True)
cii_cd = (10.**t160) * 8e20
# plt.imshow(np.log10(cii_cd), origin='lower')
# plt.show()

save_fn = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/RCW49large_CII_column_density.fits"
header['EXTNAME'] = "N(CII)"
header['BUNIT'] = "cm-2"
header['HISTORY'] = "CII column density derived from dust optical depth at 160um"
header['HISTORY'] = "From dust SED fit to Herschel 70 and 160 um with beta=1.7"

fits.writeto(save_fn, cii_cd, header=header)
