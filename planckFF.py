import numpy as np
import matplotlib.pyplot as plt
import glob

from astropy.io import fits
from astropy.coordinates import Galactic
from astropy.wcs import WCS
import astropy_healpix as hpx

hdul = fits.open(glob.glob("../ancillary_data/planck/*").pop())
nside = hdul[1].header['NSIDE']
order = hdul[1].header['ORDERING']
assert hdul[1].header['COORDSYS'] == 'Galactic'.upper()
hpmap = hpx.HEALPix(nside=nside, order=order, frame=Galactic())

column_names = hdul[1].data.names
# ML is maximum likelihood (posterior), MEAN is mean of posterior, RMS is RMS of posterior
# EMs are in pc cm-6
# Temperatures are electron temperatures in K
EMff_table = hdul[1].data['EM_MEAN']
Tff_table = hdul[1].data['TEMP_MEAN']

ref_fn = "../ancillary_data/herschel/RCW49large_3p.fits"
ref_fn = "../ancillary_data/herschel/helpssproc/processed/1342255009/SPIRE500um-image-remapped-conv.fits"
dust_SED_WCS = WCS(fits.getdata(ref_fn, header=True)[1])
i_grid, j_grid = np.meshgrid(*(np.arange(x) for x in dust_SED_WCS.array_shape), indexing='ij')
coords = dust_SED_WCS.array_index_to_world(i_grid.ravel(), j_grid.ravel())

EMff_map = hpmap.interpolate_bilinear_skycoord(coords, EMff_table).reshape(dust_SED_WCS.array_shape)
Tff_map = hpmap.interpolate_bilinear_skycoord(coords, Tff_table).reshape(dust_SED_WCS.array_shape)

plt.subplot(121)
plt.imshow(np.sqrt(EMff_map), origin='lower')
plt.title("EM")
plt.colorbar()
plt.subplot(122)
plt.imshow(Tff_map, origin='lower')
plt.title("TEMP")
plt.colorbar()
plt.show()
