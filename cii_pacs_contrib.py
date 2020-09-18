"""
I simply want to see how much [CII] contributes to the PACS 160 integrated
intensity.

Created: August 12, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 4}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy
import sys

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from spectral_cube import SpectralCube
import regions
from reproject import reproject_interp

from .mantipython.physics import instrument as mpy_instr
from . import misc_utils
from . import catalog
from . import cube_utils

cii_fn = f"{catalog.utils.ancillary_data_path}sofia/rcw49-cii.fits"
cube = cube_utils.CubeData(cii_fn)
line_center = cube.header['RESTFREQ'] * u.Hz
# invert frequency array so that integral is positive (np.trapz)
freq_axis = cube.data.spectral_axis[::-1].to(u.Hz, equivalencies=u.doppler_radio(line_center))

plt.figure(figsize=(13, 5))
ax1 = plt.subplot(131, projection=cube.wcs_flat)
img = cube.data.moment(order=0).to(u.K * u.km / u.s)
lo = misc_utils.flquantiles(img[np.isfinite(img)].to_value().ravel(), 30)[0]
up = misc_utils.flquantiles(img[np.isfinite(img)].to_value().ravel(), 500)[1]
plt.imshow(img.to_value(), origin='lower', vmin=lo, vmax=up, cmap='Greys')
plt.colorbar(label=f'{str(img.unit)}')

pacs160_data_path = f"{catalog.utils.ancillary_data_path}herschel/processed/1342255009_reproc160/PACS160um-image-remapped-conv-plus000370.fits"
pacs160_regrid, fp = reproject_interp(pacs160_data_path, cube.wcs_flat, shape_out=cube.wcs_flat.array_shape[1:])
ax3 = plt.subplot(133, projection=cube.wcs_flat)
lo = misc_utils.flquantiles(pacs160_regrid[np.isfinite(pacs160_regrid)].ravel(), 30)[0]
plt.imshow(pacs160_regrid, origin='lower', cmap='Greys', norm=LogNorm(vmin=lo, vmax=np.nanmax(pacs160_regrid)))
plt.colorbar(label='PACS160 MJy/sr')
plt.title("PACS 160um observed")


prop_cycle = plt.rcParams['axes.prop_cycle']
color_cycle = prop_cycle.by_key()['color']
colors = iter(color_cycle)

pacs160 = mpy_instr.get_instrument([160]).pop()
# s = 0 means interpolate thru all points
response_function_spline = mpy_instr.UnivariateSpline(pacs160.freq_array, pacs160.response_array, s=0)
rf = response_function_spline(freq_axis.to_value()) # response function

ax2 = plt.subplot(132)
fax_ghz = freq_axis.to(u.GHz).to_value()
vax_kms = cube.data.spectral_axis[::-1].to(u.km/u.s).to_value()
plt.plot(vax_kms, rf * (1e6 / np.max(rf)), label='PACS relative response', color='k', alpha=0.5, linestyle='--', linewidth=1.2)
# plt.axvline(line_center.to(u.GHz).to_value(), color='k', linestyle='--', label='[CII] line center', linewidth=1)
plt.xlabel(f"Velocity ({str(u.km/u.s)})")
plt.ylabel(f"[CII] brightness ({u.MJy/u.sr})")

reg_path = f"{catalog.utils.ancillary_data_path}catalogs/Ramsey/cii_spectrum_regions.reg"
cii_regs = {x.meta['label']: x for x in regions.read_ds9(reg_path)}
ax1_artists, ax1_labels = [], []
ax3_artists, ax3_labels = [], []

# selected_reg =
for selected_reg in cii_regs:
    subcube = cube.data.subcube_from_regions([cii_regs[selected_reg]])
    # invert to match frequency array so integral is positive
    spectrum = subcube.mean(axis=(1, 2))[::-1].to(u.MJy/u.sr, equivalencies=u.brightness_temperature(line_center))

    pacs_detection = (np.trapz(spectrum.to_value()*rf, x=freq_axis.to_value()) / pacs160.filter_width) * u.MJy/u.sr

    print(pacs_detection)

    reg_pixel = cii_regs[selected_reg].to_pixel(cube.wcs_flat)
    reg_pixel.visual['color'] = next(colors)
    reg_pixel.visual['width'] = 1

    reg_mask = reg_pixel.to_mask().to_image(pacs160_regrid.shape)
    pacs160_cutout = pacs160_regrid[reg_mask.astype(bool)]
    pacs160_brightness = np.nanmean(pacs160_cutout) * u.MJy/u.sr

    reg_artist_ax1_label = f"{selected_reg} [CII] in PACS160um: {pacs_detection:7.2f}"
    reg_artist_ax3_label = f"{selected_reg} observed PACS160um: {pacs160_brightness:.2E}"
    ax1_artist = reg_pixel.as_artist()
    ax3_artist = reg_pixel.as_artist()
    ax1.add_artist(ax1_artist)
    ax3.add_artist(ax3_artist)
    ax1_artists.append(ax1_artist)
    ax3_artists.append(ax3_artist)
    ax1_labels.append(reg_artist_ax1_label)
    ax3_labels.append(reg_artist_ax3_label)

    ax2.plot(vax_kms, spectrum.to_value(), label=f'\'{selected_reg}\' [CII] spectrum; {(100*pacs_detection/pacs160_brightness):.2f} % of PACS160', color=reg_pixel.visual['color'], linewidth=reg_pixel.visual['width'])


plt.sca(ax1)
plt.legend(ax1_artists, ax1_labels)
plt.title("[CII] integrated intensity and PACS detections")
plt.sca(ax2)
plt.legend()
plt.title("[CII] spectra")
plt.sca(ax3)
plt.legend(ax3_artists, ax3_labels)
plt.tight_layout()
plt.show()
