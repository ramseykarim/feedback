"""
Example of how to use regions with SpectralCube
Ramsey Karim, Oct 2, 2020

regions Python package documentation:
https://astropy-regions.readthedocs.io/en/latest/index.html

SpectralCube/regions documentation:
https://spectral-cube.readthedocs.io/en/latest/manipulating.html#extracting-a-subcube-from-a-ds9-crtf-region
"""

# Necessary packages for this script
from spectral_cube import SpectralCube
import regions

# Packages for writing spectra to file
import numpy as np

# Packages for plotting
import matplotlib.pyplot as plt
from astropy import units as u

# .reg file written by DS9
reg_filename = "/home/ramsey/Documents/Research/Feedback/rcw49_data/sofia/shell_and_ridge.reg"
# Use regions package to read the .reg file
# The "regions_list" is a list of their "regions" objects; there are 2 regions in this file
region_list = regions.read_ds9(reg_filename)
# Get the individual regions (first is "ridge", second is "shell")
ridge_reg, shell_reg = region_list

# Load the CII FITS cube using the spectral_cube package
cube_filename = "/home/ramsey/Documents/Research/Feedback/rcw49_data/sofia/rcw49-cii.fits"
cube = SpectralCube.read(cube_filename)

# Extract spectra from regions (very easy with SpectralCube!)
# The argument to subcube_from_regions MUST be a list, even if it's just one region
ridge_spectrum = cube.subcube_from_regions([ridge_reg]).mean(axis=(1, 2))
shell_spectrum = cube.subcube_from_regions([shell_reg]).mean(axis=(1, 2))


# Just writing to file; change to "True" to write to file
writing_to_file = False
if writing_to_file:
    comment_line = f"Spectral axis ({str(cube.spectral_axis.unit)}), spectrum (K)"
    spectral_axis = cube.spectral_axis.to_value()
    for spectrum, name in zip((ridge_spectrum, shell_spectrum), ('ridge', 'shell')):
        save_filename = f"/home/ramsey/Documents/Research/Feedback/rcw49_data/sofia/cii-{name}-spectrum.dat"
        spectrum = spectrum.to_value()
        array = np.column_stack((spectral_axis, spectrum))
        np.savetxt(save_filename, array, delimiter=',', header=comment_line)

# Just plotting; change to "True" to see plot
plotting = True
if plotting:
    fig = plt.figure()
    # Set up axes
    ax_img = plt.subplot2grid((1, 5), (0, 0), colspan=2, projection=cube.wcs, slices=('x', 'y', 0))
    ax_spec = plt.subplot2grid((1, 5), (0, 2), colspan=3)
    # Plot moment 0 between -25, +25
    cube._unit = u.K # the rcw49-cii.fits has a hard-to-read unit, but it's K
    mom0 = cube.spectral_slab(-25*u.km/u.s, +25*u.km/u.s).moment(order=0).to(u.K*u.km/u.s)
    im = ax_img.imshow(mom0.to_value(), origin='lower')
    fig.colorbar(im, ax=ax_img)
    ax_img.set_title("[CII] Moment 0 between [-25, 25] km/s")
    # Plot the regions
    for reg in (ridge_reg, shell_reg):
        pixel_reg = reg.to_pixel(mom0.wcs)
        pixel_reg.visual.update(reg.visual)
        reg_artist = pixel_reg.as_artist()
        ax_img.add_artist(reg_artist)
    # Plot spectra
    ax_spec.plot(cube.spectral_axis.to(u.km/u.s), ridge_spectrum, color=ridge_reg.visual['color'], label='Ridge')
    ax_spec.plot(cube.spectral_axis.to(u.km/u.s), shell_spectrum, color=shell_reg.visual['color'], label='Shell')
    ax_spec.legend()
    ax_spec.set_xlabel("v (km/s)")
    ax_spec.set_ylabel("intensity (K)")
    ax_spec.set_title("Extracted spectra from regions")
    plt.show()
