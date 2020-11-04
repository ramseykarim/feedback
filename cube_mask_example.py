import numpy as np

from reproject import reproject_interp
from astropy.io import fits
from spectral_cube import SpectralCube


"""
Fill in the mask and cube filenames
"""
mask_filename = ""
cube_filename = ""

"""
Load the mask and cube
"""
cube = SpectralCube.read(cube_filename)
mask, mask_hdr = fits.getdata(mask_filename, header=True)


"""
This is the reprojection part
Reproject the mask to the cube's WCS grid
If the mask already matches the cube grid (CII) then you can skip this
"""
# This is a trick I found to quickly get the 2D WCS info instead of 3D
flat_wcs = cube.moment0().wcs
mask = reproject_interp((mask, mask_hdr), flat_wcs, shape_out=cube.shape[1:], return_footprint=False)


"""
Apply the mask to the cube
Make sure the mask is a boolean mask (not float or int)
"""
cube_masked = cube.with_mask(mask.astype(bool))

"""
Fill in the save filename (make sure it ends with .fits)
Write the masked cube
"""
save_filename = ""
cube_masked.write(save_filename)







"""
Making masks from regions
"""
import regions

"""
Load the region file saved from DS9
This code is taken exactly from the regions_example.py file I sent earlier
"""
# .reg file written by DS9 (fill this in with a valid path)
reg_filename = "/home/ramsey/Documents/Research/Feedback/rcw49_data/sofia/shell_and_ridge.reg"
# Use regions package to read the .reg file
# The "regions_list" is a list of their "regions" objects; there are 2 regions in this file
region_list = regions.read_ds9(reg_filename)
# Get the individual regions (first is "ridge", second is "shell")
ridge_reg, shell_reg = region_list

"""
Load the cube you want to mask
We aren't going to use the spectrum, but we do need the WCS information from
the header
"""
# Same exact command as above
cube = SpectralCube.read(cube_filename)
# Trick to get the 2D WCS instead of the 3D (there is probably a better way)
flat_wcs = cube.moment0().wcs


"""
Make mask
"""
ridge_mask = ridge_reg.to_pixel(flat_wcs).to_mask().to_image(cube.shape[1:]).astype(bool)
shell_mask = shell_reg.to_pixel(flat_wcs).to_mask().to_image(cube.shape[1:]).astype(bool)
# You can combine masks with boolean operators and &, or |, xor ^, not ~
# This is how I differenced the ellipse masks: larger_ellipse & ~smaller_ellipse
# For this example, I'll "or" the masks
mask = ridge_mask | shell_mask


"""
Save mask to fits
"""
fits_header = flat_wcs.to_header()
fits_header['COMMENT'] = "Mask, November 3, 2020"
# Save the mask as a float array, FITS doesn't like boolean arrays
hdu = fits.PrimaryHDU(data=mask.astype(float), header=fits_header)
# Fill in a filename
save_filename = ""
hdu.writeto(save_filename)
