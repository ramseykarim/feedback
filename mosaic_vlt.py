import numpy as np
import matplotlib.pyplot as plt
import glob
import os, sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5, ICRS

from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd


"""
Script to parse through three different CCD chip-separated VLT images and stitch
them together in a contiuous image centered on Wd2/RCW 49.
The image is an Halpha image that we will find quite useful.

Created: 4/16/2020
"""
__author__ = "Ramsey Karim"


# Location of VLT Halpha data
data_directory = "../ancillary_data/vlt/omegacam/"
# All filenames
vlt_fns = glob.glob(f"{data_directory}/ADP*")


def make_wcs(ref_coord=None, ref_pixel=None, grid_shape=None, pixel_scale=None,
    **extra_header_kws):
    """
    Make a fresh, simple WCS object based on a few parameters. Does not need
        any existing data or WCS info.
    This will only work for 2-dimensional image data. No cubes or spectral axes.
    <HISTORY>
        This code is from g0_stars.py. I will probably need to edit this.
        Update: I edited it to accept a pixel scale matrix, but I probably won't
            even use that functionality. I am not sure why VLT and HST use matrices
            with off-diagonal elements, but IRAC uses diagonals only and I trust
            IR astronomers more than optical astronomers.
    </HISTORY>
    :param ref_coord: SkyCoord object that will match ref_pixel. This is a
        required argument.
    :param ref_pixel: 2-element integer sequence indicating reference pixel,
        which will match up with the ref_coord.
        The values should be Numpy array indices (0-indexed, ij)
        If not specified, will default to the approximate center of the grid.
    :param grid_shape: 2-element integer sequence indicating the shape of
        the data array.
        If grid shape is (10, 10) i.e. (0..9, 0..9) and you want pixel (4, 4)
        i.e. the fifth i,j pixels to be the center, specify (4, 4).
        This function will pass (4+1, 4+1) to WCS to ensure that the fifth
        pixels are chosen in this case.
        This is a required argument.
    :param pixel_scale: some indication of the pixel scale of the WCS object.
        This can be a 2x2 matrix or a scalar, in which case it can be an
        astropy Quantity. If it's a scalar but not a Quantity, it's assumed
        to be in units of arcminutes.
        This is a required argument.
    :param extra_header_kws: any additional FITS Header keywords and their
        values that you would like to add. If they're not used by WCS, they will
        be lost.
    :returns: simple astropy WCS object described by the arguments.
    """
    # Check arguments
    if any(x is None for x in (ref_coord, grid_shape, pixel_scale)):
        raise RuntimeError("You are missing required arguments.")
    if ref_pixel is None:
        ref_pixel = tuple(int(x/2) for x in grid_shape)
    # Figure out pixel scale and ultimately get a matrix
    if hasattr(pixel_scale, 'shape') and pixel_scale.shape == (2, 2):
        pixel_scale_matrix = pixel_scale
    else:
        if not isinstance(pixel_scale, u.quantity.Quantity):
            pixel_scale *= u.arcmin
        pixel_scale_matrix = np.zeros((2, 2))
        pixel_scale_matrix[0, 0] = pixel_scale_matrix[1, 1] = pixel_scale.to(u.deg).to_value()
        pixel_scale_matrix[0, 0] *= -1. # RA increasing to the left side
    # Lay out the keywords in a dictionary
    kws = {
        'NAXIS': (2, "Number of axes"),
        'NAXIS1': (grid_shape[1], "X/j axis length"),
        'NAXIS2': (grid_shape[0], "Y/i axis length"),
        'RADESYS': ('ICRS', "Interational Celestial Reference System"),
        'CRVAL1': (ref_coord.ra.deg, "[deg] RA of reference point"),
        'CRVAL2': (ref_coord.dec.deg, "[deg] DEC of reference point"),
        'CRPIX1': (ref_pixel[1] + 1, "[pix] Image reference point"),
        'CRPIX2': (ref_pixel[0] + 1, "[pix] Image reference point"),
        'CTYPE1': ('RA---TAN', "RA projection type"),
        'CTYPE2': ('DEC--TAN', "DEC projection type"),
        'PA': (0., "[deg] Position angle of axis 2 (E of N)"),
        'CD1_1': (pixel_scale_matrix[0, 0], "Transformation matrix"),
        'CD1_2': (pixel_scale_matrix[0, 1], ""),
        'CD2_1': (pixel_scale_matrix[1, 0], ""),
        'CD2_2': (pixel_scale_matrix[1, 1], ""),
        'EQUINOX': (2000., "[yr] Equatorial coordinates definition"),
    }
    kws.update(extra_header_kws)
    header = fits.Header()
    # Two lines to avoid some weird bug about reading dictionaries in the constructor
    header.update(kws)
    # Return the WCS object
    return WCS(header)


def angular_size_from_wcs(wcs_obj):
    """
    Get approximate angular size (width, height) of an image from its WCS
    information.
    :param wcs_obj: WCS object
    :returns: astropy Angles
    """
    # Get the footprint
    fp = SkyCoord(wcs_obj.calc_footprint(), unit=u.deg) # Clockwise from bottom-left corner
    # Get width (RA)
    width = fp[1].separation(fp[2])
    # Get height (Dec)
    height = fp[0].separation(fp[1])
    return width, height


def image_overlap(wcs_obj1, wcs_obj2):
    """
    Check if there is any image overlap using the image footprints
    Checking if either image contains any of the other's footprint coordinates
        should catch all edge cases
    :param wcs_obj1: one WCS object (arbitrary order)
    :param wcs_obj2: another WCS object
    :returns: True if there is overlap else False
    """
    wcs_objs = (wcs_obj1, wcs_obj2)
    is_overlap = False
    for i in range(2):
        # Do this twice, switching the WCS objects' positions in the calculation
        is_overlap |= np.any(wcs_objs[i].footprint_contains(SkyCoord(wcs_objs[1-i].calc_footprint(), unit=u.deg)))
    return is_overlap


def make_wcs_like(wcs_for_footprint, wcs_for_pixels, degrade_pixelscale_factor=1,
    shrink_height_factor=1, shrink_width_factor=1):
    """
    Create a WCS object with a simliar footprint to one existing WCS object and
        a similar pixel scale to another WCS object.
    :param wcs_for_footprint: the WCS object whose footprint we want to
        approximately copy. The new footprint will be as close to the old one
        as the resolution will allow. There are no guarantees on how close
        the footprints will be.
    :param wcs_for_pixels: the WCS object whose pixel scale will be used.
        This function will average the diagonals of the WCS's pixel_scale_matrix
        and use this as both diagonal elements of the new matrix.
        There will be no off-diagonal components in the new pixel_scale_matrix.
    :param degrade_pixelscale_factor: a factor by which to degrade the pixel
        scale. The new pixel scale will be the wcs_for_pixels scale divided by
        this factor. Use a large number here to save memory and make a
        very coarse image. Defaults to 1 for no rescaling.
    :param shrink_height_factor: factor by which to shrink the height of the new
        image. Larger factor leads to a shorter Dec span.
    :param shrink_width_factor: factor by which to shrink the width of the new
        image. Larger factor leads to shorter RA span.
    :returns: astropy WCS object
    """
    # Figure out the general location of the image.
    # Find the center coordinate of the footprint WCS
    fp_cr_coord = wcs_for_footprint.array_index_to_world(*(wcs_for_footprint.array_shape[x]//2 for x in range(2)))
    # Use the reference footprint to find the approximate height (RA) and width (Dec)
    # of the footprint in angular units
    fp_width, fp_height = angular_size_from_wcs(wcs_for_footprint) # RA, Dec order
    fp_width_height = (fp_width / float(shrink_width_factor), fp_height / float(shrink_height_factor))
    # Get the pixel scale from the pixel scale WCS (THIS ASSUMES PIXEL SCALE IS GIVEN IN DEGREES)
    # Modify with the degrade_pixelscale_factor
    pixel_scale = np.mean(np.abs(np.diag(wcs_for_pixels.pixel_scale_matrix))) * u.deg * float(degrade_pixelscale_factor)
    # Figure out the pixel height and width of the new WCS object
    grid_shape = tuple(int(round((x / pixel_scale).to_value())) for x in fp_width_height[::-1]) # ij order
    # Use make_wcs to generate an astropy WCS object based on these derived parameters
    return make_wcs(ref_coord=fp_cr_coord, grid_shape=grid_shape, pixel_scale=pixel_scale)


def iterate_over_chips(filename_list):
    """
    Generator function for cycling through all the CCD chips from multiple
        files (designed for VLT OmegaCAM data).
    There will be no indication of different files, so if you want that
        information, you will have to do this more manually.
    :param filename_list: list of valid paths to FITS files.
        Each FITS file should contain a PrimaryHDU with no data and
        at least one valid image tile.
    :returns: iterator that will return (data, WCS) on calls to next() method
    """
    # Iterate over the different files
    for i in range(len(filename_list)):
        # Check the number of chips in this file
        with fits.open(vlt_fns[i]) as hdul:
            n_chips = len(hdul) - 1
        # Iterate over the chips in this file.
        # Use fits.getdata so we don't keep HDULists open
        for j in range(1, n_chips + 1):
            data, header = fits.getdata(vlt_fns[i], j, header=True)
            yield data, WCS(header)


def list_of_chips(filename_list, wcs_reference):
    """
    Ruins the whole point of saving memory using a generator expression
    At least checks to see if we need each chip before returning it
    :param filename_list: argument to iterate_over_chips
    :param wcs_reference: WCS object that must have some overlap with the
        CCD chip in order for the CCD chip to be returned
    :returns: list of (data, WCS) tuples
    """
    list_to_return = []
    for data, w in iterate_over_chips(filename_list):
        if image_overlap(w, wcs_reference):
            list_to_return.append((data, w))
    return list_to_return


# I want the reference WCS to have roughly the same footprint as the IRAC data
irac_fn = glob.glob("../ancillary_data/spitzer/irac/300*").pop()
# IRAC pixel_scale_matrix is diagonal (0s on off-diag), so we can copy this style
irac_w = WCS(fits.getdata(irac_fn, header=True)[1])
# The pixel scale in the VLT image has RA increasing towards the RIGHT, which ultimately
# doesn't matter, but I'll have it going towards the left like IRAC and others.
vlt_w = WCS(fits.getdata(vlt_fns[0], 1, header=True)[1])

new_w = make_wcs_like(irac_w, vlt_w, degrade_pixelscale_factor=1)
# This comes out to nearly the same size as one of these VLT images, but I checked and this makes sense. IRAC is big.
# I checked the new footprint against the IRAC footprint, it's good to within 0.5 arcseconds!


def we_have_mosaic_at_home():
    """
    Wrote this before I found reproject.mosaicking
    """
    component_images = []
    footprint = np.zeros(new_w.array_shape)
    for j in range(len(vlt_fns)):
        component_images.append(np.full(new_w.array_shape, np.nan))
        sys.stdout.write(f"Opening {vlt_fns[j].split('/')[-1]} and cycling through chips.")
        with fits.open(vlt_fns[j]) as hdul:
            for i in range(len(hdul)):
                if not i:
                    sys.stdout.flush()
                    continue
                data = hdul[i].data
                header = hdul[i].header
                w = WCS(header)
                there_is_overlap = image_overlap(new_w, w)
                sys.stdout.write(f"{i}: {header['EXTNAME']}, overlap: {there_is_overlap}; ")
                if there_is_overlap:
                    sys.stdout.write("reprojecting...")
                    sys.stdout.flush()
                    new_img, rp_fp = reproject_interp((data, w), new_w, shape_out=new_w.array_shape)
                    footprint += rp_fp
                    rp_fp = rp_fp.astype(bool)
                    component_images[j][rp_fp] = new_img[rp_fp]
                    sys.stdout.write(f"filled in {np.sum(rp_fp)} / {rp_fp.size} pixels\n")
                else:
                    sys.stdout.write("\n")
                sys.stdout.flush()
        # Get rid of most of that low-value strip in one of the CCD chips
        component_images[j][component_images[j] < np.nanmedian(component_images[j])/5.] = np.nan
    sys.stdout.write("Compiling image...")
    sys.stdout.flush()
    final_new_image = np.nanmedian(component_images, axis=0)
    sys.stdout.write("done.\n")
    sys.stdout.flush()
    return final_new_image, footprint


final_new_image, mosaic_footprint = reproject_and_coadd(
    list_of_chips(vlt_fns, new_w), new_w, shape_out=new_w.array_shape,
    reproject_function=reproject_interp, combine_function='mean',
    match_background=True,
)
print("THIS FINISHED RUNNING")

# final_new_image, mosaic_footprint = we_have_mosaic_at_home()

new_header = fits.Header()
new_header.update(new_w.to_header())
new_header['COMMENT'] = "Mosiac from VLT Halpha images"

fits.writeto("../ancillary_data/vlt/omegacam/Halpa_mosaic.fits",
    final_new_image, new_header, overwrite=True)
print("WROTE FILE")

# plt.subplot(121, projection=new_w)
# plt.imshow(final_new_image, origin='lower', vmin=0.5, vmax=500)
# plt.subplot(122, projection=new_w)
# plt.imshow(mosaic_footprint, origin='lower')
# plt.show()
