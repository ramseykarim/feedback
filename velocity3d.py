import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import gaussian
import sys
import os
import pandas as pd

from mayavi import mlab


def load_fits(filename):
    # Load FITS cube
    data, hdr = fits.getdata(filename, header=True)
    # Get WCS
    w = WCS(hdr)
    return data, w


def momentn(cube, vaxis, n=0):
    # axis 0 must be velocity
    vaxis = vaxis[np.newaxis, np.newaxis, :]
    if n == 0:
        norm = len(vaxis)
        integrand = cube
    elif n == 1:
        norm = np.sum(vaxis, axis=2)
        integrand = cube*vaxis
    else:
        vaxis = vaxis**n
        norm = np.sum(vaxis**n)
        integrand = cube*vaxis
    m0 = np.nansum(integrand, axis=2) / norm
    plt.imshow(m0.T, origin='lower')
    plt.show()


def get_axis(cube, wcs_obj, axisn):
    pixel_coords = np.zeros((cube.shape[axisn], 3))
    pixel_coords[:, axisn] = np.arange(cube.shape[axisn])
    pixel_coords = pixel_coords.T
    return wcs_obj.array_index_to_world_values(*pixel_coords)[len(cube.shape) - axisn - 1]


def get_cube(filename, x_lim=None, y_lim=None, z_lim=None, vsmoothing=True,
    xysmoothing=True, ref_coord=None, velocity_conv_pix=3, spatial_conv_pix=1,
    vscale=4, contour_levels=None, cmap=None, contour_opacity=0.2):

    data, w = load_fits(filename)

    # FITS cubes tend to load out as (velocity, dec, ra)
    # Get the full-range axis arrays
    velocity_axis = get_axis(data, w, 0)/1000
    dec_axis = get_axis(data, w, 1)
    ra_axis = get_axis(data, w, 2)
    # Swap velocity and ra for (ra, dec, velocity)
    data = np.swapaxes(data, 0, 2)
    # Make RA increase the other way so we can view it correctly
    data = np.flip(data, 0)
    ra_axis = np.flip(ra_axis, 0)

    # Take a subset of the full cube, since mayavi doesn't do well with a huge
    # 3D array
    # First, setup default subset (very small)
    default_halfwidth = 50
    if x_lim is None:
        # X is RA, array index 0
        mid_x = data.shape[0]//2
        x_lim = (mid_x - default_halfwidth, mid_x + default_halfwidth)
    if y_lim is None:
        # Y is Dec, array index 1
        mid_y = data.shape[1]//2
        y_lim = (mid_y - default_halfwidth, mid_y + default_halfwidth)
    if z_lim is None:
        # Z is velocity, array index 2
        mid_z = data.shape[2]//2
        z_lim = (mid_z - default_halfwidth, mid_z + default_halfwidth)
    # Set up subset slices
    raslice = slice(*x_lim)
    decslice = slice(*y_lim)
    vslice = slice(*z_lim)

    # *******REMOVE
    # vpad = 14 # a bit of padding for velocity smoothing
    # vslice_initial = slice(max(0, z_lim[0]-vpad), min(data.shape[2], z_lim[1]+vpad)-1)
    # vslice_final = slice(max(0, z_lim[0]-vpad)+vpad, min(data.shape[2], z_lim[1]+vpad)-1-vpad)

    # Subset the axes
    ra_axis = ra_axis[raslice]
    dec_axis = dec_axis[decslice]
    velocity_axis = velocity_axis[vslice]
    velocity_axis = velocity_axis
    # Subset the data
    data = data[raslice, decslice, vslice]

    # Smooth the velocity axis to avoid a very rough contour in mayavi
    if vsmoothing:
        smooth_kernel = gaussian(5, velocity_conv_pix)
        smooth_kernel /= np.sum(smooth_kernel)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j, :] = np.convolve(data[i, j, :], smooth_kernel, mode='same')
    # Smooth spatial axes to avoid rough countours
    if xysmoothing:
        smooth_kernel = gaussian(5, spatial_conv_pix)
        smooth_kernel /= np.sum(smooth_kernel)
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data[:, j, k] = np.convolve(data[:, j, k], smooth_kernel, mode='same')
        for i in range(data.shape[0]):
            for k in range(data.shape[2]):
                data[i, :, k] = np.convolve(data[i, :, k], smooth_kernel, mode='same')

    # Use a reference coordinate to express axes in small arcminute values
    if ref_coord is None:
        reference_set = False
        ra_ref, dec_ref = ra_axis[len(ra_axis)//2], dec_axis[len(dec_axis)//2]
    else:
        ra_ref, dec_ref = ref_coord
        reference_set = True

    # Function to convert coordinates for axes and stars
    def convert_coords(ra_array, dec_array):
        ra_array = ((ra_array - ra_ref) * 60)*np.cos(np.deg2rad(dec_ref))
        dec_array = (dec_array - dec_ref) * 60
        return ra_array, dec_array

    ra_axis, dec_axis = convert_coords(ra_axis, dec_axis)  # RA positive to the W now

    # Set up range and grids for mayavi
    lims = [ra_axis[0], ra_axis[-1], dec_axis[0], dec_axis[-1], velocity_axis[0], velocity_axis[-1]]
    ragrid, dgrid, vgrid = np.meshgrid(ra_axis, dec_axis, velocity_axis, indexing='ij')

    # Function to filter star coordinates using the axis limits
    def filter_stars(ra_array, dec_array):
        stars_within_lims = (ra_array > lims[0]) & (ra_array < lims[1])
        stars_within_lims &= (dec_array > lims[2]) & (dec_array < lims[3])
        return ra_array[stars_within_lims], dec_array[stars_within_lims]


    def full_coordinate_convert(ra_array, dec_array):
        return filter_stars(*convert_coords(ra_array, dec_array))

    # Set up mayavi scalar_field source and iso_surface contours
    # Use vscale to scale back velocity axis so it lines up a bit better with RA and Dec
    src = mlab.pipeline.scalar_field(ragrid, dgrid, vgrid/vscale, data)
    if contour_levels is None:
        contour_levels = [10, 20]
    mlab.pipeline.iso_surface(src, contours=contour_levels, opacity=contour_opacity, colormap=cmap)
    # Using the volume rendering would be super cool but it's always so buggy for me
    this_never_works = True
    if not this_never_works: mlab.pipeline.volume(src, vmin=1, vmax=14)
    # Set up axis display
    mlab.axes(ranges=lims,
        zlabel="Velocity (km/s)", ylabel="Dec (min)", xlabel="RA (min)", nb_labels=4)
    # Plot a zero-velocity "plane"
    mlab.plot3d(
        [lims[0], lims[1], lims[1], lims[0], lims[0]],
        [lims[2], lims[2], lims[3], lims[3], lims[2]],
        [0, 0, 0, 0, 0],
        color=(1,1,1), tube_radius=0.05, opacity=0.4
        )
    # Reference position
    if reference_set:
        mlab.points3d([0], [0], [0], opacity=0.2, mode='sphere', color=(1,1,1), scale_factor=2)

    return full_coordinate_convert


def plot_stars(ra_array, dec_array, coord_convert_function, **points3d_kwargs):
    ra_array, dec_array = coord_convert_function(ra_array, dec_array)
    v_array = np.zeros_like(ra_array)
    mlab.points3d(ra_array, dec_array, v_array, **points3d_kwargs)

def get_stars():
    # Star positions
    path_catalog = "/jupiter/rkarim/Research/Feedback/ancillary_data/catalogs/Ramsey/"
    if not os.path.isdir(path_catalog):
        path_catalog = "/home/ramsey/Documents/Research/Feedback/ancillary_data/catalogs/Ramsey/"
    fn_catalog = path_catalog + "OBradec.csv"
    df_catalog = pd.read_csv(fn_catalog)
    ra_stars, dec_stars = df_catalog.RAdeg.values, df_catalog.DEdeg.values
    is_WR = df_catalog.SpectralType.apply(lambda x: "W" in x)
    ra_WR, dec_WR = ra_stars[is_WR], dec_stars[is_WR]
    ra_stars, dec_stars = ra_stars[~is_WR], dec_stars[~is_WR]
    return (ra_stars, dec_stars), (ra_WR, dec_WR)

# CII data
path_cii = "/n/aurora1/feedback/"
if not os.path.isdir(path_cii):
    path_cii = "/home/ramsey/Documents/Research/Feedback/ancillary_data/sofia/"
fn_cii = path_cii + "rcw49-cii.fits"

# CO data
path_co = "/jupiter/rkarim/Research/Feedback/ancillary_data/apex/"
if not os.path.isdir(path_co):
    path_co = "/home/ramsey/Documents/Research/Feedback/ancillary_data/apex/"
fn_co = path_co + "RCW49_12CO.fits"

data, w = load_fits(fn_co)
vaxis = get_axis(data, w, 0)/1000
data = np.swapaxes(data, 0, 2)
print(data.shape)
testthis = lambda : momentn(data, vaxis, n=0)



v_lim = (290, 550)
y_lim = (120, 320)
x_lim = (60, 210)
contour_levels = [7, 13, 20, 30]
Wd2_center = (155.9920833, -57.7636111)

coord_convert_function = get_cube(fn_cii, x_lim=x_lim, y_lim=y_lim, z_lim=v_lim,
    ref_coord=Wd2_center, contour_levels=contour_levels, vscale=4, cmap='Accent',
    contour_opacity=0.3)

v_lim = (460, 550)
y_lim = (0, 226)
x_lim = (0, 231)
contour_levels = [2]
contour_levels = [5, 10]

coord_convert_function = get_cube(fn_co, x_lim=x_lim, y_lim=y_lim, z_lim=v_lim,
    ref_coord=Wd2_center, contour_levels=contour_levels, vscale=4, cmap='summer',
    contour_opacity=0.3)


coords_stars, coords_WR = get_stars()
plot_stars(*coords_stars, coord_convert_function, opacity=0.5, mode='sphere', color=(1,1,1), scale_factor=0.2)
plot_stars(*coords_WR, coord_convert_function, opacity=0.5, mode='sphere', color=(0.929, 0.568, 0.047), scale_factor=0.2)

mlab.view(azimuth=0, elevation=180, focalpoint=(0,0,0), distance=60)

mlab.show()
