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


def plot_cube(filename, x_lim=None, y_lim=None, z_lim=None, vsmoothing=True,
    xysmoothing=True, ref_coord=None, velocity_conv_pix=3, spatial_conv_pix=1,
    vscale=4, contour_levels=None, cmap=None, contour_opacity=0.2,
    plot_axes=True):

    data, w = load_fits(filename)

    # FITS cubes tend to load out as (velocity, dec, ra)
    # Get the full-range axis arrays BEFORE axis swaps
    velocity_axis = get_axis(data, w, 0)/1000
    dec_axis = get_axis(data, w, 1)
    ra_axis = get_axis(data, w, 2)
    # Swap velocity and ra for (ra, dec, velocity)
    data = np.swapaxes(data, 0, 2)

    # # Make RA increase the other way so we can view it correctly
    # data = np.flip(data, 0)
    # ra_axis = np.flip(ra_axis, 0)

    # Swap Dec and velocity axes so Dec is Z axis
    data = np.swapaxes(data, 1, 2)

    # Take a subset of the full cube, since mayavi doesn't do well with a huge
    # 3D array
    # First, setup default subset (very small)
    default_halfwidth = 50
    if x_lim is None:
        # X is RA, array index 0
        mid_x = data.shape[0]//2
        x_lim = (mid_x - default_halfwidth, mid_x + default_halfwidth)
    if y_lim is None:
        # Y is velocity, array index 1
        mid_y = data.shape[1]//2
        y_lim = (mid_y - default_halfwidth, mid_y + default_halfwidth)
    if z_lim is None:
        # Z is Dec, array index 2
        mid_z = data.shape[2]//2
        z_lim = (mid_z - default_halfwidth, mid_z + default_halfwidth)
    # Set up subset slices
    raslice = slice(*x_lim)
    vslice = slice(*y_lim)
    decslice = slice(*z_lim)

    # Subset the axes
    ra_axis = ra_axis[raslice]
    velocity_axis = velocity_axis[vslice]
    dec_axis = dec_axis[decslice]
    # Subset the data
    data = data[raslice, vslice, decslice]

    # Smooth the velocity axis to avoid a very rough contour in mayavi
    if vsmoothing:
        smooth_kernel = gaussian(5, velocity_conv_pix)
        smooth_kernel /= np.sum(smooth_kernel)
        for i in range(data.shape[0]):
            for k in range(data.shape[2]):
                data[i, :, k] = np.convolve(data[i, :, k], smooth_kernel, mode='same')
    # Smooth spatial axes to avoid rough countours
    if xysmoothing:
        smooth_kernel = gaussian(5, spatial_conv_pix)
        smooth_kernel /= np.sum(smooth_kernel)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j, :] = np.convolve(data[i, j, :], smooth_kernel, mode='same')
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data[:, j, k] = np.convolve(data[:, j, k], smooth_kernel, mode='same')

    # Use a reference coordinate to express axes in small arcminute values
    if ref_coord is None:
        reference_set = False
        ra_ref, dec_ref = ra_axis[len(ra_axis)//2], dec_axis[len(dec_axis)//2]
    else:
        ra_ref, dec_ref = ref_coord
        reference_set = True

    # Function to convert coordinates for axes and stars
    def convert_coords(ra_array, dec_array):
        ra_array = (-1*(ra_array - ra_ref) * 60)*np.cos(np.deg2rad(dec_ref))
        dec_array = (dec_array - dec_ref) * 60
        return ra_array, dec_array

    ra_axis, dec_axis = convert_coords(ra_axis, dec_axis)  # RA positive to the W now

    # Specify RA and Dec limits in converted coordinates
    ra_lo, ra_hi = ra_axis[0], ra_axis[-1]
    dec_lo, dec_hi = dec_axis[0], dec_axis[-1]
    # Set up axis range (negate the negative RA) and grids for mayavi
    lims = [-ra_lo, -ra_hi, velocity_axis[0], velocity_axis[-1], dec_lo, dec_hi]
    ragrid, vgrid, dgrid = np.meshgrid(ra_axis, velocity_axis, dec_axis, indexing='ij')

    # Function to filter star coordinates using the axis limits
    def filter_stars(ra_array, dec_array):
        stars_within_lims = (ra_array > ra_lo) & (ra_array < ra_hi)
        stars_within_lims &= (dec_array > dec_lo) & (dec_array < dec_hi)
        return ra_array[stars_within_lims], dec_array[stars_within_lims]


    def full_coordinate_convert(ra_array, dec_array):
        return filter_stars(*convert_coords(ra_array, dec_array))

    # Set up mayavi scalar_field source and iso_surface contours
    # Use vscale to scale back velocity axis so it lines up a bit better with RA and Dec
    src = mlab.pipeline.scalar_field(ragrid, vgrid/vscale, dgrid, data)
    if contour_levels is None:
        contour_levels = [10, 20]
    mlab.pipeline.iso_surface(src, contours=contour_levels, opacity=contour_opacity, colormap=cmap, vmin=contour_levels[0], vmax=contour_levels[-1])
    # Using the volume rendering would be super cool but it's always so buggy for me
    this_never_works = True
    if not this_never_works: mlab.pipeline.volume(src, vmin=1, vmax=14)
    # Decorations
    if plot_axes:
        # Set up axis display
        mlab.axes(ranges=lims,
            ylabel="Velocity (km/s)", zlabel="Dec (min)", xlabel="RA (min)", nb_labels=4)
        # Plot a zero-velocity "plane"
        mlab.plot3d(
            [ra_lo, ra_hi, ra_hi, ra_lo, ra_lo],
            [0, 0, 0, 0, 0],
            [dec_lo, dec_lo, dec_hi, dec_hi, dec_lo],
            color=(1,1,1), tube_radius=0.05, opacity=0.4
            )
        # Plot reference position
        if reference_set:
            mlab.points3d([0], [0], [0], opacity=0.2, mode='sphere', color=(1,1,1), scale_factor=2)
    return full_coordinate_convert


def plot_stars(ra_array, dec_array, coord_convert_function, **points3d_kwargs):
    ra_array, dec_array = coord_convert_function(ra_array, dec_array)
    v_array = np.zeros_like(ra_array)
    mlab.points3d(ra_array, v_array, dec_array, **points3d_kwargs)

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

sizemod = 2
bg1, bg2 = tuple([0.2]*3), tuple([0.65]*3)
mlab.figure(bgcolor=bg1, fgcolor=(0.93,0.93,0.93), size=(sizemod*400, sizemod*350))

v_lim = (290, 550)
dec_lim = (120, 320)
ra_lim = (60, 210)
contour_levels = [7, 13, 20, 30]
Wd2_center = (155.9920833, -57.7636111)

# coord_convert_function = plot_cube(fn_cii, x_lim=ra_lim, y_lim=v_lim, z_lim=dec_lim,
#     ref_coord=Wd2_center, contour_levels=contour_levels, vscale=4, cmap='Accent',
#     contour_opacity=0.3, plot_axes=False)

v_lim = (460, 550)
dec_lim = (0, 226)
ra_lim = (0, 231)
# contour_levels = [2]
contour_levels = [2, 3, 5, 10]

coord_convert_function = plot_cube(fn_co, x_lim=ra_lim, y_lim=v_lim, z_lim=dec_lim,
    ref_coord=Wd2_center, contour_levels=contour_levels, vscale=4, cmap='terrain',
    contour_opacity=0.3, plot_axes=False)


coords_stars, coords_WR = get_stars()
plot_stars(*coords_stars, coord_convert_function, opacity=0.5, mode='sphere', color=(1,1,1), scale_factor=0.2)
plot_stars(*coords_WR, coord_convert_function, opacity=0.5, mode='sphere', color=(0.929, 0.568, 0.047), scale_factor=0.3)


delta_az = 1 # degrees

mlab.view(azimuth=270, elevation=90, focalpoint=(0,0,0), distance=60)

@mlab.animate(ui=False, delay=30)
def anim():
    f = mlab.gcf()
    current_az = 270
    img_num = 0
    while current_az < 270 + 360:
        mlab.view(azimuth=current_az, focalpoint=(0,0,0), elevation=85, distance=60)
        current_az += delta_az
        f.scene.render()
        mlab.savefig(f"figures/vel3d/anim_CO_{img_num:04d}.png") # dpi~300
        img_num += 1
        yield
# Saving "a" makes sense! The object needs to exist!!
a = anim()
mlab.show()
