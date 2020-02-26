import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import gaussian
import sys

from mayavi import mlab


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

fn_cii = "/n/aurora1/feedback/rcw49-cii.fits"
data, hdr = fits.getdata(fn_cii, header=True)
w = WCS(hdr)

velocity_axis = get_axis(data, w, 0)/1000
dec_axis = get_axis(data, w, 1)
ra_axis = get_axis(data, w, 2)

data = np.swapaxes(data, 0, 2)


vslice_initial = slice(290-14, 550+14)
vslice_final = slice(14, -15)
yslice = slice(120, 280)
xslice = slice(60, 210)


data = data[xslice, yslice, vslice_initial]

smoothing = True
if smoothing:
    smooth_kernel = gaussian(5, 3)
    smooth_kernel /= np.sum(smooth_kernel)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :] = np.convolve(data[i, j, :], smooth_kernel, mode='same')
xysmoothing = True
if xysmoothing:
    smooth_kernel = gaussian(5, 1)
    smooth_kernel /= np.sum(smooth_kernel)
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            data[:, j, k] = np.convolve(data[:, j, k], smooth_kernel, mode='same')
    for i in range(data.shape[0]):
        for k in range(data.shape[2]):
            data[i, :, k] = np.convolve(data[i, :, k], smooth_kernel, mode='same')

velocity_axis = velocity_axis[vslice_initial][vslice_final]
dec_axis = (dec_axis[yslice] - -57.7636111) * 60
ra_axis = (-1*(ra_axis[xslice] - 155.9920833) * 60)*np.cos(np.deg2rad(-57.7636111)) # positive to the W now

data = data[:, :, vslice_final]

vscale = 4
lims = [ra_axis[0], ra_axis[-1], dec_axis[0], dec_axis[-1], velocity_axis[0], velocity_axis[-1]]
ragrid, dgrid, vgrid = np.meshgrid(ra_axis, dec_axis, velocity_axis, indexing='ij')

src = mlab.pipeline.scalar_field(ragrid, dgrid, vgrid/vscale, data)
mlab.pipeline.iso_surface(src, contours=[7, 13, 20, 30], opacity=0.2)
# mlab.pipeline.volume(src, vmin=1, vmax=14)
mlab.axes(ranges=lims,
    zlabel="Velocity (km/s)", ylabel="Dec (min)", xlabel="-RA (min)", nb_labels=4)
mlab.plot3d(
    [lims[0], lims[1], lims[1], lims[0], lims[0]],
    [lims[2], lims[2], lims[3], lims[3], lims[2]],
    [0, 0, 0, 0, 0],
    color=(1,1,1), tube_radius=0.05, opacity=0.4
    )
mlab.points3d([0], [0], [0], opacity=0.4, mode='sphere', color=(1,1,1), scale_factor=2)
mlab.show()
