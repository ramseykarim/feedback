import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.ndimage.interpolation import rotate


"""
Created: February 14, 2020

This is my first attempt at geometric modeling. It's rough, but gave me some
ideas.
"""
__author__ = "Ramsey Karim"

"""
Big idea: I should rewrite all this code to be an "analysis suite" for PDRs,
with tools for different sorts of data.
This file should turn into a function that can produce shell models and maybe
fit them???
"""


def half_shell_mask_2d(array, pixel_scale_ang, radius_ang, thickness_ang, center_pixel, ang=0.0):
    # angle: 0.0 degrees is a half shell opening downwards. Like an upside down bowl.
    # I think the _ang arguments just need to have the same units (like degrees)
    i, j = ((np.arange(x)-c)*pixel_scale_ang for x, c in zip(array.shape, center_pixel))
    ii, jj = np.meshgrid(i, j, indexing='ij')
    ang = ang % 360
    if ang < 90:
        slope = np.tan(np.deg2rad(ang))
        plane_mask = (slope*jj - ii) < 0
    elif ang == 90.:
        plane_mask = jj < 0
    elif ang <= 180.:
        slope = np.tan(np.deg2rad(ang))
        plane_mask = (ii - slope*jj) < 0
    elif ang < 270:
        slope = np.tan(np.deg2rad(ang))
        plane_mask = (ii - slope*jj) < 0
    elif ang == 270.:
        plane_mask = jj > 0
    else:
        slope = np.tan(np.deg2rad(ang))
        plane_mask = (slope*jj - ii) < 0
    d_from_origin_squared = (ii*ii) + (jj*jj)
    within_shell = (d_from_origin_squared > (radius_ang - thickness_ang/2.)**2) & (d_from_origin_squared < (radius_ang + thickness_ang/2.)**2)
    return within_shell & plane_mask


def shell_mask_3d(array_shape, pixel_scale_ang, radius_ang, thickness_ang, center_pixel):
    """
    array_shape and center_pixel need to be len 3 (i, j, k)
    """
    # Make grid arrays using pixel scale info
    ii, jj, kk = np.meshgrid(*((np.arange(x) - c)*pixel_scale_ang for x, c in zip(array_shape, center_pixel)), indexing='ij')
    # Make distance-from-center-pixel grid
    d_from_origin_squared = (ii*ii) + (jj*jj) + (kk*kk)
    # Try to save memory as quickly as possible
    del ii, jj, kk
    # Return mask
    return (d_from_origin_squared > (radius_ang - thickness_ang/2.)**2) & (d_from_origin_squared < (radius_ang + thickness_ang/2.)**2)


"""
TESTS
"""

def test_expanding_shell():
    mask = make_test_shell()
    # somehow need to get velocities for every cell, based on angle from center


def test_3d_shell_function():
    mask = make_test_shell()
    plt.imshow(np.sum(mask, axis=-1), origin='lower')
    plt.show()


def make_test_shell():
    array_shape = (100,100,100)
    # angles in example test units (since it's arbitrary)
    pixel_scale_ang = 1.
    radius_ang = 43.2
    thickness_ang = 9.5
    center_pixel = (50, 50, 50)
    mask = shell_mask_3d(array_shape, pixel_scale_ang, radius_ang, thickness_ang, center_pixel)


def test_3d_shell():
    los_distance_pc = 1000 * 4.16 * u.pc
    # from ds9 estimate on tau160
    shell_thickness_angular_hi = 0.01 * u.deg
    shell_thickness_angular_lo = 0.0064226 * u.deg
    shell_thickness_angular = (shell_thickness_angular_lo + shell_thickness_angular_hi) / 2.
    radius_angular_hi = 0.11043 * u.deg
    radius_angular_lo = 0.1013575 * u.deg
    avg_radius_angular = (radius_angular_lo + radius_angular_hi) / 2.


    avg_radius_physical = avg_radius_angular.to(u.rad).to_value() * los_distance_pc
    shell_thickness_physical = shell_thickness_angular.to(u.rad).to_value() * los_distance_pc

    print(f"Physical shell radius: {avg_radius_physical:.2f}")
    print(f"Physical shell thickness: {shell_thickness_physical:.3f}")

    r = avg_radius_physical.to_value()
    dr = shell_thickness_physical.to_value()

    grid_size = 100
    grid_extent = 2.1*r
    grid = np.zeros(grid_size**3).reshape((grid_size, grid_size, grid_size))
    ii, jj, kk = np.meshgrid(*[(np.arange(grid_size)-(grid_size//2))*grid_extent/grid_size]*3, indexing='ij')

    plane = jj-(0.3*ii) < 0
    d_from_origin_squared = (ii*ii) + (jj*jj) + (kk*kk)
    half_shell_mask = (d_from_origin_squared > (r-(dr/2))**2) & (d_from_origin_squared < (r+(dr/2))**2) #& plane

    # half_shell_mask = rotate(half_shell_mask, 20, axes=(1, 0), reshape=False)
    # half_shell_mask = rotate(half_shell_mask, 30, axes=(0, 2), reshape=False)
    # half_shell_mask = rotate(half_shell_mask, 30, axes=(0, 2), reshape=False)

    plt.imshow(np.sum(half_shell_mask, axis=-1), origin='lower', extent=[-grid_extent/2, grid_extent/2, -grid_extent/2, grid_extent/2])
    plt.show()


if __name__ == "__main__":
    test_3d_shell_function()
