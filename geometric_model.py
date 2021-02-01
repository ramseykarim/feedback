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


def make_half_shell_mask_2d(array, pixel_scale_ang, radius_ang, thickness_ang, center_pixel, ang=0.0):
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


def make_shell_mask_3d(array_shape, pixel_scale_ang, radius_ang, thickness_ang, center_pixel):
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


def make_ellipse_volume_mask_3d(semimajor_ang, semiminor_ang, array_shape=None, pixel_scale_ang=None, center_pixel=None, meshgrid=None):
    """
    Just make the volume mask. Put in meshgrid results into that keyword to save
    time/space
    """
    if meshgrid is None:
        ii, jj, kk = np.meshgrid(*((np.arange(x) - c)*pixel_scale_ang for x, c in zip(array_shape, center_pixel)), indexing='ij')
    else:
        ii, jj, kk = meshgrid
    return ((ii*ii) / semimajor_ang**2 + ((jj*jj) + (kk*kk)) / semiminor_ang**2) < 1


def make_ellipse_mask_2d(semimajor_ang, semiminor_ang, array_shape=None, pixel_scale_ang=None, center_pixel=None, meshgrid=None):
    """
    Make a 2D ellipse mask (can be broadcast to 3D cylinder)
    """
    if meshgrid is None:
        ii, jj = np.meshgrid(*((np.arange(x) - c)*pixel_scale_ang for x, c in zip(array_shape[:2], center_pixel[:2])), indexing='ij')
    else:
        ii = meshgrid[0][:, :, 0]
        jj = meshgrid[1][:, :, 0]
    return ((ii*ii) / semimajor_ang**2 + (jj*jj) / semiminor_ang**2) < 1


def make_ellipse_shell_mask_3d(semimajor_ang, semiminor_ang, thickness_ang, array_shape=None, pixel_scale_ang=None, center_pixel=None, meshgrid=None):
    """
    array_shape and center_pixel need to be len 3 (i, j, k)
    """
    if meshgrid is None:
        # Make grid arrays using pixel scale info
        ii, jj, kk = np.meshgrid(*((np.arange(x) - c)*pixel_scale_ang for x, c in zip(array_shape, center_pixel)), indexing='ij')
    else:
        ii, jj, kk = meshgrid
    # Use the ellipse formula 1 = (x/a)^2 + (y/b)^2 to make masks of the elliptical volume
    outer_ellipse = make_ellipse_volume_mask_3d(semimajor_ang + thickness_ang/2., semiminor_ang + thickness_ang/2., meshgrid=(ii, jj, kk))
    inner_ellipse = make_ellipse_volume_mask_3d(semimajor_ang - thickness_ang/2., semiminor_ang - thickness_ang/2., meshgrid=(ii, jj, kk))
    if meshgrid is None:
        # Try to save memory as quickly as possible
        del ii, jj, kk
    # Return mask
    return outer_ellipse ^ inner_ellipse


def calculate_elliptical_correction_factor(semimajor_ang, semiminor_ang, thickness_ang, array_shape, pixel_scale_ang, center_pixel):
    """
    Elliptical shell! November 10, 2020
    Calculate the ratio of the limb-brightened shell volume to the 3D shell volume
    I won't actually use the make_ellipse_shell_mask_3d function, I'll use the
    make_ellipse_volume_mask_3d and make_ellipse_mask_2d functions only

    array_shape and center_pixel need to be len 3 (i, j, k)

    I am doing this because I found out making the elliptical correction by hand
    would be HORRENDOUS
    The semimajor axis is assumed to be the first (y or i) axis
    The j and k axes are assumed to be both semiminor (equal)
    """
    # Make grid arrays using pixel scale info
    ii, jj, kk = np.meshgrid(*((np.arange(x) - c)*pixel_scale_ang for x, c in zip(array_shape, center_pixel)), indexing='ij')
    # Make and flatten the 3D shell mask
    shell_mask_3d = make_ellipse_shell_mask_3d(semimajor_ang, semiminor_ang, thickness_ang, meshgrid=(ii, jj, kk)).sum(axis=-1)
    # Make the 2D limb brightened shell mask (really just the smaller ellipse)
    smaller_ellipse_mask_2d = make_ellipse_mask_2d(semimajor_ang - thickness_ang/2., semiminor_ang - thickness_ang/2., meshgrid=(ii, jj, kk))
    # Get sum (volume) of the 3D shell
    volume_shell_3d = shell_mask_3d.sum()
    # Mask out the smaller ellipse and get sum (volume) of limb brightened shell
    shell_mask_3d[smaller_ellipse_mask_2d] = 0
    volume_shell_lb = shell_mask_3d.sum()
    return volume_shell_3d / volume_shell_lb



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
    mask = make_ellipse_shell_mask_3d(radius_ang, radius_ang*0.8, thickness_ang, array_shape=array_shape, pixel_scale_ang=pixel_scale_ang, center_pixel=center_pixel)
    return mask # why didn't I do this already?


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


def find_elliptical_correction():
    """
    Test the elliptical correction with a few different grid sizes
    """
    los_distance_pc = 1000 * 4.16 * u.pc
    # From the paper:
    shell_semimajor_physical = 7*u.pc
    shell_semiminor_physical = 4*u.pc
    shell_thickness_physical = 1*u.pc

    trials = [
        [5.5, 5.5, 1], [7., 4., 1.]
    ]
    labels = ["Sphere r = 5 pc", "Ellipse a, b = 7, 4 pc", 'x']
    colors = ['r', 'k', 'b']

    full_grid_sizes = [50, 75, 100, 125, 150, 250, 325, 400]
    for i in range(len(trials)):
        a, b, dr = trials[i]
        corrections = []
        for full_grid_size in full_grid_sizes:
            print(full_grid_size)
            grid_shape = (full_grid_size,) + (int(round(full_grid_size*(b*1.01/a))),)*2
            pixel_scale = 2.2*a/full_grid_size
            corrections.append(calculate_elliptical_correction_factor(a, b, dr, grid_shape, pixel_scale, center_pixel=(0, 0, 0)))

        # ii, jj, kk = np.meshgrid(*[(np.arange(g) - (g//2))*pixel_scale for g in grid_shape])
        #
        # shell_mask_3d = make_ellipse_shell_mask_3d(a, b, dr, meshgrid=(ii, jj, kk))
        # plt.imshow(shell_mask_3d.sum(axis=-1), origin='lower', extent=[-grid_shape[0]*pixel_scale/2, grid_shape[0]*pixel_scale/2, -grid_shape[1]*pixel_scale/2, grid_shape[1]*pixel_scale/2])
        # plt.show()
        plt.plot(full_grid_sizes, corrections, color=colors[i], lw=2, label=labels[i])
    plt.title("Geometric correction factor as a function of grid size")
    plt.xlabel("Grid size (along semimajor axis)")
    plt.ylabel("Numerically calculated correction factor")
    plt.ylim((2.2, 2.8))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    find_elliptical_correction()
