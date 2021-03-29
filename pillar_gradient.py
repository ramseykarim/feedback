import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 10}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

from scipy.linalg import lstsq
from scipy.signal import convolve2d

from astropy import units as u
from astropy.modeling import models
from astropy.convolution import Gaussian2DKernel

from . import cube_pixel_spectra_2 as cps2
misc_utils = cps2.misc_utils
catalog = cps2.catalog
cube_utils = cps2.cube_utils
cps1 = cps2.cps1
regions = cps2.regions


"""
Created: March 2, 2021

I want to quantify the gradients along and across the pillars in M16
I don't know whether these vary along/across the pillar, but I will try to
explore that

I'm going to use the plane-fitting method of Menon et al 2021
(https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.1721M/abstract)
They make moment-1 images and then fit a plane to them, which they subtract out
of the moment-1 images to find turbulent motions

I want to extend this. I want to focus on the plane fit first, interpreting it
as the combination of the parallel and transverse gradients. I want to re-fit
in terms of alternate pillar-centric axes (along/across the pillar) so the
fit is easier to interpret.
(but I'll see if that's really possible)

Then I'll look into the residual "turbulent" motions (or whatever they might be)
and try to explain those

This will probably be easiest on Pillar 2; P3 is hardly resolved, and P1 has a
complicated gradient

March 5, 2021 update:
Generally works, but need to verify that the results mean anything
Also should run it on the CO 1-0 (and maybe 3-2 but resolution could be an issue)

To verify, I'll make a fake pillar moment 0 and 1 image, add noise, and try the
method.
Fake pillar involves making a cube from a map of Gaussian parameters,
adding noise to those spectra, and then running my code on the cube

March 18, 2021 update:
This round of testing was instructive, but the code is a bit of a mess,
so I'm going to clean it up and generalize a few things so that I can make
better plot combos
One of the things I want to be able to easily do is compare the fitted plane
to the model plane (particularly where the pillar exists)
Another thing is to streamline / clarify how the pillar is masked
"""
kms = u.km/u.s


def make_fake_pillar(base_cube, debug=False):
    """
    Make fake pillar (version 2; see last commit for previous version)
    Use astropy.modeling.models.Gaussian1D to run grid of parameters
    Shape parameters correctly

    Features to add: better definitions of pillar width, height, orientation(?),
        and gradient, so that these can be passed in as arguments
        Eventually, add other features (second set-of-Gaussian plane) to act as
        background cloud to see how this affects the answer

    This time around, generalize a little more and take in a cube whose
    shape (and wcs) will be used
    :param base_cube: SpectralCube to base shape and WCS on
    :returns: fake data SpectralCube
    """
    # Get the independent variable axis in 3D
    xarr = base_cube.spectral_axis.to_value()[:, np.newaxis, np.newaxis]
    # Get image shape (2D)
    img_shape = base_cube.shape[1:]

    # Set up boolean mask for "exact" pillar location
    # Width defined by "fraction of total width" dj_frac
    # (different than original definition, which was confusing)
    # dj_frac is the width of the pillar divided by the width of the image
    dj_frac = 0.4
    # dj is width in pixels
    dj = int(round(dj_frac * img_shape[1]))
    # Height defined same way
    di_frac = 0.75
    di = int(round(di_frac * img_shape[0]))
    bool_pillar_mask = np.zeros(img_shape, dtype=bool)
    bool_pillar_mask[:di, (img_shape[1]//2 - dj//2):(img_shape[1]//2 + dj//2)] = True

    if debug:
        fig = plt.figure()
        # Plot bool_pillar_mask
        plt.subplot2grid((2, 8), (0, 0))
        plt.imshow(bool_pillar_mask, origin='lower')
        ### can definitely turn this and the inclusive mask imgs into contour/img combo
        plt.title("Pillar source")

    # Set up a 2D image for each parameter
    # Amplitude
    A_img = np.zeros(img_shape)
    line_height = 20.
    A_img[bool_pillar_mask] = line_height
    smooth_kernel = Gaussian2DKernel(x_stddev=2)
    A_img = convolve2d(A_img, smooth_kernel, mode='same', boundary='symm')

    inclusive_bool_pillar_mask = A_img > 1

    if debug:
        # Plot inclusive_bool_pillar_mask
        plt.subplot2grid((2, 8), (1, 0))
        plt.imshow(inclusive_bool_pillar_mask, origin='lower')
        plt.title("Pillar mask")
        # Plot amplitude
        plt.subplot2grid((2, 8), (0, 1), colspan=2, rowspan=2)
        plt.imshow(A_img, origin='lower')
        plt.colorbar()
        plt.title("Amplitude")

    # Mean
    # Make img coord axes to use for velocity gradient
    ii, jj = np.mgrid[0:img_shape[0], 0:img_shape[1]]
    i_grad, j_grad, mu0 = 0.2, 0.2, 25
    # Center the gradient at the middle of the pillar
    mu_img = i_grad*(ii - di//2) + j_grad*(jj - img_shape[0]//2) + mu0

    if debug:
        # Plot mean
        plt.subplot2grid((2, 8), (0, 3), colspan=2, rowspan=2)
        plt.imshow(mu_img, origin='lower')
        plt.colorbar()
        plt.title("Mean")

    # Standard deviation
    std = 1.
    std_img = np.ones(img_shape)*std

    # Find the pillar's velocity range
    pillar_vel_limits = (mu_img[bool_pillar_mask].min() - std, mu_img[bool_pillar_mask].max() + std)

    # Put parameter images together into model plane and calculate cube
    gauss = models.Gaussian1D(mean=mu_img[np.newaxis, :], stddev=std_img[np.newaxis, :], amplitude=A_img[np.newaxis, :])
    fake_cube_data = gauss(xarr)
    # Add noise
    snr = 20.
    rng = np.random.default_rng()
    fake_cube_data += rng.normal(loc=0.0, scale=line_height/snr, size=fake_cube_data.shape)
    # Add background (DC offset)
    background_level = 10
    fake_cube_data += background_level
    # Convert to Quantity
    fake_cube_data = fake_cube_data * u.K
    # Initialize SpectralCube
    fake_cube = cube_utils.SpectralCube(data=fake_cube_data, wcs=base_cube.wcs).with_spectral_unit(kms)

    if debug:
        fake_subcube = fake_cube.spectral_slab(*(x*kms for x in pillar_vel_limits))
        fake_mom0 = fake_subcube.moment0()
        print(f"Median of moment0: {np.median(fake_mom0.to_value())}")
        print(f"Mean of moment0: {np.mean(fake_mom0.to_value())}")
        fake_mom1 = fake_subcube.moment1()
        # Plot moment 0
        plt.subplot2grid((2, 8), (0, 7))
        plt.imshow(fake_mom0.to_value(), origin='lower')
        plt.colorbar()
        plt.title("Moment 0")
        # Plot moment 1
        plt.subplot2grid((2, 8), (1, 7))
        plt.imshow(fake_mom1.to_value(), origin='lower', vmin=pillar_vel_limits[0], vmax=pillar_vel_limits[1])
        plt.colorbar()
        plt.title("Moment 1")
        # Plot the deviation of moment 1 from the model gradient
        deviation = 100*(mu_img - fake_mom1.to_value())/mu_img
        min_dev, max_dev = max(-50, deviation[bool_pillar_mask].min()), min(50, deviation[bool_pillar_mask].max())
        deviation_copy = deviation.copy()
        deviation[~inclusive_bool_pillar_mask] = np.nan
        deviation_copy = convolve2d(deviation_copy, Gaussian2DKernel(x_stddev=1), mode='same', boundary='symm')
        deviation_copy[~inclusive_bool_pillar_mask] = np.nan
        ax = plt.subplot2grid((2, 8), (0, 5), colspan=2, rowspan=2)
        im = ax.imshow(deviation, origin='lower', vmin=min_dev, vmax=max_dev)
        ax.contour(deviation_copy, levels=5, colors='r', alpha=0.7)
        fig.colorbar(im, ax=ax)
        plt.title("model deviation (%)")
    return fake_cube, (mu_img, std_img, A_img), pillar_vel_limits, (bool_pillar_mask, inclusive_bool_pillar_mask)


def vlims_moment1(cube, mom1, vel_lims):
    """
    Calculate decent visual limits for the moment1 maps
    Needs tighter velocity limits around the desired component (pillar)
    :param cube: SpectralCube object
    :param mom1: Quantity array, the moment-1 image from the cube that these
        vlims will apply to. Shape should match the cube's spatial shape
    :param vel_lims: tuple of int or float velocities (in km/s)
    :returns: dict {vmin=low, vmax=high} float velocity limits (in km/s)
        suitable for the moment-1 map of this component
    """
    subcube = cube.spectral_slab(*(v*kms for v in vel_lims))
    mom0 = subcube.moment0().to(u.K*kms).to_value()
    mom1 = mom1.to_value().copy()
    # Restrict to above median moment-0 flux (to isolate the pillar)
    mom1[mom0 < np.median(mom0)] = np.nan
    # Get visual limits for moment-1 map (10 and 90% levels within that mask)
    lo, hi = misc_utils.flquantiles(mom1[np.isfinite(mom1)].flatten(), 10)
    return dict(vmin=lo, vmax=hi)


def pillar_mask(mom0):
    """
    Make a pillar mask using median integrated flux
    :param mom0: Quantity array, the moment-0 image from the cube
    :returns: mask with "True" where there is pillar, same shape as mom0
    """
    try:
        mom0 = mom0.to_value()
    except:
        pass
    print(np.nanmin(mom0), np.nanmax(mom0), np.nanmean(mom0), np.nanmedian(mom0), np.nanstd(mom0))
    return mom0 > np.nanmedian(mom0) + np.nanstd(mom0)


def plot_moments():
    """
    Just hammer out some nice moment 0 and 1 plots
    """
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection=moment1.wcs)
    print("Vlims", vlims)
    im = ax.imshow(moment1.to_value(), cmap="RdBu_r", **vlims)
    c = ax.contour(moment0, cmap='cividis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Average velocity (km/s) between {vel_str}")
    ax.set_title(f"Moment 1 in color, moment 0 in contours, between {vel_str}")
    plt.show()


def make_full_pillar_mask():
    """
    Combine pillar_mask (just based on integrated intensity) and the hand-drawn
    boxes I made in ds9 to make the "best" pillar masks suited for plane fitting
    If Pillar 1 is selected, just return the pillar_mask (moment0-based)
    :returns: boolean mask, same shape as moment1, True where pillar exists
    """
    box_reg_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/p23_boxes1.reg"))
    make_mask = lambda reg: reg.to_pixel(moment1.wcs).to_mask().to_image(moment1.shape).astype(bool)
    if selected_component == "Pillar 1":
        print(f"{__file__}.make_full_pillar_mask: no ds9 box mask, using moment0 only")
        return pillar_mask(moment0)
    elif selected_component == "Pillar 2":
        box_mask = make_mask(box_reg_list[0])
    elif selected_component == "Pillar 3":
        box_mask = make_mask(box_reg_list[1])
    elif selected_component == "synth":
        # return moment0.to_value() > np.mean(moment0.to_value())
        return pillar_mask(moment0)
    mom0 = moment0.to_value().copy()
    # mom0[~box_mask] = np.nan
    pmask = pillar_mask(mom0) & box_mask
    return pmask


def plot_masked_moment():
    """
    Impose a hand-drawn box mask around the pillars and then combine that with a
    moment-0 flux mask. This should be the best thing to use for the plane-fit
    """
    pmask = make_full_pillar_mask()
    moment1_masked = moment1.to_value().copy()
    moment1_masked[~pmask] = np.nan
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection=moment1.wcs)
    ax.imshow(moment1_masked, cmap='RdBu_r')
    ax.contour(moment0, cmap='cividis')
    plt.show()


def make_data_points_from_image(img, origin=None):
    """
    Make arrays of x, y, and z
    Remember y is the 0th index, x is the 1st index
    NaNs in the image will be preserved in the returned array
    :param img: a 2D array whose axes and values we want.
        X is assumed to be the J axis (1st axis) and Y to be the I axis (0th)
        Adhering strictly to this ordering will allow us to make rotations
        work out intuitively.
    :param origin: tuple (Y, X) of the coordinate grid origin, if not (0,0).
        If None, (0,0) (the bottom left corner of the image) is assumed
        Again, this is (Y, X) = (I, J), native Numpy indexing order
    :returns: array of shape (3, N): (X, Y, Z). Z values can be NaN.
        Note that this needs to be transposed for something like linalg.lstsq,
        but unpacks easily as X, Y, Z = result
    """
    yx = np.mgrid[0:img.shape[0], 0:img.shape[1]].reshape(2, img.size)
    if origin is not None:
        yx[0] -= origin[0]
        yx[1] -= origin[1]
    xy = np.flip(yx, axis=0)
    z = img.flatten()[np.newaxis, :]
    xyz = np.concatenate([xy, z])
    return xyz


def temp_test_make_data_points():
    """
    Passed the test! verified that make_data_points_from_image gets X, Y, Z
    correct
    """
    arr = make_data_points_from_image(moment1.to_value())
    x, y, z = (a.reshape(moment1.shape) for a in arr)
    plt.subplot(131)
    plt.imshow(x, origin='lower')
    plt.subplot(132)
    plt.imshow(y, origin='lower')
    plt.subplot(133)
    plt.imshow(z, origin='lower')
    plt.show()


def fit_plane_directly(img=None, xyz=None):
    """
    Just fit a plane to the image
    Use scipy.linalg.lstsq
    Only one of img or xyz need to be present. If both are given, xyz is used
    :param img: the image (2D array of z values). Can contain NaN values.
    :param xyz: the direct result of running make_data_points_from_image on
        the image. A (3, N) array of (X, Y, Z)
    :returns: array of (a, b, d), where a is the x slope, b is the y slope,
        and d is the z value at the origin
    """
    if xyz is None:
        if img is None:
            raise RuntimeError("Need at least one valid input!")
        xyz = make_data_points_from_image(img) # (3, N) shape: (X, Y, Z)
    # Filter out coordinates with NaN z-values
    zmask = np.isfinite(xyz[2])
    xyz = xyz[:, zmask] # now the input xyz array is preserved
    # Switch to (N, 3), the input format for linalg.lstsq
    xyz = xyz.T
    # Put things in terms of z and A; solving for C in z = A*C
    z = xyz[:, 2].copy()
    A = xyz
    A[:, 2] = 1
    """
    All set up for z = A*C, where C is our (3,1) array of answers!
    C should be [[a], [b], [d]], where a gives the x gradient, b the y
        gradient, and d the offset (z-value) at the origin
    z = a*x + b*y + d is the expression for z, rearranged from the more general
        a*x + b*y + c*z + d = 0 when c = -1
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
    for lstsq documentation (explanations of res, rnk, s, which I won't use)
    """
    p, res, rnk, s =  lstsq(A, z)
    print(p.shape)
    return p


def plot_direct_plane_fit():
    """
    Try the direct fit!
    """
    pmask = make_full_pillar_mask()
    moment1_masked = moment1.to_value().copy()
    moment1_masked[~pmask] = np.nan
    xyz = make_data_points_from_image(moment1_masked) # (3, img.size)
    abd_arr = fit_plane_directly(xyz=xyz) # (a, b, d) is the plane solution
    # Recreate the A matrix (N, 3) = (X0..XN, Y0..YN, 1..1)
    A = xyz.T
    A[:, 2] = 1
    fitted_plane = A.dot(abd_arr).reshape(moment1.shape)

    fig = plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(131, projection=moment1.wcs)
    im = ax1.imshow(moment1_masked, cmap='RdBu_r', **vlims)
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("Average velocity (km/s)")
    ax1.contour(moment0, cmap='cividis')
    ax1.set_title(f"Moment 1 image of {selected_component} between {vel_str}")
    ax1.set_xlabel("RA")
    ax1.set_ylabel("Dec")

    ax2 = plt.subplot(132, projection=moment1.wcs)
    im = ax2.imshow(fitted_plane, cmap='RdBu_r', **vlims)
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("Average velocity (km/s)")
    ax2.contour(moment0, cmap='cividis')
    ax2.set_title("Plane fitted to masked moment 1 map")
    ax2.set_xlabel(" ")
    ax2.set_ylabel(" ")

    ax3 = plt.subplot(133, projection=moment1.wcs)
    im = ax3.imshow(moment1_masked - fitted_plane, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label("Residual velocity (km/s)")
    ax3.contour(moment0, cmap='cividis')
    ax3.set_title("Moment 1 image with plane subtracted")
    ax3.set_xlabel(" ")
    ax3.set_ylabel(" ")

    plt.subplots_adjust(left=0.05, right=0.95)
    # plt.show()
    component_stub = selected_component.replace(" ", "_")
    # plt.show()
    # fig.savefig(f"/home/ramsey/Pictures/2021-03-02-work/plane_fit_{component_stub}.png")
    return fitted_plane, moment1_masked


def make_fake_pillar_OLD(debug=False):
    """
    Make fake pillar
    Looks like I can use astropy.modeling.models.Gaussian1D to run a grid of parameters
    Just need to shape the parameter arrays correctly
    """
    if debug:
        print(cube.shape)
    xarr = cube.spectral_axis.to_value()[:, np.newaxis, np.newaxis]
    img_shape = cube.shape[1:] # just 2d for now
    # Start with peak intensity (parameter:amplitude)
    A_img = np.zeros(img_shape)
    dj_frac = 8 # make this smaller for a wider pillar. percentage of total width = 200/dj_frac. 10 means a pillar of 20% of img width
    # Put 1s for pillar (times some desired line height)
    A_img[:img_shape[0]*3//4, img_shape[1]//2-img_shape[1]//dj_frac:img_shape[1]//2+img_shape[1]//dj_frac] = 1.
    # Get a mask to use later
    bool_pillar_mask = A_img > 0
    # Multiply by desired line height
    A_img *= 20
    if debug:
        plt.figure()
        plt.subplot(131)
        plt.imshow(A_img, origin='lower')
    smooth_kernel = Gaussian2DKernel(x_stddev=2)
    A_img = convolve2d(A_img, smooth_kernel, mode='same', boundary='symm')
    if debug:
        plt.subplot(132)
        plt.imshow(A_img, origin='lower')
    # Make img coord axes to use for velocity gradient (parameter:mean)
    ii, jj = np.mgrid[0:img_shape[0], 0:img_shape[1]]
    mu_img = 0.2*(ii - img_shape[0]//2) + 0.05*(jj - img_shape[1]) + 25
    # Re-create synth velocity limits
    vel_limits['synth'] = (mu_img[bool_pillar_mask].min(), mu_img[bool_pillar_mask].max())
    print(f"Updated synth velocity limits: {vel_limits['synth']}")
    if debug:
        plt.subplot(133)
        plt.imshow(mu_img, origin='lower')
    std_img = np.ones(img_shape)*1.
    gauss = models.Gaussian1D(mean=mu_img[np.newaxis, :], stddev=std_img[np.newaxis, :], amplitude=A_img[np.newaxis, :])
    fake_cube_data = gauss(xarr)
    # Add noise, do 1 K on top of 20 K line peak, so 20 SNR
    rng = np.random.default_rng()
    fake_cube_data += rng.normal(loc=0.0, scale=1.0, size=fake_cube_data.shape)
    # Add 2 K background
    fake_cube_data += 2
    # Make units K
    fake_cube_data = fake_cube_data * u.K
    if debug:
        print(fake_cube_data.shape)
    fake_cube = cube_utils.SpectralCube(data=fake_cube_data, wcs=cube.wcs).with_spectral_unit(u.km/u.s)
    if debug:
        fake_subcube = fake_cube.spectral_slab(*(x*kms for x in vel_limits['synth']))
        fake_mom0 = fake_subcube.moment0()
        print(f"Median of moment0: {np.median(fake_mom0.to_value())}")
        print(f"Mean of moment0: {np.mean(fake_mom0.to_value())}")
        fake_mom1 = fake_subcube.moment1()
        fig = plt.figure()
        plt.subplot(231)
        im = plt.imshow(fake_mom0.to_value(), origin='lower')
        fig.colorbar(im, ax=plt.gca())
        plt.title("mom0")
        plt.subplot(232)
        im = plt.imshow(fake_mom1.to_value(), origin='lower', vmin=vel_limits['synth'][0], vmax=vel_limits['synth'][1])
        fig.colorbar(im, ax=plt.gca())
        plt.title("mom1")
        plt.subplot(233)
        im = plt.imshow(100*(mu_img - fake_mom1.to_value())/mu_img, origin='lower', vmin=-20, vmax=20)
        fig.colorbar(im, ax=plt.gca())
        plt.title("pct deviation from model")

        plt.subplot(212)
        plt.plot(fake_cube.spectral_axis.to_value(), fake_cube.unmasked_data[:, img_shape[0]//2, img_shape[1]//2])
        plt.plot(fake_cube.spectral_axis.to_value(), fake_cube.unmasked_data[:, img_shape[0]*3//4, img_shape[1]//2])
        plt.plot(fake_cube.spectral_axis.to_value(), fake_cube.unmasked_data[:, img_shape[0]*1//4, img_shape[1]//2])
        print(fake_cube)
        # plt.show()
    return fake_cube, (mu_img, std_img, A_img)


if __name__ == "__main__" and True:
    vel_limits = {'Pillar 1': (21, 28), 'Pillar 2': (19, 24), 'Pillar 3': (19, 25),
        'synth': (17, 30)}
    focused_vel_limits = {'Pillar 1': (24, 26), 'Pillar 2': (20, 23), 'Pillar 3': (21, 23),
        'synth': (20, 27)}
    reg_indices = {'Pillar 1': 0, 'Pillar 2': 1, 'Pillar 3': 2, 'synth': 0}
    selected_component = 'Pillar 2'
    vel_str_make = lambda : f"[{vel_limits[selected_component][0]}, {vel_limits[selected_component][1]}] km/s"
    vel_str = vel_str_make()
    cube = cps2.cutout_subcube(length_scale_mult=1, reg_filename="catalogs/parallelpillars_single.reg", reg_index=reg_indices[selected_component])
    # cube, param_imgs = make_fake_pillar(debug=1)
    vel_str = vel_str_make()
    subcube = cube.spectral_slab(*(v*kms for v in vel_limits[selected_component]))
    moment0 = subcube.moment0().to(u.K*kms)
    moment1 = subcube.moment1().to(kms)
    vlims = vlims_moment1(cube, moment1, focused_vel_limits[selected_component])




if __name__ == '__main__':
    make_fake_pillar(cube, debug=True)
    plt.show()

    # fitted_plane, moment1_masked = plot_direct_plane_fit()

    # print(f"New velocity limits: {vel_str}")
    # vlims = dict(vmin=param_imgs[0].min(), vmax=param_imgs[0].max())
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(param_imgs[0], origin='lower', **vlims)
    # plt.title("model gradient")
    # plt.colorbar(label="mean velocity (km/s)")
    # plt.subplot(132)
    # plt.imshow(100*(param_imgs[0] - fitted_plane)/param_imgs[0], origin='lower')
    # plt.title("(model $-$ fitted)/model")
    # plt.colorbar(label="x100 (pctg)")
    # plt.subplot(133)
    # plt.imshow(fitted_plane, origin='lower', **vlims)
    # plt.title("fitted gradient")
    # plt.colorbar(label="mean velocity (km/s)")
    # plt.show()
