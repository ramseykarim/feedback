import os, sys
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 11}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scipy.constants as cst
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.signal import gaussian

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.nddata import Cutout2D
from spectral_cube import SpectralCube
from reproject import reproject_interp
import pvextractor
import regions

from .mantipython import physics
from . import misc_utils
from . import catalog
from . import geometric_model as geomodel
from . import cube_utils


dust_path = f"{catalog.utils.feedback_path}misc_data/dust/"
herschel_path = f"{catalog.utils.ancillary_data_path}herschel/"
cii_cube = f"{catalog.utils.cii_path}rcw49-cii.fits"

"""
Constants

Note: gas-to-dust is here
"""

H_mass_amu = 1.00794
Hmass = cst.m_u * H_mass_amu * 1e3 # kg->g


"""
Other useful global variables
"""
# Some standard tau160 map
# fit2p_filename = "RCW49large_2p_2BAND_beta2.0.fits"
fit2p_filename = "RCW49large_2p_2BAND_160grid_beta2.0.fits"

# Unsure what I was using this for
# This is the SIMBAD coordinate, so at least there's that
wd2_center_coord = SkyCoord("10 23 58.1 -57 45 49", unit=(u.hourangle, u.deg))



"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%& Setup & math functions %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
"""

def Draine_fn(Rv):
    if Rv in [3.1, 4.0, 5.5]:
        return f"{dust_path}Draine_{Rv:3.1f}.dat"
    else:
        raise RuntimeError(f"Rv {Rv} not available")

def get_Draine_column_names():
    with open(dust_path+"readme", 'r') as f:
        l = '#'
        while l.strip()[0] == '#':
            l = f.readline()
        colnames = l.strip().split(' ')
    return colnames

def Draine_data(Rv):
    colnames = get_Draine_column_names()
    return np.genfromtxt(Draine_fn(Rv), names=colnames, usecols=range(len(colnames)))

def get_wl(data):
    return data['lambdaum']

def get_k(data):
    return data['K_abscm2g']

def get_C(data):
    return data['C_extHcm2H']

def get_albedo(data):
    return data['albedo']

def Draine_dustmass_per_H(Rv):
    # gram/H
    return {3.1: 1.870E-26, 4.0: 1.969E-26, 5.5: 2.199E-26}[Rv]

def Draine_gastodust(Rv):
    # includes He
    return {3.1: 1.236E+02, 4.0: 1.174E+02, 5.5: 1.051E+02}[Rv]


def convert_ktoC(k, gastodust):
    # if you use Draine's gas to dust, it includes He
    return k*Hmass / gastodust # dust cm2 / H particle

def get_val_at(wl, wl_array, val_array):
    return interp1d(wl_array, val_array)(wl)

def get_tau(filename=fit2p_filename, chisq_cut=None, flux_cut=None):
    with fits.open(herschel_path+filename) as hdul:
        tau = hdul['solutiontau'].data
        chisq = hdul['chisq'].data
        p70 = hdul['BAND70'].data
    if chisq_cut is not None:
        tau[chisq > chisq_cut] = np.nan
    if flux_cut is not None:
        tau[p70 < flux_cut] = np.nan
    return 10.**tau

def get_wcs(filename=fit2p_filename):
    with fits.open(herschel_path+filename) as hdul:
        w = WCS(hdul[1].header)
    return w


def get_physical_scale(wcs_obj, los_distance_pc):
    ps = misc_utils.get_pixel_scale(wcs_obj)
    return ps.to(u.rad).to_value() * los_distance_pc * u.pc

def get_physical_area_pixel(array, wcs_obj, los_distance_pc):
    num_valid_pixels = np.sum(np.isfinite(array).astype(int))
    print(f"{num_valid_pixels} / {array.size} valid")
    phys_scale = get_physical_scale(wcs_obj, los_distance_pc)
    print(f"physical scale: {phys_scale:.2E}")
    area_per_pixel = phys_scale ** 2
    print(f"area per pixel: {area_per_pixel.to(u.cm*u.cm):.2E}")
    return area_per_pixel.to(u.cm*u.cm)

def get_physical_image_axes(array, wcs_obj, los_distance_pc):
    phys_scale = get_physical_scale(wcs_obj, los_distance_pc).to_value()
    i, j = ((np.arange(x+1) - 0.5 - x//2)*phys_scale for x in array.shape)
    return i, j


def convert_tautomass_k(tau160, k160, gastodust):
    return gastodust * tau160/k160

def convert_tautoN_C(tau160, C160):
    return tau160 / C160

def convert_Ntomass(N):
    return N * Hmass

def make_C_plots(d=None):
    if d is None:
        d = Draine_data(3.1)
    wl = get_wl(d)
    Cext = get_C(d)
    kabs = get_k(d)
    Cabs = convert_ktoC(kabs)
    Cabs160 = get_val_at(160., wl, Cabs)
    a = get_albedo(d)
    # plt.plot(wl, Cext, label='Cext')
    # plt.plot(wl, Cabs, label='Cabs')
    plt.plot(wl, Cext/Cabs, label='ext/abs')
    totalGas_to_H = get_val_at(160., wl, Cext/Cabs)
    plt.plot(wl[a > 0], totalGas_to_H * 1./(1-a[a > 0]), label='1/(1-A) (ext/abs)')
    # plt.plot([160], [Cabs160], 'x', label='Cabs_160')
    plt.legend()
    plt.xscale('log'), plt.yscale('log')
    plt.show()



"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
    Masks
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
"""

def cii_combined_channel_mask(test_mask=False):
    # # Load in CII -12,-8
    # cii_img_blue, cii_w = catalog.utils.load_cii(1)
    # # Load in CII -8,-4
    # cii_img_red, cii_w = catalog.utils.load_cii(2)

    cii_cube = cube_utils.CubeData("sofia/rcw49-cii.fits")
    kms = u.km/u.s

    red_interval = (-6*kms, -5*kms)
    red_str = f"[{red_interval[0]:.0f}, {red_interval[1]:.0f}]"
    cii_img_red = cii_cube.data.spectral_slab(*red_interval).moment0()
    cii_w = cii_img_red.wcs
    cii_img_red = cii_img_red.to_value()

    blue_interval = (-7*kms, -6*kms)
    blue_str = f"[{blue_interval[0]:.0f}, {blue_interval[1]:.0f}]"
    cii_img_blue = cii_cube.data.spectral_slab(*blue_interval).moment0().to_value()

    # This doesn't work great, I should swap in my better convolution function
    cii_img_blue = smooth_image(cii_img_blue, kernel_length=10, std=1.5)
    cii_img_red = smooth_image(cii_img_red, kernel_length=10, std=1.5)
    # Load tau
    tau160, tau160_h = load_tau()
    tau160_w = WCS(tau160_h)
    # Project CII onto tau
    # cii_img_blue, cii_img_red = (reproject_interp((cii_img, cii_w), tau160_w, tau160.shape, return_footprint=False) for cii_img in (cii_img_blue, cii_img_red))
    tau160_cii = reproject_interp((tau160, tau160_w), cii_w, cii_img_red.shape, return_footprint=False)
    mask_vals = []
    masks = []
    for cii_img in (cii_img_blue, cii_img_red):
        cii_img[np.isnan(cii_img)] = 0
        valid_mask = cii_img > 0
        mask_val = np.median(cii_img[valid_mask]) + np.std(cii_img[valid_mask])
        print(mask_val)
        mask_vals.append(mask_val)
        masks.append(cii_img > mask_val)
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(121, projection=cii_w)
    plt.imshow(tau160_cii, origin='lower', cmap='cividis', vmin=-2.6, vmax=-1)
    plt.title(blue_str+" (blue)")
    plt.colorbar()
    lw = 0.4
    plt.contour(cii_img_blue, levels=[mask_vals[0]], colors='cyan', linewidths=lw)
    plt.contour(cii_img_red, levels=[mask_vals[1]], colors='red', linewidths=lw)

    plt.subplot(122, projection=cii_w, sharex=ax, sharey=ax)
    plt.imshow(np.all(masks, axis=0).astype(int), origin='lower', cmap='cividis')#, vmin=-2.6, vmax=-1)
    plt.title(red_str+" (red)")
    plt.colorbar()
    lw = 1.2
    plt.contour(cii_img_blue, levels=[mask_vals[0]], colors='cyan', linewidths=lw)
    plt.contour(cii_img_red, levels=[mask_vals[1]], colors='red', linewidths=lw)
    plt.show()
    # plt.savefig("/home/ramsey/Pictures/10-20-20-work/mask_at_-6_tau_v2.png")


def ellipse_region_mask(shape=None, w=None, test_mask=False, savemask=False,
    half=False):
    if shape is None or w is None:
        # Load in CII map and WCS (we need WCS)
        cii_img, cii_w = catalog.utils.load_cii(2)
        shape = cii_img.shape
        w = cii_w
    # Load the regions file
    reg_sky_list = regions.read_ds9(catalog.utils.search_for_file("catalogs/ellipse_mask.reg"))
    # Convert to pixel regions
    reg_pix_list = [reg.to_pixel(w) for reg in reg_sky_list]
    # Make masks
    mask_list = [reg.to_mask().to_image(shape).astype(bool) for reg in reg_pix_list]
    # Difference the masks
    mask = mask_list[0] ^ mask_list[1]

    if half:
        # Make a half shell; keep only the Eastern (left) half of the shell
        center_j = reg_pix_list[0].center.x
        ii, jj = np.meshgrid(*(np.arange(x) for x in shape), indexing='ij')
        mask &= jj < center_j
    if test_mask:
        plt.imshow(mask, origin='lower')
        plt.show()
    if savemask:
        m = mask.copy()
        plt.imshow(m, origin='lower')
        plt.show()
        hdr = cii_w.to_header()
        hdr['COMMENT'] = 'ellipse mask, tentative, Nov 3, 2020'
        hdu = fits.PrimaryHDU(data=m.astype(float), header=hdr)
        hdu.writeto('/home/ramsey/Downloads/half_ellipse_shell_mask.fits', overwrite=True)
        return
    return mask


"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&% Integration functions %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
"""


def integrate_shell_by_hand():
    """
    This method relies heavily on ds9 "by eye" estimates.
    See the 3/20/20 notes in my notebook for more details.

    The idea is that I sampled a few pieces of the visible shell and modeled it
    radially as a Gaussian, finding (by eye + by ds9 regions + line plotting)
    its Gaussian FWHM->std, its peak, and the approximate background level.
    Maitraiyee found the radius, which I verified looks like it lines up with
    the physical structure in the tau160 map.
    I integrate this radially-dispaced Gaussian around a semicircle and along
    the FWHM radial length. I have made this calculation by hand (in my notes).

    I assume that what I was seeing in the last step was, in reality, the limb
    brightened portion of a half shell (oriented same way as semicircle).
    What I integrated then was like a "cored sphere" with the core radius equal
    to the inner radius of the shell and sphere radius equal to outer radius.
    In the next step, I relate the volume within such a cored sphere to the
    volume of the half shell by a multiplicative factor that I derived by
    hand (in my notes).

    With these two steps, I should be able to find the approximate gas density
    within this structure, given a physical distance scale and assuming the
    shell assumptions are correct.
    """
    # Given or "observed" input values
    losD = 4.16*1000*u.pc # originally from Vargas-Alvarez 2013, used by Zeidler
    r_avg_ang = 310. * u.arcsec # from Maitraiyee
    fwhm_ang = 1.2 * u.arcmin # FWHM, from ds9 line region length + line plot
    peak_plus_bg = 10. ** (-2.12) # tau160, from ds9 line plot
    bg = 10. ** (-2.45) # same as above

    # Calculate r_avg and FWHM in physical units
    r_avg, fwhm = (r.to(u.rad).to_value() * losD for r in (r_avg_ang, fwhm_ang))
    # Calculate r1 and r2 in physical units
    r1, r2 = r_avg - (fwhm/2.), r_avg + (fwhm/2.)
    # Calculate standard deviation of Gaussian approximation in physical units
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Calculate peak value as (linear) optical depth
    peak = peak_plus_bg - bg
    # Calculate Gauss-approx 2D half shell integral using derived expression
    half_2dshell_gauss_integral = peak * ( np.pi ) * sigma * np.sqrt(2 * np.pi) * r_avg * erf(np.sqrt(np.log(2)))
    # Assume the above 2D area now equals
    # Calculate factor to convert from limb-brightened volume (cored sphere) to shell volume
    coredsph_to_shell = (r2**3. - r1**3.) / (r2**2. - r1**2.)**(3./2)
    # Calculate the 3D shell volume
    half_3dshell_integral = half_2dshell_gauss_integral * coredsph_to_shell
    # Units right now are tau160 * <length^2>

    # Convert to physical mass
    # Load in Draine beta ~ 2.0 (Rv 3.1) model
    Rv = 3.1
    d = Draine_data(Rv)
    wl = get_wl(d)
    Cext = get_C(d)
    kabs = get_k(d)
    Cabs = convert_ktoC(kabs)
    Cabs160 = get_val_at(160., wl, Cabs) * u.cm*u.cm
    kabs160 = get_val_at(160., wl, kabs) * u.cm*u.cm / u.g
    print(f"kabs160 / lambda^2 {kabs160 * 160**2:.2E}")
    # Convert tau160 to mass/<length^2> (tau160*<length^2> to mass in this case)
    mass = convert_tautomass_k(half_3dshell_integral.to(u.cm*u.cm), kabs160).to(u.solMass)
    print(f"Assuming l.o.s. distance of {losD:.2f}, Rv={Rv:.1f}")
    print(f"Shell of radius {r_avg:.2f} and average thickness {fwhm:.2f}")
    print(f"Shell mass of {mass:.2E}")
    # Convert the integral to total number of H atoms and get column density thru shell
    totalNH = convert_tautoN_C(half_3dshell_integral.to(u.cm*u.cm), Cabs160)
    NH = totalNH / (2 * np.pi * r_avg.to(u.cm)**2)
    print(f"Total number of H atoms: {totalNH:.2E}, implying N(H) = {NH:.2E}")


def integrate_shell_on_image(use_background=False, plot_anything=False):
    """
    This used to use the "geometric_model" module; see commits prior to
    Nov 3 2020 to see that version
    This is now the LATEST version of the mass estimate, using a half ellipse
    I use a geometric correction of 7/6 to account for the azimuthal extent
    I also use the same geometric correction as before (I should justify that)
    I will also plot the mask here, against the column density map
    Feb 1, 2021: improved to zoom more on the shell, per ref's comments
    """
    Rv = 3.1
    d = Draine_data(Rv)
    wl = get_wl(d)
    Cext = get_C(d)
    kabs = get_k(d)
    # k will produce slightly higher mass since g-to-d includes He
    # Albedo is 0 so ext = abs (no sca)
    Cext160 = get_val_at(160., wl, Cext)
    # Distance is the Vargas-Alvarez value
    losD = 4.16*1000

    tau160, tau160_h = load_tau()
    tau160 = 10.**tau160
    tau160_w = WCS(tau160_h)
    tau160_background = 10**(-2.6) # np.nanmedian(tau160) is really high
    if use_background:
        # Subtract an optical depth background
        # Has a factor of ~2 effect on the mass
        print(f"Using a tau160 background of {tau160_background:.2E}")
        tau160 -= tau160_background

    # mask to just the shell
    ellipse_mask = ellipse_region_mask(shape=tau160.shape, w=tau160_w, half=1)
    tau160_copy = tau160.copy()
    tau160[~ellipse_mask] = np.nan

    N = convert_tautoN_C(tau160, Cext160)
    pixel_area = get_physical_area_pixel(tau160, tau160_w, losD)
    total_mass = pixel_area * np.sum(N[np.isfinite(N)]) * Hmass * u.g / (u.cm**2)
    print(f"Mass under mask: {total_mass.to('solMass'):.3E}")
    geometric_correction_3d = 2.5
    extended_shell_correction = 7./6
    total_mass *= geometric_correction_3d
    print(f"Mass with 3D geometric correction only: {total_mass.to('solMass'):.3E}")
    print(f"Total corrected mass: {(total_mass*extended_shell_correction).to('solMass'):.3E}")

    print()
    A = 77*u.pc**2
    print(f"If the quarter-spheroid surface area is {A:.1f}, then")
    N_thru_shell = (total_mass / A / u.M_p).to(u.cm**-2)
    N_Av = 1.9e21*u.cm**-2
    print(f"Column density thru shell {N_thru_shell:.2E}, Av {(N_thru_shell/N_Av).decompose().to_value():.1f}")

    print("Column density samples:")
    tau160_sample_1 = 10.**(-2.2) * u.cm**-2
    tau160_sample_2 = 10.**(-1.6) * u.cm**-2
    tau160_background = tau160_background * u.cm**-2
    N_sample_1 = convert_tautoN_C(tau160_sample_1, Cext160)
    N_sample_2 = convert_tautoN_C(tau160_sample_2, Cext160)
    N_background = convert_tautoN_C(tau160_background, Cext160)
    print(f"Low: {N_sample_1:.2E}, Av {(N_sample_1/N_Av).decompose().to_value():.1f}")
    print(f"High: {N_sample_2:.2E}, Av {(N_sample_2/N_Av).decompose().to_value():.1f}")
    print(f"Background: {N_background:.2E}, Av {(N_background/N_Av).decompose().to_value():.1f}")

    # extent_arrays = get_physical_image_axes(N, tau160_w, losD)
    # ext = (extent_arrays[1][0], extent_arrays[1][-1], extent_arrays[0][0], extent_arrays[0][-1])
    # print(ext)

    if plot_anything:
        cmap = 'inferno'
        cbar_tick_labelsize = 12
        cbar_labelsize = 12
        cbar_labelpad = -70
        T, T_h = load_T()
        fig = plt.figure(figsize=(16, 9))
        axT = plt.subplot(121, projection=tau160_w)
        caxT = inset_axes(axT, width="100%", height="5%", loc='lower center',
            bbox_to_anchor=(0, 1.01, 1, 1), bbox_transform=axT.transAxes, borderpad=0)
        im = axT.imshow(T, origin='lower', vmin=15, vmax=55, cmap=cmap)
        cbarT = fig.colorbar(im, cax=caxT, orientation='horizontal',
            ticks=[20, 30, 40, 50])
        cbarT.set_label("Dust temperature (K)", labelpad=cbar_labelpad, fontsize=cbar_labelsize)
        caxT.xaxis.set_ticks_position('top')
        caxT.tick_params(labelsize=cbar_tick_labelsize)
        axT.tick_params(axis='x', direction='in')
        axT.tick_params(axis='y', direction='in')
        # cbarT.outline.set_edgecolor("white")
        axT.set_xlabel("Right Ascension")
        axT.set_ylabel("Declination")

        axN = plt.subplot(122, projection=tau160_w)
        caxN = inset_axes(axN, width="100%", height="5%", loc='lower center',
            bbox_to_anchor=(0, 1.01, 1, 1), bbox_transform=axN.transAxes, borderpad=0)
        N = np.log10(convert_tautoN_C(tau160_copy, Cext160))
        im = axN.imshow(N, origin='lower', vmin=21.5, vmax=23.5, cmap=cmap)
        cbarN = fig.colorbar(im, cax=caxN, orientation='horizontal',
            ticks=[22, 22.5, 23])
        caxN.set_xticklabels(['22', '23.5', '23'])
        cbarN.set_label("Hydrogen nucleus column density [cm$^{-2}$]", labelpad=cbar_labelpad-2, fontsize=cbar_labelsize)
        caxN.xaxis.set_ticks_position('top')
        caxN.tick_params(labelsize=cbar_tick_labelsize)
        axN.tick_params(axis='x', direction='in')
        axN.tick_params(axis='y', direction='in')
        axN.set_xlabel(" ")
        axN.tick_params(axis='y', labelleft=False)
        # axN.tick_params(axis='x', labelbottom=False)

        # inset_img = Cutout2D(N, (600, 500), (400, 300), wcs=tau160_w) # 300:700, 450:750 (pixel ranges)
        inset_img = Cutout2D(N, (585, 515), (280, 280), wcs=tau160_w) # modified because ref wanted more zoom (feb 1, 2021)
        ellipse_mask = ellipse_region_mask(shape=inset_img.shape, w=inset_img.wcs, half=1)

        mask_contour = ellipse_mask.astype(float)
        mask_contour_kw = dict(linewidths=1, colors='k', levels=[0.5], alpha=0.9)
        def add_mask_contour(ax):
            ax.contour(mask_contour, **mask_contour_kw)


        insetaxkwargs = dict(width="42%", height="42%", loc='upper right')
        # insetaxN = inset_axes(axN, width="36%", height="48%", loc='upper right')
        insetaxN = inset_axes(axN, **insetaxkwargs) # modified because ref wanted more zoom (feb 1, 2021)
        insetaxN.tick_params(axis='x', labelbottom=False)
        insetaxN.tick_params(axis='y', labelleft=False)
        insetaxN.set_xticks([])
        insetaxN.set_yticks([])
        N = N[inset_img.slices_original]
        insetaxN.imshow(N, origin='lower', vmin=21.5, vmax=23.5, cmap=cmap)
        add_mask_contour(insetaxN)

        insetaxT = inset_axes(axT, **insetaxkwargs)
        insetaxT.tick_params(axis='x', labelbottom=False)
        insetaxT.tick_params(axis='y', labelleft=False)
        insetaxT.set_xticks([])
        insetaxT.set_yticks([])
        T = T[inset_img.slices_original]
        insetaxT.imshow(T, origin='lower', vmin=15, vmax=55, cmap=cmap)
        add_mask_contour(insetaxT)

        plt.subplots_adjust(wspace=0, bottom=0, top=0.92, left=0.1, right=0.95)
        fig.savefig("/home/ramsey/Pictures/2021-02-01-imgs/dust_mask_zoomed_4.png")
        # plt.show()


def integrate_shell_cii_mask(n=2, test_mask=False, use_background=False, plot_anything=False, Rv=3.1,
    savemask=False):
    """
    Use a [CII]-based mask to integrate the dust optical depth across the image
    and come up with a mass estimate
    This could have a geometrical factor applied to it
    """
    # Load in CII
    cii_img, cii_w = catalog.utils.load_cii(n)
    # This doesn't work great, I should swap in my better convolution function
    cii_img = smooth_image(cii_img, kernel_length=10, std=1.5)
    # Load and prepare tau
    tau160, tau160_h = load_tau()
    tau160 = 10.**tau160
    tau160_w = WCS(tau160_h)
    if use_background:
        # Subtract an optical depth background
        # Has a factor of ~2 effect on the mass
        tau160_background = 10**(-2.6) # np.nanmedian(tau160) is really high
        print(f"Using a tau160 background of {tau160_background:.2E}")
        tau160 -= tau160_background
    # Project CII onto tau
    cii_new, fp = reproject_interp((cii_img, cii_w), tau160_w, tau160.shape)
    cii_new[np.isnan(cii_new)] = 0

    # Mask value; needs to be sort of hand-picked
    mask_val = {1: 2e4, 2: 2e4, 3: 4.5e4}[n]
    mask_val = np.median(cii_new[cii_new > 0]) + np.std(cii_new[cii_new > 0])
    print("Mask value:", mask_val)
    label_stub = {1: "-12,-8", 2: "-8,-4", 3: "-25,0"}[n]

    if savemask:
        m = cii_img > mask_val
        plt.imshow(m, origin='lower')
        plt.show()
        hdr = cii_w.to_header()
        hdr['COMMENT'] = f'cii mask from [{label_stub}] km/s'
        hdu = fits.PrimaryHDU(data=m.astype(float), header=hdr)
        hdu.writeto('/home/ramsey/Downloads/cii_shell_mask.fits')
        return

    if plot_anything:
        # Show the tau map, the CII map reprojected onto tau, and an example mask
        plt.figure(figsize=(13, 5))
        plt.subplot(121 if not test_mask else 111, projection=tau160_w)
        plt.imshow(cii_new, origin='lower')
        plt.title(f"[CII] [{label_stub}] km/s")
        plt.colorbar()
        plt.contour(cii_new, levels=[mask_val], colors='w', linewidths=0.4)

    if test_mask:
        plt.show()
        return

    # Load in useful values for calculating the mass
    d = Draine_data(Rv)
    wl = get_wl(d)
    Cext = get_C(d)
    kabs = get_k(d)
    # Albedo is 0 so ext = abs (no sca)
    Cext160 = get_val_at(160., wl, Cext)
    print(f"Cext/H(160um) = {Cext160:.2E}")
    # k will produce slightly higher mass since g-to-d includes He
    kabs160 = get_val_at(160., wl, kabs)
    # Distance is the Vargas-Alvarez value
    losD = 4.16*1000

    if plot_anything:
        # Plot the CII-contour-on-tau figure as well as the masked tau image
        plt.subplot(122, projection=tau160_w)
        plt.imshow(tau160, origin='lower', vmin=0.004, vmax=0.033)
        plt.colorbar()
        plt.contour(cii_new, levels=[mask_val], colors='w', linewidths=0.4)

    # Apply the mask and calculate the mass
    tau160[cii_new < mask_val] = np.nan
    pixel_area = get_physical_area_pixel(tau160, tau160_w, losD)

    mass_da = convert_tautomass_k(tau160, kabs160, Draine_gastodust(Rv)) * u.g / (u.cm*u.cm)
    N = convert_tautoN_C(tau160, Cext160) / (u.cm*u.cm)
    total_mass_k = pixel_area * np.sum(mass_da[np.isfinite(mass_da)])
    total_mass_C = pixel_area * np.sum(N[np.isfinite(N)]) * Hmass * u.g

    # print(f"from kappa: {total_mass_k.to('solMass'):.3E}")
    # print(f"from C / H: {total_mass_C.to('solMass'):.3E}")
    # print(f"Ratio: {total_mass_k/total_mass_C}")

    total_mass = total_mass_C

    print(f"{total_mass.to('solMass'):.3E}")

    if plot_anything:
        plt.title(f"tau_160 w/ mask contour. Mass: {total_mass.to('solMass'):.2E}")

    # Done! Show the plots
    plt.show()
    # plt.savefig(f"./figures/circa_may10_2020/dustmass_ciimask{label_stub}.png")



"""
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
    Visualization
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
"""



def load_tau():
    """
    Load the image and header
    """
    with fits.open(f"{herschel_path}{fit2p_filename}") as hdul:
        img = hdul['solutiontau'].data
        h = hdul['solutiontau'].header
    return img, h


def load_T():
    """
    Load the image and header
    """
    with fits.open(f"{herschel_path}{fit2p_filename}") as hdul:
        img = hdul['solutionT'].data
        h = hdul['solutionT'].header
    return img, h



def make_cii_mom0():
    """
    Make & save these moment 0 maps:
    0) entire velocity range
    1) -12 to -8 km/s
    2) -8 to -4 km/s
    3) -25 to 0 km/s (added 5/12/20)
    Numbers 1 & 2 are based on my 4/16/20 meeting with Xander and Maitraiyee
    Maitraiyee's R-B uses these two ranges. Xander specifically suggested
        -8 to -4 km/s as a mask for the dust optical depth.
    Number 3 is based on a conversation with Maitraiyee on 5/12/20 where she
        said it would be most correct to use -25 to 0 km/s as a mask

    I should manually set the boolean "switches" here to make sure I'm not
        overwriting files or taking time writing.
    """
    if True:
        raise RuntimeError("You already made these files on 4/22/20 on your laptop!")
    # Load the cube
    with fits.open(cii_cube) as hdul:
        h = hdul[0].header
        w = WCS(h)
        wflat = WCS(h, naxis=2)
        data = hdul[0].data * u.K / (u.m / u.s)
    cube = SpectralCube(data=data, wcs=w)
    if False:
        # First, save the whole moment 0 map
        img1 = cube.moment(order=0)
        h1 = fits.Header()
        h1.update(wflat.to_header())
        h1['COMMENT'] = "Full velocity-range moment 0 map from [CII]"
        h1['BUNIT'] = 'K'
        fits.writeto(f"{catalog.utils.cii_path}mom0_fullrange.fits", img1.to_value(), h1)
        del img1, h1
    if False:
        # Now make a subcubes from -12 to -8 and -8 to -4 km/s
        img2 = cube.spectral_slab(-12*u.km/u.s, -8*u.km/u.s).moment(order=0)
        h2 = fits.Header()
        h2.update(wflat.to_header())
        h2['COMMENT'] = "[CII] moment 0 from -12 km/s to -8 km/s"
        h2['BUNIT'] = 'K'
        fits.writeto(f"{catalog.utils.cii_path}mom0_-12to-8.fits", img2.to_value(), h2)
        del img2, h2
    if False:
        # And again for -8 to -4
        img2 = cube.spectral_slab(-8*u.km/u.s, -4*u.km/u.s).moment(order=0)
        h2 = fits.Header()
        h2.update(wflat.to_header())
        h2['COMMENT'] = "[CII] moment 0 from -8 km/s to -4 km/s"
        h2['BUNIT'] = 'K'
        fits.writeto(f"{catalog.utils.cii_path}mom0_-8to-4.fits", img2.to_value(), h2)
        del img2, h2
    if False:
        # And again for -25 to 0
        img2 = cube.spectral_slab(-25*u.km/u.s, -0*u.km/u.s).moment(order=0)
        h2 = fits.Header()
        h2.update(wflat.to_header())
        h2['COMMENT'] = "[CII] moment 0 from -25 km/s to 0 km/s"
        h2['BUNIT'] = 'K'
        fits.writeto(f"{catalog.utils.cii_path}mom0_-25to0.fits", img2.to_value(), h2)
    print("all done")


def smooth_image(img, kernel_length=20, std=2):
    """
    Smooths a map with a kernel with length kernel_length and
    standard deviation std
    """
    img = img.copy()
    # Smooth spatial axes to avoid rough countours
    smooth_kernel = gaussian(kernel_length, std)
    smooth_kernel /= np.sum(smooth_kernel)
    for i in range(img.shape[0]):
        img[i, :] = np.convolve(img[i, :], smooth_kernel, mode='same')
    for j in range(img.shape[1]):
            img[:, j] = np.convolve(img[:, j], smooth_kernel, mode='same')
    return img


def prepare_img_for_plot(img, scale=np.arcsinh, low_cutoff=np.nanmedian):
    img[img < low_cutoff(img)] = low_cutoff(img)
    if scale is not None:
        img = scale(img)
    return img




if __name__ == "__main__":
    # ellipse_region_mask(savemask=True, half=True)
    # print("CII")
    # integrate_shell_cii_mask(n=2)
    print("ELLIPSE")
    integrate_shell_on_image(plot_anything=True)
    # m1_old, m2_old = 2.42e4*u.solMass, 4.02e4*u.solMass
    # m1_new, m2_new = 1.79e4*u.solMass, 2.82e4*u.solMass
    # print(m1_new/m1_old)
    # print(m2_new/m2_old)
