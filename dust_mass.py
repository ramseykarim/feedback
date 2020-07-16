import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.signal import gaussian

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from spectral_cube import SpectralCube
from reproject import reproject_interp

from .mantipython import physics
from . import misc_utils
from . import catalog
from . import geometric_model as geomodel


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
fit2p_filename = "RCW49large_2p_2BAND_beta2.0.fits"

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


def integrate_shell_on_image():
    d = Draine_data(3.1)
    wl = get_wl(d)
    Cext = get_C(d)
    kabs = get_k(d)
    Cabs = convert_ktoC(kabs)
    Cabs160 = get_val_at(160., wl, Cabs)
    kabs160 = get_val_at(160., wl, kabs)

    losD = 4.16*1000

    w = get_wcs(filename="RCW49large_2p.fits")
    # tau = get_tau(filename="RCW49large_2p.fits", chisq_cut=7.0, flux_cut=2e3)
    tau = get_tau(filename="RCW49large_2p.fits")

    estimated_center_pixel = (103, 138)
    radius_deg = 0.09
    thickness_deg = 0.05
    pixel_scale_deg = misc_utils.get_pixel_scale(w).to_value()

    # mask to just the shell
    half_shell_mask = geomodel.half_shell_mask_2d(tau, pixel_scale_deg, radius_deg, thickness_deg, estimated_center_pixel, ang=70)
    tau[~half_shell_mask] = np.nan

    N = convert_tautoN_C(tau, Cabs160)
    mass = convert_tautomass_k(tau, kabs160) * u.g / (u.cm*u.cm)
    pixel_area = get_physical_area_pixel(tau, w, losD)
    total_mass = pixel_area * np.sum(mass[np.isfinite(mass)])
    print(f"{total_mass.to('solMass'):.3E}")

    extent_arrays = get_physical_image_axes(N, w, losD)
    ext = (extent_arrays[1][0], extent_arrays[1][-1], extent_arrays[0][0], extent_arrays[0][-1])
    print(ext)

    plt.imshow(np.log10(N), origin='lower', extent=ext)
    plt.title("column density N(H) (cm-2)")
    plt.colorbar()
    plt.show()


def integrate_shell_cii_mask(n=2, test_mask=False, use_background=False, plot_anything=False, Rv=3.1):
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
    mask_val = {2: 2e4, 3: 4.5e4}[n]
    label_stub = {2: "-8,-4", 3: "-25,0"}[n]

    if plot_anything:
        # Show the tau map, the CII map reprojected onto tau, and an example mask
        plt.figure(figsize=(13, 5))
        plt.subplot(121, projection=tau160_w)
        plt.imshow(cii_new, origin='lower')
        plt.title(f"[CII] [{label_stub}] km/s")
        plt.colorbar()

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
    integrate_shell_cii_mask(n=2, Rv=3.1, plot_anything=1, use_background=False)
    integrate_shell_cii_mask(n=3, Rv=3.1, plot_anything=1, use_background=False)
