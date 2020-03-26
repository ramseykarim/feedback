import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
from scipy.interpolate import interp1d
from scipy.special import erf
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

from mantipython import physics
import geometric_model as geomodel


ancillary_data_path = "/home/rkarim/Research/Feedback/ancillary_data/"
if not os.path.isdir(ancillary_data_path):
    ancillary_data_path = "/home/ramsey/Documents/Research/Feedback/ancillary_data/"
dust_path = f"{ancillary_data_path}dust/"
herschel_path = f"{ancillary_data_path}herschel/"

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


H_mass_amu = 1.00794
Hmass = cst.m_u * H_mass_amu * 1e3 # kg->g
gastodust = 123.6

def convert_ktoC(k):
    # (assuming gas == H)
    return k*Hmass / gastodust # dust cm2 / H particle

def get_val_at(wl, wl_array, val_array):
    return interp1d(wl_array, val_array)(wl)

fit2p_filename = "RCW49large_2p.fits"

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

def get_pixel_scale(wcs_obj):
    ps = np.mean(np.abs(np.diag(wcs_obj.pixel_scale_matrix))) * u.deg
    print(f"pixel scale: {ps.to(u.arcsec):.2E}")
    return ps

def get_physical_scale(wcs_obj, los_distance_pc):
    ps = get_pixel_scale(wcs_obj)
    return ps.to(u.rad).to_value() * los_distance_pc * u.pc

wd2_center_coord = SkyCoord("10 23 58.1 -57 45 49", unit=(u.hourangle, u.deg))

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


def convert_tautomass_k(tau160, k160):
    return gastodust * tau160/k160

def convert_tautoN_C(tau160, C160):
    return tau160 / C160

def convert_Ntomass(N):
    return N * Hmass

def make_C_plots(d):
    d = Draine_data(3.1)
    wl = get_wl(d)
    Cext = get_C(d)
    kabs = get_k(d)
    Cabs = convert_ktoC(kabs)
    Cabs160 = get_val_at(160., wl, Cabs)
    plt.plot(wl, Cext, label='Cext')
    plt.plot(wl, Cabs, label='Cabs')
    plt.plot([160], [Cabs160], 'x', label='Cabs_160')
    plt.legend()
    plt.xscale('log'), plt.yscale('log')
    plt.show()


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
    half_2dshell_gauss_integral = peak * np.pi * sigma * np.sqrt(2 * np.pi) * r_avg * erf(np.sqrt(np.log(2)))
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
    pixel_scale_deg = get_pixel_scale(w).to_value()

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

if __name__ == "__main__":
    integrate_shell_by_hand()
