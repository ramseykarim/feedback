import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
from scipy.interpolate import interp1d
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
gastodust = 100. #123.6

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


if __name__ == "__main__":
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
    half_shell_mask = geomodel.shell_mask_2d(tau, pixel_scale_deg, radius_deg, thickness_deg, estimated_center_pixel, ang=70)
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
