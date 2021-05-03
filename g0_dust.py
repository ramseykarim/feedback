import numpy as np
import matplotlib.pyplot as plt
import sys

import scipy.constants as cst
from scipy.special import factorial, zeta

from astropy.wcs import WCS
from astropy import units as u
from astropy.modeling import models
from astropy.io import fits

from reproject import reproject_interp

from .parse_FIR_fits import open_FIR_pickle, open_FIR_fits, herschel_path
from . import catalog
from . import misc_utils
"""
Currently unknown creation date (while back though)

Updated April 29, 2021 to get a L_FIR map to Maitraiyee
I am following Goicoechea 2015's prescription for L_FIR (40-500 um)
F[W m-2 Hz-1] = B(T) * (1 - e^-tau) * (solid angle per pixel)
"""

# Laptop directory
filename = "herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits"; prefix='solution'
# filename = "herschel/colorsoln_1.fits"; prefix=''
filename = catalog.utils.search_for_file(filename)

herschel_ref_filename = catalog.utils.search_for_file("herschel/RCW49large_2p_2BAND_160grid_beta2.0.fits")
cii_filename = catalog.utils.search_for_file("sofia/mom0_fullrange.fits")


"""
I need to implement this equation for G0:

G0 =
    (1 / Q_abs_avgFUV) *
    (4*15 / pi**5) *
    (beta + 3)! *
    RiemannZeta(beta + 4) *
    Q0 *
    (lambda0 * k_B * T_dust / hc)**beta *
    Stefan-Boltzmann * T_dust**4
for output in W m-2 or erg cm-2 s-1
where Q0 is 1 until further notice
beta, Td are from dust SED fit solutions
Q_abs_avgFUV is the averge Q_abs = C_abs / sigma_dust over the FUV absorption
    range that matters here
"""

def calculate_g0():
    result_dict = open_FIR_fits(filename)
    # T, tau, beta = (result_dict[x] for x in ('T', 'tau' ,'beta'))
    T, tau = (result_dict[x] for x in ('T', 'tau'))
    beta = 2.



    a = (0.2 * u.micron).to(u.cm) # micron, grain size, from Xander conversation, range between 0.1-3

    grain_density = 3. * u.g / (u.cm ** 3) # g/cm2, according to Xander (notebook pg 111)
    avg_kappa_abs_FUV = 3.e4 * (u.cm ** 2) / u.g # cm2/g, based on the plots from plot_dust()
    m_grain = grain_density * (np.pi * 4./3) * a**3.
    xr_grain = np.pi * a**2.
    avg_Cabs_FUV = avg_kappa_abs_FUV * m_grain
    avg_Qabs_FUV = avg_Cabs_FUV / xr_grain

    Q0 = 1
    k = u.Quantity(cst.k, cst.unit('Boltzmann constant'))
    h = u.Quantity(cst.h, cst.unit('Planck constant'))
    c = u.Quantity(cst.c, cst.unit('speed of light in vacuum'))
    hc = h*c
    sb = u.Quantity(cst.value('Stefan-Boltzmann constant'), cst.unit('Stefan-Boltzmann constant'))
    Td = T * u.K
    lambda0 = 2 * np.pi * a.to(u.m)

    G0 = (1 / avg_Qabs_FUV) * (4*15/(np.pi**5)) * factorial(beta+3) * zeta(beta+4) * Q0 * ((lambda0*k/hc)*Td)**beta * sb * Td**4.
    Habing = u.Quantity(1.6e-3, 'erg cm-2 s-1')
    G0 = G0.to('erg cm-2 s-1') / Habing
    return G0

def make_plot_g0():
    G0 = calculate_g0()
    result_dict = open_FIR_fits(filename)
    obs70 = result_dict['BAND70']
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.log10(obs70), origin='lower', vmin=1, vmax=5)
    plt.title("70um")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(G0.value, origin='lower', cmap='cividis', vmin=0, vmax=25)
    plt.colorbar()
    plt.title("G0")
    plt.show()


"""
Dust opacity vs wavelength plot that I probably might use again later
"""

def plot_dust():
    ### Summary: indicates dust kappa_abs ~ 2-5 * 10^4 cm2/g
    # Load Ormel dust
    # Table structure:
    #cols::[lambda,kappa-ext,kappa-sca,kappa-back,gsca]
    #colunits::[cm,cm2/g,cm2/g,cm2/g,]
    opacities_dir = "/home/ramsey/Documents/Research/Filaments/Dust/OpacTables/"
    def gen_ormel_fn(filename_stub, number):
        fn = f"{opacities_dir}{filename_stub}{number}opc.txt"
        return fn

    dust_types = [
        "(ic-sil,gra)",
        "ic-sil+gra",
        "(ic-sil,ic-gra)",
        "(sil,gra)"
    ]
    times = {
        1: "0.03",
        2: "0.1",
        3: "0.3",
        4: "1.0",
        5: "3.0",
        6: "10.",
    }
    selected_age = 5
    dt = dust_types[3]
    # for dt in dust_types:
    for selected_age in [4]:
        a = np.genfromtxt(gen_ormel_fn(dt, selected_age))
        wl_um = a[:, 0] * 1e4
        kext = a[:, 1]
        ksca = a[:, 2]
        kabs = kext - ksca
        plt.plot(wl_um, kext, '-', label="Ormel "+dt+f" {times[selected_age]}ext")
        plt.plot(wl_um, ksca, '-', label="Ormel "+dt+f" {times[selected_age]}sca")
        plt.plot(wl_um, kabs, '-', label="Ormel "+dt+f" {times[selected_age]}abs")
    plt.xscale('log'), plt.yscale('log')
    plt.legend()
    plt.show()


def fir_intensity():
    with fits.open(filename) as hdul:
        T = hdul[prefix+'T'].data
        tau = hdul[prefix+'tau'].data
        original_w = WCS(hdul[prefix+'T'].header)
    w = original_w

    with fits.open(herschel_ref_filename) as hdul:
        original_w = WCS(hdul[1].header)

    cii_data, cii_header = fits.getdata(cii_filename, header=True)
    cii_w = WCS(cii_header)

    T = reproject_interp((T, original_w), cii_w, cii_data.shape, return_footprint=False)
    tau = reproject_interp((tau, original_w), cii_w, cii_data.shape, return_footprint=False)
    w = cii_w


    # plt.subplot(121)
    # plt.imshow(T, origin='lower', vmin=20, vmax=50)
    # plt.subplot(122)
    # plt.imshow(tau, origin='lower', vmin=-2.7, vmax=-1.5)
    # plt.show()
    # return


    bb = models.BlackBody(T[:, :, np.newaxis]*u.K)
    wl_lims = np.array([40., 500.]) * u.micron
    nu_lims = wl_lims.to(u.Hz, equivalencies=u.spectral())
    nu_array = np.linspace(nu_lims[1].to_value(), nu_lims[0].to_value(), 1000) * u.Hz
    wl_array = nu_array.to(u.um, equivalencies=u.spectral())
    S_array = bb(nu_array[np.newaxis, np.newaxis, :])
    tau_array = (10.**tau[:, :, np.newaxis]) * (nu_array[np.newaxis, np.newaxis, :] / (160*u.micron).to(u.Hz, equivalencies=u.spectral()))**2.
    I_array = S_array * (1. - np.exp(-tau_array))
    F_array = np.trapz(x=nu_array, y=I_array).to('erg s-1 cm-2 sr-1')

    # pixel_area = misc_utils.get_pixel_scale(w)**2.
    # F_array = F_array * pixel_area
    # L_FIR = (4. * np.pi * (4.16*u.kpc)**2. * F_array).to('erg s-1 sr-1')

    header_kws = dict(COMMENT='I_FIR map of RCW 49 from 70 and 160 micron',
        HISTORY='from same SED fits as published paper, beta=2',
        DATE='April 30, 2021', CREATOR='Ramsey Karim', BUNIT='erg s-1 cm-2 sr-1')
    new_hdr = w.to_header()
    new_hdr.update(header_kws)
    hdu = fits.PrimaryHDU(data=F_array.to_value(), header=new_hdr)
    hdu.writeto(catalog.utils.feedback_path + 'rcw49_data/herschel/rcw49-I_FIR.fits', overwrite=True)

    plt.imshow(F_array.to_value(), origin='lower')
    plt.show()


def check_I_FIR_G0():
    I_FIR_filename = catalog.utils.search_for_file("herschel/rcw49-I_FIR.fits")
    I_FIR, hdr = fits.getdata(I_FIR_filename, header=True)
    g0_fir = 0.5 * I_FIR / 1.3e-4
    plt.imshow(g0_fir, origin='lower')
    plt.show()


if __name__ == "__main__":
    check_I_FIR_G0()
