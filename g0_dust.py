import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import datetime

import scipy.constants as cst
from scipy.special import factorial, zeta
from scipy.interpolate import UnivariateSpline

from astropy.wcs import WCS
from astropy import units as u
from astropy.modeling import models
from astropy.io import fits
from astropy.nddata.utils import Cutout2D

from reproject import reproject_interp

from .parse_FIR_fits import open_FIR_pickle, open_FIR_fits, herschel_path
from . import catalog
from . import misc_utils

from .mantipython.physics import greybody, dust, instrument
"""
Currently unknown creation date (while back though)

Updated April 29, 2021 to get a L_FIR map to Maitraiyee
I am following Goicoechea 2015's prescription for L_FIR (40-500 um)
F[W m-2 Hz-1] = B(T) * (1 - e^-tau) * (solid angle per pixel)

Updated October 7, 2022 to use the direct-calculation 70/160 method to find
temperature and optical depth (for use in M16)
"""

# Laptop directory
# filename = "herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits"; prefix='solution'
filename = "herschel/M16_2p_3BAND_beta2.0.fits"; prefix='solution'
# filename = "herschel/colorsoln_1.fits"; prefix=''
filename = catalog.utils.search_for_file(filename)

# herschel_ref_filename = catalog.utils.search_for_file("herschel/RCW49large_2p_2BAND_160grid_beta2.0.fits")
herschel_ref_filename = filename
# cii_filename = catalog.utils.search_for_file("sofia/mom0_fullrange.fits")
cii_filename = catalog.utils.search_for_file("sofia/M16_CII_mom0.fits")


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
    plt.imshow(G0.value, origin='lower', cmap='cividis', vmin=0, vmax=3)
    plt.colorbar()
    plt.title("G0 from an equation")
    # plt.show()
    plt.savefig("/home/rkarim/Pictures/2021-09-09-work/g0_fir_equation.png")


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
    """
    Reused September 9, 2021 for M16
    """
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

    header_kws = dict(COMMENT='I_FIR map of M16 from 70-160-250 micron',
        HISTORY='beta=2',
        DATE='September 9, 2021', CREATOR='Ramsey Karim', BUNIT='erg s-1 cm-2 sr-1')
    new_hdr = w.to_header()
    new_hdr.update(header_kws)
    hdu = fits.PrimaryHDU(data=F_array.to_value(), header=new_hdr)
    hdu.writeto(catalog.utils.feedback_path + 'm16_data/herschel/results/m16-I_FIR.fits', overwrite=True)

    plt.imshow(F_array.to_value(), origin='lower')
    plt.show()


def fir_intensity_2():
    """
    October 7, 2022
    Use the direct calculation method for 70 + 160 micron to find T, tau
    This is the method we used for RCW 49 in the paper.
    It will yield maps of 160 micron resolution, which is like 13ish (~11x15),
    rather than the 250 micron resolution of like 18'' which is what I had before

    per the file fit_FIR.py:
        p70_correction = 268
        p160_correction = 1055

    Then, use the T-tau solution to integrate intensity between 40 and 500 micron
    for use in PDRToolbox (following the method of fir_intensity())
    """
    # Copying this part from fir_intensity() (function above)
    solution_path = catalog.utils.search_for_file("herschel/T-tau_colorsolution.fits")
    with fits.open(solution_path) as hdul:
        T_img = hdul['T'].data
        tau_img = hdul['tau'].data
        hdr = hdul['T'].header
        original_w = WCS(hdr)
    # Extract subimage centered at (XY) (2867, 1745) with width (XY) (1192, 873)
    # try again with quarter width, lower Y
    # Cutout2D uses XY for position and YX for size
    center = (2867, 1745-30)
    size = (1192//4, 873//4)
    T_img_cutout = Cutout2D(T_img, center, size[::-1], wcs=original_w)
    T_img = T_img_cutout.data
    tau_img = tau_img[T_img_cutout.slices_original]
    new_w = T_img_cutout.wcs

    # plt.subplot(121, projection=new_w)
    # plt.imshow(T_img, origin='lower', vmin=15, vmax=35)
    # plt.subplot(122, projection=new_w)
    # plt.imshow(np.log10(tau_img), origin='lower', vmin=-3.5, vmax=-1.5)
    # plt.show()
    # return

    bb = models.BlackBody(T_img[:, :, np.newaxis]*u.K)
    wl_lims = np.array([40., 500.]) * u.micron
    nu_lims = wl_lims.to(u.Hz, equivalencies=u.spectral())
    nu_array = np.linspace(nu_lims[1].to_value(), nu_lims[0].to_value(), 1000) * u.Hz
    S_array = bb(nu_array[np.newaxis, np.newaxis, :])
    # tau(nu) = tau_160 * (nu / nu_160)**2 (using beta = 2)
    tau_array = tau_img[:, :, np.newaxis] * (nu_array[np.newaxis, np.newaxis, :] / (160*u.micron).to(u.Hz, equivalencies=u.spectral()))**2.
    I_array = S_array * (1. - np.exp(-tau_array))
    F_array = np.trapz(x=nu_array, y=I_array).to('erg s-1 cm-2 sr-1')

    # plt.subplot(111, projection=new_w)
    # plt.imshow(F_array.to_value(), origin='lower')
    # plt.show()

    hdr.update(new_w.to_header())
    hdr['HISTORY'] = f"using {solution_path}"
    hdr['HISTORY'] = f"cutout center XY {center[0]},{center[1]} size {size[0]},{size[1]}"
    hdr['HISTORY'] = "integrated 40-500 micron beta=2"
    hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    hdr['CREATOR'] = f"rkarim, via {__file__}.fir_intensity_2"
    hdr['BUNIT'] = str(F_array.unit)
    del hdr['EXTNAME']
    hdu = fits.PrimaryHDU(data=F_array.to_value(), header=hdr)
    hdu.writeto(os.path.join(os.path.dirname(solution_path), "M16_I_FIR_from70-160.fits"), overwrite=True)


    calculate_T_and_tau = False # Done on October 10, 2022, saved as T-tau_colorsolution.fits
    if calculate_T_and_tau:
        p70_correction = 268
        p160_correction = 1055

        # Set up T vs 70/160 ratio spline model using the bandpass filters
        # Following the code in color_temperature_comparison.ipynb
        model_T_arr = np.arange(1, 200, 0.1) # units of K
        # Set up PACS detectors (these use filter curves to act as detectors)
        p70_detector, p160_detector = instrument.get_instrument([70, 160])
        # Make output array (blue to red ratio)
        model_bandpass_br_ratio = np.zeros_like(model_T_arr)
        # Loop thru T array, can't use units because I didn't write mantipython with astropy.units
        # Set optical depth to something very small (10^-8) since I can't put in zero
        args = (-8., dust.TauOpacity(2.))
        for i, t in enumerate(model_T_arr):
            # # TODO: reuse Greybody
            p70_I = p70_detector.detect(greybody.Greybody(t, *args))
            p160_I = p160_detector.detect(greybody.Greybody(t, *args))
            model_bandpass_br_ratio[i] = p70_I / p160_I
        # Make spline interpolation from ratio to T
        model_bandpass_br_spline = UnivariateSpline(model_bandpass_br_ratio, model_T_arr, s=0)
        # This is our tool for interpolating from the ratio to T!

        # Make an output array for the zero-tau 160 intensity (perfect blackbody run through the detector function)
        zerotau_160intensity = np.zeros_like(model_T_arr)
        args = (0, dust.TauOpacity(2.))
        for i, t in enumerate(model_T_arr):
            p160_I = p160_detector.detect(greybody.ThinGreybody(t, *args))
            zerotau_160intensity[i] = p160_I
        # Spline fit that as a function of temperature
        zerotau_I_spline = UnivariateSpline(model_T_arr, zerotau_160intensity, s=0)

        """
        Now load the data and use it as follows:
        T_solution = model_bandpass_br_spline(observed_70/observed_160)
        tau_solution = observed_160 / zerotau_I_spline(T_solution)
        """

        # Path names
        # I did the reproc160 on October 7, 2022 (previously I had only done 250)
        pacs_obs_dir = catalog.utils.search_for_file("herschel/processed/1342218995_reproc160")
        make_pacs_fn = lambda band : os.path.join(pacs_obs_dir, f"PACS{band}um-image-remapped-conv.fits")
        p70_fn, p160_fn = make_pacs_fn(70), make_pacs_fn(160)
        # Load
        p70_img, p70_hdr = fits.getdata(p70_fn, header=True)
        p70_img += p70_correction
        p160_img, p160_hdr = fits.getdata(p160_fn, header=True)
        p160_img += p160_correction

        observed_br_ratio = p70_img / p160_img
        T_solution = model_bandpass_br_spline(observed_br_ratio)
        tau_solution = p160_img / zerotau_I_spline(T_solution)

        # fig = plt.figure()
        # axT = plt.subplot(121)
        # im = axT.imshow(T_solution, vmin=15, vmax=35)
        # fig.colorbar(im, ax=axT)
        # axtau = plt.subplot(122)
        # im = axtau.imshow(np.log10(tau_solution), vmin=-3.5, vmax=-1.5)
        # fig.colorbar(im, ax=axtau)
        # plt.show()

        new_hdr = WCS(p70_hdr).to_header()
        new_hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        new_hdr['CREATOR'] = f"rkarim, via {__file__}.fir_intensity_2"
        new_hdr['AUTHOR'] = "Ramsey Karim"
        new_hdr['OBJECT'] = "M16"
        new_hdr['HISTORY'] = "Herschel PACS 70 and 160, obsID 1342218995, 160 grid and beam"
        new_hdr['HISTORY'] = f"Zero-point offsets: {p70_correction} (70), {p160_correction} (160)"
        new_hdr['HISTORY'] = "Zero-point offsets from fit_FIR.py, calculated long ago"
        new_hdr['COMMENT'] = "T,tau calc'd using bandpasses; see color_temperature_comparison.ipynb"
        hdul = fits.HDUList([fits.PrimaryHDU(),
            fits.ImageHDU(data=T_solution, header=new_hdr.copy()),
            fits.ImageHDU(data=tau_solution, header=new_hdr.copy())])
        hdul[1].header['EXTNAME'] = 'T'
        hdul[1].header['BUNIT'] = 'K'
        hdul[2].header['EXTNAME'] = 'tau'
        hdul[2].header['BUNIT'] = 'optical depth at 160 micron'
        savepath = os.path.join("/home/ramsey/Documents/Research/Feedback/m16_data/herschel", "T-tau_colorsolution.fits")
        print("SAVING TO", savepath)
        hdul.writeto(savepath)



def check_I_FIR_G0():
    I_FIR_filename = catalog.utils.search_for_file("herschel/results/m16-I_FIR.fits")
    I_FIR, hdr = fits.getdata(I_FIR_filename, header=True)
    g0_fir = 0.5 * I_FIR / 1.3e-4
    plt.imshow(g0_fir, origin='lower', vmax=3e3)
    plt.colorbar(label="G0 in Habing units")
    plt.title("G0 from FIR intensity")
    # plt.show()
    plt.savefig("/home/rkarim/Pictures/2021-09-09-work/g0_fir.png")


if __name__ == "__main__":
    # fir_intensity()
    # check_I_FIR_G0()
    # make_plot_g0()
    fir_intensity_2()
