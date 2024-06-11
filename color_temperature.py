"""
Follow up on the RCW 49 paper ref's comments about the PACS SED reduction
This file is intended to create a publication-quality image for the paper,
and it follows directly from work in the color_temperature_comparison.ipynb
notebook (and related notebooks)
Created: March 23, 2021

Used June 11, 2024 to make Lars a dust map for RCW 49 (solve_opt_thin)
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from scipy.interpolate import UnivariateSpline
from astropy.io import fits
from astropy import units as u
from astropy.modeling import models
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS

from .mantipython.physics import greybody, dust, instrument
from .dust_mass import ellipse_region_mask
from . import misc_utils


herschel_dir = "/home/ramsey/Documents/Research/Feedback/rcw49_data/herschel"


def gen_model_br_ratio(log10_tau160, temperature_array, p70_detector, p160_detector, bb=False, tgb=False, just_160=False):
    """
    Create a T vs color relation and fit a spline to it
    :param log10_tau160: float tau160, in log10
    :param temperature_array: array of float temperature values (should be in K)
    :param p70_detector: instrument.Detector, PACS70
    :param p160_detector: instrument.Detector, PACS160
    :param bb: if False, uses TauOpacity with beta=2 law. If True, "blackbody",
        so uses ConstantOpacity. Same log10_tau160 applied to both 70 and 160
    :returns: tuple(color array, spline fit of color vs temperature)
    """
    return_array = np.zeros(temperature_array.size)
    if bb:
        args = (log10_tau160, dust.ConstantOpacity())
    else:
        args = (log10_tau160, dust.TauOpacity(2.))
    if tgb:
        gb = greybody.ThinGreybody
    else:
        gb = greybody.Greybody
    for i, t in enumerate(temperature_array):
        p160_I = p160_detector.detect(gb(t, *args))
        if just_160:
            return_array[i] = p160_I
        else:
            p70_I = p70_detector.detect(gb(t, *args))
            return_array[i] = p70_I / p160_I
    if just_160:
        return return_array, UnivariateSpline(temperature_array, return_array, s=0)
    else:
        return return_array, UnivariateSpline(return_array, temperature_array, s=0)


def gen_bb_intensity160(temperature, log10_tau160_array, p160_detector, bb=False, tgb=False):
    """
    Generate array of p160 intensities for a single temperature given an
    array of tau160s
    If bb=True, uses ConstantOpacity; if bb=False, uses TauOpacity
    """
    I_array = np.zeros(log10_tau160_array.size)
    if bb:
        d = dust.ConstantOpacity()
    else:
        d = dust.TauOpacity(2.)
    if tgb:
        gb = greybody.ThinGreybody
    else:
        gb = greybody.Greybody
    for i, tau in enumerate(log10_tau160_array):
        I_array[i] = p160_detector.detect(gb(temperature, tau, d))
    return I_array, UnivariateSpline(np.log10(I_array), log10_tau160_array, s=0)


def load_pacs_data():
    """
    Load in the RCW 49 Herschel images
    :returns: tuple of 2D arrays (pacs70, pacs160)
    """
    pacs_dir = "/home/ramsey/Documents/Research/Feedback/rcw49_data/herschel/processed/1342255009_reproc160"
    p70_fn = "PACS70um-image-remapped-conv-plus000080.fits"
    p70_fn = os.path.join(pacs_dir, p70_fn)
    assert os.path.exists(p70_fn)
    p160_fn = "PACS160um-image-remapped-conv-plus000370.fits"
    p160_fn = os.path.join(pacs_dir, p160_fn)
    assert os.path.exists(p160_fn)

    p70_img, p70_h = fits.getdata(p70_fn, header=True)
    p160_img, p160_h = fits.getdata(p160_fn, header=True)

    # Trim to just the cutout regions (since these are very large maps)
    i0, j0 = 2314, 1035
    width_i, width_j = 1286, 1286
    p70_cutout = Cutout2D(p70_img, (j0, i0), (width_i, width_j), wcs=WCS(p70_h))
    p160_cutout = Cutout2D(p160_img, (j0, i0), (width_i, width_j), wcs=WCS(p160_h))
    # WCS should be the same for each
    return p70_cutout.data, p160_cutout.data, p70_cutout.wcs


def load_original_fit():
    # Load in the original fitted parameter maps
    soln_fn = "RCW49large_2p_2BAND_160grid_beta2.0.fits"
    soln_fn = os.path.join(herschel_dir, soln_fn)
    assert os.path.exists(soln_fn)
    with fits.open(soln_fn) as hdul:
        fit_T = hdul['solutionT'].data
        fit_tau = hdul['solutiontau'].data
        # p70_img = hdul['BAND70'].data
        # p160_img = hdul['BAND160'].data
    return fit_T, fit_tau


def solve_opt_thin(p70_img, p160_img, savename=None):
    """
    Solve with the optically thin approximation
    If savename is not None, saves to that string filename
    """
    # Set stuff up
    model_T_arr = np.arange(8, 150, 0.05) # 0.1
    detectors = instrument.get_instrument([70, 160])
    model_br_thin, model_br_thin_spline = gen_model_br_ratio(-8., model_T_arr, *detectors)
    # Convert ratio image to temperature image
    br_img = p70_img / p160_img
    T_img = model_br_thin_spline(br_img)
    # Get rid of negatives
    T_img[T_img < 0] = np.nan
    # Make T -> 160_I spline so we can make a 160_I img (unnormalized)
    unnormalized_160, unnormalized_160_spline = gen_model_br_ratio(0, model_T_arr, *detectors, tgb=True, just_160=True)
    # Create 160_I img (unnormalized)
    unnormalized_160_img = unnormalized_160_spline(T_img)
    # Do arithmetic to find tau
    tau160_img = np.log10(p160_img / unnormalized_160_img)
    # Save
    if savename is not None:
        hdul1 = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=T_img), fits.ImageHDU(data=tau160_img)])
        hdul1[1].header['EXTNAME'] = 'T'
        hdul1[2].header['EXTNAME'] = 'tau'
        hdul1.writeto(os.path.join(herschel_dir, savename), overwrite=True)
    return T_img, tau160_img


def convert_tau_to_column_density_and_save(tau160, T, wcs_obj):
    """
    June 11, 2024
    Big time skip from the rest of this file. Lars requested the dust map from
    RCW 49 and I realized I have not recreated/saved a final map, I just
    calculated stuff in memory and saved an image.

    Here I will follow the shortcuts from M16. For RCW 49, I had a lengthy
    procedure in dust_mass.py, but a lot of that code isn't necessary.
    I think some of it had to do with the way I saved the optimization fit?
    I also just had way too many things hidden away in functions, so the code
    is hard to read, and I am loading Cext from a data file which is unnecessary
    for just one number.

    Following the conversion in m16_bubble.convert_pacs_tau_to_coldens
    """
    cexth = 1.9e-25 * u.cm**2
    nhtot = (tau160 / cexth).to(u.cm**-2)

    # Template for header
    hdr = wcs_obj.to_header()
    hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    hdr['CREATOR'] = f"rkarim, via {__file__}.convert_tau_to_column_density_and_save"
    hdr['AUTHOR'] = "Ramsey Karim"
    hdr['OBJECT'] = "RCW 49"
    hdr['HISTORY'] = "Herschel PACS 70 and 160, obsID 1342255009, 160 grid and beam"
    hdr['HISTORY'] = f"Zero-point offsets in MJy/sr: +80 (70), +370 (160)"

    # NH header
    hdr1 = hdr.copy()
    hdr1['EXTNAME'] = "N_H"
    hdr1['COMMENT'] = "Total H column density N_H = N(H2) + 2 N(H)"
    hdr['HISTORY'] = f"tau160 converted to N_H using Cext(160)/H = 1.9e-25 cm2"
    hdr1['BUNIT'] = nhtot.unit.to_string()

    # Temperature header
    hdr2 = hdr.copy()
    hdr2['EXTNAME'] = "T"
    hdr2['BUNIT'] = "K"

    hdu1 = fits.ImageHDU(data=nhtot.to_value(), header=hdr1)
    hdu2 = fits.ImageHDU(data=T, header=hdr2)

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2])
    hdul.writeto(os.path.join(herschel_dir, "rcw49_dust_coldens.fits"), overwrite=False)



if False:
    # # PACS 70 and 160 micron; transform to wavelengths
    pacs_freqs = ([70, 160] * u.micron).to(u.Hz, equivalencies=u.spectral())

    # Create an array of temperatures
    # model_T_arr = np.arange(8, 150, 0.1) # 0.1
    model_T_arr = np.arange(8, 150, 0.05) # 0.1
    # model_T_arr = np.arange(45, 47.6, 0.5) # 0.1

    # Set up the PACS detectors for 70 and 160
    detectors = instrument.get_instrument([70, 160])

    model_br_thin, model_br_thin_spline = gen_model_br_ratio(-8., model_T_arr, *detectors)


p70_img, p160_img, wcs_obj = load_pacs_data()
br_img = p70_img / p160_img
i0, j0 = 522, 531
# i0, j0 = 525, 541
b0, r0 = p70_img[i0, j0], p160_img[i0, j0]
original_T, original_tau = load_original_fit()
orig_T0, orig_tau0 = original_T[i0, j0], original_tau[i0, j0]

T_img, tau160_img = solve_opt_thin(p70_img, p160_img, savename='colorsoln_1-duplicate.fits')
# Tau is made log10 in solve_opt_thin, so need to undo that here
convert_tau_to_column_density_and_save(10.**tau160_img, T_img, wcs_obj)


plt.subplot(121)
plt.imshow(100.*(original_T - T_img)/original_T, origin='lower', vmin=-5, vmax=5)
# plt.imshow(original_tau, origin='lower')
plt.subplot(122)
# plt.imshow(100.*(10.**original_tau - 10.**tau160_img)/(10.**original_tau), origin='lower', vmin=-5, vmax=5)
plt.imshow(tau160_img, origin='lower')
plt.show()

def test_the_tau_array_thing():
    tau_array = np.array([-8, -6, -4, -2, -1.3, -1, -0.7, -0.5, -0.4, -0.3, 0, 0.3, 0.5, 0.7, 1, 1.5])
    # tau_array = np.log10(np.arange(0.05, 6, 0.1))
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    for t in model_T_arr[::-1]:
        I_array, _ = gen_bb_intensity160(t, tau_array, detectors[1])
        ax.plot(tau_array, np.log10(I_array), '-', label=f'{t:3.1f} K')
    ax.axhline(4.5)
    ax.set_xlabel("log10 tau160")
    ax.set_ylabel("160 micron intensity")
    ax.set_title("160um intensity vs optical depth at 160 micron")
    ax.legend()
    plt.show()


def test_iterative_fit():
    """
    This test was very educational but the result is confusing

    The iteration method WORKS, both T and tau tend to settle in at some fixed
    values that are robust to the choice of initial T/tau
    The # of iterations to settling can be improved with "damping"

    The answer tends to be very close to the original mantipython fit solution

    However, when the original observations are compared to the synthetic obs
    given the T/tau combo from this method, the residuals are higher
    The mantipython answer has very low residuals (since that's how it's solved)
    The iterative solution must be prioritizing something else...

    Debugging:
    Verified that the LS solution pixel had the same flux as the source imgs
        I'm using
    SOLVED! I forgot to set spline smoothing s=0 in the pacs160->tau160 spline
    """
    tau_array = np.arange(-4, 1, 0.005)
    print(f"fitted: T {orig_T0:.2f}, tau {orig_tau0:.3f}")
    br0 = b0/r0
    print(f"ratio0 {br0:.2f}")

    t = model_br_thin_spline(br0)
    print(f"T_{-1}: {t:.2f}", end='; ')
    tau_spline = gen_bb_intensity160(t, tau_array, detectors[1])[1]
    tau = tau_spline(np.log10(r0))
    print(f"tau_{-1} {tau:.3f}")

    # t, tau = orig_T0, orig_tau0

    T_list, tau_list = [t], [tau]
    for i in range(5):
        T_spline = gen_model_br_ratio(tau, model_T_arr, *detectors)[1]
        t = T_spline(br0)#*0.8 + t*0.2
        T_list.append(t)
        # print(f"T_{i}: {t:.2f}")
        tau_spline = gen_bb_intensity160(t, tau_array, detectors[1])[1]
        tau = tau_spline(np.log10(r0))#*0.8 + tau*0.2
        tau_list.append(tau)
        # print(f"tau_{i} {tau:.3f}")

    print(f"T_{i}: {t:.2f}", end='; ')
    print(f"tau_{i} {tau:.3f}")

    d = dust.TauOpacity(2.)
    gb0 = greybody.Greybody(orig_T0, orig_tau0, d)
    b0_, r0_ = [x.detect(gb0) for x in detectors]
    gb1 = greybody.Greybody(t, tau, d)
    b1, r1 = [x.detect(gb1) for x in detectors]

    resid0 = np.sqrt(((b0 - b0_)/b0)**2 + ((r0 - r0_)/r0)**2)
    resid1 = np.sqrt(((b0 - b1)/b0)**2 + ((r0 - r1)/r0)**2)
    print(f"LS resid: {resid0:.2E}, Iter resid: {resid1:.2E}")

    plt.subplot(121)
    plt.plot(T_list)
    plt.axhline(orig_T0)
    plt.title("T")
    plt.subplot(122)
    plt.plot(tau_list)
    plt.axhline(orig_tau0)
    plt.title("tau")
    plt.show()


def plot_1():
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)

    # ax.plot(model_T_arr, np.zeros_like(model_T_arr), '--', color='k', label='Optically thin')
    ax.plot(model_T_arr, model_br_thin, '--', color='k', label='Optically thin')

    ax.axvspan(20, 50, alpha=0.3, color='red', label='Relevant temperatures in RCW 49')

    optdepth_tup = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
    optdepth10_tup = tuple(np.log10(x) for x in optdepth_tup)
    model_br_tup, model_br_spline_tup = tuple(zip(*(gen_model_br_ratio(x, model_T_arr, *detectors) for x in optdepth10_tup)))
    for optdepth, model_br in zip(optdepth_tup, model_br_tup):
    # for optdepth, diff_T in zip(optdepth_tup, diff_T_tup):
        ax.plot(model_T_arr, model_br, '-', label=f'Tau160 = {optdepth:.2f}')
        # ax.plot(truncated_T_arr, diff_T, '-', label=f'Tau160 = {optdepth:.2f}')

    ax.set_xlabel("Temperature (K)")
    # ax.set_ylabel("$\\Delta T$ / $T$ (%)")
    ax.set_ylabel("Intensity ratio 70/160")
    ax.set_yscale('log')
    # ax.set_ylim([-10, 100])
    # ax.set_xlim([0, 60])
    ax.legend()
    # ax.set_title("Change in derived temperature, w.r.t. optically thin, due to optical depth")
    ax.set_title("Color vs temperature for different optical depths")
    # fig.savefig("/home/ramsey/Pictures/2021-03-24-work/dt.png")
    # fig.savefig("/home/ramsey/Pictures/2021-03-24-work/color.png")
    plt.show()


def plot_2():
    bb_160intensity = gen_bb_intensity160(model_T_arr, detectors[1])
    plt.plot(model_T_arr, bb_160intensity, '-', color='k', label='Blackbody')
    plt.show()


def plot_3():
    """
    This is going to (hopefully) be my publication-ready version of plot_1
    See 3/30/21 notes in notebook for steps I am going to take
    Created: March 30, 2021
    x
    """
    # Set up the initial color vs T plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    # Set up the inset zoom axis
    ax3 = ax.inset_axes(bounds=[0.32, 0.04, 0.4, 0.28])

    # Plot the optically thin line
    ax.plot(model_T_arr, np.log10(model_br_thin), '--', lw=2, color='k', label='Optically thin')
    ax3.plot(model_T_arr, np.log10(model_br_thin), '--', lw=2, color='k')

    # Plot the non-optically thin lines
    optdepth_70micron_tup = (0.05, 0.1, 0.5)
    optdepth10_160micron = np.log10(np.array(optdepth_70micron_tup) / ((160/70)**2.))
    for i in range(len(optdepth_70micron_tup)):
        tau160 = optdepth10_160micron[i]
        model_br, model_br_spline = gen_model_br_ratio(tau160, model_T_arr, *detectors)
        ax.plot(model_T_arr, np.log10(model_br), '-', lw=1, alpha=0.8, label=f'$\\tau$(70 micron)$=${optdepth_70micron_tup[i]:.2f}')
        ax3.plot(model_T_arr, np.log10(model_br), '-', lw=1, alpha=0.8)

    # Plot the optically thick (blackbody) line
    model_br_thick, model_br_thick_spline = gen_model_br_ratio(0, model_T_arr, *detectors, bb=True)
    ax.plot(model_T_arr, np.log10(model_br_thick), '-', lw=2, color='k', label='Optically thick (blackbody)')

    # Create a second use of the y axis (color) for the histogram
    ax2 = ax.twiny()
    # Plot a histogram of the observed colors
    br_arr = np.log10(br_img[np.isfinite(br_img)].ravel())
    br_lims = [-1.5, 1.]
    bins = ax2.hist(br_arr, bins=32, log=True,
        histtype='step', fill=False, color='k', range=br_lims, orientation='horizontal', label='Entire image')[1]
    shell_ellipse_mask = ellipse_region_mask(shape=br_img.shape, w=wcs_obj, half=True)
    br_arr_shell = np.log10(br_img[np.isfinite(br_img) & shell_ellipse_mask].ravel())
    bins2 = ax2.hist(br_arr_shell, bins=32, range=br_lims, log=True,
        histtype='step', fill=False, color='r', orientation='horizontal', label='Under half-ellipse mask')[1]
    ax2.set_xlim([1e20, 1])
    ax2.set_xlabel("Histogram count of observed 70/160 intensity ratio", horizontalalignment='right', x=1.0)
    ax2.set_xticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6])
    # ax2.legend(loc='lower right', title='Histograms')
    ax2.legend(bbox_to_anchor=(1., 0.4), loc='center right', title='Histograms')

    ax.axhspan(br_arr_shell.min(), br_arr_shell.max(), alpha=0.2, color='r')

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("70/160 band intensity ratio (log10)")
    ax.set_ylim(br_lims)
    ax.set_xlim([10, 80])
    ax.legend(loc='upper left', title='Color vs T curves')

    # Adjust inset axis
    inset_ylims = [0.4, 0.45]
    inset_xlims = [model_br_thin_spline(10.**inset_ylims[0]), model_br_spline(10.**inset_ylims[1])]
    ax3.set_xlim(inset_xlims)
    ax3.set_ylim(inset_ylims)
    ax3.tick_params(axis='both', direction='in')

    # Plot inset axis footprint in main plot
    ax.plot(
        [inset_xlims[0], inset_xlims[1], inset_xlims[1], inset_xlims[0], inset_xlims[0]],
        [inset_ylims[0], inset_ylims[0], inset_ylims[1], inset_ylims[1], inset_ylims[0]],
        lw=1, color='k')

    inset_topright_corner_data_coords = ax.transData.inverted().transform(ax.transAxes.transform((0.32+0.4, 0.04+0.28)))
    inset_topleft_corner_data_coords = ax.transData.inverted().transform(ax.transAxes.transform((0.32, 0.04+0.28)))
    ax.plot([inset_xlims[1], inset_topright_corner_data_coords[0]],
        [inset_ylims[0], inset_topright_corner_data_coords[1]],
        [inset_xlims[0], inset_topleft_corner_data_coords[0]],
        [inset_ylims[0], inset_topleft_corner_data_coords[1]],
        lw=0.6, color='k')

    plt.tight_layout()
    plt.show()
    # fig.savefig('/home/ramsey/Pictures/2021-03-30-work/color_vs_T_NEW.png')


def plot_4():
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    model_tau_array = np.arange(-4, 1, 0.005)
    T_tup = (30, 39, 40, 42, 50)
    for t in T_tup:
        I160, logI160_spline = gen_bb_intensity160(t, model_tau_array, detectors[1], bb=False, tgb=False)
        p = ax.plot(model_tau_array + np.log10((160/70.)**2.), np.log10(I160), '-', label=f'T={t} K')
        I160, logI160_spline = gen_bb_intensity160(t, model_tau_array, detectors[1], bb=False, tgb=True)
        ax.plot(model_tau_array + np.log10((160/70.)**2.), np.log10(I160), '--', color=p[0].get_c(), label=f'T={t} K, linear approx')
    ax.legend()

    ax2 = ax.twiny()
    I160_arr = np.log10(p160_img[np.isfinite(p160_img)].ravel())
    # I160_lims = []
    bins = ax2.hist(I160_arr, bins=32, log=True, histtype='step', fill=False,
        color='k', range=None, orientation='horizontal', label='Entire image')[1]
    shell_ellipse_mask = ellipse_region_mask(p160_img.shape, w=wcs_obj, half=True)
    I160_arr_shell = np.log10(p160_img[np.isfinite(p160_img) & shell_ellipse_mask].ravel())
    bins2 = ax2.hist(I160_arr_shell, bins=bins, log=True, histtype='step', fill=False,
        color='r', orientation='horizontal', label='Under half-ellipse mask')[1]
    ax2.set_xlim([1e20, 1])
    ax2.set_xlabel("Histogram count of observed 160 micron intensity", horizontalalignment='right', x=1.0)
    ax2.set_xticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6])
    ax2.legend(bbox_to_anchor=(1., 0.2), loc='center right', title='Histograms')

    ax.axhspan(I160_arr_shell.min(), I160_arr_shell.max(), alpha=0.2, color='r')

    ax.set_xlabel("Optical depth at 160 micron $\\tau(160)$")
    ax.set_ylabel("160 micron band intensity (log10 MJy/sr)")

    plt.tight_layout()
    plt.show()

# args = plot_4()
