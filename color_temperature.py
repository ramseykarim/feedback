"""
Follow up on the RCW 49 paper ref's comments about the PACS SED reduction
This file is intended to create a publication-quality image for the paper,
and it follows directly from work in the color_temperature_comparison.ipynb
notebook (and related notebooks)
Created: March 23, 2021
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.interpolate import UnivariateSpline
from astropy.io import fits
from astropy import units as u
from astropy.modeling import models
from astropy.nddata.utils import Cutout2D

from .mantipython.physics import greybody, dust, instrument


def gen_model_br_ratio(log10_tau160, temperature_array, p70_detector, p160_detector):
    """
    Create a T vs color relation and fit a spline to it
    :param log10_tau160: float tau160, in log10
    :param temperature_array: array of float temperature values (should be in K)
    :param p70_detector: instrument.Detector, PACS70
    :param p160_detector: instrument.Detector, PACS160
    :returns: tuple(color array, spline fit of color vs temperature)
    """
    return_array = np.zeros(temperature_array.size)
    args = (log10_tau160, dust.TauOpacity(2.))
    for i, t in enumerate(temperature_array):
        p70_I = p70_detector.detect(greybody.Greybody(t, *args))
        p160_I = p160_detector.detect(greybody.Greybody(t, *args))
        return_array[i] = p70_I / p160_I
    return return_array, UnivariateSpline(return_array, temperature_array, s=0)


def gen_bb_intensity160(temperature_array, p160_detector):
    """
    honestly not sure what this is gonna do yet, or if it's the right thing
    """
    # Create optically thin "tau=1" 160 micron intensities as a function of temperature
    # Tau=1 won't make the expression optically thick;
    #   rather, it's mathematically equivalent to pulling tau160 out of the equation
    zerotau_160intensity = np.zeros(temperature_array.size)
    args = (0, dust.TauOpacity(2.)) # log10 tau, so 0 -> 1 in linear
    for i, t in enumerate(temperature_array):
        p160_I = p160_detector.detect(greybody.ThinGreybody(t, *args))
        zerotau_160intensity[i] = p160_I
    # Spline fit that as a function of temperature
    # then use that to convert the new temperature map to "zero optical depth intensities"
    zerotau_I_spline = UnivariateSpline(model_T_arr.to_value()[:, 0], zerotau_160intensity, s=0)


def gen_bb_intensity160_alt(temperature, log10_tau160_array, p160_detector):
    """
    Generate array of p160 intensities for a single temperature given an
    array of tau160s
    """
    I_array = np.zeros(log10_tau160_array.size)
    d = dust.TauOpacity(2.)
    for i, tau in enumerate(log10_tau160_array):
        I_array[i] = p160_detector.detect(greybody.Greybody(temperature, tau, d))
    return I_array, UnivariateSpline(np.log10(I_array), log10_tau160_array)


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

    p70_img = fits.getdata(p70_fn)
    p160_img = fits.getdata(p160_fn)

    # Trim to just the cutout regions (since these are very large maps)
    i0, j0 = 2314, 1035
    width_i, width_j = 1286, 1286
    p70_img = Cutout2D(p70_img, (j0, i0), (width_i, width_j)).data
    p160_img = Cutout2D(p160_img, (j0, i0), (width_i, width_j)).data
    return p70_img, p160_img


def load_original_fit():
    # Load in the original fitted parameter maps
    herschel_dir = "/home/ramsey/Documents/Research/Feedback/rcw49_data/herschel"
    soln_fn = "RCW49large_2p_2BAND_160grid_beta2.0.fits"
    soln_fn = os.path.join(herschel_dir, soln_fn)
    assert os.path.exists(soln_fn)
    with fits.open(soln_fn) as hdul:
        fit_T = hdul['solutionT'].data
        fit_tau = hdul['solutiontau'].data
    return fit_T, fit_tau


# # PACS 70 and 160 micron; transform to wavelengths
# pacs_freqs = ([70, 160] * u.micron).to(u.Hz, equivalencies=u.spectral())

# Create an array of temperatures
model_T_arr = np.arange(8, 150, 0.1) # 0.1
# model_T_arr = np.arange(45, 47.6, 0.5) # 0.1

# Set up the PACS detectors for 70 and 160
detectors = instrument.get_instrument([70, 160])

# Set up output array
model_bandpass_br_ratio = np.zeros(model_T_arr.size)
# Loop through the temperature array. Need to do everything unitless this time for mantipython
# Since we can't put in tau = 0, put tau = very small (log10(tau) = -8, for example)

model_br_thin, model_br_thin_spline = gen_model_br_ratio(-8., model_T_arr, *detectors)
optdepth_tup = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
optdepth10_tup = tuple(np.log10(x) for x in optdepth_tup)
model_br_tup, model_br_spline_tup = tuple(zip(*(gen_model_br_ratio(x, model_T_arr, *detectors) for x in optdepth10_tup)))

truncated_T_arr = model_T_arr[model_T_arr > 10]
truncated_br_thin = model_br_thin[model_T_arr > 10]
diff_T_tup = tuple((100*(x(truncated_br_thin) - truncated_T_arr)/truncated_T_arr for x in model_br_spline_tup))

p70_img, p160_img = load_pacs_data()
i0, j0 = 522, 531
# i0, j0 = 525, 541
b0, r0 = p70_img[i0, j0], p160_img[i0, j0]
original_T, original_tau = load_original_fit()
orig_T0, orig_tau0 = original_T[i0, j0], original_tau[i0, j0]


def test_the_tau_array_thing():
    tau_array = np.array([-8, -6, -4, -2, -1.3, -1, -0.7, -0.5, -0.4, -0.3, 0, 0.3, 0.5, 0.7, 1, 1.5])
    # tau_array = np.log10(np.arange(0.05, 6, 0.1))
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    for t in model_T_arr[::-1]:
        I_array, _ = gen_bb_intensity160_alt(t, tau_array, detectors[1])
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
    """
    tau_array = np.arange(-8, 1, 0.05)
    print(f"fitted: T {orig_T0:.2f}, tau {orig_tau0:.3f}")
    br0 = b0/r0
    print(f"ratio0 {br0:.2f}")

    t = model_br_thin_spline(br0)
    print(f"T_{-1}: {t:.2f}")
    tau_spline = gen_bb_intensity160_alt(t, tau_array, detectors[1])[1]
    tau = tau_spline(np.log10(r0))
    print(f"tau_{-1} {tau:.3f}")

    # t, tau = orig_T0, orig_tau0

    T_list, tau_list = [t], [tau]
    for i in range(5):
        T_spline = gen_model_br_ratio(tau, model_T_arr, *detectors)[1]
        t = T_spline(br0)*0.8 + t*0.2
        T_list.append(t)
        # print(f"T_{i}: {t:.2f}")
        tau_spline = gen_bb_intensity160_alt(t, tau_array, detectors[1])[1]
        tau = tau_spline(np.log10(r0))*0.8 + tau*0.2
        tau_list.append(tau)
        # print(f"tau_{i} {tau:.3f}")

    print(f"T_{i}: {t:.2f}")
    print(f"tau_{i} {tau:.3f}")

    d = dust.TauOpacity(2.)
    gb0 = greybody.Greybody(orig_T0, orig_tau0, d)
    b0_, r0_ = [x.detect(gb0) for x in detectors]
    gb1 = greybody.Greybody(t, tau, d)
    b1, r1 = [x.detect(gb1) for x in detectors]

    resid0 = np.sqrt((b0 - b0_)**2 + (r0 - r0_)**2)
    resid1 = np.sqrt((b0 - b1)**2 + (r0 - r1)**2)
    print(resid0, resid1)

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

plot_1()
