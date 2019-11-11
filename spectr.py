import numpy as np
import matplotlib.pyplot as plt
import sys
import misc_utils
"""
Utility functions for examining spectra, specifically from the MUSE IFU
Created: October 2, 2019
"""

__author__ = "Ramsey Karim"


def extract_background(wavl_array, flux_array,
    p=3, valid_mask=(lambda x: x < 8200), debug=False):
    """
    Extracts background from a spectrum using a low-order polynomial fit.
    Background RMS is calculated after background subtraction.
    Not designed for delicate extraction, assumes high SNR (>10) on lines.
    :param wavl_array: Wavelength array, in whatever units
    :param flux_array: Flux array, in whatever units
    :param p: polynomial fit degree. Should not be too high (~<5)
    :param valid_mask: function operating on wavelength array (in correct units)
        Should return boolean array in argument shape that is TRUE where the
        spectrum's background can be modeled as a low-order polynomial.
        Examples of invalid are: >8200 A at the Paschen break.
        Defaults to lambda x: x > 8200
    :param debug: return dictionary with lots of useful debug info
    :return: If debug is FALSE, returns a 2-element tuple:
        (
            1) function that takes wavelength array as arg and returns
                the background in flux units,
            2) float background RMS in flux units
        )
        If debug is TRUE, then returns a dictionary with keys:
            backgroundf: the background function described above
            rms: background RMS described above
            pfit: polynomial coefficients of fit
            th_prelim: preliminary background threshold.
                Used to produce mprelim mask.
            th: (lower, upper) limits used to calculate background RMS.
                Used to produce mbackg mask.
            mvalid: tuple(wavl, flux) under valid mask.
                Applied to the input wavl, flux arrays.
            mprelim: tuple(wavl, flux) under preliminary background mask.
                Applied to the "mvalid" wavl, flux arrays.
            mbackg: tuple(wavl, flux) under final background mask, used to
                derive RMS. Applied to the "mprelim" wavl, flux arrays.
    """
    # Apply valid mask, make copies of arrays
    mvalid = valid_mask(wavl_array)
    flux_valid = flux_array[mvalid].copy()
    wavl_valid = wavl_array[mvalid].copy()
    # Preliminary background ID and fitting
    # Calculating typical span of the spectrum using first-to-last
    #  hexadecile difference
    # Adding the span to the last hexadecile to get a rough upper limit
    #  for background
    first_quant, last_quant = misc_utils.flquantiles(flux_valid, 16)
    # Typical span is (last - first) quant
    prelim_threshold = last_quant + (last_quant - first_quant)
    mprelim = np.abs(flux_valid) < prelim_threshold
    background_fit_coeffs = np.polyfit(wavl_valid[mprelim], flux_valid[mprelim],
        deg=p)
    # Generate background function to return
    def background_f(x):
        return misc_utils.polynomial(x, background_fit_coeffs)
    # Correct background within valid range
    flux_valid_mBG = flux_valid - background_f(wavl_valid)
    # Repeat quantile analysis with 25-quantiles to get background RMS estimate
    first_quant, last_quant = misc_utils.flquantiles(flux_valid_mBG, 25)
    # Final background mask is between first and last 25-quantiles
    mbackg = (flux_valid_mBG > first_quant) & (flux_valid_mBG < last_quant)
    background_rms = np.std(flux_valid_mBG[mbackg])
    if not debug:
        return background_f, background_rms
    else:
        debug_dict = {
            "backgroundf": backgrounf_f,
            "rms": background_rms,
            "pfit": background_fit_coeffs,
            "th_prelim": prelim_threshold,
            "th": (first_quant, last_quant),
            "mvalid": mvalid,
            "mprelim": mprelim,
            "mbackg": mbackg,
        }
        return debug_dict


def acquire_line(line_wavl, wavl_array, flux_array, background_rms):
    """
    Line wavelength and wavl_array in same units
    Flux array already background-subtracted
    Background RMS in flux units; feel free to add a tolerance of your own
    Returns slice object such that indexing flux_array[slice] works
    """
    # Get nearest wavelength index
    line_i = np.searchsorted(wavl_array, line_wavl)
    # Refine peak location using data
    line_i += np.argmax(flux_array[line_i-1:line_i+2]) - 1
    line_start = line_end = line_i
    while flux_array[line_start] > background_rms:
        line_start -= 1
    while flux_array[line_end] > background_rms:
        line_end += 1
    return slice(line_start, line_end+1)



def test_characterize_bg_and_noise():
    """
    Terrible draft version of what I will do more formally
    proof of concept, really
    """
    cols = np.genfromtxt("muse/test_spec.dat")
    wl, flux = cols[:, 0], cols[:, 1]
    del cols
    testmask = (wl<8200) # & (wl>5500)
    filtered_testmask = testmask & (np.abs(flux) < 3*np.nanmedian(flux))
    bg_polyfit = np.polyfit(wl[filtered_testmask], flux[filtered_testmask], deg=3)
    bg1 = lambda x: misc_utils.polynomial(x, bg_polyfit)
    bg1_bgsub = misc_utils.polynomial(wl[filtered_testmask], bg_polyfit)
    # Should try to fit this background with a Planck curve...
    flux_bgsub = flux[filtered_testmask] - bg1_bgsub
    wl_bgsub = wl[filtered_testmask]
    std_thresh = np.std(flux_bgsub)**2
    f2mask = flux_bgsub**2 < 3*std_thresh
    bg_flux = flux_bgsub[f2mask]
    bg_wl = wl_bgsub[f2mask]
    bg_rms = np.std(bg_flux)
    plt.figure()
    plt.plot(wl, flux, color='k', linewidth=0.5)
    plt.plot(wl[filtered_testmask], flux[filtered_testmask], color='g')
    plt.plot(wl_bgsub, bg1_bgsub, color='r', linewidth=0.5)
    plt.ylim((-20, 200))
    plt.show()
    plt.figure()
    plt.subplot(121)
    print(std_thresh*3)
    plt.plot(wl[testmask], flux[testmask] - bg1(wl[testmask]), color='k', linewidth=0.5)
    plt.plot(bg_wl, bg_flux, color='g')
    plt.ylim([-20, 50])
    plt.subplot(122)
    print(bg_rms)
    plt.plot(bg_wl, bg_flux, color='g')
    plt.show()


if __name__ == "__main__":
    # File in
    cols = np.genfromtxt("muse/test_spec.dat")
    wavl, flux = cols[:, 0], cols[:, 1]
    del cols
    # Get background
    background_f, bg_rms = extract_background(wavl, flux)
    flux = flux[wavl < 8200]
    wavl = wavl[wavl < 8200]
    flux -= background_f(wavl)

    print(f"BG RMS: {bg_rms}")
    print(bg_rms*3)
    line1_wavl = 6563
    line2_wavl = 4861

    line1_sl = acquire_line(line1_wavl, wavl, flux, 2*bg_rms)
    line2_sl = acquire_line(line2_wavl, wavl, flux, 2*bg_rms)

    ja = np.sum(flux[line1_sl])
    jb = np.sum(flux[line2_sl])
    print(ja/jb)

    # plt.plot(wavl_bg, flux_bg, color='g')
    plt.plot(wavl, flux, color='r')
    plt.plot(wavl[line1_sl], flux[line1_sl], linewidth=2.5, color='blue')
    plt.plot(wavl[line2_sl], flux[line2_sl], linewidth=2.5, color='blue')
    # plt.ylim([-100, 200])
    plt.show()
