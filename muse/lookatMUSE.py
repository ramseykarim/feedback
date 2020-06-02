# script to examine MUSE cubes
# created: September 19, 2019
__author__ = "Ramsey Karim"

import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
import matplotlib.pyplot as plt
import sys
import misc_utils
import extinction
import spectr
import muse_convenience as mcon


def open_muse(stub):
    """
    Given the "filename stub", open a file
    Returns the cube and the header info dict
    """
    with fits.open(mcon.gen_filename_raw(stub)) as hdul:
        header_global = hdul[0].header
        header_img = hdul[1].header
        cube_raw = hdul[1].data
    header_info = mcon.unpack_header(header_global, header_img)
    return cube_raw, header_info


def make_flat_image(stub, cube, header_info, q=4, getval=False):
    img = np.nanmean(cube, axis=0)
    vmin, vmax = misc_utils.flquantiles(img[np.isfinite(img)].ravel(), q)
    if getval:
        return img, vmin, vmax
    else:
        plt.imshow(img, origin='lower', vmin=0, vmax=vmax)
        plt.colorbar()
        plt.title(f"Integrated MUSE map, {header_info['wavl'][0]:.0f} - {header_info['wavl'][-1]:.0f} A, {stub}")
        plt.show()


def make_avg_spectrum(stub, cube, header_info, slices=None, mask=None, getval=False):
    if slices is not None:
        spectrum = np.nanmean(cube[:, slices[0], slices[1]], axis=(1, 2))
    elif mask is not None:
        spectrum = np.nanmean(cube[:, mask], axis=1)
    else:
        spectrum = np.nanmean(cube, axis=(1, 2))
    if getval:
        return spectrum
    else:
        plt.plot(header_info['wavl'], spectrum, color='k')
        plt.xlabel(header_info['wavl_unit'])
        plt.ylabel(header_info['flux_unit'])
        plt.title(f"Spectrum from {stub}")
        plt.show()


if __name__ == "__main__":
    selected_stub = "east-large-long"
    cube, header_info = open_muse(selected_stub)
    args = (selected_stub, cube, header_info)
    islice = slice(515, 532)
    jslice = slice(107, 125)
    flatimg, vmin, vmax = make_flat_image(*args, q=8, getval=True)
    print(f"min/max = {vmin:.1f}/{vmax:.1f}")
    # mask = (flatimg > vmin) & (flatimg < vmax)
    mask = flatimg < vmin
    # mask = (flatimg > -1) & (flatimg < 1)
    plt.subplot(141)
    plt.imshow(flatimg, origin='lower', vmin=0, vmax=vmax)
    plt.colorbar()
    plt.subplot(142)
    plt.imshow(mask, origin='lower')
    plt.subplot(122)
    make_avg_spectrum(*args, mask=mask)





if False:
    """
    This is the old code (from October 2019)
    I have returned to this subproject in March 2020
    I'll delete this block eventually, and will rewrite
    in functions above -rkarim March 5 2020
    """
    ### RCW49, randomly selected cube. PI Peter Zeidler
    with fits.open(mcon.current_filename) as hdul:
        global_header = hdul[0].header
        raw_header = hdul[1].header
        cube_raw = hdul[1].data

    # cube = fits.getdata(mcon.cube_filename_conv)

    # if True:
    #     # RESAMPLE CUBE
    #     mcon.resample_img(cube, raw_header, box_size=5)
    #     # sys.exit()
    # print("FINISHED RESAMPLE")

    header_info = mcon.unpack_header(global_header, raw_header)
    ### Load in the convolved (7 pixel FWHM) data. SHORTER FREQ AXIS and ALREADY MASKED
    ### Load in the resampled (9 original pixels to a single one here) data. same notes as above^
    # cube, header = fits.getdata(mcon.cube_filename_resamp, header=True)
    # cube_samp = fits.getdata(mcon.cube_filename_rsmperr)

    ########## MOMENT 0 MAP
    # plt.imshow(np.nanmean(cube_raw, axis=0), origin='lower', vmin=0, vmax=160)
    # plt.colorbar()
    # plt.show()

    wavl_array = header_info['wavl']
    # cube_raw = mcon.trim_MUSE(cube_raw, wavl_array)
    # wavl_array = mcon.trim_MUSE(wavl_array, wavl_array)

    # ratio_map = np.full(cube.shape[1:], np.nan)

    # for i, j in zip(*np.where(np.sum(~np.isnan(cube), axis=0)/cube.shape[0] > 0.95)):
    #     ja2b = mcon.get_Hab(wavl_array, cube[:, i, j])
    #     ratio_map[i, j] = ja2b
    #     if (ratio_map[i, j] < 0) or (ratio_map[i, j] > 250):
    #         ratio_map[i, j] = np.nan
    # NH_map = extinction.Hab_reddening(ratio_map, Rv=5.5, Hab_ratio=2.85, method="Draine")
    # Av_map = extinction.extinction(1e4/extinction.VBAND_WAVELENGTH_A, method="Draine", Rv=5.5) * 1.086 * NH_map

    ########### PLOT SPECTRUM
    plt.plot(wavl_array, np.nanmean(cube_raw, axis=(1, 2)), color='k')
    plt.xlabel(header_info['wavl_unit'])
    plt.ylabel(header_info['flux_unit'])
    # plt.legend()
    plt.show()

    # fits.writeto(mcon.NH_filename, NH_map, header)
    # fits.writeto(mcon.Av_filename, Av_map, header)

    # plt.subplot(131)
    # plt.title("Observed Intensity")
    # plt.imshow(np.nanmean(cube, axis=0), origin='lower', cmap='inferno')
    # plt.subplot(132)
    # plt.title("N(H) (cm-2)")
    # plt.imshow(NH_map, origin='lower', vmin=7e21, vmax=2e22, cmap='plasma')
    # plt.subplot(133)
    # plt.title("Av")
    # plt.imshow(Av_map, origin='lower', vmin=4, vmax=12, cmap='magma')
    # plt.show()


    """
    STEPS TO TAKE TOMORROW: (oct 2)

    Automatic line fitting
    examine one example spectrum and select an H beta and H alpha window (line and nearby region)
    mask out line and get RMS and linear fit (or just mean? if flat enough?)
    subtract floor and integrate lines, get ratio
    ***** good up to here! can get better noise if we --dont-- get rid of layers..
    cut out just these layers from entire cube (avoid unneccessary smoothing)
    smooth cube to a fair radius (7 pixels?); try to do it in one go with numpy
    ratio should be single image
    should go into the Hab_reddening function just fine, return N map
    draw up C_ext/H curve and multiply (in new dimension) by N map; gives full redenning correction (tau)
    multiply by 1.086 for Av approx map
    can do the math to convert original C_ext curve to some operation that will yield extinction directly
        and convert N map to whatever unit that needs to be too (avoid taking exp of huge cube? may be unavoidable)
    """

    ### Histogram of flat image
    # dhist, dedges = np.histogram(flatimg[~np.isnan(flatimg)].ravel(), bins=128, range=(-30, 200))
    # prep_arr = lambda a, b: np.array([a, b]).T.flatten()
    # histx, histy = prep_arr(dedges[:-1], dedges[1:]), prep_arr(dhist, dhist)
    # bin_centers = (dedges[:-1]+dedges[1:])/2
    # plt.plot(histx, histy, '-')


    """
    for paschen_line in line_series:
        name, transition, line_wl = paschen_line
        if line_wl > whi*10:
            continue
        plt.plot([line_wl, line_wl], list(plt.ylim()), '--', label=f"{name[:2]} {transition}")
    pa_inf_wl = 820.4 * 10
    plt.plot([pa_inf_wl, pa_inf_wl], list(plt.ylim()), '-.', linewidth=3, color='red', label=r'Pa $\inf$')
    """
    ## plt.ylim((-100,200))
    ## plt.plot(wlrange_A, medianspec, '-', color='k', label="_nolabel_")

    # plt.plot(wlrange_A, slice_spec, '-', color='k', label="_nolabel_")
    # for Ba_line in line_series[-2:]:
    #     line_wl = Ba_line[-1]
    #     plt.plot([line_wl, line_wl], list(plt.ylim()), color='red')
    # plt.plot(wlrange_A[line_slices[1]], slice_spec[line_slices[1]], color='blue', label="H beta")
    # plt.plot(wlrange_A[line_slices[0]], slice_spec[line_slices[0]], color='green', label="H alpha")
    # plt.xlabel(cunit3)
    # plt.ylabel(bunit)
    # plt.legend()

    # plt.figure()
    # iarr, jarr = (np.arange(flatimg.shape[0]) - flatimg.shape[0]//2), (np.arange(flatimg.shape[1]) - flatimg.shape[1]//2)
    # conv_kernel = misc_utils.gaussian(iarr, 0, 10/2.35, 1)[:, np.newaxis] * misc_utils.gaussian(jarr, 0, 10/2.35, 1)[np.newaxis, :]
    # conv_kernel /= np.sum(conv_kernel)
    # flatimg[star_mask] = np.nan
    # plt.imshow(misc_utils.convolve_properly(flatimg, conv_kernel), origin='lower', vmin=-100, vmax=100)
    # plt.plot([center[0]-delta, center[0]+delta], [center[1]-delta, center[1]+delta], [center[0]-delta, center[0]+delta], [center[1]+delta, center[1]-delta], '-', color='red')
    # plt.show()
