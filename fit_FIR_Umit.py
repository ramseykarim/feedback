"""
Created: December 12, 2023

Fit the NGC 7538 PACS + SPIRE images that Umit sent me.
"""
__author__ = "Ramsey Karim"

import os
# mantipython is Ramsey's software for Herschel SED fitting
import mantipython

from astropy.io import fits
from astropy import units as u
from radio_beam import Beam


data_dir = "/home/ramsey/Documents/Research/Feedback/ngc7538_data/herschel/"
data_set_select="Feb2024_Umit"

def data_fn_format(band, suffix=None):
    """
    :param band: int PACS or SPIRE wavelength
    :param data_set_select: string describing which dataset to use.
        options:
            Dec2023_Umit - Umit's original December data
            Feb2024_Umit - Umit's new February data
    :returns: path to the NGC 7538 data.
        relative to the data_dir defined globally.
    """
    if data_set_select == "Feb2024_Umit":
        if band == 500:
            raise RuntimeError("There's no 500 micron in the Feb 2024 set.")
        prefix = "ngc7538_"
        if suffix is None:
            suffix = "_cgs.fits"
        band_str = f"{band:d}mu"
    elif data_set_select == "Dec2023_Umit":
        if suffix is None:
            suffix = "_repro_500.fits"
        if band > 200:
            prefix = "spire"
        else:
            prefix = "pacs"
        band_str = f"{band:03d}"
    return f"{prefix}{band_str}{suffix}"


bands = [70, 160, 250, 350]




"""
Check all the units and convert them to MJy/sr, which mantipython uses
"""
def check_unit(b):
    fn = data_dir + data_fn_format(b)
    header = fits.getheader(fn)
    print(b)
    print("  ", header['BUNIT'])
    try:
        print("  ", u.Unit(header['BUNIT']))
    except Exception as e:
        print(e)

def run_check_units():
    for b in bands:
        try:
            check_unit(b)
        except Exception as e:
            print(e)

def convert_unit_from_jybeam(b):
    """
    Original December 2023 version, assuming that the unit was per beam
    """
    fn = data_dir + data_fn_format(b)
    data, header = fits.getdata(fn, header=True)
    data = data * u.Jy / u.beam
    """
    500 micron beam is 36.7 arcsec FWHM
    """
    beam_fwhm = 36.7*u.arcsec
    beam = Beam(beam_fwhm)
    beam_equiv = u.beam_angular_area(beam.sr)
    data = data.to(u.MJy / u.sr, equivalencies=beam_equiv)
    header['BUNIT'] = data.unit.to_string()
    header['COMMENT'] = f"converted to MJy/sr using beam fwhm {beam_fwhm:.2f}"
    fits.PrimaryHDU(data=data.to_value(), header=header).writeto(data_dir + data_fn_format(b, suffix="-img.fits"))

def convert_unit_from_cgs(b):
    """
    March 11, 2024
    For Umit's Feb 2024 data which is in CGS erg/cm2/sr/s/Hz
    """
    fn = data_dir + data_fn_format(b)
    data, header = fits.getdata(fn, header=True)
    unit = u.Unit(header['BUNIT'])
    data = (data * unit).to(u.MJy / u.sr)
    header['BUNIT'] = data.unit.to_string()
    header['COMMENT'] = f"converted from {unit.to_string()} to {data.unit.to_string()}"
    fits.PrimaryHDU(data=data.to_value(), header=header).writeto(data_dir + data_fn_format(b, suffix="-img.fits"))

def convert_unit(b):
    """
    March 11, 2024
    Wrapper for both convert unit functions
    """
    if data_set_select == "Feb2024_Umit":
        return convert_unit_from_cgs(b)
    elif data_set_select == "Dec2023_Umit":
        return convert_unit_from_jybeam(b)
    else:
        raise RuntimeError(f"Invalid global data_set_select: <{data_set_select}>")


def run_convert_units():
    for b in bands:
        convert_unit(b)

def data_set_stub():
    """
    March 11, 2024
    filename differentiator. Leave the December ones blank since they already are
    """
    if data_set_select == "Feb2024_Umit":
        return "_feb24"
    elif data_set_select == "Dec2023_Umit":
        return ""
    else:
        raise RuntimeError(f"Invalid global data_set_select: <{data_set_select}>")


# Both data and error filenames in the same dict
data_fn_dict = {b: (data_fn_format(b, "-img.fits"), data_fn_format(b, "-err.fits")) for b in bands}

"""
Save 5% error maps for everyone.
mantipython needs error maps and they won't affect the direct result, only the
stated uncertainties. (Well I think the fitting function uses them so they do kind of matter...)
"""
def make_and_save_uncertainty_map(b, pct=5, add=0):
    """
    Save an error map which is pct x the data
    :param b: int band
    :param pct: number, error percentage
        i.e. "5" means 5% error, or 0.05 x data
    :param add: add a flat error in MJy/sr. Default 0. Helps with errors that
        might end up really small because the data is close to 0.
    """
    data_fn, err_fn = data_fn_dict[b]
    data, hdr = fits.getdata(data_dir + data_fn, header=True)
    err = data*pct/100. + add
    hdr['COMMENT'] = f"ERROR MAP, {pct} pct of data"
    fits.PrimaryHDU(data=err, header=hdr).writeto(data_dir + err_fn, overwrite=True)


def run_save_all_uncertainty_maps():
    for b in bands:
        make_and_save_uncertainty_map(b, pct=5, add=50)


def run_mantipython():
    allow_check_and_refit = True
    check_and_refit_stub = "" if allow_check_and_refit else "_norefit"
    write_fn = os.path.join(data_dir, f"NGC7538{data_set_stub()}_2p_4band_beta2.0_standardfit{check_and_refit_stub}.fits")
    log_fn_creator = lambda s: os.path.join(data_dir, f"log{s}.log")
    n_processes = 6 # My laptop has 8 threads, so up to 6 should be fine
    mantipython.fit_entire_map(data_fn_dict, [70, 160, 250, 350],
        ("T", "tau"), initial_param_vals={'beta': 2.0},
        data_directory=data_dir, log_name_func=log_fn_creator,
        n_procs=n_processes, destination_filename=write_fn,
        fitting_function='standard', allow_check_and_refit=allow_check_and_refit,
    )


def convert_tau_to_coldens(fn):
    """
    March 11, 2024
    Calculate coldens using the tau from the given solution.
    Keep the tau extension in there, but push it to the back
    """
    unit_dict = {
        "solutionT": u.K,
        "solutiontau": "optical depth at 160um",
        "model_flux70": u.MJy / u.sr,
        "model_flux160": u.MJy / u.sr,
        "model_flux250": u.MJy / u.sr,
        "model_flux350": u.MJy / u.sr,
        "diff_flux70": u.MJy / u.sr,
        "diff_flux160": u.MJy / u.sr,
        "diff_flux250": u.MJy / u.sr,
        "diff_flux350": u.MJy / u.sr,
        "chisq": u.dimensionless_unscaled,
        "errorT": u.K,
        "errortau": "optical depth at 160um",
        "n_iter": u.dimensionless_unscaled,
        "success": u.dimensionless_unscaled,
        "BAND70": u.MJy / u.sr,
        "dBAND70": u.MJy / u.sr,
        "BAND160": u.MJy / u.sr,
        "dBAND160": u.MJy / u.sr,
        "BAND250": u.MJy / u.sr,
        "dBAND250": u.MJy / u.sr,
        "BAND350": u.MJy / u.sr,
        "dBAND350": u.MJy / u.sr,
        "errorNH2_lo": u.cm**-2,
        "errorNH2_hi": u.cm**-2,
        "solutionNH2": u.cm**-2,
    }
    ext_order = [
        "solutionT",
        "solutionNH2",
        "solutiontau",
        "errorT",
        "errorNH2_lo",
        "errorNH2_hi",
        "errortau",
        "model_flux70",
        "model_flux160",
        "model_flux250",
        "model_flux350",
        "diff_flux70",
        "diff_flux160",
        "diff_flux250",
        "diff_flux350",
        "chisq",
        "n_iter",
        "success",
        "BAND70",
        "dBAND70",
        "BAND160",
        "dBAND160",
        "BAND250",
        "dBAND250",
        "BAND350",
        "dBAND350",
    ]

    fn = data_dir + fn
    Cext160 = 1.9e-25 * u.cm**2
    with fits.open(fn) as hdul:
        # for i, hdu in enumerate(hdul):
        #     if i > 0:
        #         print(hdu.header['EXTNAME'])
        tau_name = "solutiontau"
        tau_err_name = "errortau"
        log10_tau = hdul[tau_name].data
        log10_tau_err = hdul[tau_err_name].data
        log10_tau_lo = log10_tau - log10_tau_err
        log10_tau_hi = log10_tau + log10_tau_err
        tau = 10.**log10_tau
        tau_lo = 10.**log10_tau_lo
        tau_hi = 10.**log10_tau_hi
        # Coldens
        nh2 = tau / Cext160
        nh2_lo = tau_lo / Cext160
        nh2_hi = tau_hi / Cext160
        nh2_err_lo = nh2 - nh2_lo
        nh2_err_hi = nh2_hi - nh2
        nh2_name = "solutionNH2"
        nh2_lo_name = "errorNH2_lo"
        nh2_hi_name = "errorNH2_hi"

        hdu_dict = {hdu.header['EXTNAME']: (hdu.data, hdu.header) for hdu in hdul[1:]}


    hdr_template = hdu_dict[tau_name][1]
    for nh2_extname, nh2_img in zip((nh2_name, nh2_lo_name, nh2_hi_name), (nh2, nh2_err_lo, nh2_err_hi)):
        hdr = hdr_template.copy()
        hdr['EXTNAME'] = nh2_extname
        hdu_dict[nh2_extname] = (nh2_img.to_value(), hdr)

    for extname, (d, hdr) in hdu_dict.items():
        # print(hdu.header['BUNIT'])
        hdr['BUNIT'] = str(unit_dict[extname])
        # print(hdu.header['BUNIT'])

    hdu_list_to_save = [fits.PrimaryHDU()] + [fits.ImageHDU(*hdu_dict[extname]) for extname in ext_order]
    hdul = fits.HDUList(hdu_list_to_save)
    savename = fn.replace(".fits", "_withNH2.fits")
    hdul.writeto(savename, overwrite=True)


if __name__ == "__main__":
    # run_save_all_uncertainty_maps()
    # run_mantipython()
    convert_tau_to_coldens("NGC7538_feb24_2p_4band_beta2.0_standardfit.fits")
