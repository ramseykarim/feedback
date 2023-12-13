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

def data_fn_format(band, suffix=None):
    """
    :param band: int PACS or SPIRE wavelength
    :returns: path to the NGC 7538 data.
        relative to the data_dir defined globally.
    """
    if suffix is None:
        suffix = "_repro_500.fits"
    if band > 200:
        prefix = "spire"
    else:
        prefix = "pacs"
    return f"{prefix}{band:03d}{suffix}"


bands = [70, 160, 250, 350, 500]




"""
Check all the units and convert them to MJy/sr, which mantipython uses
"""
def check_unit(b):
    fn = data_dir + data_fn_format(b)
    header = fits.getheader(fn)
    print(header['BUNIT'])

def run_check_units():
    for b in bands:
        check_unit(b)

def convert_unit(b):
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


def run_convert_units():
    for b in bands:
        convert_unit(b)



# Both data and error filenames in the same dict
data_fn_dict = {b: (data_fn_format(b, "-img.fits"), data_fn_format(b, "-err.fits")) for b in bands}

"""
Save 5% error maps for everyone.
mantipython needs error maps and they won't affect the direct result, only the
stated uncertainties.
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
        make_and_save_uncertainty_map(b, pct=10, add=50)


def run_mantipython():
    write_fn = os.path.join(data_dir, "NGC7538_2p_4band_long_beta2.0_standardfit_norefit_softerrors.fits")
    log_fn_creator = lambda s: os.path.join(data_dir, f"log{s}.log")
    n_processes = 6 # My laptop has 8 threads, so up to 6 should be fine
    mantipython.fit_entire_map(data_fn_dict, [160, 250, 350, 500],
        ("T", "tau"), initial_param_vals={'beta': 2.0},
        data_directory=data_dir, log_name_func=log_fn_creator,
        n_procs=n_processes, destination_filename=write_fn,
        fitting_function='standard', allow_check_and_refit=False,
    )

if __name__ == "__main__":
    # run_save_all_uncertainty_maps()
    run_mantipython()
