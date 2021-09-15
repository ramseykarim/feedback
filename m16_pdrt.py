"""
Created: August 24, 2021
Putting the relevant parts of PDR_Example_Model_Plotting-RamseyCopy.ipynb
in here as a script that can be run and modified quickly.
"""
__author__ = "Ramsey Karim"

import os

from pdrtpy.modelset import ModelSet
from pdrtpy.plot.modelplot import ModelPlot
from pdrtpy.measurement import Measurement
import pdrtpy.pdrutils as utils
from astropy.nddata import StdDevUncertainty
import astropy.units as u


data_dir = "/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt"
if not os.path.isdir(data_dir):
    data_dir = "/home/rkarim/Research/Feedback/m16_data/catalogs/pdrt"

pillar = 1


def make_plot():
    """
    Created: Aug 24, 2021
    """
    # Boilerplate from the notebook
    m = ModelSet(name='wk2006', z=1)
    mp = ModelPlot(m)
    # create list of measurements for the plot
    m16_measurements = []
    pillar = 1
    # Loop through the 10 regions
    for i in range(10):
        # Add to the list the CII/CO1-0 measurement and the CII/FIR measurement with the correct labels
        cii_m = Measurement.read(os.path.join(data_dir, f"cii_pillar{pillar}_{i}.fits"), identifier='CII_158')
        co10_m = Measurement.read(os.path.join(data_dir, f"co10_pillar{pillar}_{i}.fits"), identifier='CO_10')
        fir_m = Measurement.read(os.path.join(data_dir, f"fir_pillar{pillar}_{i}.fits"), identifier='FIR')
        cii_m_converted = utils.convert_integrated_intensity(cii_m)
        # cii_m_converted.header['BUNIT'] = 'erg / (cm2 s sr)' # Marc's fix
        cii_to_fir_m = cii_m_converted / fir_m
        #### at this point, the cii_to_fir_m has a unit of cm2/cm2 which should have reduced
        #### Marc recommends papering over it by assigning everything to unitless, but I'll see if it'll run without
        co10_m_converted = utils.convert_integrated_intensity(co10_m)
        cii_to_co10_m = cii_m_converted / co10_m_converted
        m16_measurements.append(cii_to_co10_m)
        m16_measurements.append(cii_m_converted)
    # Create the plot using the same labels
    mp.phasespace(['CII_158', 'CII_158/CO_10'], nax1_clip=[1E2,1E5]*u.Unit("cm-3"), nax2_clip=[1E1,1E6]*utils.habing_unit, measurements=m16_measurements, label=None,fmt=['k+', 'r+', 'g+', 'b+', 'y+', 'c+', 'ko', 'ro', 'go', 'yo'], title=f'M16 Pillar {pillar}')
    mp.savefig(f"/home/ramsey/Pictures/2021-09-14-work/cii_co_pdrt_p{pillar}.png")


def make_measurements_from_fits():
    """
    Created: Sept 8, 2021
    This is a direct follow-up to m16_deepdive.prepare_pdrt_tables_2

    Updated: Sept 14, 2021 following Marc's email working through my pilot error
    The issue was that I need to manually encourage a unit conversion
    I was also not making tables, I was making "measurements" which are FITS.
    That's fine, Marc said, it'll work fine for the phase space plots.
    """
    # Filepath to the masked FITS images
    tmp_path = "/home/ramsey/Downloads/tmp"
    if not os.path.isdir(tmp_path):
        tmp_path = "/home/rkarim/Downloads/tmp"
    # Loop through the 10 regions
    for i in range(10):
        # Create and save CII measurement from masked images
        cii_m_fn = os.path.join(data_dir, f"cii_pillar{pillar}_{i}.fits")
        Measurement.make_measurement(os.path.join(tmp_path, f"reg_p{pillar}_cii_{i}.fits"), error='10%', outfile=cii_m_fn)

        # Create and save CO1-0 measurement from masked images
        co10_m_fn = os.path.join(data_dir, f"co10_pillar{pillar}_{i}.fits")
        Measurement.make_measurement(os.path.join(tmp_path, f"reg_p{pillar}_co10_{i}.fits"), error='10%', outfile=co10_m_fn)

        # Load the CII and CO1-0 measurements I just made
        cii_m = Measurement.read(cii_m_fn)
        co10_m = Measurement.read(co10_m_fn)
        # Create and save the ratio measurement
        cii_to_co10_m = cii_m / co10_m
        cii_to_co10_m.write(os.path.join(data_dir, f"cii_to_co10_pillar{pillar}_{i}.fits"))

        # Create and save FIR measurement from the masked image
        fir_m_fn = os.path.join(data_dir, f"fir_pillar{pillar}_{i}.fits")
        Measurement.make_measurement(os.path.join(tmp_path, f"reg_p{pillar}_fir_{i}.fits"), error='10%', outfile=fir_m_fn)
        # Load the FIR measurement I just made
        fir_m = Measurement.read(fir_m_fn)
        # Create and save the CII to FIR ratio measurement
        cii_to_fir_m = cii_m / fir_m
        cii_to_fir_m.write(os.path.join(data_dir, f"cii_to_fir_pillar{pillar}_{i}.fits"))


if __name__ == "__main__":
    make_plot()
