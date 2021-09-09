"""
Created: August 24, 2021
Putting the relevant parts of PDR_Example_Model_Plotting-RamseyCopy.ipynb
in here as a script that can be run and modified more quickly.
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


def make_plot():
    """
    Created: Aug 24, 2021
    """
    m = ModelSet(name='wk2006', z=1)
    mp = ModelPlot(m)
    m16_measurements = []
    pillar = 1
    for i in range(10):
        m16_measurements.append(Measurement.read(os.path.join(data_dir, f"cii_pillar{pillar}_{i}.fits"), identifier='CII_158'))
        m16_measurements.append(Measurement.read(os.path.join(data_dir, f"cii_to_co10_pillar{pillar}_{i}.fits"), identifier='CII_158/CO_10'))
    mp.phasespace(['CII_158', 'CII_158/CO_10'], nax1_clip=[1E2,1E5]*u.Unit("cm-3"),
        nax2_clip=[1E1,1E6]*utils.habing_unit, measurements=m16_measurements,
        label=None,fmt=['k+', 'r+', 'g+', 'b+', 'y+', 'c+', 'ko', 'ro', 'go', 'yo'], title='M16 Pillar 1')
    mp.savefig("/home/ramsey/Pictures/2021-09-08-work/cii_co_pdrt_fixedG0_p1.png")


def make_measurement_tables_from_fits():
    """
    Created: Sept 9, 2021
    This is a direct follow-up to m16_deepdive.prepare_pdrt_tables_2
    """
    tmp_path = "/home/ramsey/Downloads/tmp"
    pillar = 1
    for i in range(10):
        cii_m_fn = os.path.join(data_dir, f"cii_pillar{pillar}_{i}.fits")
        Measurement.make_measurement(os.path.join(tmp_path, f"reg_p{pillar}_cii_{i}.fits"), error='10%', outfile=cii_m_fn)
        cii_m = Measurement.read(cii_m_fn)
        co10_m_fn = os.path.join(data_dir, f"co10_pillar{pillar}_{i}.fits")
        Measurement.make_measurement(os.path.join(tmp_path, f"reg_p{pillar}_co10_{i}.fits"), error='10%', outfile=co10_m_fn)
        co10_m = Measurement.read(co10_m_fn)
        cii_to_co10_m = cii_m / co10_m
        cii_to_co10_m.write(os.path.join(data_dir, f"cii_to_co10_pillar{pillar}_{i}.fits"))

if __name__ == "__main__":
    make_plot()
