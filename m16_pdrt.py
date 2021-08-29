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

m = ModelSet(name='wk2006', z=1)
mp = ModelPlot(m)
data_dir = "/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt"

m16_measurements = []
for i in range(10):
    m16_measurements.append(Measurement.from_table(os.path.join(data_dir, f"cii_pillar1_{i}.txt")))
    m16_measurements.append(Measurement.from_table(os.path.join(data_dir, f"cii_to_co10_pillar1_{i}.txt")))

mp.phasespace(['CII_158', 'CII_158/CO_10'], nax1_clip=[1E2,1E5]*u.Unit("cm-3"),
    nax2_clip=[1E1,1E6]*utils.habing_unit, measurements=m16_measurements,
    label=None,fmt=['k+', 'r+', 'g+', 'b+', 'y+', 'c+', 'ko', 'ro', 'go', 'yo'], title='M16 Pillar 1')

mp.savefig("/home/ramsey/Pictures/2021-08-24-work/cii_co_pdrt_fixedG0_p1.png")
