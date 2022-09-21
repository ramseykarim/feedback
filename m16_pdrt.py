"""
Created: August 24, 2021
Putting the relevant parts of PDR_Example_Model_Plotting-RamseyCopy.ipynb
in here as a script that can be run and modified quickly.

August 15, 2022:
I will make spaghetti plots, and I will build a measurement-creating function
in m16_deepdive.prepare_pdrt_tables_3
"""
__author__ = "Ramsey Karim"

import os

from pdrtpy.modelset import ModelSet
from pdrtpy.plot.modelplot import ModelPlot
from pdrtpy.measurement import Measurement
import pdrtpy.pdrutils as utils
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from astropy.table import Table


data_dir = "/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt"
if not os.path.isdir(data_dir):
    data_dir = "/home/rkarim/Research/Feedback/m16_data/catalogs/pdrt"

def get_measurement_filename(line_stub):
    return os.path.join(data_dir, f"{line_stub}__pillar1_pointsofinterest_v3.txt")

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
    mp.phasespace(['CII_158', 'CII_158/CO_10'], nax1_clip=[1E2,1E5]*u.Unit("cm-3"), nax2_clip=[1E1,1E6]*utils.habing_unit, measurements=m16_measurements, label=None, fmt=['k+', 'r+', 'g+', 'b+', 'y+', 'c+', 'ko', 'ro', 'go', 'yo'], title=f'M16 Pillar {pillar}')
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


def make_phase_space_plot_2(l1, l2):
    """
    Created: September 21, 2022
    Making phase space plots with my new measurements/tables

    I didn't realize Marc already had a function utils.convert_integrated_intensity
    I should probably check my conversion using his function
    """
    fn1 = get_measurement_filename(l1)
    fn2 = get_measurement_filename(l2)
    m1 = Measurement.from_table(fn1, array=False)
    m2 = Measurement.from_table(fn2, array=False)
    ratios = m1 / m2

    m = ModelSet(name='wk2020', z=1)
    # for x in m._supported_ratios['ratio label']:
    #     print(x)
    mp = ModelPlot(m)

    mp.phasespace([m1.id, ratios.id], nax1_clip=[1E2,1E5]*u.Unit("cm-3"), nax2_clip=[1E1,1E6]*utils.habing_unit, measurements=[m1, ratios])
    mp.savefig(f"/home/ramsey/Pictures/2022-09-21/phasespace_{m1.id}_{ratios.id.replace('/', 'over')}_1.png")


def make_phase_space_plot_3(meas_x, meas_y):
    """
    Created: September 21, 2022
    Making phase space plots with my new measurements/tables
    This time, 2 ratios at a time
    """

    meas_list = [collect_measurements_from_tables(meas) for meas in (meas_x, meas_y)]

    m = ModelSet(name='wk2020', z=1)
    # for x in m._supported_ratios['ratio label']:
    #     print(x)
    mp = ModelPlot(m)

    mp.phasespace([meas_list[0].id, meas_list[1].id], nax1_clip=[1E2,1E5]*u.Unit("cm-3"), nax2_clip=[1E1,1E6]*utils.habing_unit, measurements=meas_list)
    mp.savefig(f"/home/ramsey/Pictures/2022-09-21/phasespace_{meas_list[0].id.replace('/', 'over')}_{meas_list[1].id.replace('/', 'over')}.png")



def collect_measurements_from_tables(line_or_ratio, reg_name=None):
    """
    Created: September 21, 2022
    Get any Measurement from the tables, make any valid ratio, and
    organize them by region.
    :param reg_name: If a reg_name is specified, only return those
    :returns: a Measurement; single pixel if reg_name specified
    """
    supported_line_stubs = ['cii', '12co10CONV', '13co10CONV', '12co32', '13co32', 'co65CONV', 'FIR']
    if '/' in line_or_ratio:
        line_or_ratio = [x.strip() for x in line_or_ratio.split('/')]
    elif isinstance(line_or_ratio, str):
        line_or_ratio = [line_or_ratio]

    if len(line_or_ratio) == 1:
        is_ratio = False
    elif len(line_or_ratio) == 2:
        is_ratio = True
    else:
        raise ValueError(f"Number of molecular lines can't be {len(line_or_ratio)} ({line_or_ratio})")

    fns = [get_measurement_filename(l) for l in line_or_ratio]
    meas_list = []
    if reg_name is not None:
        # Find the location of this region's row
        reg_i = None
        for fn in fns:
            if reg_i is None:
                t = Table.read(fn, format='ipac')
                reg_name_list = t['region']
                reg_i = list(reg_name_list).index(reg_name)
            meas_list.append(Measurement.from_table(fn, array=True)[reg_i])
    else:
        for fn in fns:
            # Get all regions in one Measurement
            meas_list.append(Measurement.from_table(fn, array=False))
    # Take ratio if we're doing that, otherwise just get the single value
    if is_ratio:
        meas = meas_list[0] / meas_list[1]
    else:
        meas = meas_list[0]
    return meas



def get_g0_values_at_locations(reg_name):
    """
    Created: September 21, 2022
    Get the G0 from Herschel (Nicola made this) and also the one from the
    stars that I made, return the two values as a tuple
    :param reg_name: the name of the region
    """
    fns = ["uv_m16_repro_CII", "g0_hillenbrand_stars_fuvgt4-5_ltxarcmin"]
    # for loop
    t = Table.read(fn, format='ipac')
    reg_name_list = t['region']
    reg_i = list(reg_name_list).index(reg_name)
    # # TODO: FINISH THIS! return a tuple




if __name__ == "__main__":
    # make_phase_space_plot_3('cii/FIR', '12co32/12co10CONV')
    get_g0_values()
