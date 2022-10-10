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
from pdrtpy.tool.lineratiofit import LineRatioFit
from pdrtpy.plot.lineratioplot import LineRatioPlot

from astropy.nddata import StdDevUncertainty
import astropy.units as u
from astropy.table import Table, QTable

from copy import deepcopy

data_dir = "/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt"
if not os.path.isdir(data_dir):
    data_dir = "/home/rkarim/Research/Feedback/m16_data/catalogs/pdrt"

# Ratio of beam areas of the CO(3-2) to CII beam
# CO(3-2) is around 19'' fwhm, CII is around 15''
APEX_to_SOFIA_beam_area_ratio = 1.54853702


# Whether to just use my converted units, or start from K km/s and let pdrtpy
# make the conversion to cgs
# I checked my conversions, they're good for at least CII and CO(1-0) and I
# expect that they're good for all the others.
use_my_unit_conversions = True

def get_measurement_filename(line_stub):
    """
    Late September/early October 2022
    Get the measurement table filename for a given line
    If I switch to a different set of tables (e.g. from P1a to the other pillars),
    I can just change that here
    """
    default_fn = os.path.join(data_dir, f"{line_stub}__pillar1_pointsofinterest_v3.txt")
    if use_my_unit_conversions:
        # Use the tables which I already converted to cgs
        # This should be the most common behavior, since the conversions are fine
        fn = default_fn
    else:
        # Use the K km/s tables (only available for some lines in P1a right now)
        fn = os.path.join(data_dir, f"{line_stub}_v2__pillar1_pointsofinterest_v3.txt")
        if not os.path.exists(fn):
            fn = default_fn
    return fn

# Models that I created from existing models and saved as FITS
user_models = {'CO_65/FIR': ('CO65_FIR.fits', "CO(J=6-5) / I$_{FIR}$")}

default_supported_line_stubs = {'cii', 'co65CONV', 'FIR', '12co10CONV'} #  '12co32', '13co32', '13co10CONV'
def set_supported_lines(line_name_list):
    """
    October 7, 2022
    Reset supported lines dynamically. This lets me pick lines at the bottom of
    this file in a function call rather than changing the contents of a function.
    Reuse the same set instance (same memory location) rather than reassigning
    the default_supported_line_stubs variable to a new set
    """
    global default_supported_line_stubs
    default_supported_line_stubs.clear()
    default_supported_line_stubs |= set(line_name_list)



# Useful for switching to other sets of measurement tables
pillar = 1




"""
Plotting functions
"""



def add_model():
    """
    Created: September 28, 2022
    Add a model to the set and save it to FITS somwhere
    """
    ms = ModelSet("wk2020", z=1)

    cii_fir = ms.get_model("CII_158/FIR")
    cii_co65 = ms.get_model("CII_158/CO_65")
    co65_fir = cii_fir / cii_co65
    text1 = "CO (J=6-5)/FIR"
    text2 = "CII/FIR / CII/CO65"
    savename = "CO65_FIR"

    old_model = cii_fir
    new_model = co65_fir

    new_model.header = deepcopy(old_model.header)
    new_model.header["TITLE"] = text1
    new_model.header["DATAMAX"] = new_model.data.max()
    new_model.header["DATAMIN"] = new_model.data.min()
    new_model.header["HISTORY"] = f"Computed arithmetically from {text2}"
    new_model.header["DATE"] = utils.now()

    new_model.write(os.path.join(data_dir, savename+".fits"))


def load_user_model(ms, modelname):
    """
    September 28, 2022
    load a model I saved as FITS into a ModelSet
    """
    fn_stub, title = user_models[modelname]
    fn = os.path.join(data_dir, fn_stub)
    ms.add_model(modelname, fn, title=title)


def load_all_user_models(ms):
    """
    September 28, 2022
    load all models I created
    """
    for modelname in user_models:
        load_user_model(ms, modelname)


def make_phase_space_plot_3(meas_x, meas_y):
    """
    Created: September 21, 2022
    Making phase space plots with my new measurements/tables
    This time, 2 ratios at a time

    # TODO: need to fix this since collect_measurement_from_tables no longer makes ratios
    """

    meas_list = []
    reg_name_list = get_region_names()
    for reg_name in reg_name_list:
        meas_list += [collect_measurement_from_tables(meas, reg_name=reg_name) for meas in (meas_x, meas_y)]
    # ['k+', 'r+', 'g+', 'b+', 'y+', 'c+', 'ko', 'ro', 'go', 'yo']
    markers = ['k^', 'k<', 'k+', 'r+', 'r^', 'kv', 'rv', 'r<']

    m = ModelSet(name='wk2020', z=1)
    # for x in m._supported_ratios['ratio label']:
    #     print(x)
    mp = ModelPlot(m)

    mp.phasespace([meas_list[0].id, meas_list[1].id], nax1_clip=[1E2,1E5]*u.Unit("cm-3"), nax2_clip=[1E1,1E6]*utils.habing_unit, measurements=meas_list, fmt=markers, label=reg_name_list)
    # 2022-09-21,
    mp.savefig(f"/home/ramsey/Pictures/2022-09-29/phasespace_{meas_list[0].id.replace('/', 'over')}_{meas_list[1].id.replace('/', 'over')}.png")


def collect_measurement_from_tables(line_name, reg_name=None):
    """
    Created: September 21, 2022
    Get any Measurement from the tables and organize them by region.
    :param line_name: name of line (or other single quantity, like FIR)
    :param reg_name: If a reg_name is specified, only return those
        If this region isn't found in the file (e.g. for CI or OI), return None
    :returns: a Measurement; single pixel if reg_name specified
    """
    supported_line_stubs = ['cii', '12co10CONV', '13co10CONV', '12co32', '13co32', 'co65CONV', 'FIR']
    fn = get_measurement_filename(line_name)

    if use_my_unit_conversions:
        # I converted from K km/s using my own function when I created the tables.
        # I checked the conversion using the pdrtpy function and they give the
        # same answer.
        if reg_name is not None:
            # Find the location of this region's row
            t = Table.read(fn, format='ipac')
            reg_name_list = t['region']
            reg_i = list(reg_name_list).index(reg_name)
            meas = Measurement.from_table(fn, array=True)[reg_i]
        else:
            # Get all regions in one Measurement
            meas = Measurement.from_table(fn, array=False)
        return_val = meas

    else:
        # Use the pdrtpy function convert_integrated_intensity to convert from
        # K km/s to cgs
        t = QTable.read(fn, format='ipac')
        # # TODO: this is untested; not sure if t[i] returns Table-like (probably should)
        if reg_name is not None:
            reg_name_list = t['region']
            reg_i = list(reg_name_list).index(reg_name)
            t = t[reg_i] # Try to make a single-row table, not just a "Row"
        # Create measurement from table manually so we can set the rest_freq
        meas = Measurement(
            data=t['data'].value, uncertainty=StdDevUncertainty(t['uncertainty'].value),
            identifier=t['identifier'][0], restfreq=str(t['rest_freq'][0]),
            unit=str(t['data'].unit)
        )
        return_val = utils.convert_integrated_intensity(meas)
    # Modify the CO(3-2) value by the beam ratio
    # This assumes that CO(3-2) is beam-diluted and that the area of the beam
    # not covered by pillar sees no emission (rather than non-zero background)
    # It's a good enough assumption to try to get something out of the CO(3-2)
    # but it is probably only valid along threads and definitely not towards
    # the head of P1 (towards anything larger than an APEX beam)
    if 'co32' in line_name:
        return_val = return_val * APEX_to_SOFIA_beam_area_ratio
    return return_val


def collect_all_measurements_for_region(reg_name):
    """
    Created: September 27, 2022
    Run collect_measurement_from_tables on every supported line intensity
    """
    result = []
    for line in default_supported_line_stubs:
        result.append(collect_measurement_from_tables(line, reg_name=reg_name))
    # Filter out None values, which will be returned by collect_measurement_from_tables
    # in the case that that region isn't available (CI and OI don't have full coverage)
    return result


def get_region_names():
    """
    September 29, 2022
    Use the g0 tables to get a list of region names
    """
    fn = get_measurement_filename("uv_m16_repro_CII")
    t = Table.read(fn, format='ipac')
    return list(t['region'])


def get_g0_values_at_locations(reg_name):
    """
    Created: September 21, 2022
    Get the G0 from Herschel (Nicola made this) and also the one from the
    stars that I made, return the two values as a tuple
    :param reg_name: the name of the region
    :return: dict(dict, dict)
        main dictionary keys 'Herschel_G0', 'Stars_G0'
        sub-dictionaries keys 'data', 'uncertainty', 'region'
        in a tuple ordered (Herschel, Stars)
    """
    fns = ["uv_m16_repro_CII", "g0_hillenbrand_stars_fuvgt4-5_ltxarcmin"]
    result = {}
    for raw_fn in fns:
        fn = get_measurement_filename(raw_fn)
        t = Table.read(fn, format='ipac')
        reg_name_list = t['region']
        reg_i = list(reg_name_list).index(reg_name)
        result[t['identifier'][reg_i]] = dict(t[reg_i])
    return result


def make_spaghetti_plot(reg_name, chisq=False):
    """
    Created: September 27, 2022
    Following the notebook: https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Find_n_G0_Single_Pixel.ipynb
    Make a spaghetti plot for a given region
    """
    meas_list = collect_all_measurements_for_region(reg_name)
    """
    # TODO: the KOSMA-tau models work a little differently, and I should make sure
    that I am working within those bounds

    I think I will have to remove the overlays of intensities (says CII_158 is not supported)
    """
    # modelset_name = "kt2013wd01-7" # "wk2020"
    # ms = ModelSet(modelset_name, z=1, medium='clumpy', mass=100.0)
    modelset_name = 'wk2020'
    ms = ModelSet(modelset_name, z=1)
    # print(ms._supported_lines)
    # print(ms._supported_ratios)
    # return
    load_all_user_models(ms)
    p = LineRatioFit(ms, measurements=meas_list)
    p.run()
    lrp_plot = LineRatioPlot(p)

    # add to the default color cycle since I have lots of lines/ratios. Must be done before I call the overlay plot function
    lrp_plot._CB_color_cycle += ['#66ffb3', '#ff8000', '#e600e6', '#8533ff', '#999900']

    if chisq:
        lrp_plot.reduced_chisq(cmap='gray_r',norm='log',label=True,colors='white',
            legend=True,vmax=8E4,figsize=(15,10),yaxis_unit='Habing')
    else:
        lrp_plot.overlay_all_ratios(yaxis_unit="Habing",
            figsize=(15, 10), loc='upper left',
            measurements=[x for x in meas_list if "FIR" not in x.id])
            # loc='upper left',
            # bbox_to_anchor=(1.05,0.9))

    g0_dict = get_g0_values_at_locations(reg_name)
    g0_plot_params = {'Stars_G0': ('#1f77b4', 'bottom'), 'Herschel_G0': ('#ff7f0e', 'top')}
    for g0_name in g0_dict:
        color, va = g0_plot_params[g0_name]
        lrp_plot._plt.axhline(g0_dict[g0_name]['data'], linestyle='--', color=color)
        lrp_plot._plt.text(15, g0_dict[g0_name]['data'], g0_name.replace('_G0', ' $G_0$'), color=color, fontsize='large', va=va)


    dens, dens_unc = p.density.value, p.density.uncertainty.array
    radfield_meas = utils.to(utils.habing_unit, p.radiation_field)
    radfield, radfield_unc = radfield_meas.value, radfield_meas.uncertainty.array
    lrp_plot._plt.errorbar(dens, radfield, xerr=dens_unc, yerr=radfield_unc, color='k')

    # 2022-09-28, (27 ?), 29, 10-05,6,7
    save_path = f"/home/ramsey/Pictures/2022-10-07/{modelset_name}"
    if not os.path.exists(save_path):
        print("Creating directory ", save_path)
        os.makedirs(save_path)

    if chisq:
        # lrp_plot._plt.subplots_adjust(right=0.6)
        lrp_plot.savefig(os.path.join(save_path, f"chisq_{reg_name}.png"),
            metadata={'Author': "Ramsey Karim", 'Source': f'{__file__}.make_spaghetti_plot',
                'Title': f'{reg_name} chisq plot'})
    else:
        lrp_plot.savefig(os.path.join(save_path, f"spaghetti_{reg_name}.png"),
            metadata={'Author': "Ramsey Karim", 'Source': f'{__file__}.make_spaghetti_plot',
                'Title': f'{reg_name} spaghetti plot'})



# line names
# 'cii', 'co65CONV', 'FIR', '12co10CONV', '12co32', '13co32', '13co10CONV'

if __name__ == "__main__":
    set_supported_lines(['cii', 'co65CONV', 'FIR', '12co10CONV', '12co32'])
    # for d in ['E', 'S', 'W']:
    #     for i in range(2):
    #         make_spaghetti_plot(d+'-peak', chisq=i)

    # r = 'broad-line'
    # for i in range(2):
    #     make_spaghetti_plot(r, chisq=i)

    for ns in 'NS':
        for ew in 'EW':
            for i in range(2):
                make_spaghetti_plot(f'{ns}{ew}-thread', chisq=i)
