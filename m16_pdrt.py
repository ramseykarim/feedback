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
from astropy.table import Table

from copy import deepcopy

data_dir = "/home/ramsey/Documents/Research/Feedback/m16_data/catalogs/pdrt"
if not os.path.isdir(data_dir):
    data_dir = "/home/rkarim/Research/Feedback/m16_data/catalogs/pdrt"

def get_measurement_filename(line_stub):
    return os.path.join(data_dir, f"{line_stub}__pillar1_pointsofinterest_v3.txt")

user_models = {'CO_65/FIR': ('CO65_FIR.fits', "CO(J=6-5) / I$_{FIR}$")}

pillar = 1


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

    meas_list = []
    reg_name_list = get_region_names()
    for reg_name in reg_name_list:
        meas_list += [collect_measurements_from_tables(meas, reg_name=reg_name) for meas in (meas_x, meas_y)]
    # ['k+', 'r+', 'g+', 'b+', 'y+', 'c+', 'ko', 'ro', 'go', 'yo']
    markers = ['k^', 'k<', 'k+', 'r+', 'r^', 'kv', 'rv', 'r<']

    m = ModelSet(name='wk2020', z=1)
    # for x in m._supported_ratios['ratio label']:
    #     print(x)
    mp = ModelPlot(m)

    mp.phasespace([meas_list[0].id, meas_list[1].id], nax1_clip=[1E2,1E5]*u.Unit("cm-3"), nax2_clip=[1E1,1E6]*utils.habing_unit, measurements=meas_list, fmt=markers, label=reg_name_list)
    # 2022-09-21,
    mp.savefig(f"/home/ramsey/Pictures/2022-09-29/phasespace_{meas_list[0].id.replace('/', 'over')}_{meas_list[1].id.replace('/', 'over')}.png")



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


def collect_all_measurements_for_region(reg_name):
    """
    Created: September 27, 2022
    Run collect_measurements_from_tables on every supported line intensity
    """
    supported_line_stubs = ['cii', 'co65CONV', 'FIR', '12co10CONV', '12co32'] #  , '13co32', '13co10CONV'
    result = []
    for line in supported_line_stubs:
        result.append(collect_measurements_from_tables(line, reg_name=reg_name))
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
    for m in meas_list:
        print(m.id, m)
    return
    ms = ModelSet("wk2020", z=1)
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
        # co65_meas = [x for x in meas_list if "65" in x.id].pop()
        # fir_meas = [x for x in meas_list if "FIR" in x.id].pop()
        # new_meas_list = list(meas_list) + [co65_meas / fir_meas]
        # print([x.id for x in new_meas_list])
        # return
        new_meas_list = [x for x in meas_list if "FIR" not in x.id]
        lrp_plot.overlay_all_ratios(yaxis_unit="Habing", figsize=(15, 10),
            loc='upper left', measurements=new_meas_list)
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

    # 2022-09-28, (27 ?), 29
    save_path = "/home/ramsey/Pictures/2022-09-29"

    if chisq:
        # lrp_plot._plt.subplots_adjust(right=0.6)
        lrp_plot.savefig(os.path.join(save_path, f"chisq_{reg_name}.png"),
            metadata={'Author': "Ramsey Karim", 'Source': f'{__file__}.make_spaghetti_plot',
                'Title': f'{reg_name} chisq plot'})
    else:
        lrp_plot.savefig(os.path.join(save_path, f"spaghetti_{reg_name}.png"),
            metadata={'Author': "Ramsey Karim", 'Source': f'{__file__}.make_spaghetti_plot',
                'Title': f'{reg_name} spaghetti plot'})


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


if __name__ == "__main__":
    make_phase_space_plot_3("cii/FIR", "12co32/12co10CONV")
