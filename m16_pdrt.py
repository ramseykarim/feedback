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
import numpy as np

from pdrtpy.modelset import ModelSet
from pdrtpy.plot.modelplot import ModelPlot
from pdrtpy.measurement import Measurement
import pdrtpy.pdrutils as utils
from pdrtpy.tool.lineratiofit import LineRatioFit
from pdrtpy.plot.lineratioplot import LineRatioPlot

from astropy.nddata import StdDevUncertainty
import astropy.units as u
from astropy.table import Table, QTable

from matplotlib.lines import Line2D

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

# For use when we do not have access to tables or FITS headers; values in Hz
line_frequencies = {'cii': 0.1900536900000E+13, '12co10': 1.15271204000E+11, '12co32': 0.3457959899000E+12, 'co65': 0.6914730000000E+12,}
# Copy of the dict in cube_utils.py; not importing anything to this file, and I only need a few of these. Rarely used except when I'm using the Gaussian-modeled intensities
cubeIDs_pdrt = {
    'cii': 'CII_158',
    '12co10': 'CO_10', '12co32': 'CO_32', 'co65': 'CO_65', '13co10': '13CO_10', '13co32': '13CO_32',
    'oi': 'OI_63', 'ci': 'CI_609',
}

filename_gen = [] # fill this each time
available_filenames = {
    'p1a': (lambda line_stub: f"{line_stub}__pillar1_pointsofinterest_v3.txt"),
    'misc': (lambda line_stub: f"{line_stub}__pillar123_pointsofinterest_v1.txt"),
    'all': (lambda line_stub: f"{line_stub}__pillar123_pointsofinterest_v2.txt"),
}

"""
A collection of manually entered Measurements; these are primarily from the
Gaussian-fitted line profiles, so each Measurement represents not a directly
measured integrated intensity, but one derived from modeled Gaussian parameters.
Multiple Gaussians were fitted to each line profile, so the solutions are not
necessarily unique, but may be interesting in some cases.
These are taken from the spreadsheet line_info_2.ods; the image model_fit_table_viz.png
is also made from this table and is a good way to vizualize all these fitted components.
All values in K km/s; will need to be converted using line frequencies (see line_frequencies dictionary)
"""
manually_entered_observations = {
    # First broad line entry is an approximation of the blue component; CII and CO65 are blue, CO10 and CO32 are green
    # Second broad line entry is red
    'broad-line': (
        (('cii', 67.4009783393804), ('12co10', 60.6854705288165), ('12co32', 21.9329974030212), ('co65', 12.8810613776738)),
        (('cii', 28.0278640527865), ('12co10', 55.165875068079), ('12co32', 16.2479644761581), ('co65', 16.2249034960315)),
    ),
    # First E-peak component is blue; all components are blue
    # Second E-peak component is red
    'E-peak': (
        (('cii', 23.7538121817132), ('12co10', 29.1646199753317), ('12co32', 15.1776342028907), ('co65', 5.7604824379295)),
        (('cii', 41.1839025521873), ('12co10', 96.6994522645774), ('12co32', 54.6219367324841), ('co65', 20.4555906979537))
    ),
    # 'S-peak': ()
}
def make_measurement_list_from_manual_entry(region_stub_with_index):
    """
    October 12, 2022
    Use the dictionary manually_entered_observations to create a Measurement
    :returns: single-pixel Measurement
    """
    # Parse the region stub with index; like 'broad-line-1' or 'broad-line-2' (1-indexed)
    split_region_stub_with_index = region_stub_with_index.split('-')
    region_stub = '-'.join(split_region_stub_with_index[:-1])
    index = int(split_region_stub_with_index[-1]) - 1 # transform 1-index to 0-index
    print(f"Loading manually entered data for {region_stub} # {index}:")
    # Get the tuple of length number_of_lines
    # Each value into the tuple is a 2-tuple of (line_stub, integrated_intensity_Kkms)
    tuple_of_info = manually_entered_observations[region_stub][index]
    print(tuple_of_info)
    meas_list = []
    # Iterate thru and fill meas_list with one Measurement from each line
    for line_stub, int_intens_kkms in tuple_of_info:
        # Get rest_freq from the dictionary
        rest_freq = line_frequencies[line_stub] * u.Hz
        # Initialize the Measurement, assume a 10% uncertainty (for now at least)
        meas = Measurement(
            data=int_intens_kkms, uncertainty=StdDevUncertainty(int_intens_kkms/10.), # 10% uncertainty (total guess, not scientifically estimated)
            identifier=cubeIDs_pdrt[line_stub], restfreq=str(rest_freq),
            unit=u.K*u.km/u.s
        )
        meas_list.append(utils.convert_integrated_intensity(meas))
    return meas_list


def get_measurement_filename(line_stub):
    """
    Late September/early October 2022
    Get the measurement table filename for a given line
    If I switch to a different set of tables (e.g. from P1a to the other pillars),
    I can just change that here
    """
    try:
        default_fn = os.path.join(data_dir, filename_gen[0](line_stub))
    except IndexError as e:
        raise RuntimeError("You probably forgot to fill in filename_gen with an available_filenames value") from e
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

default_supported_line_stubs = {'cii', 'oiCONV', 'co65CONV', 'FIR', '12co10CONV'} #  '12co32', '13co32', '13co10CONV'
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
        # I am not going to use this option often, use the other one (cgs tables)
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

    # Correct FIR for background of 0.14 +/- 0.02 (added this in November 22, 2022)
    # if 'FIR' in line_name:
    #     fir_background_meas = Measurement(data=0.14, uncertainty=StdDevUncertainty(0.02), unit=(u.erg/(u.s * u.sr * u.cm**2)))
    #     return_val = return_val - fir_background_meas
    #     return_val.identifier('FIR') # the subtraction changes the id, so we have to reset it

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
    result = [x for x in result if not np.isnan(x.data)]
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
        in a tuple ordered (Herschel, Stars)(? was this line left in?)
    """
    if reg_name[-1].isdigit() and reg_name[-2]=='-':
        # Assume we're doing something like 'broad-line-1' and get rid of '-1'
        reg_name = reg_name[:-2]
    fns = ["uv_m16_repro_CII", "g0_hillenbrand_stars_fuvgt4-5_ltxarcmin"]
    result = {}
    for raw_fn in fns:
        fn = get_measurement_filename(raw_fn)
        t = Table.read(fn, format='ipac')
        reg_name_list = t['region']
        reg_i = list(reg_name_list).index(reg_name)
        result[t['identifier'][reg_i]] = dict(t[reg_i])

    # Correct the Herschel_G0 for the background of ~620
    # I used to use 620, I am trying 800 now based on a second look (rkarim, 2023-03-29)
    result['Herschel_G0']['data'] = result['Herschel_G0']['data'] - 800 #620
    return result


def make_spaghetti_plot(reg_name, plot_setting=0):
    """
    Created: September 27, 2022
    Following the notebook: https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Find_n_G0_Single_Pixel.ipynb
    Make a spaghetti plot for a given region
    :param plot_setting: the one knob to turn. values are as follows:
        0: regular overlay plot with ratios AND intensities
        1: just the chi squared plot
        2: overlay plot with only ratios
        3: overlay plot with only intensities
    """
    if reg_name[-1].isdigit() and reg_name[-2]=='-':
        # Use the manually entered data for broad-line and E-peak
        # I'm no longer using this option, this was when i tried modeling a single Gaussian component after a decomposition of the line profile
        meas_list = make_measurement_list_from_manual_entry(reg_name)
    else:
        # Use the regular observed tables
        meas_list = collect_all_measurements_for_region(reg_name)

    # modelset_name = "kt2013wd01-7"
    # ms = ModelSet(modelset_name, z=1, medium='clumpy', mass=100.0)
    modelset_name = 'wk2020'
    ms = ModelSet(modelset_name, z=1)
    # print(ms._supported_lines)
    # print(ms._supported_ratios)
    # return
    p = LineRatioFit(ms, measurements=meas_list)
    p.run()

    if plot_setting == 3:
        plot = ModelPlot(ms)
    else:
        plot = LineRatioPlot(p)

    # add to the default color cycle since I have lots of lines/ratios. Must be done before I call the overlay plot function
    plot._CB_color_cycle += ['#66ffb3', '#ff8000', '#e600e6', '#8533ff', '#999900']

    # make the list in case we need it
    plottable_intensities = [x for x in meas_list if "FIR" not in x.id]

    """
    Handle the plot setting; this is which type of plot we want
    chi squared, or overlay of either ratios, intensities, or both
    """
    textsize = 18
    kwargs = {'figsize': (18, 10), 'loc': 'upper left', 'yaxis_unit': "Habing"}
    # chi squared only
    if plot_setting == 1:
        plot.reduced_chisq(cmap='gray_r',norm='log',label=True,colors='white',
            legend=False,vmax=8E4,figsize=kwargs['figsize'],yaxis_unit='Habing', aspect='auto')
    # includes ratios
    elif plot_setting%2 == 0:
        if plot_setting == 0:
            # yes intensities
            kwargs['measurements'] = plottable_intensities
        # kwargs['assigned_colors'] = {k: 'Magenta' for k in ('OI_63/CII_158', 'CO_65/CO_10', 'CO_32/CO_10', 'CII_158/CO_65', 'CII_158/CO_32', 'CII_158/CO_10')}
        kwargs['assigned_colors'] = {k: plot._CB_color_cycle[i] for i, k in enumerate(('OI_63/CII_158', 'CO_65/CO_10', 'CO_32/CO_10', 'CII_158/CO_65', 'CII_158/CO_32', 'CII_158/CO_10'))}
        plot.overlay_all_ratios(**kwargs)
    # intensities only
    elif plot_setting == 3:
        plot.overlay(measurements=plottable_intensities, **kwargs)
    else:
        raise RuntimeError(f"unsupported plot setting: {plot_setting}")

    g0_dict = get_g0_values_at_locations(reg_name)
    g0_plot_params = {'Stars_G0': ('#1f77b4', 'bottom'), 'Herschel_G0': ('#ff7f0e', 'top')}
    for g0_name in g0_dict:
        color, va = g0_plot_params[g0_name]
        plot.axis[0].axhline(g0_dict[g0_name]['data'], linestyle='--', color=color)
        plot.axis[0].text(120, g0_dict[g0_name]['data'], g0_name.replace('_G0', ' $G_0$'), color=color, fontsize=textsize, va=va, ha='left')


    dens, dens_unc = p.density.value, p.density.uncertainty.array
    radfield_meas = utils.to(utils.habing_unit, p.radiation_field)
    radfield, radfield_unc = radfield_meas.value, radfield_meas.uncertainty.array
    plot.axis[0].errorbar(dens, radfield, xerr=dens_unc, yerr=radfield_unc, color='k')

    # set x and y limits because we know what they should probably be
    plot.axis[0].set_xlim([100, 1e6])
    plot.axis[0].set_ylim([30, 3e4])

    plot.axis[0].tick_params(labelsize=textsize)
    plot.axis[0].xaxis.label.set(fontsize=textsize)
    plot.axis[0].yaxis.label.set(fontsize=textsize)
    """
    alt:
    hploto._plt.rcParams["xtick.major.size"] = 7
    hploto._plt.rcParams["xtick.minor.size"] = 4
    hploto._plt.rcParams["ytick.major.size"] = 7
    hploto._plt.rcParams["ytick.minor.size"] = 4
    hploto._plt.rcParams['font.size'] = 14
    hploto._plt.rcParams['axes.linewidth'] =1.5
    plot.overlay_all_ratios(yaxis_unit="Habing",figsize=(15,5),ncols=2,reset=True,index=1)
    plot.overlay_all_ratios(yaxis_unit="Habing",figsize=(15,5),ncols=2,reset=False,index=2)
    plot._plt.subplots_adjust(wspace=0)
    legend=False and make my own
    """

    # 2022-09-28, (27 ?), 29, 10-05,6,7,11,12,13,14,17, 11-22
    # 2023-03-29, 04-04,05,06
    save_path = f"/home/ramsey/Pictures/2023-04-07" # removed modelset name because we won't do anymore kosma-tau models
    if not os.path.exists(save_path):
        print("Creating directory ", save_path)
        os.makedirs(save_path)

    if plot_setting == 1:
        # plot._plt.subplots_adjust(right=0.6)
        plot.savefig(os.path.join(save_path, f"chisq_{reg_name}.png"),
            metadata={'Author': "Ramsey Karim", 'Source': f'{__file__}.make_spaghetti_plot',
                'Title': f'{reg_name} chisq plot'})
    else:
        setting_stub = ''
        if plot_setting in [0, 2]:
            setting_stub += 'ratio'
        if plot_setting in [0, 3]:
            setting_stub += 'intens'
        if setting_stub:
            setting_stub = '_' + setting_stub
        plot.savefig(os.path.join(save_path, f"spaghetti{setting_stub}_{reg_name}.png"),
            metadata={'Author': "Ramsey Karim", 'Source': f'{__file__}.make_spaghetti_plot',
                'Title': f'{reg_name} spaghetti plot'})


def make_paper_spaghetti_plot():
    """
    April 7, 2023
    Make a single figure, 8 panel spaghetti plot.
    Use the original regions since the paper regions are sampled towards high H2 column regions, not PDRs
    """
    # list of region names
    region_list = ['NE-thread', 'Western-Horn', 'P2', 'P3']
    # ignore this list, it's just for file handling
    filename_keys = ['p1a'] + ['misc']*3 # because of where the data are stored

    # We know in advance that these ratios are possible
    ratio_ids = ('OI_63/CII_158', 'CO_65/CO_10', 'CO_32/CO_10', 'CII_158/CO_65', 'CII_158/CO_32', 'CII_158/CO_10')
    g0_plot_params = {'Stars_G0': ('#1f77b4', 'bottom'), 'Herschel_G0': ('#ff7f0e', 'top')}

    # Set up persistent ModelSet
    ms = ModelSet('wk2020', z=1)
    # Save ratio table for legend labels later
    ratio_table = ms.supported_ratios.copy()
    # Set up variable for persistent ModelPlot
    mp = None

    textsize = 18
    plot_kwargs = {'figsize': (20, 10), 'legend': False, 'yaxis_unit': 'Habing'}
    grid_shape = (2, 4)

    for i, reg_name in enumerate(region_list):
        ########## This stuff looks confusing but it's to select the correct table filenames
        # Specific to my scripts here, not general to pdrtpy
        filename_gen.clear()
        filename_gen.append(available_filenames[filename_keys[i]])
        # meas_list is a list of single-value Measurements
        meas_list = collect_all_measurements_for_region(reg_name)
        # g0_dict is keyed with "Stars_G0" and "Herschel_G0" and stores G0 values wrapped in a second dictionary (for some good reason)
        g0_dict = get_g0_values_at_locations(reg_name)
        ########## Back to normal-looking stuff

        # Remake the LineRatioFit and LineRatioPlot for each unique measurement set
        p = LineRatioFit(ms, measurements=meas_list)
        p.run()
        plot = LineRatioPlot(p)

        # Store the fit result for plotting a cross
        dens, dens_unc = p.density.value, p.density.uncertainty.array
        radfield_meas = utils.to(utils.habing_unit, p.radiation_field)
        radfield, radfield_unc = radfield_meas.value, radfield_meas.uncertainty.array
        print(f"Reduced chisq {reg_name}: {p.reduced_chisq(min=True):.4f}")

        if 'assigned_colors' not in plot_kwargs:
            plot_kwargs['assigned_colors'] = {k: plot._CB_color_cycle[i] for i, k in enumerate(('OI_63/CII_158', 'CO_65/CO_10', 'CO_32/CO_10', 'CII_158/CO_65', 'CII_158/CO_32', 'CII_158/CO_10'))}

        # Here's the trick; the LineRatioPlot __init__ is pretty light, so we just swap back in the old ModelPlot instance if it exists
        if mp is None:
            # Save it for later; we're only using one
            mp = plot._modelplot
        else:
            # Swap in the old one!
            plot._modelplot = mp

        # Add 1 to indices because they're 1 indexed in pdrtpy
        plot.overlay_all_ratios(index=i+1, reset=(i==0), nrows=grid_shape[0], ncols=grid_shape[1], **plot_kwargs)
        plot.reduced_chisq(index=i+1+grid_shape[1], reset=False, nrows=grid_shape[0], ncols=grid_shape[1], cmap='gray_r', norm='log', label=True, colors='white', legend=False, vmax=8e4, plot_cross=False, yaxis_unit=plot_kwargs['yaxis_unit'], aspect='auto')



        # indices are 0 indexed in the plot.axis (1d) array
        for plot_idx, ax in enumerate([plot.axis[x] for x in (i, i+grid_shape[1])]):
            ax.set_xlim([200, 2e6])
            ax.set_ylim([30, 3e4])
            ax.xaxis.label.set(fontsize=textsize)
            ax.yaxis.label.set(fontsize=textsize)

            # Hide both x and y labels and use sup(x/y)label
            ax.set_xlabel("")
            ax.set_ylabel("")

            if i > 0:
                # Hide ticklabels on y axes that aren't first column
                ax.tick_params(axis='y', labelleft=False)

            # Increase textsize on x and y ticks
            ax.tick_params(axis='y', labelsize=textsize)
            ax.tick_params(axis='x', labelsize=textsize)

            if plot_idx == 0:
                # Hide x ticklabels on top row (overlay plots)
                ax.tick_params(axis='x', labelbottom=False)
                # Plot region names
                ax.text(240, 2e4, reg_name.replace('-', ' ').replace("thread", "Thread"), color='k', fontsize=textsize-2, va='center', ha='left')
                # Fold x ticks inwards on top row
                ax.tick_params(axis='x', direction='in', which='both') # which=both means both major and minor ticks

            # Plot G0 estimates on both overlay and chisq as horizontal lines
            for g0_name in g0_dict:
                color, va = g0_plot_params[g0_name]
                ax.axhline(g0_dict[g0_name]['data'], linestyle='--', color=color)
                if plot_idx == 0:
                    # Only label the G0 horizontal on top row (overlay)
                    ax.text(240, g0_dict[g0_name]['data'], g0_name.replace('_G0', ' $G_0$').replace('Herschel', 'FIR'), color=color, fontsize=textsize-2, va=va, ha='left')

            # Plot the fit result as a cross on both plots
            ax.errorbar(dens, radfield, xerr=dens_unc, yerr=radfield_unc, color='k')


    plot.figure.supxlabel("n (cm$^{-3}$)", size=textsize)
    plot.figure.supylabel("$G_0$ (Habing)", size=textsize)

    legend_handles = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in plot._CB_color_cycle[:len(ratio_ids)]]
    legend_labels = [str(ratio_table.loc['ratio label', x]['title']) for x in ratio_ids]
    plot.figure.legend(legend_handles, legend_labels, loc='upper center', ncols=len(ratio_ids), fontsize=12)
    plot._plt.subplots_adjust(top=0.95, wspace=0.05, hspace=0, bottom=0.08, left=0.06)

    # 2023-04-07,10,11
    save_path = f"/home/ramsey/Pictures/2023-04-11" # removed modelset name because we won't do anymore kosma-tau models
    if not os.path.exists(save_path):
        print("Creating directory ", save_path)
        os.makedirs(save_path)
    plot.savefig(os.path.join(save_path, "multipanel_overlay.png"),
        metadata={'Author': "Ramsey Karim", 'Source': f'{__file__}.make_paper_spaghetti_plot',
            'Title': 'multipanel spaghetti plot'})




# line names
# 'cii', 'co65CONV', 'FIR', '12co10CONV', '12co32', '13co32', '13co10CONV'

if __name__ == "__main__":
    set_supported_lines(['cii', 'oiCONV', 'co65CONV', '12co10CONV', '12co32'])
    # for d in ['E', 'S', 'W']:
    #     for i in range(0, 4):
    #         make_spaghetti_plot(d+'-peak', plot_setting=i)

    """
    The pillar123_pointsofinterest_v2 regions look worse than v1.
    I think the reason is that the v1 were sampled towards edges (PDRs) and v2 from the dense molecular peaks where CII and OI are weaker
    I'm sticking with the v1s
    """
    make_paper_spaghetti_plot()

    # reg_name_list = ['Western-Horn', 'P2', 'P3']; filename_gen.append(available_filenames['misc'])
    # for reg_name in reg_name_list:
    #     for i in [1, 2]: #[1, 2]:
    #         make_spaghetti_plot(reg_name, plot_setting=i)
    # filename_gen.pop()
    # reg_name_list = ['NE-thread']; filename_gen.append(available_filenames['p1a'])
    # for reg_name in reg_name_list:
    #     for i in [1, 2]: #[1, 2]:
    #         make_spaghetti_plot(reg_name, plot_setting=i)

    # r = 'broad-line'
    # for i in range(0, 4):
    #     make_spaghetti_plot(r, plot_setting=i)

    # for ns in 'NS':
    #     for ew in 'EW':
    #         for i in range(1, 4):
    #             make_spaghetti_plot(f'{ns}{ew}-thread', plot_setting=i)

    # Use the manually entered observations from the Gaussian modeling adventure
    # make_spaghetti_plot('broad-line-1', chisq=0)
    # for idx in [1, 2]:
    #     for reg_name in manually_entered_observations.keys():
    #         for i in range(2):
    #             make_spaghetti_plot(f"{reg_name}-{idx}", chisq=i)
