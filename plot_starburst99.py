import numpy as np
import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 6}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import pandas as pd

import catalog_utils

sb99_output_path = f"{catalog_utils.ancillary_data_path}catalogs/starburst99/"
o_star_names = {'O2': 'O2', 'O3': 'O3', 'test1': 'standard', 'Wd2': 'standard'}
o_star_output_directories = {k: f"{sb99_output_path}{k}/" for k in o_star_names}

def get_o_star_run_info(n):
    return o_star_output_directories[n], o_star_names[n]

def open_power(directory, name):
    """
    Open the "power" output, with mechanical luminosity and momentum flux
    :param directory: string directory with starburst99 output
    :param name: the "name" of the starburst99 run (<name>.input1, etc)
    :returns: pandas DataFrame
    """
    df = pd.read_csv(f"{directory}{name}.power1", sep='\s+', header=4, mangle_dupe_cols=True, index_col='TIME')
    new_cols = {}
    for i, c in enumerate(df.columns):
        if i < 5:
            c_new = "POWER_" + c
        elif i == 5:
            c_new = "ENERGY_" + c.split('.')[0]
        else:
            c_new = "MOMFLUX_" + c.split('.')[0]
        new_cols[c] = c_new
    df.rename(columns=new_cols, inplace=True)
    return df


def open_snr(directory, name):
    """
    Open the "snr" output, with SN mechanical luminosity and momentum flux
    :param directory: string directory with starburst99 output
    :param name: the "name" of the starburst99 run (<name>.input1, etc)
    :returns: pandas DataFrame
    """
    filename = f"{directory}{name}.snr1"
    with open(filename, 'r') as f:
        for i in range(6):
            f.readline()
        colnames_raw = f.readline().split('  ')
    colnames_raw = [x for x in colnames_raw if x]
    colnames = []
    for i, c in enumerate(colnames_raw):
        if c:
            c_new = c.strip()
            if i >= 1 and i < 4:
                c_new = c_new + "_SN"
            elif i >= 4 and i < 7:
                c_new = c_new + "_IB"
            elif i >= 9 and i < 11:
                c_new = c_new + "_Stars+SN"
            colnames.append(c_new)
    df = pd.read_csv(filename, sep='\s+', skiprows=7, names=colnames, index_col='TIME')
    return df


def open_both(directory, name):
    """
    Open both "power" and "snr" files and make a combined DataFrame
    """
    power_df = open_power(directory, name)
    snr_df = open_snr(directory, name)
    for col in ['POWER_SN', 'ENERGY_SN', "POWER_Stars+SN", "ENERGY_Stars+SN"]:
        power_df[col] = snr_df[col]
    power_df['MOMFLUX_SN'] = np.log10((10.**power_df['POWER_SN']) / (5000. * 1e5)) # Converting erg/s to dyne (erg / cm) (using 5000 km/s SN velocity?)
    power_df['MOMFLUX_Stars+SN'] = np.log10((10.**power_df['MOMFLUX_ALL']) + (10.**power_df['MOMFLUX_SN']))
    return power_df


def plot_value(df, name=None, value_name_long=None, value_name_short=None, add_factor=0.):
    """
    :param df: pandas DataFrame with <name>.power1
    :param name: the name of the starburst99 run, optional
    """
    if name is None:
        name = f"Starburst99 simulation, {value_name_long}"
    else:
        name = f"Starburst99 simulation {name}, {value_name_long}"
    t = df.index
    lines = {
        "OB winds": df[f'{value_name_short}_OB'],
        "Red supergiants": df[f'{value_name_short}_RSG'],
        "Lum blue variables": df[f'{value_name_short}_LBV'],
        "Wolf-Rayet": df[f'{value_name_short}_WR'],
        "SN": df[f'{value_name_short}_SN'],
        "Total": df[f'{value_name_short}_Stars+SN'],
    }
    colors = {
        "OB winds": 'red',
        "Wolf-Rayet": 'orange',
        "Lum blue variables": 'green',
        "Red supergiants": 'yellow',
        "SN": 'black',
        "Total": 'blue',
    }
    for l in lines:
        plt.plot(t/1e6, lines[l] + add_factor, label=l, alpha=0.7, color=colors[l],
            linestyle=('--' if l == "Total" else '-'))
    plt.title(name)
    if value_name_short == "POWER":
        plt.ylabel("Power (erg/s)")
        plt.ylim((35, 42))
    else:
        plt.ylabel("Momentum flux (dyn)")
        plt.ylim((27, 34))
    plt.xlabel("Time (Myr)")
    plt.legend()



def run_plot_standard():
    directory, name = get_o_star_run_info('test1')
    df = open_both(directory, name)


    plt.figure(figsize=(7, 5))
    # plt.subplot(121)
    plot_value(df, name=name, value_name_long='momentum flux', value_name_short='MOMFLUX')
    plt.title("IMF 0.5-120, 10^6 total")
    plt.ylim((30, 33))
    # plt.subplot(122)
    # plot_value(df, name=name, value_name_long='mechanical luminosity', value_name_short='POWER')
    # plt.ylim((37, 41))


def run_plot_O3():
    directory, name = get_o_star_run_info('O3')
    df = open_both(directory, name)


    plt.figure(figsize=(7, 5))
    # plt.subplot(121)
    plot_value(df, name=name, value_name_long='momentum flux', value_name_short='MOMFLUX', add_factor=np.log10(74./1e6))
    plt.title("IMF 72-75, 10^6 total (curves x 74/1e6)")
    plt.ylim((28, 31))
    # plt.xlim((0, 4))
    # plt.subplot(122)
    # plot_value(df, name=name, value_name_long='mechanical luminosity', value_name_short='POWER')
    # plt.ylim((37, 41))


def run_plot_O2():
    directory, name = get_o_star_run_info('O2')
    df = open_both(directory, name)


    plt.figure(figsize=(7, 5))
    # plt.subplot(121)
    plot_value(df, name=name, value_name_long='momentum flux', value_name_short='MOMFLUX', add_factor=np.log10(115./1e6))
    plt.title("IMF 110-120, 10^6 total (curves x 115/1e6)")
    plt.ylim((28, 31))


def run_plot_Wd2():
    directory, name = get_o_star_run_info('Wd2')
    df = open_both(directory, name)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plot_value(df, name=name, value_name_long='momentum flux', value_name_short='MOMFLUX')
    plt.title("Westerlund 2 parameters; total mass $1 \\times 10^{4}~M_{\\odot}$")
    plt.ylim((29, 32))
    plt.subplot(122)
    plot_value(df, name=name, value_name_long='mechanical luminosity', value_name_short='POWER')
    plt.title("IMF 0.65-80 $M_{\\odot}$, $\\alpha = 2.03$ (Zeidler+2017)")
    plt.ylim((37, 40))



if __name__ == "__main__":
    pass
