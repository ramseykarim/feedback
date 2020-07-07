import numpy as np
import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 6}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import pandas as pd
from collections.abc import Sequence

from astropy import units as u

from .catalog import utils as catalog_utils

sb99_output_path = f"{catalog_utils.ancillary_data_path}catalogs/starburst99/"
sb99_run_names = {'O2': 'O2', 'O3': 'O3', 'test1': 'standard',}
sb99_run_names.update({f'Wd2_{i}': 'standard' for i in list(range(1, 6))+list(range(10, 14))+list(range(20, 24))})
sb99_output_directories = {k: f"{sb99_output_path}{k}/" for k in sb99_run_names}

def get_sb99_run_info(n):
    return sb99_output_directories[n], sb99_run_names[n]

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
        plt.ylabel("Power [erg/s]")
        plt.ylim((35, 42))
    else:
        plt.ylabel("Momentum flux [dyn]")
        plt.ylim((27, 34))
    plt.xlabel("Time (Myr)")
    plt.legend()



def run_plot_standard():
    directory, name = get_sb99_run_info('test1')
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
    directory, name = get_sb99_run_info('O3')
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
    directory, name = get_sb99_run_info('O2')
    df = open_both(directory, name)


    plt.figure(figsize=(7, 5))
    # plt.subplot(121)
    plot_value(df, name=name, value_name_long='momentum flux', value_name_short='MOMFLUX', add_factor=np.log10(115./1e6))
    plt.title("IMF 110-120, 10^6 total (curves x 115/1e6)")
    plt.ylim((28, 31))


"""
THE RUNS WITH 3x10^4 ARE INCORRECT: THE CLUSTER IS NOT THAT MASSIVE
"""
run_descriptions = {
    1: "3.1e4 stellar masses [0.65, 80] and Zeidler pdmf (a = 2.03)",
    2: "3.1e4, kroupa imf (2.3), [0.65, 80]",
    3: "3.1e4, kroupa imf, [0.65, 126]",
    4: "4500, kroupa imf, [0.65, 126]",
    5: "10^4, kroupa imf, [0.65, 126]",
}

def run_plot_Wd2_n(n):
    directory, name = get_sb99_run_info(f'Wd2_{n}')
    df = open_both(directory, name)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plot_value(df, name=name, value_name_long='momentum flux', value_name_short='MOMFLUX')
    plt.title("STARBURST99; Momentum Flux")
    alt_title = "Westerlund 2 parameters; total mass $1 \\times 10^{4}~M_{\\odot}$"
    plt.ylim((29, 32))
    plt.subplot(122)
    plot_value(df, name=name, value_name_long='mechanical luminosity', value_name_short='POWER')
    plt.title("STARBURST99; Mechanical Luminosity")
    alt_title = "IMF 0.65-80 $M_{\\odot}$, $\\alpha = 2.03$ (Zeidler+2017)"
    plt.ylim((37, 40))


def check_power_linear(n, age_lim_Myr=2.):
    directory, name = get_sb99_run_info(f'Wd2_{n}')
    df = open_both(directory, name)
    if isinstance(age_lim_Myr, Sequence):
        age_lims = np.array(age_lim_Myr)
    else:
        age_lims = np.array([age_lim_Myr])
    age_lims = (age_lims + 0.02) * 1e6
    for age_lim in age_lims:
        print(f"AGE: {age_lim/1e6:.1f} Myr")
        powerOB = df.loc[(df.index <= age_lim), 'POWER_OB']
        power_values = (10.**powerOB.values) * u.erg / u.s
        time_values = ([0.]+list(powerOB.index.values)) * u.year
        time_steps = np.diff(time_values)
        energy_steps = (power_values * time_steps).to(u.erg)
        total_energy = np.cumsum(energy_steps)
        avg_power = np.mean(power_values)
        print("power")
        print(avg_power)
        print("energy")
        # print(total_energy)
        print((avg_power * time_values[-1]).to(u.erg))


def compare_runs():
    runs = [2, 3, 4]
    colors = ['blue', 'red', 'k']
    run_descriptions = {
        2: "3.1e4, [0.65, 80]",
        3: "3.1e4, [0.65, 126]",
        4: "4500, [0.65, 126]",
    }
    plt.figure(figsize=(10, 5))
    pow_ax = plt.subplot(121)
    mom_ax = plt.subplot(122)
    for i in range(len(runs)):
        n = runs[i]
        directory, name = get_sb99_run_info(f'Wd2_{n}')
        df = open_both(directory, name)
        powerOB = df['POWER_OB']
        powerWR = df['POWER_WR']
        mvfluxOB = df['MOMFLUX_OB']
        mvfluxWR = df['MOMFLUX_WR']
        t = df.index
        l = f" ({run_descriptions[n]})"
        plt.sca(pow_ax)
        plt.plot(t/1e6, powerOB, label='OB'+l, alpha=0.7, color=colors[i], linestyle='-')
        plt.plot(t/1e6, powerWR, label='WR'+l, alpha=0.7, color=colors[i], linestyle='--')
        plt.sca(mom_ax)
        plt.plot(t/1e6, mvfluxOB, label='OB'+l, alpha=0.7, color=colors[i], linestyle='-')
        plt.plot(t/1e6, mvfluxWR, label='WR'+l, alpha=0.7, color=colors[i], linestyle='--')
    plt.sca(pow_ax)
    plt.title("Starburst99 simulation, power")
    plt.ylabel("Power [erg/s]")
    plt.ylim((36, 40))
    plt.xlabel("Time (Myr)")
    plt.sca(mom_ax)
    plt.title("Momentum flux")
    plt.ylabel("Momentum flux [dyn]")
    plt.ylim((28, 32))
    plt.xlabel("Time (Myr)")
    plt.legend()
    plt.show()



def runs_with_uncertainty(prefix_n, axes=None, colorOB='green', colorWR='orange',
    setup=False, show=False, label_suffix=None):
    if axes is None:
        pow_ax, mom_ax = plt.subplot(121), plt.subplot(122)
        axes = pow_ax, mom_ax
    else:
        pow_ax, mom_ax = axes
    if label_suffix is None:
        label_suffix = ''
    powerOB_list, mvfluxOB_list = [], []
    powerWR_list, mvfluxWR_list = [], []
    time_list = []
    for i in range(3):
        args = get_sb99_run_info(f'Wd2_{prefix_n}{i}')
        df = open_both(*args)
        powerOB_list.append(df['POWER_OB'])
        powerWR_list.append(df['POWER_WR'])
        mvfluxOB_list.append(df['MOMFLUX_OB'])
        mvfluxWR_list.append(df['MOMFLUX_WR'])
        time_list.append(df.index)
    t = time_list[0] / 1e6

    labelOB = f"OB {label_suffix}"
    labelWR = f"WR {label_suffix}"

    fill_kwargs = dict(alpha=0.2, lw=0.02)
    fillOB_kwargs = {**fill_kwargs, 'color': colorOB}
    fillWR_kwargs = {**fill_kwargs, 'color': colorWR, 'hatch': '////'}

    plot_kwargs = dict(alpha=0.9, lw=1)
    plotOB_kwargs = {**plot_kwargs, 'color': colorOB, 'linestyle': '-', 'label': labelOB}
    plotWR_kwargs = {**plot_kwargs, 'color': colorWR, 'linestyle': '--', 'label': labelWR}


    plt.sca(pow_ax)
    plt.fill_between(t, powerOB_list[2], powerOB_list[1], **fillOB_kwargs)
    plt.plot(t, powerOB_list[0], **plotOB_kwargs)
    plt.fill_between(t, powerWR_list[2], powerWR_list[1], **fillWR_kwargs)
    plt.plot(t, powerWR_list[0], **plotWR_kwargs)
    if setup:
        plt.title("Starburst99 simulation, power")
        plt.ylabel("Power [erg/s]")
        plt.ylim((36, 39))
        plt.xlabel("Time (Myr)")
        plt.xlim((0, 6))

    plt.sca(mom_ax)
    plt.fill_between(t, mvfluxOB_list[2], mvfluxOB_list[1], **fillOB_kwargs)
    plt.plot(t, mvfluxOB_list[0], **plotOB_kwargs)
    plt.fill_between(t, mvfluxWR_list[2], mvfluxWR_list[1], **fillWR_kwargs)
    plt.plot(t, mvfluxWR_list[0], **plotWR_kwargs)
    if setup:
        plt.title("Momentum flux")
        plt.ylabel("Momentum flux [dyn]")
        plt.ylim((28, 31))
        plt.xlabel("Time (Myr)")
        plt.xlim((0, 6))
    if show:
        plt.sca(mom_ax)
        plt.legend()
        plt.show()
    return axes

def fill_between_horiz(ax, center, low, high, color='k', linestyle='-', hatch=None, label=None):
    plt.sca(ax)
    plt.axhspan(low, high, color=color, hatch=hatch, alpha=0.05, lw=0.02)
    plt.axhline(y=center, color=color, alpha=0.7, lw=0.7, linestyle=linestyle, label=label)


def make_nice_sb99_figure():
    axes = runs_with_uncertainty(1, setup=True, label_suffix='[1, 120]')
    runs_with_uncertainty(2, axes=axes, colorOB='blue', colorWR='red', label_suffix='[1, 80]')
    ob_pow = (np.log10(8.3) + 37, np.log10(8.3 - 0.5) + 37, np.log10(8.3 + 0.5) + 37)
    # wr_pow = (np.log10(3.6) + 37, np.log10(3.6 - 1.1) + 37, np.log10(3.6 + 1.3) + 37) # WR20a
    wr_pow = (np.log10(5.3) + 37, np.log10(5.3 - 1.3) + 37, np.log10(5.3 + 1.7) + 37) # WR20a + WR20b

    ob_mom = (np.log10(5.6) + 29, np.log10(5.6 - 0.4) + 29, np.log10(5.6 + 0.4) + 29)
    # wr_mom = (np.log10(3.0) + 29, np.log10(3.0 - 0.6) + 29, np.log10(3.0 + 0.7) + 29) # WR20a
    wr_mom = (np.log10(4.2) + 29, np.log10(4.2 - 0.7) + 29, np.log10(4.2 + 0.8) + 29) # WR20a + WR20b

    pow_ax, mom_ax = axes
    fill_between_horiz(pow_ax, *ob_pow)
    fill_between_horiz(pow_ax, *wr_pow, linestyle='--', hatch='\\'*4)
    fill_between_horiz(mom_ax, *ob_mom, label='OB catalog $r<3\'$')
    fill_between_horiz(mom_ax, *wr_mom, linestyle='--', hatch='\\'*4, label='WR20a+b')
    plt.sca(mom_ax)
    plt.legend()
    plt.show()


def total_energy_over_time(prefix_n, axes=None, color='green',
    setup=False, show=False, label_suffix=None, log=True):
    if axes is None:
        erg_ax = plt.subplot(111)
        axes = erg_ax
    else:
        erg_ax = axes
    if label_suffix is None:
        label_suffix = ''
    if log:
        f = np.log10
    else:
        f = lambda x: x
    power_list = []
    time_list = []
    convert_power = lambda x: (10.**x.values) * u.erg/u.s
    for i in range(3):
        args = get_sb99_run_info(f'Wd2_{prefix_n}{i}')
        df = open_both(*args)
        powerOB = convert_power(df['POWER_OB'])
        powerWR = convert_power(df['POWER_WR'])
        power_list.append(powerOB + powerWR)
        time_list.append(df.index)
    t = time_list[0].values
    t_axis = t/1e6
    time_steps = np.diff(([0.] + list(t)) * u.year)
    total_energy = [f(np.cumsum((p*time_steps).to(u.erg)).to_value()) for p in power_list]
    shared_kwargs = dict(color=color)
    plt.sca(erg_ax)
    plt.fill_between(t_axis, total_energy[2], total_energy[1], **shared_kwargs, lw=0.02, alpha=0.2)
    plt.plot(t_axis, total_energy[0], linestyle='-', lw=1, **shared_kwargs, label=f"OB+WR {label_suffix}")
    if setup:
        plt.title("Starburst99, Cumulative stellar wind energy over time")
        plt.ylabel("Cumulative energy (ergs)")
        plt.xlabel("Time (Myr)")
        # plt.xlim((0, 6))
    if show:
        plt.legend()
        plt.show()
    return axes

def make_nice_energy_figure(ax=None, log=True):
    if log:
        f = np.log10
    else:
        f = lambda x: x
    ax = total_energy_over_time(1, axes=ax, setup=True, color='green', label_suffix='[1, 120]', log=log)
    total_energy_over_time(2, axes=ax, color='blue', label_suffix='[1, 80]', log=log)
    shell_KE = f(2e50)
    plasma_TE = f(2.4e48)
    plt.axhline(y=shell_KE, color='k', alpha=0.7, lw=0.7, linestyle='-', label='Shell KE')
    plt.axhline(y=plasma_TE, color='k', alpha=0.7, lw=0.7, linestyle='--', label='Plasma TE')
    plt.legend()
    if log:
        txt = ax.get_ylabel()
        txt = txt.replace('(', '[').replace(')', ']')
        ax.set_ylabel(txt)

if __name__ == "__main__":
    make_nice_sb99_figure()
    # check_power_linear(10, age_lim_Myr=[2, 2.5, 3, 3.5])
    # make_nice_energy_figure(ax=plt.subplot(121), log=False)
    # make_nice_energy_figure(ax=plt.subplot(122))
    # plt.show()
