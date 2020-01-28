import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
from itertools import cycle

from parse_FIR_fits import open_FIR_pickle, herschel_path, get_vlims
from mantipython.v3.src import dust, greybody, instrument



def plot_points_and_greybody(ax, x_limited, x, obs, model, gb, nu, color):
    ax.plot(x_limited, obs, '.', color=color, marker='x', label='Observed')
    model_plots_iter = (([xpos*0.99, xpos*1.01], [ypos, ypos]) for xpos, ypos in zip(x_limited, model))
    model_plots = []
    for mpx, mpy in model_plots_iter:
        model_plots.append(mpx)
        model_plots.append(mpy)
    ax.plot(*model_plots, '-', color=color, linewidth=1)
    ax.plot(x, gb.radiate(nu), color=color)

def create_SED_figure(figsize=(7, 7)):
    fig_SED, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
    ax_SED_lin, ax_SED_log = axes
    setup_SED_ax(ax_SED_lin, ax_SED_log)
    return fig_SED, ax_SED_lin, ax_SED_log

def setup_SED_ax(ax_SED_lin, ax_SED_log):
    wl_ticks = [40, 70, 160, 250, 350, 500, 1000]
    ax_SED_log.set_xlim([np.log(x) for x in [40, 1000]])
    ax_SED_log.set_xticks([np.log(x) for x in wl_ticks])
    ax_SED_log.set_xticklabels(list(map(str, wl_ticks)))
    ax_SED_log.set_xlabel("Wavelength (micron)")
    ax_SED_lin.set_ylabel("Flux (MJy/sr)")
    ax_SED_log.set_yscale('log')


if __name__ == "__main__":

    print("I think 70um offset needs 20less, and 160um needs 20more.")
    print("look at these plots for more info....")
    print("try model-obs160um/obs160um")

    filename = herschel_path+"RCW49large_350grid_3p_TILEFULL.pkl"
    displaytxt = input("display on? ")
    if not displaytxt:
        displaytxt='chisq'


    plt.ion()
    """
    use matplotlib interactive mode to plot up SED fits to data for a given
    pixel, selected by mouse click. run in while loop to explore many points
    """
    colors = {"PACS70um": 'violet', "PACS160um": 'blue', "SPIRE250um": 'green', "SPIRE350um": 'red',}
    # Load in the data (the data is in a format that sucks so I should fix that)
    fit_result_dict = open_FIR_pickle(filename)
    # Set up image figure. This figure will be clicked on
    fig_img = plt.figure(figsize=(7, 5))
    ax_img = plt.subplot(111)
    if '/' in displaytxt:
        d1, d2 = displaytxt.split('/')
        display = fit_result_dict[d1] / fit_result_dict[d2]
    else:
        display = fit_result_dict[displaytxt]
    plt.imshow(display, **get_vlims(displaytxt))
    plt.title(displaytxt)
    # Get the SED plot components
    fig_SED, ax_SED_lin, ax_SED_log = create_SED_figure()
    # Set up wavelength axes and such
    wavelengths = sorted([int(k.replace('model', '').replace('um', '')) for k in fit_result_dict if ('model' in k) and ('obs' not in k)])
    wavelengths_plot = [np.log(x) for x in wavelengths]
    wl_range_plot = np.linspace(np.log(40), np.log(1000), 200)
    wl_range = np.exp(wl_range_plot)
    nu_range = cst.c / (wl_range*1e-6)
    # Get bandpass response functions in plottable form
    herschel = instrument.get_instrument(wavelengths)
    herschel_responses = []
    for detector in herschel:
        rsp = detector.response_array/np.trapz(detector.response_array, x=detector.freq_array)
        rsp /= np.max(rsp)
        rsp_x = np.log(1e6*cst.c/detector.freq_array)
        herschel_responses.append((rsp_x, rsp))

    gen_color_list = lambda : plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_info = {
        'colorcycle': cycle(gen_color_list()),
        'horiz_offset': 0.02,
    }


    def onclick(event):
        try:
            j, i = int(round(event.xdata)), int(round(event.ydata))
        except Exception as e:
            print("something wrong...", e)
            return
        if event.button == 1:
            ax_SED_lin.clear()
            ax_SED_log.clear()
            setup_SED_ax(ax_SED_lin, ax_SED_log)
            plot_new_SED(i, j)
            plot_info['colorcycle'] = cycle(gen_color_list())
            plot_info['horiz_offset'] = 0.02
        elif event.button == 3:
            plot_info['horiz_offset'] += 0.2
            plot_new_SED(i, j, color=next(plot_info['colorcycle']), horiz_offset=plot_info['horiz_offset'])


    def plot_new_SED(i, j, color='k', horiz_offset=0.05):
        # Iterate through bands to do get the model and observed data
        model_fluxes, obs_fluxes = [], []
        for idx, b in enumerate(wavelengths):
            # Get model and observed data
            model_flux = fit_result_dict[f'model{b}um'][i, j]
            diff_flux = fit_result_dict[f'model-obs{b}um'][i, j]
            obs_flux = model_flux - diff_flux
            model_fluxes.append(model_flux)
            obs_fluxes.append(obs_flux)
        # Make and plot model for this pixel
        T, tau, beta = (fit_result_dict[k][i, j] for k in ('T', 'tau', 'beta'))
        gb = greybody.Greybody(T, tau, dust.TauOpacity(beta))
        plot_points_and_greybody(ax_SED_lin, wavelengths_plot, wl_range_plot, obs_fluxes, model_fluxes, gb, nu_range, color=color)
        plot_points_and_greybody(ax_SED_log, wavelengths_plot, wl_range_plot, obs_fluxes, model_fluxes, gb, nu_range, color=color)
        # Add legend and text describing some things
        ax_SED_lin.legend(loc='upper right', shadow=True)
        desc = f"{T:4.1f} K\n{tau:4.2f}\n{beta:4.2f}"
        ax_SED_log.text(horiz_offset, 0.2, desc, fontsize=6, transform=ax_SED_log.transAxes, verticalalignment='center')
        ymax = ax_SED_lin.get_ylim()[1]
        for idx, b in enumerate(wavelengths):
            # Plot detector curve
            rsp_x, rsp = herschel_responses[idx]
            ax_SED_lin.fill(rsp_x, rsp*ymax*0.9,
                color=colors[herschel[idx].name], alpha=0.2)

    cid = fig_img.canvas.mpl_connect('button_press_event', onclick)
