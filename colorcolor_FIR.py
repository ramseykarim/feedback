import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
import pickle

from astropy.io import fits

from mantipython.physics import greybody, dust, instrument

"""
Generating a grid of measurements for a fixed optical depth.
Grid varies by beta and temperature
Purpose is to look at a color/color diagram
"""

fila_dir = "/home/ramsey/Documents/Research/Filaments/"
if not os.path.isdir(fila_dir):
    fila_dir = "/home/rkarim/Research/Filaments/"

readme = """## README ##
This is a 3D grid of log10(tau(160micron)), T [K], beta
Return value is a dictionary; check the keys
## END ##
"""
print(readme)



filename = "tb_grid_tau.pkl"
filename = "tb_grid_kappa1Av.pkl"
# filename = "tb_grid_t160-0.001.pkl"
# filename = "tb_grid_t160-0.01.pkl"
# filename = "tb_grid_t160-0.1.pkl"
# filename = "tb_grid_t160-0.3.pkl"
filename = "tb_grid_3D.pkl"

filename = fila_dir + filename

WINDOW_TITLE = filename.split('t160-')[-1]

datasets = {
    'Perseus': {'offsets': [None, '000045'],
        'data_dir': "/n/sgraraid/filaments/Perseus/Herschel/processed/1342190326/"},
    'RCW49': {'offsets': ['000102', '000343'],
        'data_dir': "/n/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009/"}
}

def make_grid():
    herschel = instrument.get_all_Herschel()

    # set up parameter grid

    # log10(Tau160) in logarithmic steps of 0.5%
    tau_range = np.arange(np.log10(0.002), np.log10(0.4), np.log10(1.005))
    # Temperature from 5 to 150 in geometric steps of 3%
    T_range = np.arange(5, 60, 0.05)
    # Beta in arithmetic steps of 0.1
    b_range = np.arange(1.2, 2.4, 0.05)
    """
    a 28 million element grid will take
    """
    print("Grid shape: (tau, T, beta)", tau_range.shape, T_range.shape, b_range.shape)

    #### If setting tau160 to a single value (2D grid)
    # N1Av = np.log10(1.1) + 21
    # kappa2 = dust.Dust(beta=2.0, k0=0.05625, nu0=750*1e9)
    # tau160 = np.log10(kappa2(dust.nu0_160)*(10**N1Av))
    # print("Tau160: {:.3E}".format(10**tau160))
    # print("Tau350: {:.3E}".format(kappa2(dust.cst.c/(350*1e-6))*(10**N1Av)))
    # tau160 = np.log10(0.3)
    # print(f"Tau160 = {tau160:.3f}")

    # I switched the indexing from xy to ij on June 13, 2020.
    # See numpy docs on 3D meshgrid indexing; unintuitive
    tt, TT, bb = np.meshgrid(tau_range, T_range, b_range, indexing='ij')
    result = {d.name: np.full(TT.shape, np.nan).ravel() for d in herschel}

    total_i = TT.size

    for i, t160, T, b in zip(range(TT.size), tt.flat, TT.flat, bb.flat):
        for d in herschel:
            # result[d.name][i] = d.detect(greybody.Greybody(T, N1Av, dust.Dust(beta=b, k0=0.05625, nu0=750*1e9)))
            result[d.name][i] = d.detect(greybody.Greybody(T, t160, dust.TauOpacity(b)))
            if (i % 5000 == 0):
                sys.stdout.write(f"{float(i+1)*100/float(total_i):6.2f}\r")
                sys.stdout.flush()
    print()
    for d in herschel:
        result[d.name] = result[d.name].reshape(TT.shape)
    result['tau160'] = tt
    result['T'] = TT
    result['beta'] = bb
    result['README'] = readme
    print("grid shape:", TT.shape)
    with open(filename, 'wb') as f:
        pickle.dump(result, f)
    return result


def plot_data(dataset_name, SNR_MINIMUM=10, force160250=0, additional_mask=None, coverage_plot=True, extra_title=""):
    # More or less automatic
    dataset = datasets[dataset_name]
    valid_bands = [1 if dataset_name[:3] == "RCW" else 0, 1, 1, 1, 1]
    if dataset_name[:3] == "Per":
        force160250 = 1
    first_band = 0 + force160250
    bands = [70, 160, 250, 350, 500]
    data_dir = dataset['data_dir']
    suffix, fitsstub = "-image-remapped-conv", ".fits"
    errsuffix = "-error-remapped-conv"
    p70_offset = "-plus"+dataset['offsets'][0] if dataset['offsets'][0] is not None else 'INVALID'
    p160_offset = "-plus"+dataset['offsets'][-1]
    offsets = [p70_offset, p160_offset] + ['']*3
    prefixes = [f'SPIRE{wl}um' if wl > 200 else f'PACS{wl}um' for wl in bands]
    img_names = {pre: data_dir+pre+suffix+offset+fitsstub for pre, offset in zip(prefixes, offsets)}
    err_names = {pre: data_dir+pre+errsuffix+fitsstub for pre in prefixes}
    SNRs, imgs = {}, {}
    for i in range(len(bands)):
        if not valid_bands[i]:
            continue
        img = fits.getdata(img_names[prefixes[i]])
        imgs[prefixes[i]] = img
        SNR = img / fits.getdata(err_names[prefixes[i]])
        SNRs[prefixes[i]] = SNR
    # Make mask
    valid_snr_mask = np.all([SNRs[prefixes[i]] > SNR_MINIMUM for i in range(len(bands)) if valid_bands[i]], axis=0)
    if additional_mask is not None:
        additional_mask = additional_mask(imgs)
        valid_snr_mask &= additional_mask
    if coverage_plot:
        # Plot image of what will be included in scatter plot
        old_fig = plt.gcf()
        new_fig = plt.figure()
        plt.imshow(valid_snr_mask, origin='lower')
        title = f"SNR > {SNR_MINIMUM}" + extra_title
        plt.title(title)
        plt.figure(old_fig.number)
    # Make ratios
    y_ratio = imgs[prefixes[first_band]] / imgs[prefixes[first_band+1]]
    print(f"Y ratio: {prefixes[0+int(force160250)]} to {prefixes[1+int(force160250)]}")
    x_ratio = imgs[prefixes[-2]] / imgs[prefixes[-1]]
    print(f"X ratio: {prefixes[-2]} to {prefixes[-1]}")
    y_ratio, x_ratio = (ratio[valid_snr_mask].flatten() for ratio in (y_ratio, x_ratio))
    # Plot scatter plot
    plt.plot(x_ratio, y_ratio, '.', alpha=0.1, markersize=3, color='k')
    return ", " + title


def color_color(dataset_name=None, force160250=0, SNR_MINIMUM=10, additional_mask=None, extra_title=""):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    p70p160 = result['PACS70um'] / result['PACS160um']
    p160s250 = result['PACS160um'] / result['SPIRE250um']

    if (dataset_name is not None and dataset_name[:3] == "Per") or force160250:
        selected_y = p160s250
        ylabel = "PACS 160/SPIRE 250"
    else:
        selected_y = p70p160
        ylabel = "PACS 70/160"
    s350s500 = result['SPIRE350um'] / result['SPIRE500um']
    plt.figure(figsize=(8, 8))

    extra_title = plot_data(dataset_name, SNR_MINIMUM=SNR_MINIMUM, force160250=force160250, additional_mask=additional_mask, coverage_plot=True, extra_title=extra_title)

    # TEMPERATURE
    n_T = selected_y.shape[1]
    scalarMap = cmx.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=n_T),
        cmap=plt.get_cmap('autumn'))
    for i in reversed(range(0, n_T, 30)):
        plt.plot(s350s500[:, i], selected_y[:, i], linestyle='-', linewidth=4, color=scalarMap.to_rgba(i), label='{:4.1f}K'.format(result['T'][0, i]))
    plt.legend(title="Isotherms", loc=2, fontsize='small', fancybox=True)

    # BETA
    n_beta = selected_y.shape[0]
    scalarMap = cmx.ScalarMappable(norm=mcolors.Normalize(vmin=1, vmax=2.5),
        cmap=plt.get_cmap('cool'))
    # for i in range(0, n_beta, 4):
    print('using beta: ', end='')
    for i in [12, 16, 20]:
        b = result['beta'][i, 0]
        print("{:.2f}, ".format(b), end='')
        plt.plot(s350s500[i], selected_y[i], linestyle='-', color=scalarMap.to_rgba(b))
        # for i, b in zip(range(n_beta), result['beta'][:, 0]):
        #     plt.plot(s350s500[i], selected_y[i], label="{:.2f}".format(b))
        # plt.legend()
    print()
    # plt.yscale('log')#, plt.xscale('log')
    # plt.xlim([0.2, 5]), plt.ylim([10**(-8), 5])
    cbar = plt.gcf().colorbar(scalarMap)
    cbar.set_label("beta")



    plt.gcf().canvas.set_window_title(WINDOW_TITLE)
    if dataset_name is not None:
        dataset_name = ", " + dataset_name
    else:
        dataset_name = ""
    plt.title("Herschel color-color diagram" + dataset_name + extra_title)
    plt.xlabel("SPIRE 350/500")
    plt.ylabel(ylabel)

    # print(f"plt.xlim({plt.xlim()}), plt.ylim({plt.ylim()})")
    plt.xlim((0.1561884485339575, 4.895977203736461)), plt.ylim((-0.3239731111304286, 7.395308040999753))

    plt.show()



if __name__ == "__main__":
    make_grid()
    # def additional_mask(imgs):
    #     s500 = imgs['SPIRE500um']
    #     return s500 > 20
    # color_color("Perseus", SNR_MINIMUM=0, additional_mask=additional_mask, extra_title=", 500um > 20")
    # color_color("Perseus", SNR_MINIMUM=10)
