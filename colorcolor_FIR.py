import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
import pickle

from mantipython.v3.src import greybody, dust, instrument

"""
Generating a grid of measurements for a fixed optical depth.
Grid varies by beta and temperature
Purpose is to look at a color/color diagram
"""

def make_grid():
    herschel = instrument.get_all_Herschel()

    # set up parameter grid

    # Temperature from 5 to 150 in geometric steps of 3%
    T_range = np.exp(np.arange(np.log(5), np.log(150), np.log(1.03)))
    # Beta in arithmetic steps of 0.1
    b_range = np.arange(1., 2.51, 0.1)

    TT, bb = np.meshgrid(T_range, b_range, indexing='xy')
    result = {d.name: np.full(TT.shape, np.nan).ravel() for d in herschel}

    for i, T, b in zip(range(TT.size), TT.flat, bb.flat):
        for d in herschel:
            result[d.name][i] = d.detect(greybody.Greybody(T, 1, dust.TauOpacity(b)))
    for d in herschel:
        result[d.name] = result[d.name].reshape(TT.shape)
    result['T'] = TT
    result['beta'] = bb

    with open("/home/ramsey/Documents/Research/Filaments/tb_grid.pkl", 'wb') as f:
        pickle.dump(result, f)


def color_color():
    with open("/home/ramsey/Documents/Research/Filaments/tb_grid.pkl", 'rb') as f:
        result = pickle.load(f)
    p70p160 = result['PACS70um'] / result['PACS160um']
    s350s500 = result['SPIRE350um'] / result['SPIRE500um']
    plt.figure(figsize=(8, 8))

    # TEMPERATURE
    n_T = p70p160.shape[1]
    scalarMap = cmx.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=n_T),
        cmap=plt.get_cmap('autumn'))
    for i in range(0, n_T, 20):
        plt.plot(s350s500[:, i], p70p160[:, i], linestyle='-', linewidth=4, color=scalarMap.to_rgba(i), label='{:4.1f}K'.format(result['T'][0, i]))
    plt.legend(title="Isotherms", loc=4, fontsize='small', fancybox=True)

    # BETA
    n_beta = p70p160.shape[0]
    scalarMap = cmx.ScalarMappable(norm=mcolors.Normalize(vmin=1, vmax=2.5),
        cmap=plt.get_cmap('cool'))
    for i in range(n_beta):
        b = result['beta'][i, 0]
        plt.plot(s350s500[i], p70p160[i], linestyle='-', color=scalarMap.to_rgba(b))
        # for i, b in zip(range(n_beta), result['beta'][:, 0]):
        #     plt.plot(s350s500[i], p70p160[i], label="{:.2f}".format(b))
        # plt.legend()
    plt.yscale('log')#, plt.xscale('log')
    # plt.xlim([0.2, 3.5]), plt.ylim([10**(-8), 10])
    cbar = plt.gcf().colorbar(scalarMap)
    cbar.set_label("beta")

    plt.title("Herschel color-color diagram")
    plt.xlabel("SPIRE 350/500")
    plt.ylabel("PACS 70/160")
    plt.show()

if __name__ == "__main__":
    color_color()
