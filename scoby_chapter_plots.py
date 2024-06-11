"""
Make images for the thesis chapter about scoby

Created: April 11, 2024
"""
__author__ = "Ramsey Karim"

import os
import sys
import socket
import pwd
import numpy as np
import importlib.util
import matplotlib.pyplot as plt
from . import catalog
from .m16_deepdive import marcs_colors

from astropy.modeling.models import BlackBody
import astropy.units as u
import astropy.constants as const

scoby_import_path = "/home/ramsey/Documents/Research/scoby-all/scoby"
if os.path.isdir(scoby_import_path):
    sys.path.append(scoby_import_path)
    import scoby
else:
    scoby = None

def create_debug_img_metadata(file=None, func_name=None) -> dict:
    """
    Create a PNG-appropriate metadata dictionary
    which will be passed to matplotlib's savefig function
    :param file: the __file__ variable. Easy, just pass it through.
    :param func_name: name of the current function. This is not trivial
        to get automatically, so just pass it in here
    :returns: dict, appropriate to provide PNG metadata
    """
    source = []
    if file is not None:
        source.append(os.path.basename(file).replace('.py', ''))
    if func_name is not None:
        source.append(func_name)
    source = '.'.join(source)
    if not source:
        source = 'unspecified location'
    source = f'({pwd.getpwuid(os.getuid())[0]}@{socket.gethostname()}) {source} (scoby v{scoby.__version__})'
    return {"Title": "func_name", "Source": source}


def test_plot_sptype_calibration():
    """
    Copied from the unit tests

    Plot Log L vs Teff for Martins and Sternberg tables
    (This used to plot log g vs Teff and included Vacca, but Vacca doesn't do log L)
    TODO: Include Leitherer?
    """
    dfs, col_units = scoby.spectral.sternberg.load_tables_df()
    dfs2, col_units2 = scoby.spectral.martins.load_tables_df()
    colors = {'I': marcs_colors[0], 'III': marcs_colors[1], 'V': marcs_colors[2]}
    plt.figure(figsize=(14, 9))
    Teff, log_g = 'Teff', 'log_g'
    # The characteristics to go on each axis
    char_x, char_y = Teff, log_g #"log_L"
    for lc in scoby.spectral.parse_sptype.luminosity_classes:
        st_sb03 = np.array([scoby.spectral.parse_sptype.st_to_number(i) for i in dfs[lc].index])
        st_m05 = np.array([scoby.spectral.parse_sptype.st_to_number(i) for i in dfs2[lc].index])
        print(lc)
        print(st_sb03)
        print(st_m05)
        print('--')
        independent, dependent = dfs[lc][char_x], dfs[lc][char_y]
        ind2, dep2 = dfs2[lc]['Teff'], dfs2[lc][char_y]
        plt.plot(independent, dependent, 'x', color=colors[lc], label=f'Sternberg+2003 (S03) {lc}')
        plt.plot(ind2, dep2, '.', color=colors[lc], label=f'Martins+2005 {lc}')
        # Interpolate to one of them (Martins)
        fit = scoby.spectral.sternberg.interp1d(ind2, dep2, kind='linear')
        x = np.linspace(ind2.min(), ind2.max(), 50)
        plt.plot(x, fit(x), '--', label=f'fit to S03 class {lc}', color=colors[lc])

    plt.legend()
    plt.ylabel(char_y), plt.xlabel(char_x)
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(catalog.utils.todays_image_folder(), "test_plot_sptype_calibration_stuff.png"),
                metadata=create_debug_img_metadata(file=__file__, func_name="test_plot_sptype_calibration_stuff"))



def plot_Teff_vs_type():
    """
    April 11, 2024
    Try it again but from scratch
    Include Sternberg, Vacca, and Martins and plot against spectral type, not Teff
    """
    sp_numbers = np.arange(3, 11.5)
    sptypes = [(*scoby.spectral.parse_sptype.number_to_st(i), "V") for i in sp_numbers]
    martins_cal = scoby.spectral.sttable.STTable(*scoby.spectral.martins.load_tables_df())
    sternberg_cal = scoby.spectral.sttable.STTable(*scoby.spectral.sternberg.load_tables_df())
    T_m05 = [martins_cal.lookup_characteristic('Teff', st_tuple) for st_tuple in sptypes]
    T_s03 = [sternberg_cal.lookup_characteristic('Teff', st_tuple) for st_tuple in sptypes]
    # Vacca temperatures are exactly the same as Sternberg (can write this into the caption)
    # T_v96 = [scoby.spectral.vacca.vacca_calibration('Teff', st_tuple) for st_tuple in sptypes]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sp_numbers, T_m05, label="Martins+05")
    ax.plot(sp_numbers, T_s03, label="Sternberg+03")
    # ax.plot(sp_numbers, T_v96, label="Vacca+96", linestyle="--")
    plt.show()


def plot_logG_vs_type():
    """
    April 11, 2024
    Try it again but from scratch
    Include Sternberg, Vacca, and Martins and plot against spectral type, not Teff
    """
    sp_numbers = np.arange(3, 11.5)
    sptypes = [(*scoby.spectral.parse_sptype.number_to_st(i), "V") for i in sp_numbers]
    martins_cal = scoby.spectral.sttable.STTable(*scoby.spectral.martins.load_tables_df())
    sternberg_cal = scoby.spectral.sttable.STTable(*scoby.spectral.sternberg.load_tables_df())
    T_m05 = [martins_cal.lookup_characteristic('log_g', st_tuple) for st_tuple in sptypes]
    T_s03 = [sternberg_cal.lookup_characteristic('log_g', st_tuple) for st_tuple in sptypes]
    # Vacca temperatures are exactly the same as Sternberg (can write this into the caption)
    # T_v96 = [scoby.spectral.vacca.vacca_calibration('Teff', st_tuple) for st_tuple in sptypes]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sp_numbers, T_m05, label="Martins+05")
    ax.plot(sp_numbers, T_s03, label="Sternberg+03")
    # ax.plot(sp_numbers, T_v96, label="Vacca+96", linestyle="--")
    plt.show()


def test_L_vs_T_vs_g():
    """
    I want to see if I can use logL and T to map cleanly to log g
    (Sept 23, 2020 for use with Cyg OB2)

    This actually tests PoWR and the fancy interpolation method I use (see powr.py)
    """
    from scipy.optimize import minimize

    df1, u1 = scoby.spectral.martins.load_tables_df()  # dictionary of dfs
    fig = plt.figure()
    colors = {'I': 'blue', 'III': 'green', 'V': 'red'}
    ax = fig.add_subplot(111, projection='3d')
    all_T, all_L, all_g = [], [], []
    for lc in scoby.spectral.parse_sptype.luminosity_classes:
        Teff = df1[lc]['Teff'] / 1000
        all_T.extend(list(Teff))
        logL = df1[lc]['log_L']
        all_L.extend(list(logL))
        log_g = df1[lc]['log_g']
        all_g.extend(list(log_g))
        ax.scatter(Teff, logL, log_g, c=colors[lc], marker='o')

    tbl = scoby.spectral.powr.PoWRGrid('OB')
    # Teff = tbl.grid_info['T_EFF']/1000.
    # logL = tbl.grid_info['LOG_L']
    # log_g = tbl.grid_info['LOG_G']
    # ax.scatter(Teff, logL, log_g, c='k', marker='o')

    Teff = np.arange(20., 50., 1.5) * 1000
    logL = np.linspace(4.4, 6.2, Teff.size)
    tbl.interp_g(Teff[0], logL[0])
    Teff, logL = np.meshgrid(Teff, logL, indexing='xy')
    TL_grid = np.stack([np.log10(Teff), logL], axis=-1)
    log_g = tbl.TL_interp(TL_grid)
    Teff /= 1000
    ax.plot_surface(Teff, logL, log_g, color='orange', alpha=0.3)
    """
    This interpolation is better than the plane fit below. It covers a much
    larger area and reflects some of the curvature of the log_g surface.
    But both are roughly consistent with each other!
    """

    ax.set_xlabel('Teff')
    ax.set_ylabel('log L')
    ax.set_zlabel('log g')

    # From stackoverflow: https://stackoverflow.com/a/20700063
    def plane(x, y, params):
        a, b, d = params[:3]
        z = a * x + b * y + d
        return z

    points = np.array([all_T, all_L, all_g])

    def fit_plane(params):
        residuals = points[2, :] - plane(points[0, :], points[1, :], params)
        return np.sum(residuals ** 2)

    res = minimize(fit_plane, [1, 1, 1])
    print("<test_L_vs_T_vs_g>")
    print(res.x)
    print("</test_L_vs_T_vs_g>\n")
    """
    THE FIT IS:
    [ 0.05727171 -0.65728093  5.20380702]
    We only need to run this once!!
    This is a fit to T/1000, logL for log_g
    """

    xx, yy = np.meshgrid(np.array([27, 45]), np.array([4.5, 6.1]))
    zz = plane(xx, yy, res.x)

    ax.plot_surface(xx, yy, zz, alpha=0.3)

    plt.show()
    # plt.savefig(os.path.join(catalog.utils.todays_image_folder(), "test_L_vs_T_vs_g.png"),
    #             metadata=create_debug_img_metadata(file=__file__, func_name="test_L_vs_T_vs_g"))

def plot_T_vs_g_vs_L():
    """
    April 11, 2024
    get a good plot going for thesis
    """
    df1, u1 = scoby.spectral.martins.load_tables_df()  # dictionary of dfs
    tbl = scoby.spectral.powr.PoWRGrid('OB')
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    sc = ax.scatter(tbl.grid_info['T_EFF']/1000, tbl.grid_info['LOG_G'], c=tbl.grid_info['LOG_L'], marker='s', s=50, cmap='cool')

    cax = ax.inset_axes([1, 0, 0.02, 1])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("log $L$ [$L_{\\odot}$]", size=16)
    cbar.ax.tick_params(labelsize=13)


    sp_numbers = np.arange(3, 13., 0.5)
    martins_cal = scoby.spectral.sttable.STTable(*scoby.spectral.martins.load_tables_df())

    sternberg_cal = scoby.spectral.sttable.STTable(*scoby.spectral.sternberg.load_tables_df())

    for lc in ['I', 'III', 'V']:
        sptypes = [(*scoby.spectral.parse_sptype.number_to_st(i), lc) for i in sp_numbers]

        T_extd_s03 = [sternberg_cal.lookup_characteristic('Teff', st_tuple) for st_tuple in sptypes]
        logg_extd_s03 = [sternberg_cal.lookup_characteristic('log_g', st_tuple) for st_tuple in sptypes]
        ax.plot(np.array(T_extd_s03)/1000, logg_extd_s03, color='k', linestyle='none', marker='x')


        T_extd_m05 = [martins_cal.lookup_characteristic('Teff', st_tuple) for st_tuple in sptypes]
        logg_extd_m05 = [martins_cal.lookup_characteristic('log_g', st_tuple) for st_tuple in sptypes]
        ax.plot(np.array(T_extd_m05)/1000, logg_extd_m05, color='grey', marker='o')
        print(T_extd_m05)

        T_m05 = df1[lc]['Teff']
        logg_m05 = df1[lc]['log_g']
        ax.plot(T_m05/1000, logg_m05, color='k', marker='o')

        ax.text(T_extd_m05[-1]*0.95/1000, logg_extd_m05[-1]*0.99, lc, fontsize=23, fontweight='bold', font='serif', ha='left', va='center')


    ax.set_xlabel("T$_{\\rm eff}$ (kK)", fontsize=16)
    ax.set_ylabel("log $g$ [cm s$^{-2}$]", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.invert_xaxis()
    fig.subplots_adjust(top=0.96, left=0.07, right=0.92, bottom=0.09)
    # plt.show()
    # return
    plt.savefig(os.path.join(catalog.utils.todays_image_folder(), "scoby_TvsgvsL.png"),
        metadata=catalog.utils.create_png_metadata(title="O3 to B2.5 inclusive",
            file=__file__, func="plot_T_vs_g_vs_L"))

def compare_G0():
    """
    April 11, 2024
    Compare scoby G0 for all class V spectral types to integrating the boltzmann function

    I started looking up stellar radii because I thought I needed to convert
    the blackbody flux using it, but Nicola's 2023 paper uses the stefan boltzmann
    law to get around it??
    """
    df, udf = scoby.spectral.martins.load_tables_df()
    # a = df['V']['R']
    # print(a)

    scoby.spectral.stresolver.random.seed(1312)
    scoby.spectral.stresolver.UNCERTAINTY = False  # toggle the half-type/sampling
    np.random.seed(1312)
    powr_grids = {x: scoby.spectral.powr.PoWRGrid(x) for x in ('OB',)}
    cal_tables = scoby.spectral.sttable.STTable(*scoby.spectral.martins.load_tables_df())
    ltables = scoby.spectral.leitherer.LeithererTable()

    sternberg_cal = scoby.spectral.sttable.STTable(*scoby.spectral.sternberg.load_tables_df())

    # b = cal_tables.lookup_characteristic('R', ("B", "0.5", "V"))
    # print(b)
    # return

    sp_numbers = np.arange(3, 11.5)
    sptypes = [(*scoby.spectral.parse_sptype.number_to_st(i), "V") for i in sp_numbers]
    teffs = []
    fuv_lum_bb = []
    Q_martins = []
    Q_sternberg = []
    for num, sptype in zip(sp_numbers, sptypes):
        teff = cal_tables.lookup_characteristic("Teff", sptype)
        logL = cal_tables.lookup_characteristic("log_L", sptype)
        logQ0 = cal_tables.lookup_characteristic("Qo", sptype)
        logQ0_s = sternberg_cal.lookup_characteristic("log_QH", sptype)

        energies = (np.linspace(6, 13.6, 1000)*u.eV)
        freqs = (energies).to(u.Hz, equivalencies=u.spectral())
        t = teff*u.K
        bb = BlackBody(temperature=t)
        spec = bb(freqs)
        fuv_flux = np.trapz(spec, x=freqs)
        # stefan boltzmann; see Nicola's 2023 methods
        total_flux = const.sigma_sb * t**4
        fuv_frac = (np.pi*u.sr * fuv_flux / total_flux).decompose() # pi steradians from Wikipedia Stefan Boltzmann law
        fuv_lum = (10.**logL * u.solLum) * fuv_frac

        teffs.append(teff)
        fuv_lum_bb.append(fuv_lum)
        Q_martins.append(logQ0)
        Q_sternberg.append(logQ0_s)

    fuv_lum_bb = u.Quantity(fuv_lum_bb)
    joined_types = [scoby.spectral.parse_sptype.st_tuple_to_string(x) for x in sptypes]
    catr = scoby.spectral.stresolver.CatalogResolver(joined_types,
        calibration_table=cal_tables, leitherer_table=ltables,
        powr_dict=powr_grids)
    fuv_lum_scoby = u.Quantity([x for x, y in catr.get_array_FUV_flux()])
    Q_scoby = u.Quantity([x for x, y in catr.get_array_ionizing_flux()])
    Q_scoby = np.log10(Q_scoby * u.s) # log10 it to compare with Martins

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(211)
    ax1.plot(sp_numbers, fuv_lum_bb/1e4, label="Blackbody", color=marcs_colors[1])
    ax1.plot(sp_numbers, fuv_lum_scoby/1e4, label="PoWR via scoby", color='k', linewidth=3)
    ax1.tick_params(axis='x', labelbottom=False, bottom=False)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax1.legend()
    ax1.set_ylabel("$L_{\\rm FUV}$ ($10^4$ $L_\\odot$)", fontsize=16)
    # ax.set_yscale('log')
    ax2 = fig.add_subplot(212)
    ax2.plot(sp_numbers, Q_martins, label="Martins+05", color=marcs_colors[1])
    ax2.plot(sp_numbers, Q_sternberg, label="Sternberg+03", color=marcs_colors[2])
    ax2.plot(sp_numbers, Q_scoby, label="PoWR via scoby", color='k', linewidth=3)
    ax2.legend()
    ax2.set_xticks(sp_numbers, labels=[x[:-1] for x in joined_types])
    ax2.set_xlabel("Main sequence (class V) spectral type", fontsize=16)
    ax2.set_ylabel("$Q_0$ [s$^{-1}$]", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    fig.subplots_adjust(top=0.97, left=0.1, right=0.97, bottom=0.1, hspace=0.04)
    # plt.show()
    plt.savefig(os.path.join(catalog.utils.todays_image_folder(), "scoby_compare_GandQ.png"),
        metadata=catalog.utils.create_png_metadata(title="O3 to B2.5 inclusive",
            file=__file__, func="compare_G0"))


def plot_spectrum_O3():
    """
    April 11, 2024
    Just to see
    """
    tbl = scoby.spectral.powr.PoWRGrid("OB")
    teff = 45000
    wf1 = tbl.get_model_spectrum(teff, 3.8)
    wl, flux = wf1


    # tbl.plot_spectrum(*wf1, xunit=u.eV, fuv=True, show=False, ylog=False)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    energy = wl.to(u.eV, equivalencies=u.spectral())
    fuv_mask = (energy > 6*u.eV) & (energy < 13.6*u.eV)
    energy_fuv = energy[fuv_mask]
    flux_fuv = flux[fuv_mask]
    ax.plot(energy_fuv, flux_fuv)
    # ax.set_xscale('log')
    # ax.set_xlim((1, 38))

    t = teff*u.K
    energies = (np.linspace(6, 13.6, 1000)*u.eV)
    # energies = energy
    freqs = (energies).to(u.Hz, equivalencies=u.spectral())
    bb = BlackBody(temperature=t, scale=1*u.Unit("erg / (cm^2 Angstrom s sr)"))
    spec = bb(freqs)
    ax2.plot(energies, spec)
    # ax2.set_xscale('log')
    # ax2.set_xlim((1, 38))


    plt.show()


if __name__ == "__main__":
    compare_G0()
