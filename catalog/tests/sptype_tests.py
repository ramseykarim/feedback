"""
Tests for the spectral module, dealing with spectral type reading and stellar
parameter assignment.

Split from catalog_spectral.py (previously readstartypes.py) on June 2, 2020

Created: June 2, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy import units as u

from .. import utils
from .. import spectral


def main():
    """
    Easier to have this at the top, never have to scroll down.
    "args" variable will contain any return values
    """
    return test_powr_totalL_accuracy()


def plot_sptype_calibration_stuff():
    dfs, col_units = spectral.sternberg.load_tables_df()
    dfs2, col_units2 = spectral.martins.load_tables_df()
    colors = {'I':'blue','III':'green','V':'red'}
    plt.figure(figsize=(14, 9))
    for lc in spectral.parse_sptype.luminosity_classes:
        Teff, log_g = 'Teff', 'log_g'
        charX, charY = Teff, "log_L"
        st_sb03 = np.array([spectral.parse_sptype.st_to_number(i) for i in  dfs[lc].index])
        st_m05 = np.array([spectral.parse_sptype.st_to_number(i) for i in  dfs2[lc].index])
        independent, dependent = dfs[lc][charX], dfs[lc][charY]
        ind2, dep2 = dfs2[lc]['teff'], dfs2[lc][charY]
        plt.plot(independent, dependent, 'x', color=colors[lc], label='SB03')
        plt.plot(ind2, dep2, '.', color=colors[lc], label='M05')
        fit = spectral.sternberg.interp1d(independent, dependent, kind='linear')
        x = np.linspace(independent.min(), independent.max(), 50)
        plt.plot(x, fit(x), '--', label=f'fit to Sternberg+2003 class {lc}', color=colors[lc])
        # plt.plot(x, [spectral.vacca.vacca_calibration((*spectral.parse_sptype.number_to_st(i), lc), charY) for i in x], '-', color=colors[lc], label='Vacca+1996 calib. class {}'.format(lc))
        # plt.plot(x, [vacca_calibration(i, {'I':1,'III':3,'V':5}[lc], charY) for i in x], '-', color=colors[lc], label='Vacca+1996 calib. class {}'.format(lc))
    # plt.legend()
    plt.ylabel(charY), plt.xlabel(charX)
    plt.gca().invert_xaxis()
    plt.show()


def test_martins_calibration_load():
    df1, u1 = spectral.martins.load_tables_df()
    df2, u2 = spectral.sternberg.load_tables_df()
    for i in u2.Units:
        print(i, u.Unit(i))
    for i in u1.Units:
        print(i, u.Unit(i))


def test_martins_calibration():
    df1, u1 = spectral.martins.load_tables_df()
    print(df1['V'])
    df2, u2 = spectral.sternberg.load_tables_df()
    print(df2['V'])
    print('-----')
    print(u1)
    print(u2)


def test_sttables():
    """
    I used this to confirm that STTable gives good looking results
    for both Sternberg and Martins
    """
    df1, u1 = spectral.martins.load_tables_df()
    df2, u2 = spectral.sternberg.load_tables_df()
    stt1 = spectral.sttable.STTable(df1, u1)
    stt2 = spectral.sttable.STTable(df2, u2)
    return stt1, stt2


def test_leitherer_open():
    """
    open_tables works, as far as I can tell
    """
    df1, u1 = spectral.leitherer.open_tables()
    # print(df1)
    print(u1)
    return df1, u1


def test_leitherer_grid_smoothness():
    """
    Check interpolation via CloughTocher2DInterpolator (and inherent smoothness
    of these characteristics across the T, L model grid)

    Results: Mdot is smooth, interpolation looks good. scipy.interpolate.interp2d
    is much worse than this CloughTocher2DInterpolator I am now using, which
    looks pretty good.
    v_inf is a little rockier, looks like there are some jumps (we know about these)
    The interp still deals with them reasonably well. Nothing is that extreme.
    Mass is also a little rocky, similar valleys to v_inf, but same deal.
    Nothing too bad.
    Radius is remarkably smooth and well behaved.
    """
    df1, u1 = spectral.leitherer.open_tables()
    T = np.log10(df1['T_eff'])
    Tlabel = f"log T_eff ({u1.loc['T_eff', 'Units']})"
    L = df1['log_L']
    Llabel = f"log L ({u1.loc['log_L', 'Units']})"
    z = df1['R']
    TL = np.array([T.values, L.values]).T
    interp = spectral.leitherer.CloughTocher2DInterpolator(TL, z, fill_value=np.nan)
    print(u1)
    plt.subplot(121)
    plt.scatter(T, L, marker='o', c=z, vmin=np.min(z), vmax=np.max(z))
    plt.colorbar()
    plt.xlabel(Tlabel), plt.ylabel(Llabel)
    plt.gca().invert_xaxis()
    aspect = plt.gca().get_aspect()
    xlim, ylim = (T.min(), T.max()), (L.min(), L.max())
    plt.subplot(122)
    zz = interp(T, L)
    print(T.shape, L.shape, zz.shape)
    print(zz.shape)
    plt.scatter(T, L, c=zz, vmin=np.min(z), vmax=np.max(z))
    xx, yy = np.meshgrid(np.linspace(*xlim, 50), np.linspace(*ylim, 50))
    zz = interp(xx, yy)
    plt.imshow(zz, origin='lower', vmin=np.min(z), vmax=np.max(z),
        extent=[*xlim, *ylim])
    plt.colorbar()
    plt.gca().invert_xaxis()
    plt.gca().set_aspect(aspect)
    plt.show()


def test_leitherer_sptypes():
    """
    This test shows that I can connect the Martins (or Sternberg) tables to
    the Leitherer tables and thus reliably map spectral type to mass loss rate
    via T & L. I can also get vinf, so I can get momentum flux.
    """
    ltables = spectral.leitherer.LeithererTable()
    mtables = spectral.sttable.STTable(*spectral.martins.load_tables_df())
    sptypes_n = np.arange(3, 13., 0.5)
    print(mtables.column_units)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for lc in spectral.parse_sptype.luminosity_classes:
        sptypes = [spectral.parse_sptype.number_to_st(x)+(lc,) for x in sptypes_n]
        T_arr = [mtables.lookup_characteristic('Teff', x) for x in sptypes]
        logL_arr = [mtables.lookup_characteristic('log_L', x) for x in sptypes]
        axes[0].plot(np.log10(T_arr), logL_arr, '--', label=lc)
        mdot_arr = ltables.lookup_characteristic('log_Mdot', T_arr, logL_arr)
        vinf_arr = ltables.lookup_characteristic('v_inf', T_arr, logL_arr)
        c0 = axes[0].scatter(np.log10(T_arr), logL_arr, marker='o', c=mdot_arr, vmin=-9, vmax=-5)
        pdot_arr = np.log10((10.**np.array(mdot_arr))*(1.988e33/3.154e7) * np.array(vinf_arr) * 1e5)
        print(pdot_arr)
        c1 = axes[1].scatter(np.log10(T_arr), logL_arr, marker='o', c=pdot_arr, vmin=24, vmax=29)
    for i in range(2):
        plt.sca(axes[i])
        plt.xlabel("log T"), plt.ylabel("log L")
        plt.gca().invert_xaxis()
        if not i:
            plt.legend()
            plt.colorbar(c0, label='log $\dot M$ ($M_{\\odot} yr-1$)')
            plt.title("O-type L vs T and mass loss rate")
        else:
            plt.colorbar(c1, label='log $\dot p$ (dyne)')
            plt.title("momentum transfer rate")
    plt.tight_layout()
    plt.show()




old_catalog_fn = f"{utils.ancillary_data_path}catalogs/Ramsey/old_catalogs/OBradec.pkl"

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Older tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def test_st_parse_slashdash():
    """
    Tested the new subtype dash behavior, it looks like it works!
    """
    cat = pd.read_pickle(old_catalog_fn)
    tests = [cat.SpectralType[19], cat.SpectralType[5], cat.SpectralType[7], 'B5/6.5III', 'O4II/III', cat.SpectralType[26], cat.SpectralType[27], 'O4-5.5V/III*']
    for t in tests:
        l = spectral.parse_sptype.st_parse_slashdash(t)
        print(t, '\t', l)
        print('\t', [spectral.parse_sptype.st_parse_type(x) for x in l])
        print()


def test_full_st_parse():
    """
    This is no longer the "full" parse
    """
    cat = pd.read_pickle(old_catalog_fn)
    count = 0
    for st in cat.SpectralType:
        assert isinstance(st, str)
        print(f"{st:30s}", end=": ")
        if st == 'ET':
            st = 'O9.5V'
            count += 1
        stars = [[spectral.parse_sptype.st_parse_type(x) for x in spectral.parse_sptype.st_parse_slashdash(y)] for y in spectral.parse_sptype.st_parse_binary(st)]
        types = [x[0][0:2] for x in stars]
        if types and all(types[0]):
            print([i[0]+i[1] for i in types])
        else:
            print(stars)
    print(f"There are {count} ET stars")


def test_powr_totalL_accuracy():
    tbl = spectral.powr.PoWRGrid('OB')
    count = 0
    for model_info in tbl.iter_models():
        count += 1
        wl, flux = tbl.get_model(model_info)
        lum = np.trapz(flux, x=wl).to('solLum')
        print(f"Model {model_info.MODEL} (Teff{model_info.T_EFF}/log_g{model_info.LOG_G}:")
        print(f" -> Integrated luminosity (numpy): {lum.to_string()}")
        print(f" -> Difference: {((lum.to_value() / (10**model_info.LOG_L))-1)*100.:.5} %")
        print()
        if count > 20:
            break

"""

def test_powr_totalL_plot():
    # tbl.plot_grid_space(tbl.grid_info.LOG_L, "log L")
    tbl = PoWRGrid('WNE')
    tbl.plot_grid_space(c='blue', show=False, alpha=0.3)
    tbl = PoWRGrid('WNL')
    tbl.plot_grid_space(c='red', setup=False, alpha=0.3)


def testSuite_PoWR():
    tbl = PoWRGrid('WNL')
    wf1 = tbl.get_model(4.4, 1.7)
    wf2 = tbl.get_model(5.0, 0.0)
    tbl.plot_spectrum(*wf1, show=False, xunit=u.eV, fuv=True, ylog=False)
    tblob = PoWRGrid('OB')
    wf3 = tblob.get_model(43000, 3.8)
    tbl.plot_spectrum(*wf2, show=False, setup=False, xunit=u.eV, fuv=True, ylog=False)
    wf4 = tblob.get_model(16000, 3.0)
    tbl.plot_spectrum(*wf3, show=False, setup=False, xunit=u.eV, fuv=True, ylog=False)
    tbl.plot_spectrum(*wf4, setup=False, xunit=u.eV, fuv=True, ylog=False)



def test_STResolver():
    def f(x):
        # function to pass NaNs and print MODEL from DataFrames
        try:
            return x['MODEL']
        except:
            return "[NO MODEL]"
    tbls = {x: PoWRGrid(x) for x in ('OB', "WNE", "WNL")}
    sb_tables = S03_OBTables()
    # cat = pd.read_pickle(f"{catalogs_directory}/Ramsey/OBradec.pkl")
    tests = ['B5/6.5III', 'O4I/III', 'WN', 'C*']
    count = 0
    for t in tests:
        if t == "ET":
            t = "O7.5/B1"
        s = STResolver(t)
        # print(t, '\n\t', s)
        # print('\t', s.__repr__())
        s.link_powr_grids(tbls)
        v, lu = s.get_mass_loss_rate(sb_tables)
        l, u = lu
        dl, du = v-l, u-v
        # vtxt = "---N/A---" if ((v == 0.) or np.isnan(v)) else f"{v:.2E}"
        # etxt = "" if ((dl == 0.) and (du == 0.)) else f" (-{dl:.2E}, +{du:.2E})"
        vtxt = str(v) + " +/- "
        etxt = str(dl) + "," + str(du)
        print(f"{s.__repr__():.<30s}: {vtxt}{etxt}")
        count += 1
        if count > 95:
            break
    return s
"""

if __name__ == "__main__":
    args = main()
