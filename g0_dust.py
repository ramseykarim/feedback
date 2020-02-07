import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
from astropy import units as u
from scipy.special import factorial, zeta
import sys

from parse_FIR_fits import open_FIR_pickle, open_FIR_fits, herschel_path


# Laptop directory
filename = "RCW49large_3p_secondCal_sysErr_jac.fits"

"""
I need to implement this equation for G0:

G0 =
    (1 / Q_abs_avgFUV) *
    (4*15 / pi**5) *
    (beta + 3)! *
    RiemannZeta(beta + 4) *
    Q0 *
    (lambda0 * k_B * T_dust / hc)**beta *
    Stefan-Boltzmann * T_dust**4
for output in W m-2 or erg cm-2 s-1
where Q0 is 1 until further notice
beta, Td are from dust SED fit solutions
Q_abs_avgFUV is the averge Q_abs = C_abs / sigma_dust over the FUV absorption
    range that matters here
"""

def calculate_g0():
    result_dict = open_FIR_fits(herschel_path+filename)
    T, tau, beta = (result_dict[x] for x in ('T', 'tau' ,'beta'))


    a = (0.2 * u.micron).to(u.cm) # micron, grain size, from Xander conversation, range between 0.1-3

    grain_density = 3. * u.g / (u.cm ** 3) # g/cm2, according to Xander (notebook pg 111)
    avg_kappa_abs_FUV = 3.e4 * (u.cm ** 2) / u.g # cm2/g, based on the plots from plot_dust()
    m_grain = grain_density * (np.pi * 4./3) * a**3.
    xr_grain = np.pi * a**2.
    avg_Cabs_FUV = avg_kappa_abs_FUV * m_grain
    avg_Qabs_FUV = avg_Cabs_FUV / xr_grain

    Q0 = 1
    k = u.Quantity(cst.k, cst.unit('Boltzmann constant'))
    h = u.Quantity(cst.h, cst.unit('Planck constant'))
    c = u.Quantity(cst.c, cst.unit('speed of light in vacuum'))
    hc = h*c
    sb = u.Quantity(cst.value('Stefan-Boltzmann constant'), cst.unit('Stefan-Boltzmann constant'))
    Td = T * u.K
    lambda0 = 2 * np.pi * a.to(u.m)

    G0 = (1 / avg_Qabs_FUV) * (4*15/(np.pi**5)) * factorial(beta+3) * zeta(beta+4) * Q0 * ((lambda0*k/hc)*Td)**beta * sb * Td**4.
    Habing = u.Quantity(1.6e-3, 'erg cm-2 s-1')
    G0 = G0.to('erg cm-2 s-1') / Habing
    return G0

def make_plot_g0():
    G0 = calculate_g0()
    obs70 = result_dict['BAND70']
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.log10(obs70), origin='lower', vmin=1, vmax=5)
    plt.title("70um")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.log10(G0.value), origin='lower', cmap='cividis', vmin=0.5, vmax=4)
    plt.colorbar()
    plt.title("G0")
    plt.show()





if __name__ == "__main__":
    make_plot_g0()





"""
Dust opacity vs wavelength plot that I probably might use again later
"""

def plot_dust():
    ### Summary: indicates dust kappa_abs ~ 2-5 * 10^4 cm2/g
    # Load Ormel dust
    # Table structure:
    #cols::[lambda,kappa-ext,kappa-sca,kappa-back,gsca]
    #colunits::[cm,cm2/g,cm2/g,cm2/g,]
    opacities_dir = "/home/ramsey/Documents/Research/Filaments/Dust/OpacTables/"
    def gen_ormel_fn(filename_stub, number):
        fn = f"{opacities_dir}{filename_stub}{number}opc.txt"
        return fn

    dust_types = [
        "(ic-sil,gra)",
        "ic-sil+gra",
        "(ic-sil,ic-gra)",
        "(sil,gra)"
    ]
    times = {
        1: "0.03",
        2: "0.1",
        3: "0.3",
        4: "1.0",
        5: "3.0",
        6: "10.",
    }
    selected_age = 5
    dt = dust_types[3]
    # for dt in dust_types:
    for selected_age in times:
        a = np.genfromtxt(gen_ormel_fn(dt, selected_age))
        wl_um = a[:, 0] * 1e4
        kext = a[:, 1]
        ksca = a[:, 2]
        kabs = kext - ksca
        # plt.plot(wl_um, kext, '-', label="Ormel "+dt+"ext")
        # plt.plot(wl_um, ksca, '-', label="Ormel "+dt+"sca")
        plt.plot(wl_um, kabs, '-', label="Ormel "+dt+f" {times[selected_age]}abs")
    plt.xscale('log'), plt.yscale('log')
    plt.legend()
    plt.show()
