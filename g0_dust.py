import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
from astropy import units as u

from parse_FIR_fits import open_FIR_pickle


# Laptop directory
herschel_path = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/"
filename = "RCW49large_350grid_3p_TILEFULL.pkl"

# result_dict = open_FIR_pickle(herschel_path+filename)
# T, tau, beta = (result_dict[x] for x in ('T', 'tau' ,'beta'))


a = (0.2 * u.micron).to(u.cm) # micron, grain size, from Xander conversation, range between 0.1-3
grain_density = 3. * u.g / (u.cm ** 3) # g/cm2, according to Xander (notebook pg 111)
avg_kappa_abs_FUV = 3.e4 * (u.cm ** 2) / u.g # cm2/g, based on the plots from plot_dust()
m_grain = grain_density * (np.pi * 4./3) * a**3.
xr_grain = np.pi * a**2.
avg_Cabs_FUV = avg_kappa_abs_FUV * m_grain
avg_Qabs_FUV = avg_Cabs_FUV / xr_grain

"""
I need to implement this equation for G0:

G0 =
    (1 / Q_abs_avgFUV) *
    (4*15 / pi) *
    (beta + 3)! *
    RiemannZeta(beta + 4) *
    Q0 *
    (lambda0 * k_B * T_dust / hc)**beta
where Q0 is 1 until further notice
beta, Td are from dust SED fit solutions
Q_abs_avgFUV is the averge Q_abs = C_abs / sigma_dust over the FUV absorption
    range that matters here
"""













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
