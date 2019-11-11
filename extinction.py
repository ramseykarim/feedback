import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

"""
Code producing extinction curves that are valid at least in the NIR/Optical/UV (for sure have H alpha and beta)
Supported models are
    1) Cardelli+1989 (commonly CCM 89), from polynomial fits described in the paper
        Supports any Rv, since not read from table
    2) Draine 2003, read from the tables on Draine's website.
        Supports Rv=[3.1, 4.0, 5.5], since 3 discrete tables
Created: October 1, 2019
"""
__author__ = "Ramsey Karim"


VBAND_WAVELENGTH_A = 5448

def extinction_CCM(x, Rv=3.1, normalized=True):
    """
    Extinction based on CCM89 (Cardelli+1989)
    x is reciprocal wavelength in micron^-1
    Function is piecewise in IR / nearIR-optical / UV
    Function has limits; only works between 3.5 micron and 1250 A
    x should be an array; array of the same shape is returned
    normalized is always True due to nature of this function
        call signature must match Draine
    return value is NaN where function is out of bounds
    return value: A(lambda) / Av (Av is centered at 5448 A)
    """
    # Set up regime masks and solution array
    maskIR = (x >= (1/3.5)) & (x < 1.1)
    maskNIROPT = (x >= 1.1) & (x < 3.3)
    maskUV = (x >= 3.3) & (x <=8)
    mask_notcalculated = (x < (1/3.5)) | (x > 8)
    solution = np.full(x.shape, np.nan)
    
    # Infrared case
    xIR = x[maskIR]
    a = 0.574 * (xIR ** 1.61)
    b = -0.527 * (xIR ** 1.61)
    solution[maskIR] = a + b/Rv

    # NIR/Optical case
    xNIROPT = x[maskNIROPT]
    yNIROPT = xNIROPT - 1.82
    a_coeff = (1, 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999)
    b_coeff = (0, 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002)
    a, b = [], []
    for exponent, coeff in enumerate(a_coeff):
        a.append(coeff * (yNIROPT ** exponent))
    for exponent, coeff in enumerate(b_coeff):
        b.append(coeff * (yNIROPT ** exponent))
    a = np.sum(a, axis=0)
    b = np.sum(b, axis=0)
    solution[maskNIROPT] = a + b/Rv

    # UV case
    xUV = x[maskUV]
    Fa = (-0.04473 * (xUV - 5.9)**2) - (0.009779 * (xUV - 5.9)**3)
    Fb = (0.2130 * (xUV - 5.9)**2) + (0.1207 * (xUV - 5.9)**3)
    Fa[xUV < 5.9] = 0
    Fb[xUV < 5.9] = 0
    a = 1.752 - (0.316*xUV) - (0.104/(((xUV - 4.67)**2) + 0.341)) + Fa
    b = -3.090 + (1.825 * xUV) + (1.206/(((xUV - 4.62)**2) + 0.263)) + Fb
    solution[maskUV] = a + b/Rv

    return solution


def extinction_Draine(x, Rv=3.1, normalized=False):
    """
    Extinction based on Draine 2003 (from website)
    x is reciprocal wavelength in units of micron^-1
    Function only works in Draine table limits (most of EM spectrum)
        1 cm to 12.4 keV
    x should be a value or array (input to scipy.interpolate.UnivariateSpline)
    return value is of the same shape as x
    return value is NaN where the function is out of bounds
    return value:
        IF NORMALIZED is TRUE:
        C(lambda) / Cv (Cv is referenced at 5448 Angstroms)
        C is cross section, but is linearly related to A (extinction) as
            A_lambda = 1.086 * (column density cm-2) * C [cm 2]
        Thus the ratio C_lambda/Cv should be equivalent to A_lambda/Av
        as given by CCM87 (extinction_CCM defined above)
        IF NORMALIZED IS FALSE:
        extinction cross section C(lambda) in units of cm-2 / H nucleon
    """
    availableRv = [3.1, 4.0, 5.5]
    # Reference to V band center, like Av (5448 Angstroms)
    # Uses the "V filter" comment in the Draine table
    reference_vals = [4.896E-22, 5.787E-22, 6.715E-22]
    # Check if valid Rv
    if Rv not in availableRv:
        raise RuntimeError("Rv={} not supported by Draine".format(Rv))
    filenames = ["kext_albedo_WD_MW_3.1_60_D03.all", "kext_albedo_WD_MW_4.0A_40_D03.all", "kext_albedo_WD_MW_5.5A_30_D03.all"]
    filename = "dustmodels/" + filenames[availableRv.index(Rv)]
    n_wl, n_C = "lambda", "C_ext"
    table = np.genfromtxt(filename, skip_header=80, usecols=[0,3], names=(n_wl, n_C))
    # Wavelength column is in microns; convert to micron^-1 to match input (and CCM)
    rwl_column = 1./table[n_wl]
    # Avoid unnecessary spline interp by trimming to x query bounds.
    # Draine generally steps in 0.97 geometrical increments,
    #  so 0.9 should grab us ~4 extra points on each side
    query_min, query_max = np.min(x)*.9, np.max(x)*1.1
    query_mask = (rwl_column >= query_min) & (rwl_column <= query_max)
    # Create spline fit that interpolates thru all points (s=0)
    spline = UnivariateSpline(rwl_column[query_mask], table[n_C][query_mask], s=0, ext="zeros")
    solution = spline(x)
    # Draine curves don't have zeros, so zeros indicate out-of-bounds. Replace with NaN.
    solution[solution == 0] = np.nan
    if normalized:
        # Normalize to V band cross section
        solution /= reference_vals[availableRv.index(Rv)]
    return solution


def extinction(x, method="Draine", **kwargs):
    """
    Generalized wrapper for CCM and Draine extinctions
    Valid methods are "Draine" and "CCM"
    x is reciprocal wavelength in units of micron^-1
    return value is of the same shape as x
    return value is NaN where the function is out of bounds
        Draine has more EM spectrum coverage
    return value: A(lambda) / Av (Av is centered at 5448 A)
    See extinction_Draine and extinction_CCM doc for details
    """
    extinction_functions = {"Draine": extinction_Draine, "CCM": extinction_CCM}
    if method not in extinction_functions:
        raise RuntimeError(f"Method {method} is not valid. Accepted extinction models are {list(extinction_functions.keys())}")
    return extinction_functions[method](x, **kwargs)



def Hab_reddening(ja2jb, Rv=3.1, Hab_ratio=2.89, method="Draine"):
    """
    Calculates reddening by extinction given the ratio of H alpha to H beta intensities
    Can be calculated either using CCM 87 dust models (any Rv) or Draine 2003 (Rv 3.1, 4.0, or 5.5)
    Hab_ratio 2.89 used in Weilbacher+2015
        Osterbrock text(1989): 2.8(6,5,1) for N:10^(2,4,6) cm-3 and T:10^4 K
    method kwarg is either "CCM" or "Draine"
    Result is in the most natural units to the method. CCM: Av, Draine: N [cm-2]
    Results are designed to be directly multiplied to the result of "extinction" function to yield a meaningful curve.
    Result matches the shape of input ja2jb.
    """
    if method == "CCM":
        # Using CCM:
        wl_Halpha, wl_Hbeta = 6563., 4861.
        A_Avs = extinction_CCM(1e4/np.array([wl_Halpha, wl_Hbeta]), Rv=Rv)
        Avdiff = A_Avs[1] - A_Avs[0]  # H beta minus H alpha extinction at Av=1
        # Now use ja/jb ratio to solve for Av describing reddening
        Av = 2.5 * np.log10(ja2jb/Hab_ratio) / Avdiff # 2.5 from mag system :)))
        return Av
    else:
        # Using Draine:
        availableRv = [3.1, 4.0, 5.5]
        if Rv not in availableRv:
            raise RuntimeError("Rv={} not supported by Draine".format(Rv))
        # C_beta - C_alpha, from the rows marked "H alpha" and "H beta"
        C_diffs = [5.704E-22 - 3.801E-22, 6.568E-22 - 4.615E-22, 7.427E-22 - 5.494E-22]
        C_diff = C_diffs[availableRv.index(Rv)]
        # Now use ja/jb ratio to solve for column density N [cm-2] describing extinction
        N = np.log(ja2jb / Hab_ratio) / C_diff  # natural log is intentional
        return N



def test_demonstrate_success_CCM(muse=False):
    if muse:
        rwl_range_A = np.arange(4000, 9550, 1.25)
        rwl_range = 1e4/rwl_range_A
    else:
        rwl_range = np.arange(0.5, 8.01, 0.05)
        rwl_range_A = 1e4/rwl_range
    x_plot = rwl_range_A if muse else rwl_range
    for Rv in [3.1, 3.2, 4.1, 5.1, 5.5]:
        Av = extinction_CCM(rwl_range, Rv=Rv)
        plt.plot(x_plot, Av, label="Rv={:.1f}".format(Rv))
    xlabel = r"$\lambda$ (Angstroms)" if muse else r"1/$\lambda$ ($\mu m^{-1}$)"
    plt.xlabel(xlabel)
    plt.ylabel(r"A$_{\lambda}$ / Av")
    plt.legend()
    plt.show()


def test_demonstrate_success_DrainevCCM(muse=False):
    if muse:
        rwl_range_A = np.arange(4000, 9550, 1.25)
        rwl_range = 1e4/rwl_range_A
    else:
        rwl_range = np.arange(0.5, 8.01, 0.05)
        rwl_range_A = 1e4/rwl_range
    x_plot = rwl_range_A if muse else rwl_range
    for Rv in [3.1, 4.0, 5.5]:
        for method in ["Draine", "CCM"]:
            plt.plot(x_plot, extinction(rwl_range, method=method, Rv=Rv, normalized=True), label=f"{method}: Rv={Rv}", linestyle="--" if method=="CCM" else "-")
    xlabel = r"$\lambda$ (Angstroms)" if muse else r"1/$\lambda$ ($\mu m^{-1}$)"
    plt.xlabel(xlabel)
    plt.ylabel(r"A$_{\lambda}$ / Av")
    plt.legend()
    plt.show()


def test_match_Bolatto17():
    Rv = 3.075
    wl_Halpha, wl_Hbeta = 6563., 4861.
    A_Avs = extinction_CCM(1e4/np.array([wl_Halpha, wl_Hbeta]), Rv=Rv)
    Avdiff = A_Avs[1] - A_Avs[0]  # H beta minus H alpha extinction at Av=1
    print("Halpha", A_Avs[0])
    print("Bolatto+17 gives 5.86; we get {:.2f}".format(2.5 * A_Avs[0] / Avdiff))


def Av2ext(Av):
    return 10 ** (-Av/2.5)

if __name__ == "__main__":
    test_demonstrate_success_DrainevCCM()
