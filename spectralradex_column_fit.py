"""
Fitting column densities using spectralradex

Created: October 13, 2023 (Friday the 13th)
"""
__author__ = "Ramsey Karim"

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from spectralradex import radex

ratio_12co_to_H2 = 8.5e-5 # from Tielens book
ratio_12co_to_13co = 44.65 # call it 45 in paper, the difference will be miniscule


def get_T_R_column_values(df, filter=True, jup=(1, 3)):
    """
    Get the "T_R (K)" column values of the input DataFrame as a list
    :param df: pandas DataFrame with "T_R (K)" column
    :param filter: bool, True if filtering for Qup in (1, 3)
    :returns: list of floats. len of list == number of rows in df
        or len list == 2 if filter == True
    """
    colname = "T_R (K)"
    if filter:
        df = df.loc[df['Qup'].apply(int).isin(jup)]
    vals = df[colname].values
    return list(vals)



class TestSpectralRadex:
    """
    Not an official unit test, but something like it so I can organize and not
    delete tests
    """
    def __init__(self, verbose=True):
        """
        Set a "verbose" boolean parameter; controls more frequent printouts
        """
        print("TESTING")
        self.verb = verbose
        self.molfile = None

    def list_data_files(self):
        radex.list_data_files()

    def set_params(self):
        """
        Get the default params into a dictionary and set the frequency limits
        to contain the 1-0 through 3-2 lines
        """
        params = radex.get_default_parameters()
        params['fmin'] = 100.
        params['fmax'] = 400.
        self.params = params
        if self.molfile is None:
            self.molfile = params['molfile']
        else:
            params['molfile'] = self.molfile

    def print_params(self, table=False):
        """
        Print params, perhaps in an Obsidian-ready table
        """
        l = max([len(p) for p in self.params])
        if table:
            print("| Key | Default Value |") # cool table making for Obsidian
            print("|---|---|")
        for k in self.params:
            if table:
                print(f"| {k:>{l+1}} | {self.params[k]} |")
            else:
                print(f"{k:>{l+1}}", "-"*4, self.params[k])

    def run_radex_basic(self):
        """
        Run using default params (which should already be set with .set_params)
        """
        self.result = radex.run(self.params)
        if self.verb:
            print(self.result)
            print(type(self.result))

    def select_rows(self):
        """
        Select the 1-0 and 3-2 rows.
        Qup and Qlow are Jup and Jlow.
        The E_UP (K) energies match expected.
        """
        self.selected_rows = self.result.loc[self.result['Qup'].apply(int).isin([1, 3])]
        if self.verb:
            print(self.selected_rows)

    def test_basic(self):
        """
        Roundup of the last few tests
        """
        self.set_params()
        if self.verb:
            self.print_params()
        self.run_radex_basic()
        self.select_rows()

    def verify_that_flux_is_integrated_gaussian(self):
        """
        Show that FLUX is the Gaussian integral using T_R and the linewidth
        """
        self.test_basic()
        tr = self.selected_rows["T_R (K)"]
        flux = self.selected_rows["FLUX (K*km/s)"]
        if self.verb:
            print(tr)
            print(flux)
        print((flux/(tr/2.355))/np.sqrt(2*np.pi))


    def find_basic_results(self):
        """
        Get the T_R from the default run (.test_basic)
        """
        self.molfile = "13co.dat"
        self.set_params()
        self.params['cdmol'] = 1e21 * ratio_12co_to_H2
        self.params['h2'] = 1e4
        if '13' in self.molfile:
            self.params['cdmol'] /= ratio_12co_to_13co
        self.run_radex_basic()
        self.select_rows()
        tr_colname = "T_R (K)"
        tr = self.selected_rows[tr_colname]
        if self.verb:
            for col in self.selected_rows.columns:
                print('-')
                print(col)
                print(self.selected_rows[col])
                print('-')
        else:
            print(self.molfile)
            print(tr_colname)
            print(tr)


    def test_fit_params_fixed_lw(self):
        """
        Test out fitting to observations with fixed linewidth (1.0 km/s for all lines)
        """
        # set up "observations" based on outputs from .find_basic_results()
        # I'll always set up observations in the order [12co10, 12co32, 13co10, 13co32]
        # observations are T_R in K
        ##### DEFAULT TESTS (known answers)
        # observations = np.array([0.006275, 0.021294, 0.000129, 0.000462]) # too low for fit to work; based on defaults
        # observations = np.array([22.775128, 18.294886, 1.543410, 2.442767]) # works ok for fit; based on realistic defaults
        ##### REAL TEST (unknown answer, real data)
        # observations = np.array([27.39, 39.41, 7.84, 20.01]) # first point, western side
        observations = np.array([28.87, 22.67, 11.81, 10.29]) # second point, eastern side, smooth spatial emission, similar Tex from coldens estimate but dissimilar coldens
        errors = observations * 0.1 # 5 or 10% uncertainty
        e2 = errors**2 # we always use error^2
        # parameters are LOG10 N(H2), LOG10 n(H2), T_K

        def run_12_and_13(x):
            """
            Separate from goodness_of_fit_f so that we can find out what the output is for the solution
            """
            # unpack parameters
            log10_coldens_h2, log10_dens_h2, t_k = x
            coldens_h2 = 10**log10_coldens_h2
            dens_h2 = 10**log10_dens_h2
            # transform coldens_h2 to coldens_12co
            coldens_12co = coldens_h2 * ratio_12co_to_H2
            # Run 12CO
            params_12 = radex.get_default_parameters()
            params_12.update({'molfile': "co.dat", 'tkin': t_k, 'cdmol': coldens_12co, 'h2': dens_h2, 'fmin': 100, 'fmax': 400, 'linewidth': 2.3})
            result_12 = radex.run(params_12)
            t_r_12 = get_T_R_column_values(result_12)
            # Run 13CO
            coldens_13co = coldens_12co / ratio_12co_to_13co
            params_13 = radex.get_default_parameters()
            params_13.update({'molfile': "13co.dat", 'tkin': t_k, 'cdmol': coldens_13co, 'h2': dens_h2, 'fmin': 100, 'fmax': 400, 'linewidth': 1.5})
            result_13 = radex.run(params_13)
            t_r_13 = get_T_R_column_values(result_13)
            # combine modeled observations into single array
            model_data = np.array(t_r_12 + t_r_13)
            return model_data

        def goodness_of_fit_f(x):
            # use run wrapper to get model data
            model_data = run_12_and_13(x)
            # calculate chi squared
            chisq = np.sum((model_data - observations)**2 / e2)
            return chisq #np.sqrt(chisq)

        # self.params['cdmol'] = 1e21 * ratio_12co_to_H2
        # self.params['h2'] = 1e4

        if False:
            x_test = [np.log10(1e21), np.log10(1e4), 30]
            x_bounds = [(np.log10(1e20), np.log10(9e23)),
                (1, 6), (10, 150)
            ]
            print(x_test)
            print(goodness_of_fit_f(x_test))
            result = minimize(goodness_of_fit_f,
                x0=np.array(x_test)*1.2,
                bounds=x_bounds
            )
            print(result)
            N_H2, nh2, t_k = result.x
            print(f"N(H2) {10**N_H2:.2E} cm-2")
            print(f"n_H2 {10**nh2:.2E} cm-3")
            print(f"T_K {t_k:.1f} K")


        # print(observations)
        # print(run_12_and_13(result.x))
        if True:
            t_k = 32.6
            N_H2 = np.log10(1.47e22)
            nh2 = np.log10(5.6e3)
            params_12 = radex.get_default_parameters()
            params_12.update({'molfile': "13co.dat", 'tkin': t_k, 'cdmol': (10**N_H2)*ratio_12co_to_H2/ratio_12co_to_13co, 'h2': 10**nh2, 'fmin': 100, 'fmax': 400, 'linewidth': 2.3})
            print(params_12)
            result_12 = radex.run(params_12)
            print(result_12['FLUX (K*km/s)'])
            print(result_12['tau'])
            print(result_12['T_ex'])

    def test_fit_params_co32_only(self, n=1e2, tk=30, verb=False):
        """
        December 5, 2023
        Do the same as test_fit_params_fixed_lw
        but only use CO 3-2, no 1-0. Assume that 3-2 traces different gas (different density)
        """
        # observations (12co32, 13co32)
        observations = np.array([22.67, 10.29]) # second point, see method above
        errors = np.array([0.55, 0.68]) # from cube_utils.onesigmas
        e2 = errors**2
        # parameters are LOG10 N(H2), LOG10 n(H2), T_K
        # fixing n and T_K. Only fit for N(H2)

        def run_12_and_13(*x):
            """
            Separate from goodness_of_fit_f so that we can find out what the output is for the solution
            """
            # unpack parameters
            log10_coldens_h2, log10_dens_h2, t_k = (float(parameter) for parameter in x)
            coldens_h2 = 10**log10_coldens_h2
            dens_h2 = 10**log10_dens_h2
            # transform coldens_h2 to coldens_12co
            coldens_12co = coldens_h2 * ratio_12co_to_H2
            # Run 12CO
            params_12 = radex.get_default_parameters()
            params_12.update({'molfile': "co.dat", 'tkin': t_k, 'cdmol': coldens_12co, 'h2': dens_h2, 'fmin': 100, 'fmax': 400, 'linewidth': 2.3})
            result_12 = radex.run(params_12)
            t_r_12 = get_T_R_column_values(result_12, jup=(3,))
            # Run 13CO
            coldens_13co = coldens_12co / ratio_12co_to_13co
            params_13 = radex.get_default_parameters()
            params_13.update({'molfile': "13co.dat", 'tkin': t_k, 'cdmol': coldens_13co, 'h2': dens_h2, 'fmin': 100, 'fmax': 400, 'linewidth': 1.5})
            result_13 = radex.run(params_13)
            t_r_13 = get_T_R_column_values(result_13, jup=(3,))
            # combine modeled observations into single array
            model_data = np.array(t_r_12 + t_r_13)
            return model_data

        def goodness_of_fit_f(*x):
            # use run wrapper to get model data
            model_data = run_12_and_13(*x)
            # calculate chi squared
            chisq = np.sum((model_data - observations)**2 / e2)
            return chisq #np.sqrt(chisq)

        x_test = [np.log10(5e21), np.log10(n), tk]
        x_bounds = [(np.log10(1e20), np.log10(9e23))]
        if verb:
            print("params", x_test)
            print("obs", observations)
            print("init", run_12_and_13(*x_test))
            print("chisq init", goodness_of_fit_f(*x_test))
            print()
        result = minimize(goodness_of_fit_f,
            x0=x_test[0],
            bounds = x_bounds,
            args=tuple(x_test[1:]))
        N_H2 = float(result.x)
        chisq = goodness_of_fit_f(result.x, *x_test[1:])
        if verb:
            print(result)
            print(f"N(H2) {10**N_H2:.2E} cm-2")
            print("chisq result", chisq)
            print("obs result", run_12_and_13(result.x, *x_test[1:]))
            print("success", result.success)
        #### Return N(H2), chisq (dof=1), and result.success
        return 10**N_H2, chisq, result.success

    def fit_co32_vary_params(self):
        """
        December 5, 2023
        Use test_fit_params_co32_only repeatedly over a range of either tk or n
        """

        setup = 2
        """
        Proof of concept:
        vary Tk coarsely.
        n = 1e2: Clean single minimum around 28--36 K. N(H2) hits max of 9e23, unlikely
        n = 1e3: Clean single minimum around 28--35. N(H2) high-ish 1e23
        n = 1e4: Clean single minimum around 28--35. N(H2) moderate 1e22
        """
        if setup == 1:
            tk_arr = np.arange(22, 53, 4)
        elif setup == 2:
            tk_arr = np.arange(28, 36, 1)
        elif setup == 3:
            n_arr = np.arange(2, 7, 1)

        result = []

        if setup in (1, 2):
            n_val = 1e4
            stub = f"n_1e{np.log10(n_val):.0f}"
            for tk_val in tk_arr:
                result.append(self.test_fit_params_co32_only(n=n_val, tk=tk_val))
            xaxis = tk_arr
            xlabel = "T_kin (K)"
        elif setup == 3:
            tk_val = 32
            stub = f"tk_{tk_val:.0f}"
            for n_val in 10**n_arr:
                result.append(self.test_fit_params_co32_only(n=n_val, tk=tk_val))
            xaxis = n_arr
            xlabel = "log10 n [cm-3]"
        N_H2_arr, chisq_arr, success_arr = zip(*result)
        print(success_arr)
        print(chisq_arr)
        ax1 = plt.subplot(211)
        plt.plot(xaxis, N_H2_arr)
        plt.title(f"n = {n_val:.1E} cm-3")
        plt.ylabel("N(H2) (cm-2)")
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(xaxis, chisq_arr)
        plt.ylabel("chisq (dof=1)")
        plt.xlabel(xlabel)
        plt.show()
        return
        if setup == 1:
            plt.savefig(f"/home/ramsey/Pictures/2023-12-05/radex_obs2_{stub}_varytk_coarse.png")
        elif setup == 2:
            plt.savefig(f"/home/ramsey/Pictures/2023-12-05/radex_obs2_{stub}_varytk_med.png")
        elif setup == 3:
            plt.savefig(f"/home/ramsey/Pictures/2023-12-05/radex_obs2_{stub}_varyn_coarse.png")




    def test_radex_run_grid(self):
        self.set_params()
        self.params['cdmol'] = [9e16, 1e17, 1.1e17]
        return radex.run_grid(self.params, target_value="T_R (K)")

    def test_chisq_grid(self):
        """
        This will be time consuming to code and I am questioning if it will be
        worthwhile. I'd end up with a 3d chisq space, I remember doing this with
        manticore / mantipython. It was helpful to understand the shape of the
        chisq region, but very time consuming. I think for this, I might just
        want to learn what I learned and try something else.
        """
        ...


if __name__ == "__main__":
    test = TestSpectralRadex(verbose=False)
    # test.list_data_files()
    # test.find_basic_results()
    test.test_fit_params_co32_only(n=1e4, tk=32, verb=True)
    # test.fit_co32_vary_params()
