import os

import mantipython

data_dir = "/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009/"
if not os.path.isdir(data_dir):
    # Laptop path
    data_dir = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/1342255009/"
data_fns = {
    70: "PACS70um-image-remapped-conv-plus000080.fits",
    160: "PACS160um-image-remapped-conv-plus000370.fits",
    250: "SPIRE250um-image-remapped-conv.fits",
    350: "SPIRE350um-image-remapped-conv.fits",
    500: "SPIRE500um-image-remapped-conv.fits",
}
err_fns = {
    70: "PACS70um-error-remapped-conv-plus6.0pct.fits",
    160: "PACS160um-error-remapped-conv-plus8.0pct.fits",
    250: "SPIRE250um-error-remapped-conv-plus5.5pct.fits",
    350: "SPIRE350um-error-remapped-conv-plus5.5pct.fits",
    500: "SPIRE500um-error-remapped-conv-plus5.5pct.fits",
}

# organize filenames
data_dictionary = {}
for k in data_fns:
    data_dictionary[k] = (data_fns[k], err_fns[k])
# select small cutout area

if "350" in data_dir:
    i0, j0 = 726, 466
    width_i, width_j = 220, 280
else:
    # 500 grid
    i0, j0 = 519, 334
    # width_i, width_j = 157, 200
    width_i, width_j = 5, 5
# decide whether or not this is parallel
n_processes = 10
path = "/home/rkarim/Research/Feedback/ancillary_data/herschel/"
if not os.path.isdir(path):
    path = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/"
# write_fn = "RCW49large_3p_not70_500grid.fits"
write_fn = "TEST_4p.fits"
mantipython.fit_entire_map(data_dictionary, [70,160,250,350,500],
    ('T', 'tau', 'T_bg', 'tau_bg'), initial_param_vals={'beta': 2.0},
    param_bounds={'T': (25, None), 'T_bg': (10, 25)},
    data_directory=data_dir, log_name_func=lambda s: f"{path}log{s}.log",
    n_procs=n_processes, destination_filename=path+write_fn,
    cutout=((i0, j0), (width_i, width_j)), fitting_function='standard',
)
