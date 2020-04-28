import os

import mantipython

data_dir = "/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009_reproc160/"
if not os.path.isdir(data_dir):
    # Laptop path
    data_dir = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/processed/1342255009/"
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
elif "160" in data_dir:
    i0, j0 = 2314, 1035
    width_i, width_j = 1286, 1286
    width_i, width_j = 15, 15 # just the test region
else:
    # 500 grid
    # i0, j0 = 519, 334 # OLD CUTOUT (retired as of March 9 2020)
    # width_i, width_j = 157, 200
    i0, j0 = 543, 337
    width_i, width_j = 294, 294 # NEW CUTOUT (as of March 9 2020, include extended stuff to the W)
    width_i, width_j = 15, 15 # just the test region
# decide whether or not this is parallel
n_processes = 6
path = "/home/rkarim/Research/Feedback/ancillary_data/herschel/"
if not os.path.isdir(path):
    path = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/"
# write_fn = "RCW49large_2p_2BAND_beta2.0.fits"
write_fn = "TEST.fits"
mantipython.fit_entire_map(data_dictionary, [70,160],
    ('T', 'tau'), initial_param_vals={'beta': 2.0},
    param_bounds={'T': (0, None),},
    data_directory=data_dir, log_name_func=lambda s: f"{path}log{s}.log",
    n_procs=n_processes, destination_filename=path+write_fn,
    cutout=((i0, j0), (width_i, width_j)), fitting_function='jac',
)
