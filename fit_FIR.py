import os

import mantipython.v3 as mantipython

data_dir = "/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009_reproc350/"
if not os.path.isdir(data_dir):
    # Laptop path
    data_dir = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/"
data_fns = {
    70: "PACS70um-image-remapped-conv-plus000080.fits",
    160: "PACS160um-image-remapped-conv-plus000370.fits",
    250: "SPIRE250um-image-remapped-conv.fits",
    350: "SPIRE350um-image-remapped-conv.fits",
    # 500: "SPIRE500um-image-remapped-conv.fits",
}
err_fns = {
    70: "PACS70um-error-remapped-conv.fits",
    160: "PACS160um-error-remapped-conv.fits",
    250: "SPIRE250um-error-remapped-conv.fits",
    350: "SPIRE350um-error-remapped-conv.fits",
    # 500: "SPIRE500um-error-remapped-conv.fits",
}

# organize filenames
data_dictionary = {}
for k in data_fns:
    data_dictionary[k] = (data_fns[k], err_fns[k])
# select small cutout area
i0, j0 = 726, 466
width_i, width_j = 220, 280
# decide whether or not this is parallel
n_processes = 10
path = "/home/rkarim/Research/Feedback/ancillary_data/herschel/"
write_fn = "RCW49large_3p_secondCal_jac.fits"
mantipython.fit_entire_map(data_dictionary, [70, 160, 250, 350], ('T', 'tau', 'beta'),
    data_directory=data_dir, log_name_func=lambda s: f"{path}log{s}.log",
    n_procs=n_processes, destination_filename=path+write_fn,
    cutout=((i0, j0), (width_i, width_j)), fitting_function='jac',
)

