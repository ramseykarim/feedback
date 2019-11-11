import numpy as np
import astropy.units as u
from astroquery.mast import Catalogs

"""
Query source catalog
"""
catalog_data = Catalogs.query_object("10:24:17.509 -57:45:29.28", radius="0.0122628 deg", catalog="HSC")





"""
Query atomic lines
"""
# from astroquery.atomic import AtomicLineList

# candidates = "S II\nCl II\nP II\nC II\nO I\nO II\nH I\nH II\nHe I\nN I"
# weirdline1 = [7316.47*u.Angstrom, 7323.97*u.Angstrom]  # Likely SII 7319.15 A
# weirdline2 = [7326.47*u.Angstrom, 7333.97*u.Angstrom]  # Maybe SII? Maybe OII or PII?
# weirdline3 = [8577.72*u.Angstrom, 8580.22*u.Angstrom]  # ClII 8579.74 seems most likely
# line_tbl = AtomicLineList.query_object(wavelength_range=weirdline1, wavelength_type='Vacuum',
#     element_spectrum=candidates,
#     depl_factor=1)

