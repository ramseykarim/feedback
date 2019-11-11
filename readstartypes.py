# script to read the Sternberg et al spectral types tables
# created: November 11, 2019
__author__ = "Ramsey Karim"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spectypes_directory = "../ancillary_data/catalogs/SpecTypes/"
spectral_classes = ('V', 'III', 'I')

def table_name(spectral_class):
    # Spectral classes V, III, and I are available
    return f"{spectypes_directory}class{spectral_class}.txt"

spectypes_table = f"{spectypes_directory}spectypes.txt"
column_name_table = f"{spectypes_directory}colnames.txt"

with open(column_name_table, 'r') as f:
    colnames = f.readline().split()
    units = f.readline().split()
with open(spectypes_table, 'r') as f:
    spectypes = f.readline().split()

col_units = pd.DataFrame(units, index=colnames, columns=['Units'])
dfs = {spectral_class: pd.read_table(table_name(spectral_class), index_col=0, names=colnames) for spectral_class in spectral_classes}
for spectral_class in dfs:
    dfs[spectral_class]['Teff'] = dfs[spectral_class].apply(lambda row: float(row['Teff'].replace(',', '')), axis=1)

