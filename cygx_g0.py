import numpy as np
import pandas as pd
import os
import glob

import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 6}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy import units as u

from . import catalog

data_dir = os.path.abspath(os.path.join(catalog.utils.feedback_path, "cygx_data/catalogs"))
berlanas_table_filenames = glob.glob(os.path.join(data_dir, "berlanas*"))
berlanas_table_filenames = {
    'newstars': [x for x in berlanas_table_filenames if 'A1' in x].pop(),
    'oldstars': [x for x in berlanas_table_filenames if 'A2' in x].pop(),
}
mytable = os.path.join(data_dir, "cygx_all.pkl")


# Helper function for "Sample" column in both tables
def find_cygob2(s):
    return s.strip() == 'Cygnus OB2'


def open_and_reduce_newstars():
    """
    Open Berlanas 2018 Table A1, new stars from that work
    Return a DataFrame with (at least) columns:
        coords: SkyCoord
        Teff: float, K
        LogL: float, log L/Lsun
    """
    # Load everything as string first, since there's junk in there
    newstars = pd.read_csv(berlanas_table_filenames['newstars'], sep=';', comment='#', dtype=str)
    # Drop the unit and hyphen rows
    newstars.drop(index=[0, 1], inplace=True)
    # We won't use the Av column
    newstars.drop(columns=['AV'], inplace=True)
    # Convert these columns from string to float
    for c in ['Teff', 'LogL', '_RA', '_DE']:
        newstars[c] = newstars[c].astype(float)
    # Remove whitespace from SpType
    newstars['SpType'] = newstars['SpType'].apply(lambda x: x.strip())

    # Reduce the table to only the Cyg OB2 entries
    newstars = newstars.loc[newstars['Sample'].apply(find_cygob2)]
    # Drop that column, since they're all the same now
    newstars.drop(columns=['Sample'], inplace=True)

    # Make SkyCoords from the RA and DEC
    def make_skycoords(row):
        return SkyCoord(row['_RA'], row['_DE'], unit=u.deg, frame='fk5')
    newstars['coords'] = newstars.apply(make_skycoords, axis=1)
    # Drop the RA, DEC columns
    newstars.drop(columns=['_RA', '_DE'], inplace=True)

    # Set the index column to Name, making sure they're unique (should be)
    newstars.set_index('Name', inplace=True, verify_integrity=True)

    # Make a provenance column to keep track of the old and new stars
    newstars['origin'] = 'A1-new'

    return newstars


def open_and_reduce_oldstars():
    """
    Open Berlanas 2018 Table A2, known stars from Comeron & Pasquali 2012
    Only stars from "Cygnus OB2" are included
    Return a DataFrame with (at least) columns:
        coords: SkyCoord
        Teff: float, K
        LogL: float, log L/Lsun
    and Name (usually 2MASS) as the index
    """
    # Load everything as string first, since there's junk in there
    oldstars = pd.read_csv(berlanas_table_filenames['oldstars'], sep=';', comment='#', dtype=str)
    # Drop the unit and hyphen rows
    oldstars.drop(index=[0, 1], inplace=True)

    # Convert these columns from string to float
    for c in ['Teff', 'LogL']:
        oldstars[c] = oldstars[c].astype(float)
    # Remove whitespace from SpType
    oldstars['SpType'] = oldstars['SpType'].apply(lambda x: x.strip())

    # Reduce the table to only the Cyg OB2 entries
    oldstars = oldstars.loc[oldstars['Sample'].apply(find_cygob2)]
    # Drop that column, since they're all the same now
    oldstars.drop(columns=['Sample'], inplace=True)

    # Make SkyCoords from the RA and DEC
    def make_skycoords(row):
        return SkyCoord(row['RAJ2000'], row['DEJ2000'], unit=(u.hourangle, u.deg), frame='fk5')
    oldstars['coords'] = oldstars.apply(make_skycoords, axis=1)
    # Drop the RA, DEC columns
    oldstars.drop(columns=['RAJ2000', 'DEJ2000'], inplace=True)

    # Set the index column to Name, making sure they're unique (should be)
    oldstars.set_index('Name', inplace=True, verify_integrity=True)

    # Make a provenance column to keep track of the old and new stars
    oldstars['origin'] = 'A2-old'

    return oldstars


def combine_tables():
    """
    Combine tables A1 and A2 from Berlanas 2018 and ensure they don't contain
    duplicates. They don't, I checked, nor should they.
    Save the result as a pickle file. That's the "mytable" variable up top.

    There are ~20 from the "new" table and ~100 from the "old" table
    """
    raise RuntimeError("I ran this already on September 24, 2020")
    newstars = open_and_reduce_newstars()
    oldstars = open_and_reduce_oldstars()

    assert newstars.index.intersection(oldstars.index).empty
    allstars = pd.concat([newstars, oldstars])

    allstars.to_pickle(os.path.join(data_dir, "cygx_all_NEW.pkl"))


df = pd.read_pickle(mytable)
coords = SkyCoord(df.coords.values)
T = df.Teff.values
plt.scatter(coords.ra.deg, coords.dec.deg, c=np.sqrt(T/1000))
plt.show()
