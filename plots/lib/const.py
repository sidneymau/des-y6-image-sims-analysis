import glob
import os

import numpy as np


BANDS = ["g", "r", "i", "z"]
TRUTH_BANDS = ["u", "g", "r", "i", "z", "Y", "J",  "H", "Ks"]
DEEPFIELD_BANDS = ["u", "g", "r", "i", "z", "Y", "J", "H", "K"]

SHEAR_STEPS = [
    "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0",
    "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8",
    "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0",
]

MDET_STEPS = [
    "noshear",
    "1p",
    "1m",
    "2p",
    "2m",
]

MDET_CATALOG = "/global/cfs/projectdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V6_UNBLINDED/metadetect_cutsv6_all.h5"

DEEPFIELD_CATALOG = "/global/cfs/cdirs/desbalro/DES_DF_COSMOS.fits"
DES_COSMOS_CATALOG = "/global/cfs/cdirs/desbalro/des_cosmos_data_v2.fits"
SOURCE_CATALOG = "/global/cfs/cdirs/desbalro/input_cosmos_v4.fits"
TRUTH_CATALOG = "/global/cfs/cdirs/desbalro/input_cosmos_v4_montara_simcat_v7_seed42.fits"

TRUTH_DIR = "/global/cfs/cdirs/desbalro/cosmos_simcat/"
TRUTH_CATALOGS = {}
for _truth_file in glob.glob(f"{TRUTH_DIR}/*.fits"):
    _tilename = _truth_file.split("/")[-1].split("_")[3]
    TRUTH_CATALOGS[_tilename] = _truth_file

IMSIM_DIR = "/global/cfs/cdirs/des/y6-image-sims/fiducial/"
IMSIM_CATALOGS = {
    shear_step: os.path.join(
        IMSIM_DIR,
        shear_step,
        "metadetect_cutsv6_all.h5",
    )
    for shear_step in SHEAR_STEPS
}

MATCH_DIR = "/global/cfs/cdirs/des/y6-image-sims/fiducial-matches/"
MATCH_CATALOGS = {
    shear_step: os.path.join(
        MATCH_DIR,
        f"match_{shear_step}.hdf5",
    )
    for shear_step in SHEAR_STEPS
}

TRUTH_MATCH_CATALOG = "/global/cfs/cdirs/des/y6-image-sims/fiducial-matches/truth-match.hdf5"

REDSHIFT_DIR = "/global/cfs/cdirs/des/y6-redshift/imsim_400Tile/fidbin_S005/"
REDSHIFT_CATALOGS = {
    shear_step: os.path.join(
        REDSHIFT_DIR,
        f"{shear_step}_sompz_unblind_fidbin.h5",
    )
    for shear_step in SHEAR_STEPS
}

ZEROPOINT = 30

TOMOGRAPHIC_BINS = [0, 1, 2, 3]

SOM_SHAPE = (32, 32)
CELL_IDS = list(range(1024 + 1))

ZVALS = np.array([
    0.   , 0.035, 0.085, 0.135, 0.185, 0.235, 0.285, 0.335, 0.385,
    0.435, 0.485, 0.535, 0.585, 0.635, 0.685, 0.735, 0.785, 0.835,
    0.885, 0.935, 0.985, 1.035, 1.085, 1.135, 1.185, 1.235, 1.285,
    1.335, 1.385, 1.435, 1.485, 1.535, 1.585, 1.635, 1.685, 1.735,
    1.785, 1.835, 1.885, 1.935, 1.985, 2.035, 2.085, 2.135, 2.185,
    2.235, 2.285, 2.335, 2.385, 2.435, 2.485, 2.535, 2.585, 2.635,
    2.685, 2.735, 2.785, 2.835, 2.885, 2.935, 2.985])

ZEDGES = np.array([0.  , 0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46,
       0.51, 0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86, 0.91, 0.96, 1.01,
       1.06, 1.11, 1.16, 1.21, 1.26, 1.31, 1.36, 1.41, 1.46, 1.51, 1.56,
       1.61, 1.66, 1.71, 1.76, 1.81, 1.86, 1.91, 1.96, 2.01, 2.06, 2.11,
       2.16, 2.21, 2.26, 2.31, 2.36, 2.41, 2.46, 2.51, 2.56, 2.61, 2.66,
       2.71, 2.76, 2.81, 2.86, 2.91, 2.96, 3.01])
