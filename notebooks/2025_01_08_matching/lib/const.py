import glob
import os

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

ZEROPOINT = 30