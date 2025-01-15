import logging

import fitsio
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from . import const, util


logger = logging.getLogger(__name__)

# We augment the wide-field catalog (i.e., detections & measurements) by
# matching to the truth catalog and the deep-field catalog.
# Truth matching is done spatially via a ball tree, with multiple associations
# broken via choosing the object whose chi squared (computed according to
# fluxes) is smallest.
# For the truth matched objects, we can get the appropriate match from the deep
# field by matching on the DES ID between each catalogs.
#

def get_deepfield_ids():
    with fitsio.FITS(const.DEEPFIELD_CATALOG) as fits:
        deepfield_ids = fits[1]["ID_DES"].read()

    return deepfield_ids

def get_deepfield_table():
    with fitsio.FITS(const.DEEPFIELD_CATALOG) as fits:
        _deepfield_table = fits[1].read()

    return _deepfield_table

def get_knn():
    _deepfield_table = get_deepfield_table()
    
    _table = {}
    for band in const.DEEPFIELD_BANDS:

        _mag = _deepfield_table[f"MAG_{band}"]
        _mag_err = _deepfield_table[f"ERR_MAG_{band}"]

        # _flux = np.power(
        #     10,
        #     -(_mag - 30) / 2.5,
        # )
        # _flux_err = np.log(10) / 2.5 * _flux * _mag_err
        _flux, _flux_err = util.mag_to_flux_with_error(_mag, _mag_err)

        _table[f"flux_{band}"] = _flux
        _table[f"flux_err_{band}"] = _flux_err

    _X = np.array(
        [
            _table[f"flux_{band}"]
            for band in const.DEEPFIELD_BANDS
        ]
    ).T

    _y = np.array(
        [
            _table[f"flux_err_{band}"]
            for band in const.DEEPFIELD_BANDS
        ]
    ).T

    knn = KNeighborsRegressor(weights="distance")
    knn.fit(
        _X,
        _y,
    )

    return knn
