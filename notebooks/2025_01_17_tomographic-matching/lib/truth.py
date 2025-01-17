import logging

import fitsio

from . import const


logger = logging.getLogger(__name__)


def get_truth_ids():
    with fitsio.FITS(const.TRUTH_CATALOG) as fits:
        _ids = fits[1]["des_id"].read()

    return _ids

def get_truth_table():
    with fitsio.FITS(const.TRUTH_CATALOG) as fits:
        _table = fits[1].read()

    return _table

