import functools
import logging
import os

import numpy as np
import h5py

logger = logging.getLogger(__name__)

def get_tomography(
    hf_imsim,
    hf_redshift,
    mdet_step,
):
    # redshift_unsorted_indices         : indices into the redshift catalog index
    # redshift_sorted_indices           : indices that sort the redshift catalog index
    # redshift_sorted_indices_mapping   : indices that map from unsorted to sorted indices
    # redshift_unsorted_indices_mapping : indices that map from sorted to unsorted indices

    # because hdf5 datasets require monotonically increasing indices, we must take
    # the original indices, apply sorting, extract the values, and map from
    # sorted to unsorted indices; e.g.,
    # values[redshift_index[redshift_sorted_indices]][redshift_sorted_indices_mapping]

    imsim_id = hf_imsim["mdet"][mdet_step]["uid"][:]
    redshift_id = hf_redshift["sompz"][mdet_step]["coadd_object_id"][:]

    logger.info(f"Checking if shear object ids are unique")
    _, _counts = np.unique(imsim_id, return_counts=True)
    assert np.max(_counts) == 1

    logger.info(f"Checking if redshift object ids are unique")
    _, _counts = np.unique(redshift_id, return_counts=True)
    assert np.max(_counts) == 1

    logger.info(f"Intersecting shear and redshift object ids")
    _, imsim_index, redshift_index = np.intersect1d(
        imsim_id,
        redshift_id,
        return_indices=True
    )

    logger.info(f"Checking if imsim indices are monotonic")
    assert np.all(np.diff(imsim_index) > 0)

    logger.info(f"Checking if redshift indices are monotonic")
    assert np.all(np.diff(redshift_index) > 0)

    logger.info(f"Getting indices for redshift")
    redshift_unsorted_indices = np.indices(redshift_index.shape).ravel()

    logger.info(f"Getting sorted indices for redshift")
    redshift_sorted_indices = np.argsort(redshift_index)

    logger.info(f"Intersecting redshift sorting indices")
    _, redshift_unsorted_indices_mapping, redshift_sorted_indices_mapping = np.intersect1d(
        redshift_unsorted_indices,
        redshift_sorted_indices,
        return_indices=True
    )

    # these are the bhat values for objects indixed in the imsim and redshift
    # catalogs by imsim_index and redshift_index, respectively
    logger.info(f"Extracting tomographic binning from redshift catalog")
    bhat_sorted = hf_redshift["sompz"][mdet_step]["bhat"][redshift_index[redshift_sorted_indices]]

    logger.info(f"Reindexing tomographic binning")
    bhat = bhat_sorted[redshift_sorted_indices_mapping]

    out = np.full(imsim_id.shape, np.nan)
    out[imsim_index] = bhat

    # bhat = hf_redshift["sompz"][mdet_step]["bhat"][redshift_index]
    # out = np.full(imsim_id.shape, np.nan)
    # out[imsim_index] = bhat

    return out

def get_assignment(
    hf_imsim,
    hf_redshift,
    mdet_step,
):
    # redshift_unsorted_indices         : indices into the redshift catalog index
    # redshift_sorted_indices           : indices that sort the redshift catalog index
    # redshift_sorted_indices_mapping   : indices that map from unsorted to sorted indices
    # redshift_unsorted_indices_mapping : indices that map from sorted to unsorted indices

    # because hdf5 datasets require monotonically increasing indices, we must take
    # the original indices, apply sorting, extract the values, and map from
    # sorted to unsorted indices; e.g.,
    # values[redshift_index[redshift_sorted_indices]][redshift_sorted_indices_mapping]

    imsim_id = hf_imsim["mdet"][mdet_step]["uid"][:]
    redshift_id = hf_redshift["sompz"][mdet_step]["coadd_object_id"][:]

    logger.info(f"Checking if shear object ids are unique")
    _, _counts = np.unique(imsim_id, return_counts=True)
    assert np.max(_counts) == 1

    logger.info(f"Checking if redshift object ids are unique")
    _, _counts = np.unique(redshift_id, return_counts=True)
    assert np.max(_counts) == 1

    logger.info(f"Intersecting shear and redshift object ids")
    _, imsim_index, redshift_index = np.intersect1d(
        imsim_id,
        redshift_id,
        return_indices=True
    )

    logger.info(f"Checking if imsim indices are monotonic")
    assert np.all(np.diff(imsim_index) > 0)

    logger.info(f"Checking if redshift indices are monotonic")
    assert np.all(np.diff(redshift_index) > 0)

    logger.info(f"Getting indices for redshift")
    redshift_unsorted_indices = np.indices(redshift_index.shape).ravel()

    logger.info(f"Getting sorted indices for redshift")
    redshift_sorted_indices = np.argsort(redshift_index)

    logger.info(f"Intersecting redshift sorting indices")
    _, redshift_unsorted_indices_mapping, redshift_sorted_indices_mapping = np.intersect1d(
        redshift_unsorted_indices,
        redshift_sorted_indices,
        return_indices=True
    )

    # these are the bhat values for objects indixed in the imsim and redshift
    # catalogs by imsim_index and redshift_index, respectively
    logger.info(f"Extracting SOM cell assignment from redshift catalog")
    cell_sorted = hf_redshift["sompz"][mdet_step]["cell_wide"][redshift_index[redshift_sorted_indices]]

    logger.info(f"Reindexing cell assignment")
    cell = cell_sorted[redshift_sorted_indices_mapping]

    out = np.full(imsim_id.shape, np.nan)
    out[imsim_index] = cell

    # bhat = hf_redshift["sompz"][mdet_step]["bhat"][redshift_index]
    # out = np.full(imsim_id.shape, np.nan)
    # out[imsim_index] = bhat

    return out


