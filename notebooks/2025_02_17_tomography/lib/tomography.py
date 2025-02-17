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

    logger.info(f"Checking if shear object ids are unique")
    _, _counts = np.unique(hf_imsim["mdet"][mdet_step]["uid"], return_counts=True)
    assert np.max(_counts) == 1

    logger.info(f"Checking if redshift object ids are unique")
    _, _counts = np.unique(hf_redshift["sompz"][mdet_step]["coadd_object_id"], return_counts=True)
    assert np.max(_counts) == 1

    logger.info(f"Intersecting shear and redshift object ids")
    imsim_id = hf_imsim["mdet"][mdet_step]["uid"]
    redshift_id = hf_redshift["sompz"][mdet_step]["coadd_object_id"]
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

    logger.info(f"Checking if shear object ids are unique")
    _, _counts = np.unique(hf_imsim["mdet"][mdet_step]["uid"], return_counts=True)
    assert np.max(_counts) == 1

    logger.info(f"Checking if redshift object ids are unique")
    _, _counts = np.unique(hf_redshift["sompz"][mdet_step]["coadd_object_id"], return_counts=True)
    assert np.max(_counts) == 1

    logger.info(f"Intersecting shear and redshift object ids")
    imsim_id = hf_imsim["mdet"][mdet_step]["uid"]
    redshift_id = hf_redshift["sompz"][mdet_step]["coadd_object_id"]
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


# if __name__ == "__main__":
#     shear_steps = [
#         'g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1',
#         'g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8',
#         # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0',
#     ]

#     imsim_base = "/global/cfs/cdirs/des/y6-image-sims/fiducial-400/"
#     imsim_catalogs = {
#         shear_step: os.path.join(
#             imsim_base,
#             shear_step,
#             "metadetect_cutsv6_all.h5",
#         )
#         for shear_step in shear_steps
#     }

#     redshift_base = "/global/cfs/cdirs/des/y6-redshift/imsim_400Tile/fidbin/"
#     redshift_catalogs = {
#         shear_step: os.path.join(
#             redshift_base,
#             f"{shear_step}_sompz_unblind_fidbin.h5"
#         )
#         for shear_step in shear_steps
#     }

#     for shear_step in shear_steps:
#         hf_imsim = h5py.File(imsim_catalogs[shear_step], mode="r")
#         hf_redshift = h5py.File(redshift_catalogs[shear_step], mode="r")

#         mdet_step = "noshear"
#         tomography = get_tomography(
#             hf_imsim,
#             hf_redshift,
#             mdet_step,
#         )
#         for i, uid in enumerate(hf_redshift["sompz"][mdet_step]["coadd_object_id"][:][::1000]):
#             redshift_index = np.nonzero(hf_redshift["sompz"][mdet_step]["coadd_object_id"][:] == uid)
#             bhat_redshift = hf_redshift["sompz"][mdet_step]["bhat"][redshift_index]

#             imsim_index = np.nonzero(hf_imsim["mdet"][mdet_step]["uid"][:] == uid)
#             bhat_imsim = tomography[imsim_index]

#             assert bhat_imsim == bhat_redshift
#             print(f"{i}: uid {uid} ok")
#         print("total", hf_imsim["mdet"]["noshear"]["uid"].size)
#         for tomographic_bin in [0, 1, 2, 3]:
#             print(tomographic_bin, np.sum(tomography == tomographic_bin))
