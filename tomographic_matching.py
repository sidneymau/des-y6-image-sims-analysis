import functools
import logging
import glob
import os

import numpy as np
from scipy import stats

import util
import matching
import tomography
import weights


logger = logging.getLogger(__name__)


SHEAR_STEPS = [
    'g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
    'g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0',
]
TOMOGRAPHIC_BINS = [0, 1, 2, 3]
CELL_IDS = list(range(1024 + 1))
MDET_STEPS = ["noshear", "1p", "1m", "2p", "2m"]

BANDS = ["g", "r", "i", "z"]

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


def get_tilenames(hf_imsim_list):
    return functools.reduce(
        np.intersect1d,
        [
            np.unique(hf["mdet"]["noshear"]["tilename"][:].astype(str))
            for hf in hf_imsim_list
        ],
    )

def get_bhat(hf_imsim, hf_redshift):
    return tomography.get_tomography(
        hf_imsim,
        hf_redshift,
        "noshear",
    )


def get_assignments(hf_imsim, hf_redshift):
    return {
        mdet_step: tomography.get_assignment(
            hf_imsim,
            hf_redshift,
            mdet_step,
        )
        for mdet_step in MDET_STEPS
    }

def get_statistical_weights(hf_imsim):
    return {
        mdet_step: weights.get_shear_weights(
            hf_imsim["mdet"][mdet_step],
        )
        for mdet_step in MDET_STEPS
    }

def get_g1(hf_imsim):
    return {
        mdet_step: hf_imsim["mdet"][mdet_step]["gauss_g_1"][:]
        for mdet_step in MDET_STEPS
    }

def get_g2(hf_imsim):
    return {
        mdet_step: hf_imsim["mdet"][mdet_step]["gauss_g_2"][:]
        for mdet_step in MDET_STEPS
    }

def get_response_grid(g1, g2, assignments, statistical_weights):
    mean_g_1 = {}
    mean_g_2 = {}
    for mdet_step in ["1p", "1m"]:
        _sum_weight, _, _ = stats.binned_statistic(
            assignments[mdet_step],
            statistical_weights[mdet_step],
            statistic="sum",
            bins=CELL_IDS,
        )

        _sum_g_1, _, _ = stats.binned_statistic(
            assignments[mdet_step],
            statistical_weights[mdet_step] * g1[mdet_step],
            statistic="sum",
            bins=CELL_IDS,
        )
        mean_g_1[mdet_step] = _sum_g_1 / _sum_weight

        _sum_g_2, _, _ = stats.binned_statistic(
            assignments[mdet_step],
            statistical_weights[mdet_step] * g2[mdet_step],
            statistic="sum",
            bins=CELL_IDS,
        )
        mean_g_2[mdet_step] = _sum_g_2 / _sum_weight

    mean_response = (mean_g_1["1p"] - mean_g_1["1m"]) / (2 * 0.01)
    
    return mean_response


def get_weight_grid(assignments, statistical_weights):

    weight_grid, _, _ = stats.binned_statistic(
        assignments["noshear"],
        statistical_weights["noshear"],
        statistic="sum",
        bins=CELL_IDS,
    )
    
    return weight_grid


def get_redshift_distribution(hf_match, assignments, statistical_weights):

    _nz = []
    
    # extend the last bin and "pileup"
    zedges = np.copy(ZEDGES)
    zedges[-1] = 4.
    
    tilenames = hf_match.keys()

    for i, tilename in enumerate(tilenames):
        print(f"{tilename} ({i + 1} / {len(tilenames)})", end="\r", flush=True)

        match_indices = hf_match[tilename]["noshear"]["indices"][:]
        redshift = hf_match[tilename]["noshear"]["redshift"][:]

        _sum_weight, _, _, _ = stats.binned_statistic_2d(
            assignments["noshear"][match_indices],
            redshift,
            statistical_weights["noshear"][match_indices],
            statistic="sum",
            bins=[CELL_IDS, zedges],
        )
        
        _nz.append(_sum_weight)

    nz = np.nansum(_nz, axis=0) / np.nansum(_nz, axis=(0, -1))[:, np.newaxis] / np.diff(ZEDGES)
    
    return nz, ZEDGES


# N. B. not used
# def reweight_redshift_distribution(nz, response_grid):
#     return np.sum(nz * response_grid[:, np.newaxis], axis=0) / np.sum(response_grid)


def get_tomographic_redshift_distribution(nz, response_grid, weight_grid, assignments, bhat):
    cell_assignments = {}
    for tomographic_bin in TOMOGRAPHIC_BINS:
        cell_assignments[tomographic_bin] = np.unique(
            assignments["noshear"][bhat == tomographic_bin]
        ).astype(int)
    
    assert len(
        functools.reduce(
            np.intersect1d,
            [
                cells
                for cells in cell_assignments.values()
            ],
        )
    ) == 0
    
    nz_tomo = {}
    for tomographic_bin in TOMOGRAPHIC_BINS:
        nz_tomo[tomographic_bin] = np.sum(
            nz[cell_assignments[tomographic_bin]] \
            * response_grid[cell_assignments[tomographic_bin], np.newaxis] \
            * weight_grid[cell_assignments[tomographic_bin], np.newaxis],
            axis=0
        ) / np.sum(
            response_grid[cell_assignments[tomographic_bin]] \
            * weight_grid[cell_assignments[tomographic_bin]]
        )
        # manually force n(0) = 0
        nz_tomo[tomographic_bin][0] = 0
    
    return nz_tomo


def do_nz(hf_imsim, hf_redshift, hf_match):
    bhat = get_bhat(hf_imsim, hf_redshift)
    assignments = get_assignments(hf_imsim, hf_redshift)
    statistical_weights = get_statistical_weights(hf_imsim)
    g1 = get_g1(hf_imsim)
    g2 = get_g2(hf_imsim)
    
    response_grid = get_response_grid(g1, g2, assignments, statistical_weights)
    weight_grid = get_weight_grid(assignments, statistical_weights)
    
    nz, zedges = get_redshift_distribution(hf_match, assignments, statistical_weights)
    nz_tomo = get_tomographic_redshift_distribution(nz, response_grid, weight_grid, assignments, bhat)
    
    return nz_tomo, zedges
    
