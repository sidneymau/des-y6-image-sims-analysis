import functools
from pathlib import Path

import tqdm
import numpy as np
import h5py

import weights


SHEAR_STEPS = ["noshear", "1p", "1m", "2p", "2m"]


def concatenate_catalogs(data):
    _dp_fiducial, _dm_fiducial, _dp_median_color, _dm_median_color = np.stack(data, axis=1)

    dp_fiducial = {
        shear_step: np.hstack(
            [_d[shear_step] for _d in _dp_fiducial],
        )
        for shear_step in SHEAR_STEPS
    }
    dm_fiducial = {
        shear_step: np.hstack(
            [_d[shear_step] for _d in _dm_fiducial],
        )
        for shear_step in SHEAR_STEPS
    }

    dp_median_color = {
        shear_step: np.hstack(
            [_d[shear_step] for _d in _dp_median_color],
        )
        for shear_step in SHEAR_STEPS
    }
    dm_median_color = {
        shear_step: np.hstack(
            [_d[shear_step] for _d in _dm_median_color],
        )
        for shear_step in SHEAR_STEPS
    }

    return dp_fiducial, dm_fiducial, dp_median_color, dm_median_color


def process_file(*, dset, tile):
    mdet = dset["mdet"]

    res = {}
    for shear_step in SHEAR_STEPS:
        mdet_cat = mdet[shear_step]
        in_tile = mdet_cat["tilename"][:] == tile
        _w = weights.get_shear_weights(mdet_cat, in_tile)
        n = np.sum(_w)
        if n > 0:
            g1 = np.average(mdet_cat["gauss_g_1"][in_tile], weights=_w)
            g2 = np.average(mdet_cat["gauss_g_2"][in_tile], weights=_w)
        else:
            g1 = np.nan
            g2 = np.nan

        res[shear_step] = np.array(
            [(n, g1, g2)],
            dtype=[
                ("n", "f8"),
                ("g1", "f8"),
                ("g2", "f8"),
            ],
        )
    return res


def process_file_pair(
    dset_plus_fiducial,
    dset_minus_fiducial,
    dset_plus_median_color,
    dset_minus_median_color,
    *,
    tile,
):
    dp_fiducial = process_file(dset=dset_plus_fiducial, tile=tile)
    dm_fiducial = process_file(dset=dset_minus_fiducial, tile=tile)
    dp_median_color = process_file(dset=dset_plus_median_color, tile=tile)
    dm_median_color = process_file(dset=dset_minus_median_color, tile=tile)

    return dp_fiducial, dm_fiducial, dp_median_color, dm_median_color


def compute_shear_pair(dp, dm):
    g1_p = np.nansum(dp["noshear"]["g1"] * dp["noshear"]["n"]) / np.nansum(dp["noshear"]["n"])
    g1p_p = np.nansum(dp["1p"]["g1"] * dp["1p"]["n"]) / np.nansum(dp["1p"]["n"])
    g1m_p = np.nansum(dp["1m"]["g1"] * dp["1m"]["n"]) / np.nansum(dp["1m"]["n"])
    R11_p = (g1p_p - g1m_p) / 0.02

    g1_m = np.nansum(dm["noshear"]["g1"] * dm["noshear"]["n"]) / np.nansum(dm["noshear"]["n"])
    g1p_m = np.nansum(dm["1p"]["g1"] * dm["1p"]["n"]) / np.nansum(dm["1p"]["n"])
    g1m_m = np.nansum(dm["1m"]["g1"] * dm["1m"]["n"]) / np.nansum(dm["1m"]["n"])
    R11_m = (g1p_m - g1m_m) / 0.02

    g2_p = np.nansum(dp["noshear"]["g2"] * dp["noshear"]["n"]) / np.nansum(dp["noshear"]["n"])
    g2p_p = np.nansum(dp["2p"]["g2"] * dp["2p"]["n"]) / np.nansum(dp["2p"]["n"])
    g2m_p = np.nansum(dp["2m"]["g2"] * dp["2m"]["n"]) / np.nansum(dp["2m"]["n"])
    R22_p = (g2p_p - g2m_p) / 0.02

    g2_m = np.nansum(dm["noshear"]["g2"] * dm["noshear"]["n"]) / np.nansum(dm["noshear"]["n"])
    g2p_m = np.nansum(dm["2p"]["g2"] * dm["2p"]["n"]) / np.nansum(dm["2p"]["n"])
    g2m_m = np.nansum(dm["2m"]["g2"] * dm["2m"]["n"]) / np.nansum(dm["2m"]["n"])
    R22_m = (g2p_m - g2m_m) / 0.02

    # return (g1_p - g1_m) / (R11_p + R11_m) / 0.02 - 1., (g2_p + g2_m) / (R22_p + R22_m)
    # return (g1_p - g1_m) / (R11_p + R11_m) / 0.02 - 1., (g1_p + g1_m) / (R11_p + R11_m)
    return (
        (g1_p - g1_m) / (R11_p + R11_m) / 0.02 - 1.,  # m1
        (g1_p + g1_m) / (R11_p + R11_m),  # c1
        (g2_p + g2_m) / (R22_p + R22_m),  # c2
    )

def compute_shear_pair_difference(dp_fiducial, dm_fiducial, dp_median_color, dm_median_color):
    m_fiducial, c1_fiducial, c2_fiducial = compute_shear_pair(dp_fiducial, dm_fiducial)
    m_median_color, c1_median_color, c2_median_color = compute_shear_pair(dp_median_color, dm_median_color)

    return (
        m_fiducial - m_median_color,
        c1_fiducial - c1_median_color,
        c2_fiducial - c2_median_color,
    )


def process_pair(
    catalog_p_fiducial,
    catalog_m_fiducial,
    catalog_p_median_color,
    catalog_m_median_color,
    seed=None,
    resample="jackknife",
):

    hf_plus_fiducial = h5py.File(
        catalog_p_fiducial,
        mode="r",
        locking=False,
    )

    hf_minus_fiducial = h5py.File(
        catalog_m_fiducial,
        mode="r",
        locking=False,
    )

    hf_plus_median_color = h5py.File(
        catalog_p_median_color,
        mode="r",
        locking=False,
    )

    hf_minus_median_color = h5py.File(
        catalog_m_median_color,
        mode="r",
        locking=False,
    )

    tilenames_p_fiducial = np.unique(hf_plus_fuducial["mdet"]["noshear"]["tilename"][:])
    tilenames_m_fiducial = np.unique(hf_minus_fiducial["mdet"]["noshear"]["tilename"][:])
    tilenames_p_median_color = np.unique(hf_plus_median_color["mdet"]["noshear"]["tilename"][:])
    tilenames_m_median_color = np.unique(hf_minus_median_color["mdet"]["noshear"]["tilename"][:])

    tilenames = functools.reduce(
        np.intersect1d,
        [tilenames_p_fiducial, tilenames_m_fiducial, tilenames_p_median_color, tilenames_m_median_color],
    )
    ntiles = len(tilenames)

    data = [
        process_file_pair(
            hf_plus_fiducial,
            hf_minus_fiducial,
            hf_plus_median_color,
            hf_minus_median_color,
            tile=tile,
        )
        for tile in tqdm.tqdm(
            tilenames,
            total=ntiles,
            desc="processing",
            ncols=80,
        )
    ]

    if resample == "bootstrap":
        ns = 1000  # number of bootstrap resamples
        rng = np.random.RandomState(seed=seed)

        dp_fiducial, dm_fiducial, dp_median_color, dm_median_color = concatenate_catalogs(data)
        dm_mean, dc1_mean, dc2_mean = compute_shear_pair_difference(dp_fiducial, dm_fiducial, dp_median_color, dm_median_color)

        bootstrap = []
        for i in tqdm.trange(ns, desc="bootstrap", ncols=80):
            rind = rng.choice(data.shape[0], size=data.shape[0], replace=True)
            _bootstrap = data[rind]
            _dp_fiducial, _dm_fiducial, _dp_median_color, _dm_median_color = concatenate_catalogs(_bootstrap)
            bootstrap.append(compute_shear_pair_difference(_dp_fiducial, _dm_fiducial, _dp_median_color, _dm_median_color))

        bootstrap = np.array(bootstrap)
        dm_std, dc_std_1, dc_std_2 = np.std(bootstrap, axis=0)

    elif resample == "jackknife":
        jackknife = []
        for i in tqdm.trange(len(data), desc="jackknife", ncols=80):
            _pre = data[:i]
            _post = data[i + 1:]
            _jackknife = _pre + _post
            _dp_fiducial, _dm_fiducial, _dp_median_color, _dm_median_color = concatenate_catalogs(_jackknife)
            jackknife.append(compute_shear_pair_difference(_dp_fiducial, _dm_fiducial, _dp_median_color, _dm_median_color))

        _n = len(jackknife)
        # m_mean, c_mean_1, c_mean_2 = np.mean(jackknife, axis=0)
        jackknife_mean = np.mean(jackknife, axis=0)
        # jackknife_var = ((_n - 1) / _n) * np.sum(np.square(np.subtract(jackknife, jackknife_mean)), axis=0)
        jackknife_std = np.sqrt(((_n - 1) / _n) * np.sum(np.square(np.subtract(jackknife, jackknife_mean)), axis=0))

        dm_mean, dc1_mean, dc2_mean = jackknife_mean
        dm_std, dc1_std, dc2_std = jackknife_std

    results = np.array(
        [(dm_mean, dm_std, dc1_mean, dc1_std, dc2_mean, dc2_std, ntiles)],
        dtype=[
            ("dm_mean", "f8"),
            ("dm_std", "f8"),
            ("dc1_mean", "f8"),
            ("dc1_std", "f8"),
            ("dc2_mean", "f8"),
            ("dc2_std", "f8"),
            ("ntiles", "i8"),
        ],
    )

    print("\n")
    print(f"fiducial - median_color")
    print(f"| dm mean | 3 * dm std | dc_1 mean | 3 * dc_1 std | dc_2 mean | 3 * dc_2 std | # tiles |")
    print(f"| {dm_mean:0.3e} | {3 * dm_std:0.3e} | {dc1_mean:0.3e} | {3 * dc1_std:0.3e} | {dc2_mean:0.3e} | {3 * dc2_std:0.3e} | {ntiles} |")
    print("\n")

    return results


def main():
    input_dir = Path("/pscratch/sd/s/smau/y6-image-sims-cats")

    kwargs = {"seed": None, "resample": "jackknife"}

    sim_variants = [
        "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0",
        "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0",
    ]

    fiducial_sims = {}
    median_color_sims = {}
    for variant in sim_variants:
        fiducial_catalog_file = input_dir / "fiducial" / variant / "metadetect_cutsv6_all.h5"
        assert fiducial_catalog_file.is_file()
        fiducial_sims[variant] = fiducial_catalog_file

        median_color_catalog_file = input_dir / "median_color" / variant / "metadetect_cutsv6_all.h5"
        assert median_color_catalog_file.is_file()
        median_color_sims[variant] = median_color_catalog_file

    fiducial_pair = (
        fiducial_sims["g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"],
        fiducial_sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
    )

    median_color_pair = (
        median_color_sims["g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"],
        median_color_sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
    )

    process_pair(
        *fiducial_pair,
        *median_color_pair,
    )


if __name__ == "__main__":
    main()
