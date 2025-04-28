import jax  # noqa: I001

jax.config.update("jax_enable_x64", True)

import numpyro  # noqa: E402, I001

numpyro.set_host_device_count(4)

import hashlib  # noqa: E402, I001
import os  # noqa: E402, I001

import h5py  # noqa: E402, I001
import numpy as np  # noqa: E402, I001
import tqdm  # noqa: E402, I001
import yaml  # noqa: E402, I001

from des_y6_imsim_analysis import interpolant  # noqa: E402, I001
from des_y6_imsim_analysis.stats import (  # noqa: E402, I001
    compute_model_chi2_info,
    measure_map,
    run_mcmc,
)
from des_y6_imsim_analysis.utils import (  # noqa: E402, I001
    compute_eff_nz_from_data,
    read_data,
)

fit_stats = {}
fit_data = {}

model_module = interpolant
models = [
    {
        "num_pts": 10,
        "model_kind": "g",
    },
    {
        "num_pts": 9,
        "model_kind": "f",
    },
]
fnames = [
    "../../data/N_gamma_alpha_v3_statistical.hdf5",
    "../../data/N_gamma_alpha_v3_statistical-occupancy.hdf5",
    "../../data/N_gamma_alpha_v3_statistical-neighbor.hdf5",
    "../../data/N_gamma_alpha_v3_statistical-nz.hdf5",
]
keys = [os.path.basename(fname).replace(".hdf5", "").split("_")[-1] for fname in fnames]

with tqdm.tqdm(ncols=80, total=len(models) * len(keys), desc="fitting models") as pbar:
    for model in models:
        model_kind = model["model_kind"]
        seeds = [
            (
                abs(
                    int(
                        hashlib.sha256(
                            (key + str(model_kind)).encode("utf-8")
                        ).hexdigest(),
                        16,
                    )
                )
                + 1
            )
            % 2**32
            for key in keys
        ]
        fit_data[model_kind] = {}
        fit_stats[model_kind] = {}

        for key, fname, seed in zip(keys, fnames, seeds):
            data = read_data(fname)

            model_data = model_module.make_model_data(
                z=data.z,
                nzs=data.nzs,
                mn=data.mn,
                cov=data.cov,
                mn_pars=data.mn_pars,
                zbins=data.zbins,
                **model,
            )

            map_params = measure_map(
                model_module=model_module,
                model_data=model_data,
                seed=seed,
                progress_bar=False,
                num_steps=50_000,
                learning_rate=1e-3,
            )

            chi2_info = compute_model_chi2_info(
                model_module=model_module,
                model_data=model_data,
                data=data,
                params=map_params,
            )

            mcmc = run_mcmc(
                model_module=model_module,
                model_data=model_data,
                init_params=map_params,
                seed=seed,
                progress_bar=False,
                dense_mass=True,
                num_samples=2500,
            )

            fit_data[model_kind][key] = {
                "map_params": map_params,
                "mcmc": mcmc,
                "model_data": model_data,
                "data": data,
                "seed": seed,
            }

            fit_stats[model_kind][key] = {
                "key": key,
                "model_kind": model_kind,
                "chi2": float(chi2_info["chi2"]),
                "dof": int(chi2_info["dof"]),
                "p_value": float(chi2_info["p_value"]),
                "seed": seed,
            }
            pbar.write(
                yaml.dump(
                    fit_stats[model_kind][key],
                    default_flow_style=False,
                    indent=2,
                )
            )
            fit_stats[model_kind][key]["map_params"] = {
                k: float(v) for k, v in map_params.items()
            }

            pbar.update(1)

with h5py.File("imsim_v3_mcmc_samples.h5", "w") as fp:
    for model in models:
        model_kind = model["model_kind"]
        for key in keys:
            samples = fit_data[model_kind][key]["mcmc"].get_samples()
            for param in samples:
                fp.create_dataset(
                    f"{model_kind}-{key}/{param}",
                    data=np.asarray(samples[param], dtype=np.float64),
                    compression="gzip",
                )


with open("fit_stats.yml", "w") as f:
    yaml.dump(fit_stats, f, default_flow_style=False, indent=2)


desnz = np.load("../../data/Tz_realizations_WZ_bq_pile4_0d05.npy")

corr_nz = {}
with tqdm.tqdm(
    ncols=80,
    total=len(models) * len(keys) * 2,
    desc="adjusting data n(z) and writing model data",
) as pbar:
    for skip_nn in [True, False]:
        for model in models:
            model_kind = model["model_kind"]
            corr_nz[model_kind] = {}

            for key in keys:
                corr_nz[model_kind][key] = {}

                mvals, dzvals, finalnzs = compute_eff_nz_from_data(
                    model_module=model_module,
                    mcmc_samples=fit_data[model_kind][key]["mcmc"].get_samples(),
                    model_data=fit_data[model_kind][key]["model_data"],
                    input_nz=desnz,
                    rng=np.random.default_rng(fit_data[model_kind][key]["seed"]),
                    shift_negative=skip_nn,
                )
                corr_nz[model_kind][key]["mvals"] = mvals
                corr_nz[model_kind][key]["dzvals"] = dzvals
                corr_nz[model_kind][key]["finalnzs"] = finalnzs

                m_mn = np.mean(mvals, axis=0)
                m_sd = np.std(mvals, axis=0)

                dz_mn = np.mean(dzvals, axis=0)
                dz_sd = np.std(dzvals, axis=0)
                pbar.write(f"""\
model-kind|key|shift_nonneg: {model_kind}|{key}|{skip_nn}
|--------------------------------------------|
| bin | m [10^-3, 3sigma] | dz [1sigma]      |
|--------------------------------------------|
| 0   | {m_mn[0] / 1e-3:+5.1f} +/- {3 * m_sd[0] / 1e-3:-5.1f}   | {dz_mn[0]:+0.3f} +/- {dz_sd[0]:0.3f} |
| 1   | {m_mn[1] / 1e-3:+5.1f} +/- {3 * m_sd[1] / 1e-3:-5.1f}   | {dz_mn[1]:+0.3f} +/- {dz_sd[1]:0.3f} |
| 2   | {m_mn[2] / 1e-3:+5.1f} +/- {3 * m_sd[2] / 1e-3:-5.1f}   | {dz_mn[2]:+0.3f} +/- {dz_sd[2]:0.3f} |
| 3   | {m_mn[3] / 1e-3:+5.1f} +/- {3 * m_sd[3] / 1e-3:-5.1f}   | {dz_mn[3]:+0.3f} +/- {dz_sd[3]:0.3f} |
|--------------------------------------------|
""")

                pbar.update(1)

        if skip_nn:
            dfname = "des_y6_nz_SOMPZ_WZ_imsim_v3_nonneg.h5"
        else:
            dfname = "des_y6_nz_SOMPZ_WZ_imsim_v3.h5"

        with h5py.File(dfname, "w") as fp:
            for model in models:
                model_kind = model["model_kind"]
                for key in keys:
                    fp.create_dataset(
                        f"{model_kind}-{key}/m",
                        data=corr_nz[model_kind][key]["mvals"],
                        compression="gzip",
                    )
                    fp.create_dataset(
                        f"{model_kind}-{key}/dz",
                        data=corr_nz[model_kind][key]["dzvals"],
                        compression="gzip",
                    )
                    fp.create_dataset(
                        f"{model_kind}-{key}/nz",
                        data=corr_nz[model_kind][key]["finalnzs"],
                        compression="gzip",
                    )
