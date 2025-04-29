import numpy as np
import pytest
import scipy.stats

from des_y6_imsim_analysis import bump, interpolant, interpolant_variants
from des_y6_imsim_analysis.stats import (
    compute_model_chi2_info,
    measure_map,
    run_mcmc,
)
from des_y6_imsim_analysis.utils import (
    GMODEL_COSMOS_Z,
    ZBIN_HIGH,
    ZBIN_LOW,
    ModelData,
)


def _make_fake_bump_data(*, num_pts, w, seed, model_kind, add_noise=False):
    """Make fake data with an F(z) model."""

    rng = np.random.default_rng(seed)
    true_params = {
        "w": w,
    }
    for i in range(4):
        true_params[f"g_b{i}"] = 0.0
        for j in range(num_pts):
            true_params[f"a{j}_b{i}"] = rng.uniform(0, 1)

    z = np.asarray(GMODEL_COSMOS_Z.copy()[1:-1], dtype=np.float64)

    mns = (0.5, 0.7, 0.9, 1.1)
    nzs = []
    for mn in mns:
        nzs.append(scipy.stats.lognorm.pdf(z / mn, 0.3))
        nzs[-1] = nzs[-1] / np.sum(nzs[-1])
    nzs = np.array(nzs, dtype=np.float64)

    zbins = np.array(
        [[ZBIN_LOW[0], ZBIN_HIGH[-1]]]
        + [[ZBIN_LOW[i], ZBIN_HIGH[i]] for i in range(ZBIN_LOW.shape[0])],
        dtype=np.float64,
    )

    pts = bump.make_bump_pts(num_pts=num_pts, zbins=zbins)

    mn_pars = []
    for sind in range(-1, 10):
        for bind in range(4):
            mn_pars.append((sind, bind))
    mn_pars = np.array(mn_pars, dtype=np.int32)

    mn = bump.model_mean(
        pts=pts,
        n_pts=num_pts,
        z=z,
        nz=nzs,
        mn_pars=mn_pars,
        zbins=zbins,
        params=true_params,
        model_kind=model_kind,
    )
    cov = np.diagflat((mn * 0.01) ** 2)
    if add_noise:
        mn += rng.normal(size=mn.shape) * np.sqrt(np.diag(cov))

    model_data = bump.make_model_data(
        z=z,
        nzs=nzs,
        mn=mn,
        cov=cov,
        mn_pars=mn_pars,
        zbins=zbins,
        fixed_param_values={
            k: v for k, v in true_params.items() if k == "w" or k.startswith("g_b")
        },
        num_pts=num_pts,
        model_kind=model_kind,
    )

    data = ModelData(
        z=z,
        nzs=nzs,
        mn_pars=mn_pars,
        zbins=zbins,
        mn=mn,
        cov=cov,
    )

    return {
        "model_data": model_data,
        "true_params": true_params,
        "data": data,
    }


@pytest.mark.parametrize("model_kind", ["f", "g"])
def test_integration_bump_map(model_kind):
    fake_data = _make_fake_bump_data(num_pts=9, w=0.1, seed=42, model_kind=model_kind)

    model_data = fake_data["model_data"]
    true_params = fake_data["true_params"]

    map_params = measure_map(
        model_module=bump,
        model_data=model_data,
        seed=42,
    )

    for k in set(map_params.keys()) | set(true_params.keys()):
        assert k in map_params
        assert k in true_params
        np.testing.assert_allclose(
            map_params[k],
            true_params[k],
            rtol=0,
            atol=5e-5,
            err_msg=k,
        )


@pytest.mark.parametrize("model_kind", ["f", "g"])
@pytest.mark.parametrize("add_noise", [True, False])
def test_integration_bump_map_chi2(add_noise, model_kind):
    fake_data = _make_fake_bump_data(
        num_pts=9, w=0.1, seed=42, add_noise=add_noise, model_kind=model_kind
    )

    model_data = fake_data["model_data"]
    data = fake_data["data"]

    if add_noise:
        map_params = measure_map(
            model_module=bump,
            model_data=model_data,
            seed=42,
        )
    else:
        map_params = fake_data["true_params"]

    chi2_info = compute_model_chi2_info(
        model_module=bump,
        model_data=model_data,
        data=data,
        params=map_params,
    )
    print(chi2_info)

    if not add_noise:
        assert np.allclose(chi2_info["chi2"], 0.0), chi2_info
    else:
        assert np.abs(chi2_info["p_value"] - 0.5) < 0.45, chi2_info


@pytest.mark.parametrize("model_kind", ["f", "g"])
def test_integration_bump_mcmc(capsys, model_kind):
    with capsys.disabled():
        fake_data = _make_fake_bump_data(
            num_pts=9, w=0.1, seed=42, model_kind=model_kind
        )

        model_data = fake_data["model_data"]
        true_params = fake_data["true_params"]

        map_params = measure_map(
            model_module=bump,
            model_data=model_data,
            seed=42,
        )

        mcmc = run_mcmc(
            model_module=bump, model_data=model_data, init_params=map_params, seed=42
        )

    mcmc.print_summary(exclude_deterministic=True)

    for line in capsys.readouterr().out.splitlines():
        if not line.strip():
            continue
        if "mean" in line:
            continue
        if "divergences" in line:
            continue

        parts = line.strip().split()
        rhat = float(parts[-1])
        assert np.abs(rhat - 1.0) < 0.01, line

    samples = mcmc.get_samples()
    for k in set(samples.keys()) | set(true_params.keys()):
        assert k in samples
        assert k in true_params
        np.testing.assert_allclose(
            np.mean(samples[k]),
            true_params[k],
            rtol=0,
            atol=5e-3,
            err_msg=k,
        )


def _make_fake_interpolant_data(*, num_pts, model_kind, seed, add_noise=False):
    """Make fake data with an F(z) model."""

    rng = np.random.default_rng(seed)
    true_params = {}
    for i in range(4):
        for j in range(num_pts):
            true_params[f"a{j}_b{i}"] = rng.uniform(0, 1)

    z = np.asarray(GMODEL_COSMOS_Z.copy()[1:-1], dtype=np.float64)

    mns = (0.5, 0.7, 0.9, 1.1)
    nzs = []
    for mn in mns:
        nzs.append(scipy.stats.lognorm.pdf(z / mn, 0.3))
        nzs[-1] = nzs[-1] / np.sum(nzs[-1])
    nzs = np.array(nzs, dtype=np.float64)

    zbins = np.array(
        [[ZBIN_LOW[0], ZBIN_HIGH[-1]]]
        + [[ZBIN_LOW[i], ZBIN_HIGH[i]] for i in range(ZBIN_LOW.shape[0])],
        dtype=np.float64,
    )

    mn_pars = []
    for sind in range(-1, 10):
        for bind in range(4):
            mn_pars.append((sind, bind))
    mn_pars = np.array(mn_pars, dtype=np.int32)

    mn = interpolant.model_mean(
        n_pts=num_pts,
        z=z,
        nz=nzs,
        mn_pars=mn_pars,
        zbins=zbins,
        params=true_params,
        model_kind=model_kind,
    )
    cov = np.diagflat((mn * 0.01) ** 2)
    if add_noise:
        mn += rng.normal(size=mn.shape) * np.sqrt(np.diag(cov))

    model_data = interpolant.make_model_data(
        z=z,
        nzs=nzs,
        mn=mn,
        cov=cov,
        mn_pars=mn_pars,
        zbins=zbins,
        fixed_param_values={},
        num_pts=num_pts,
        model_kind=model_kind,
    )

    data = ModelData(
        z=z,
        nzs=nzs,
        mn_pars=mn_pars,
        zbins=zbins,
        mn=mn,
        cov=cov,
    )

    return {
        "model_data": model_data,
        "true_params": true_params,
        "data": data,
    }


@pytest.mark.parametrize("model_kind", ["f", "g"])
def test_integration_interpolant_map(model_kind):
    fake_data = _make_fake_interpolant_data(num_pts=9, model_kind=model_kind, seed=42)

    model_data = fake_data["model_data"]
    true_params = fake_data["true_params"]

    map_params = measure_map(
        model_module=interpolant,
        model_data=model_data,
        seed=42,
    )

    for k in set(map_params.keys()) | set(true_params.keys()):
        assert k in map_params
        assert k in true_params
        np.testing.assert_allclose(
            map_params[k],
            true_params[k],
            rtol=0,
            atol=5e-5,
        )


@pytest.mark.parametrize("model_kind", ["f", "g"])
@pytest.mark.parametrize("add_noise", [True, False])
def test_integration_interpolant_map_chi2(add_noise, model_kind):
    fake_data = _make_fake_interpolant_data(
        num_pts=9, model_kind=model_kind, seed=42, add_noise=add_noise
    )

    model_data = fake_data["model_data"]
    data = fake_data["data"]

    if add_noise:
        map_params = measure_map(
            model_module=interpolant,
            model_data=model_data,
            seed=654,
            num_steps=100_000,
        )
    else:
        map_params = fake_data["true_params"]

    chi2_info = compute_model_chi2_info(
        model_module=interpolant,
        model_data=model_data,
        data=data,
        params=map_params,
    )
    if not add_noise:
        assert np.allclose(chi2_info["chi2"], 0.0), chi2_info
    else:
        assert np.abs(chi2_info["p_value"] - 0.5) < 0.45, chi2_info


@pytest.mark.parametrize("model_kind", ["f", "g"])
def test_integration_interpolant_mcmc(capsys, model_kind):
    with capsys.disabled():
        fake_data = _make_fake_interpolant_data(
            num_pts=9, model_kind=model_kind, seed=42
        )

        model_data = fake_data["model_data"]
        true_params = fake_data["true_params"]

        map_params = measure_map(
            model_module=interpolant,
            model_data=model_data,
            seed=42,
        )

        mcmc = run_mcmc(
            model_module=interpolant,
            model_data=model_data,
            init_params=map_params,
            seed=42,
        )

    mcmc.print_summary(exclude_deterministic=True)

    for line in capsys.readouterr().out.splitlines():
        if not line.strip():
            continue
        if "mean" in line:
            continue
        if "divergences" in line:
            continue

        parts = line.strip().split()
        rhat = float(parts[-1])
        assert np.abs(rhat - 1.0) < 0.01, line

    samples = mcmc.get_samples()
    for k in set(samples.keys()) | set(true_params.keys()):
        assert k in samples
        assert k in true_params
        np.testing.assert_allclose(
            np.mean(samples[k]),
            true_params[k],
            rtol=0,
            atol=5e-3,
        )


def _make_fake_interpolant_variants_data(*, num_pts, model_kind, seed, add_noise=False):
    """Make fake data with an F(z) model."""

    assert model_kind == "fgcconv"

    rng = np.random.default_rng(seed)
    true_params = {}
    for i in range(4):
        for j in range(num_pts):
            true_params[f"f{j}_b{i}"] = rng.uniform(0, 1)
    true_params["grl"] = rng.uniform(0, 1) * 0.1
    true_params["grh"] = rng.uniform(0, 1) * 0.1

    z = np.asarray(GMODEL_COSMOS_Z.copy()[1:-1], dtype=np.float64)

    mns = (0.5, 0.7, 0.9, 1.1)
    nzs = []
    for mn in mns:
        nzs.append(scipy.stats.lognorm.pdf(z / mn, 0.3))
        nzs[-1] = nzs[-1] / np.sum(nzs[-1])
    nzs = np.array(nzs, dtype=np.float64)

    zbins = np.array(
        [[ZBIN_LOW[0], ZBIN_HIGH[-1]]]
        + [[ZBIN_LOW[i], ZBIN_HIGH[i]] for i in range(ZBIN_LOW.shape[0])],
        dtype=np.float64,
    )

    mn_pars = []
    for sind in range(-1, 10):
        for bind in range(4):
            mn_pars.append((sind, bind))
    mn_pars = np.array(mn_pars, dtype=np.int32)

    mn = interpolant_variants.model_mean(
        n_pts=num_pts,
        z=z,
        nz=nzs,
        mn_pars=mn_pars,
        zbins=zbins,
        params=true_params,
        model_kind=model_kind,
    )
    efac = 1e-2
    cov = np.diagflat((mn * efac) ** 2)
    if add_noise:
        mn += rng.normal(size=mn.shape) * np.sqrt(np.diag(cov))

    model_data = interpolant_variants.make_model_data(
        z=z,
        nzs=nzs,
        mn=mn,
        cov=cov,
        mn_pars=mn_pars,
        zbins=zbins,
        fixed_param_values={},
        num_pts=num_pts,
        model_kind=model_kind,
    )

    data = ModelData(
        z=z,
        nzs=nzs,
        mn_pars=mn_pars,
        zbins=zbins,
        mn=mn,
        cov=cov,
    )

    return {
        "model_data": model_data,
        "true_params": true_params,
        "data": data,
    }


@pytest.mark.parametrize("model_kind", ["fgcconv"])
def test_integration_interpolant_variants_map(model_kind):
    fake_data = _make_fake_interpolant_variants_data(
        num_pts=9, model_kind=model_kind, seed=42
    )

    model_data = fake_data["model_data"]
    true_params = fake_data["true_params"]

    map_params = measure_map(
        model_module=interpolant_variants,
        model_data=model_data,
        seed=42,
        learning_rate=1e-2,
        num_steps=50_000,
        progress_bar=True,
    )

    for k in set(map_params.keys()) | set(true_params.keys()):
        assert k in map_params
        assert k in true_params
        np.testing.assert_allclose(
            map_params[k],
            true_params[k],
            rtol=1e-2,
            atol=1e-2,
            err_msg=k,
        )


@pytest.mark.parametrize("model_kind", ["fgcconv"])
@pytest.mark.parametrize("add_noise", [True, False])
def test_integration_interpolant_variants_map_chi2(add_noise, model_kind):
    fake_data = _make_fake_interpolant_variants_data(
        num_pts=9, model_kind=model_kind, seed=42, add_noise=add_noise
    )

    model_data = fake_data["model_data"]
    data = fake_data["data"]

    if add_noise:
        map_params = measure_map(
            model_module=interpolant_variants,
            model_data=model_data,
            seed=654,
            learning_rate=1e-2,
            num_steps=100_000,
            progress_bar=True,
        )
    else:
        map_params = fake_data["true_params"]

    chi2_info = compute_model_chi2_info(
        model_module=interpolant_variants,
        model_data=model_data,
        data=data,
        params=map_params,
    )
    if not add_noise:
        assert np.allclose(chi2_info["chi2"], 0.0), chi2_info
    else:
        assert np.abs(chi2_info["p_value"] - 0.5) < 0.45, chi2_info


@pytest.mark.parametrize("model_kind", ["fgcconv"])
def test_integration_interpolant_variants_mcmc(capsys, model_kind):
    with capsys.disabled():
        fake_data = _make_fake_interpolant_variants_data(
            num_pts=9, model_kind=model_kind, seed=42
        )

        model_data = fake_data["model_data"]
        true_params = fake_data["true_params"]

        map_params = measure_map(
            model_module=interpolant_variants,
            model_data=model_data,
            seed=42,
            learning_rate=1e-2,
            num_steps=100_000,
            progress_bar=True,
        )

        mcmc = run_mcmc(
            model_module=interpolant_variants,
            model_data=model_data,
            init_params=map_params,
            seed=42,
            progress_bar=True,
            dense_mass=True,
            num_samples=2000,
            num_chains=2,
        )

    mcmc.print_summary(exclude_deterministic=True)

    for line in capsys.readouterr().out.splitlines():
        if not line.strip():
            continue
        if "mean" in line:
            continue
        if "divergences" in line:
            continue

        parts = line.strip().split()
        rhat = float(parts[-1])
        assert np.abs(rhat - 1.0) < 0.01, line

    samples = mcmc.get_samples()
    for k in set(samples.keys()) | set(true_params.keys()):
        assert k in samples
        assert k in true_params
        np.testing.assert_allclose(
            np.mean(samples[k]),
            true_params[k],
            rtol=0.5,
            atol=0.10,
            err_msg=k,
        )
