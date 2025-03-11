import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from jax.nn import sigmoid  # noqa: E402

from des_y6_nz_modeling import (  # noqa: E402
    sompz_integral,
    GMODEL_COSMOS_NZ,
)


def _bump(z, a, b, w):
    # always use jax.nn.sigmoid here to ensure stable autodiff
    return sigmoid((z - a) / w) * (1 - sigmoid((z - b) / w))


def model_parts_smooth(
    *, params, pts, z, nz=None, mn_pars=None, zbins=None, mn=None, cov=None, fixed_param_values=None,
):
    gtemp = GMODEL_COSMOS_NZ[:z.shape[0]]
    gtemp = gtemp / gtemp.sum()

    fixed_param_values = fixed_param_values or {}
    for k, v in fixed_param_values.items():
        params[k] = numpyro.deterministic(k, v)
    model_parts = {}
    for i in range(4):
        model_parts[i] = {}
        fvals = jnp.zeros_like(z)
        for j in range(pts.shape[1]):
            fvals += params[f"a{j}_b{i}"] * _bump(z, pts[i, j, 0], pts[i, j, 1], params["w"])
        model_parts[i]["F"] = fvals

        g = params.get(f"g_b{i}", 0.0)
        model_parts[i]["G"] = g * gtemp

    return model_parts


def model_mean_smooth(*, pts, z, nz, mn_pars, zbins, params, mn=None, cov=None, fixed_param_values=None):
    model_parts = model_parts_smooth(
        pts=pts,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
        fixed_param_values=fixed_param_values,
    )
    ngammas = []
    for i in range(4):
        ngamma = (1.0 + model_parts[i]["F"]) * nz[i] + model_parts[i]["G"]
        ngammas.append(ngamma)

    return jnp.stack(ngammas)


def model_mean(*, pts, z, nz, mn_pars, zbins, params, mn=None, cov=None, fixed_param_values=None):
    ngammas = model_mean_smooth(
        pts=pts,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
        fixed_param_values=fixed_param_values,
    )

    def _scan_func(mn_pars, ind):
        si, bi = mn_pars[ind]
        zlow, zhigh = zbins[si + 1]
        val = sompz_integral(ngammas[bi], zlow, zhigh)
        return mn_pars, val

    inds = jnp.arange(len(mn_pars))
    _, model = jax.lax.scan(_scan_func, mn_pars, inds)
    return model


def model_mean_smooth_tomobin(
    *, pts, z, nz, mn_pars, zbins, params, tbind, mn=None, cov=None, fixed_param_values=None
):
    model_mn = model_mean_smooth(
        pts=pts, z=z, nz=nz, mn_pars=mn_pars, zbins=zbins, params=params, fixed_param_values=fixed_param_values,
    )
    return np.asarray(model_mn)[tbind]


def model(pts=None, z=None, nz=None, mn=None, cov=None, mn_pars=None, zbins=None, fixed_param_values=None):
    assert pts is not None
    assert nz is not None
    assert mn is not None
    assert cov is not None
    assert mn_pars is not None
    assert zbins is not None
    assert z is not None

    fixed_param_values = fixed_param_values or {}

    params = {}
    if "w" not in fixed_param_values:
        params["w"] = numpyro.sample("w", dist.LogNormal(np.log(0.1), 0.1))
    for i in range(4):
        if f"g_b{i}" not in fixed_param_values:
            params[f"g_b{i}"] = numpyro.sample(f"g_b{i}", dist.Normal(0.0, 1.0))
        for j in range(pts.shape[1]):
            params[f"a{j}_b{i}"] = numpyro.sample(f"a{j}_b{i}", dist.Uniform(-10, 10))

    model_mn = model_mean(
        pts=pts,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
        fixed_param_values=fixed_param_values,
    )
    numpyro.sample(
        "model", dist.MultivariateNormal(loc=model_mn, covariance_matrix=cov), obs=mn
    )


def make_model_data(*, z, nzs, mn, cov, mn_pars, zbins, fixed_param_values=None, num_bins=-1):
    """Create the dict of model data.

    Parameters
    ----------
    z : array
        The redshift values.
    nzs : dict mapping bin index to n(z).
        The input n(z) data.
    mn : array
        The measured N_gamma_alpha values.
    cov : array
        The covariance matrix for the measured N_gamma_alpha values.
    mn_pars : array
        The mapping of N_gamma_alpha values to (shear bin, tomographic bin)
        indices.
    zbins : array
        The shear bin edges.
    fixed_param_values : dict, optional
        The values of fixed parameters. Default is None.
    num_bins : int, optional
        The number of bins to use. If not positive, then the bins are set to the sheared
        ranges from the image sims. Otherwise, they are set to uniformly cover the range
        of 0 to 2.7 for num_bins-1 and then a single bin from 2.7 to 6.01. The default is -1.

    Returns
    -------
    data : dict
        The model data. Pass to the functions using `**data`.
    """
    if num_bins <= 0:
        pts = []
        for i in range(4):
            pts.append(zbins[1:, :].copy())
        pts = np.array(pts)
    else:
        pts = []
        for i in range(4):
            zmid = np.linspace(0.0, 2.7, num_bins)[1:-1]
            be = np.concatenate(
                [
                    [0.0],
                    zmid,
                    [2.7],
                    [6.01],
                ]
            )
            assert be.shape[0] == num_bins+1
            _pts = []
            for i in range(num_bins):
                _pts.append(be[i:i+2])
            pts.append(_pts)

        pts = np.array(pts)

    assert pts.shape[0] == 4
    if num_bins > 0:
        assert pts.shape[1] == num_bins
    assert pts.shape[2] == 2

    return dict(
        pts=np.array(pts),
        z=z,
        nz=nzs,
        mn=mn,
        cov=cov,
        mn_pars=jnp.asarray(mn_pars, dtype=np.int32),
        zbins=jnp.asarray(zbins),
        fixed_param_values=fixed_param_values,
    )
