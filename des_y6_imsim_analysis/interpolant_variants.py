"""
!!! NOT USED !!!
"""

import jax

jax.config.update("jax_enable_x64", True)

import functools  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from jax.nn import sigmoid  # noqa: E402

from des_y6_imsim_analysis.utils import (  # noqa: E402
    GMODEL_COSMOS_NZ,
    sompz_integral,
)

N_PTS_F = 1


def _lognormal(x, mu, sigma):
    xs = (jnp.log(x) - mu) / sigma
    return jnp.exp(-0.5 * xs * xs) / sigma / x / jnp.sqrt(2.0 * jnp.pi)


@jax.jit
def _bump(z, a, b, w):
    # always use jax.nn.sigmoid here to ensure stable autodiff
    return sigmoid((z - a) / w) * (1 - sigmoid((z - b) / w))


@jax.jit
def _kern(z, zc, ml, mh):
    _s = 0.05
    return ml * _bump(z, -1, zc - 7 * _s, _s) + mh * _bump(z, zc + 7 * _s, 6, _s)


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts", "model_kind"))
def model_parts_smooth(
    *,
    params,
    n_pts,
    model_kind,
    z,
    nz,
    mn_pars=None,
    zbins=None,
    mn=None,
    cov=None,
    extra_kwargs=None,
):
    gtemp = GMODEL_COSMOS_NZ[: z.shape[0]]
    gtemp = gtemp / gtemp.sum()

    model_parts = {}
    for i in range(4):
        model_parts[i] = {}

        if model_kind == "fgln":
            min_lin_f = -0.02
            max_lin_f = 0.02
            xp = make_interpolant_pts(n_pts)
            yp = jnp.array(
                [
                    (params[f"f{j}_b{i}"]) * (max_lin_f - min_lin_f) + min_lin_f
                    for j in range(n_pts)
                ],
                dtype=jnp.float64,
            )
            fvals = jnp.interp(z, xp, yp, left=None, right=None)

            min_gmu = 0.3
            max_gmu = 0.5
            gmu = jnp.log(params[f"gmu_b{i}"] * (max_gmu - min_gmu) + min_gmu)

            min_gnrm = 1e-5
            max_gnrm = 0.1
            gnrm = params[f"gnrm_b{i}"] * (max_gnrm - min_gnrm) + min_gnrm

            min_gsigma = 0.6
            max_gsigma = 1.0
            gsigma = params[f"gsigma_b{i}"] * (max_gsigma - min_gsigma) + min_gsigma
            gvals = gnrm * _lognormal(z, gmu, gsigma)

        elif model_kind == "fg":
            min_lin_f = -0.05
            max_lin_f = 0.05
            xp = make_interpolant_pts(N_PTS_F)
            yp = jnp.array(
                [
                    (params[f"f{j}_b{i}"]) * (max_lin_f - min_lin_f) + min_lin_f
                    for j in range(N_PTS_F)
                ],
                dtype=jnp.float64,
            )
            fvals = jnp.interp(z, xp, yp, left=None, right=None)

            min_lin_g = 0.0
            max_lin_g = 10.0
            xp = jnp.concatenate(
                [jnp.zeros(1), make_interpolant_pts(n_pts), jnp.array([4.0])]
            )
            yp = jnp.array(
                [0.0]
                + [
                    (params[f"g{j}_b{i}"]) * (max_lin_g - min_lin_g) + min_lin_g
                    for j in range(n_pts)
                ]
                + [0.0],
                dtype=jnp.float64,
            )
            gvals = jnp.interp(z, xp, yp, left=0.0, right=0.0)

        elif model_kind == "fgsig":
            min_lin_f = -1
            max_lin_f = 10
            xp = make_interpolant_pts(n_pts)
            yp = jnp.array(
                [
                    (params[f"f{j}_b{i}"]) * (max_lin_f - min_lin_f) + min_lin_f
                    for j in range(n_pts)
                ],
                dtype=jnp.float64,
            )
            fvals = jnp.interp(z, xp, yp, left=None, right=None)

            gr = params[f"gr_b{i}"]

            max_gz = 1.0
            min_gz = 0.1
            gz = params[f"gz_b{i}"] * (max_gz - min_gz) + min_gz

            max_gw = 1.0
            min_gw = 0.1
            gw = params[f"gw_b{i}"] * (max_gw - min_gw) + min_gw

            gvals = gtemp * (gr * (1.0 - sigmoid((z - gz) / gw)))

        elif model_kind == "fgcconv":
            min_lin_f = -1.0
            max_lin_f = 10
            xp = make_interpolant_pts(n_pts)
            yp = jnp.array(
                [
                    (params[f"f{j}_b{i}"]) * (max_lin_f - min_lin_f) + min_lin_f
                    for j in range(n_pts)
                ],
                dtype=jnp.float64,
            )
            fvals = jnp.interp(z, xp, yp, left=None, right=None)

            grl = params["grl"]
            grh = params["grh"]

            gvals = jnp.sum(
                _kern(
                    z.reshape((1, -1)),
                    z.reshape((-1, 1)),
                    grl,
                    grh,
                )
                * gtemp.reshape((1, -1))
                * nz[i].reshape((-1, 1)),
                axis=0,
            )

        model_parts[i]["F"] = fvals
        model_parts[i]["G"] = gvals

    return model_parts


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts", "model_kind"))
def model_mean_smooth(
    *,
    n_pts,
    model_kind,
    z,
    nz,
    mn_pars,
    zbins,
    params,
    mn=None,
    cov=None,
    extra_kwargs=None,
):
    model_parts = model_parts_smooth(
        n_pts=n_pts,
        model_kind=model_kind,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )
    ngammas = []
    for i in range(4):
        ngamma = (1.0 + model_parts[i]["F"]) * nz[i] + model_parts[i]["G"]
        ngammas.append(ngamma)

    return jnp.stack(ngammas)


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts", "model_kind"))
def model_mean(
    *,
    n_pts,
    model_kind,
    z,
    nz,
    mn_pars,
    zbins,
    params,
    mn=None,
    cov=None,
    fixed_param_values=None,
    extra_kwargs=None,
):
    ngammas = model_mean_smooth(
        n_pts=n_pts,
        model_kind=model_kind,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )

    def _scan_func(mn_pars, ind):
        si, bi = mn_pars[ind]
        zlow, zhigh = zbins[si + 1]
        val = sompz_integral(ngammas[bi], zlow, zhigh)
        return mn_pars, val

    inds = jnp.arange(mn_pars.shape[0])
    _, model = jax.lax.scan(_scan_func, mn_pars, inds)
    return model


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts", "model_kind"))
def model_mean_smooth_tomobin(
    *,
    n_pts,
    model_kind,
    z,
    nz,
    mn_pars,
    zbins,
    params,
    tbind,
    mn=None,
    cov=None,
    fixed_param_values=None,
    extra_kwargs=None,
):
    model_mn = model_mean_smooth(
        n_pts=n_pts,
        model_kind=model_kind,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )
    return jnp.asarray(model_mn)[tbind]


def model(
    n_pts=None,
    model_kind=None,
    z=None,
    nz=None,
    mn=None,
    cov=None,
    mn_pars=None,
    zbins=None,
    fixed_param_values=None,
    extra_kwargs=None,
):
    assert n_pts is not None
    assert model_kind is not None
    assert nz is not None
    assert mn is not None
    assert cov is not None
    assert mn_pars is not None
    assert zbins is not None
    assert z is not None
    assert extra_kwargs is not None

    fixed_param_values = fixed_param_values or {}

    params = {}
    for i in range(4):
        if model_kind == "fgln":
            for j in range(n_pts):
                params[f"f{j}_b{i}"] = numpyro.sample(f"f{j}_b{i}", dist.Uniform(0, 1))

            params[f"gnrm_b{i}"] = numpyro.sample(f"gnrm_b{i}", dist.Uniform(0, 1))
            params[f"gmu_b{i}"] = numpyro.sample(f"gmu_b{i}", dist.Uniform(0, 1))
            params[f"gsigma_b{i}"] = numpyro.sample(f"gsigma_b{i}", dist.Uniform(0, 1))

        elif model_kind == "fg":
            for j in range(n_pts):
                if j < N_PTS_F:
                    params[f"f{j}_b{i}"] = numpyro.sample(
                        f"f{j}_b{i}", dist.Uniform(0, 1)
                    )
                params[f"g{j}_b{i}"] = numpyro.sample(f"g{j}_b{i}", dist.Uniform(0, 1))

        elif model_kind == "fgsig":
            for j in range(n_pts):
                params[f"f{j}_b{i}"] = numpyro.sample(f"f{j}_b{i}", dist.Uniform(0, 1))

            params[f"gr_b{i}"] = numpyro.sample(f"gr_b{i}", dist.Uniform(0, 1))
            params[f"gz_b{i}"] = numpyro.sample(f"gz_b{i}", dist.Uniform(0, 1))
            params[f"gw_b{i}"] = numpyro.sample(f"gw_b{i}", dist.Uniform(0, 1))

        elif model_kind == "fgcconv":
            for j in range(n_pts):
                params[f"f{j}_b{i}"] = numpyro.sample(f"f{j}_b{i}", dist.Uniform(0, 1))

            if i == 0:
                params["grh"] = numpyro.sample("grh", dist.Uniform(0, 1))
                params["grl"] = numpyro.sample("grl", dist.Uniform(0, 1))

    for k, v in fixed_param_values.items():
        params[k] = numpyro.deterministic(k, v)

    model_mn = model_mean(
        n_pts=n_pts,
        model_kind=model_kind,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )
    numpyro.sample(
        "model", dist.MultivariateNormal(loc=model_mn, covariance_matrix=cov), obs=mn
    )


def make_interpolant_pts(n_pts):
    """Make the array of linear interpolation points.

    Parameters
    ----------
    n_pts : int
        The number of points to use.

    Returns
    -------
    pts : array, dimension (n_pts)
        The array of linear interpolation points.
    """
    return jnp.concatenate(
        [(jnp.arange(n_pts - 1) + 0.5) * 2.7 / (n_pts - 1), jnp.array([2.7])],
        axis=0,
        dtype=jnp.float64,
    )


def make_model_data(
    *,
    z,
    nzs,
    mn,
    cov,
    mn_pars,
    zbins,
    fixed_param_values=None,
    num_pts=-1,
    model_kind="F",
):
    """Create the dict of model data.

    Parameters
    ----------
    z : array
        The redshift values.
    nzs : array, dimension (4, n_z)
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
    num_pts : int, optional
        The number of points to use.
    model_kind : str, optional
        The model kind. Default is "F" meaning n_eff = (1+F) * n. If "G", then
        n_eff = n + G.

    Returns
    -------
    data : dict
        The model data. Pass to the functions using `**data`.
    """
    model_kind = model_kind.lower()

    if model_kind == "fg":
        dof = len(mn) - num_pts * 4 - N_PTS_F * 4 + len(fixed_param_values or {})
    elif model_kind == "fgln":
        dof = len(mn) - (num_pts + 3) * 4 + len(fixed_param_values or {})
    elif model_kind == "fgsig":
        dof = len(mn) - (num_pts + 3) * 4 + len(fixed_param_values or {})
    elif model_kind == "fgcconv":
        dof = len(mn) - (num_pts) * 4 - 2 + len(fixed_param_values or {})

    print(f"model {model_kind} has {dof} degrees of freedom")

    return dict(
        n_pts=num_pts,
        model_kind=model_kind,
        z=z,
        nz=nzs,
        mn=mn,
        cov=cov,
        mn_pars=np.asarray(mn_pars, dtype=np.int32),
        zbins=np.asarray(zbins),
        fixed_param_values=fixed_param_values,
        extra_kwargs=("n_pts", "model_kind"),
    )
