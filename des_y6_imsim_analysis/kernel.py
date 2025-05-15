"""
!!! NOT USED !!!
"""

import jax

jax.config.update("jax_enable_x64", True)

import functools  # noqa: E402

import interpax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from des_y6_imsim_analysis.utils import (  # noqa: E402
    GMODEL_COSMOS_NZ,
    sompz_integral,
)


def build_rmat(*, params, n_pts, zbins, z):
    min_lin_r = -1
    max_lin_r = 10
    xy = jnp.concatenate(
        [
            jnp.array([0.0]),
            make_interpolant_pts(n_pts, zbins),
            jnp.array([4.0]),
        ],
        dtype=jnp.float64,
    )
    rmat = jnp.array(
        [[0.0 for _ in range(n_pts + 2)]]
        + [
            [0.0]
            + [
                params[f"R{i}_{j}"] * (max_lin_r - min_lin_r) + min_lin_r
                for j in range(n_pts)
            ]
            + [0.0]
            for i in range(n_pts)
        ]
        + [[0.0 for _ in range(n_pts + 2)]],
        dtype=jnp.float64,
    )
    zmat = jnp.tile(z.reshape((-1, 1)), (1, z.shape[0]))
    rmat = interpax.interp2d(
        zmat.ravel(),
        zmat.T.ravel(),
        xy,
        xy,
        rmat,
        method="linear",
        extrap=0.0,
    ).reshape((z.shape[0], z.shape[0]))

    return rmat


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

    max_m = 0.1
    min_m = -0.1

    rmat = build_rmat(params=params, n_pts=n_pts, zbins=zbins, z=z)

    ones = jnp.ones_like(nz[0])

    wgts = jnp.array([0.28604739, 0.25258379, 0.28342376, 0.17794506])

    model_parts = {}
    for i in range(nz.shape[0]):
        model_parts[i] = {}

        gvals = jnp.sum(
            rmat * gtemp.reshape((-1, 1)) * nz[i].reshape((1, -1)) * wgts[i],
            axis=0,
        )

        model_parts[i]["F"] = ((params[f"m{i}"] + 0.5) * (max_m - min_m) + min_m) * ones
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
    for i in range(nz.shape[0]):
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
    for i in range(nz.shape[0]):
        if f"m{i}" not in fixed_param_values:
            params[f"m{i}"] = numpyro.sample(f"m{i}", dist.Uniform(-0.5, 0.5))
    for i in range(n_pts):
        for j in range(n_pts):
            params[f"R{i}_{j}"] = numpyro.sample(f"R{i}_{j}", dist.Uniform(0, 1))

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


def make_interpolant_pts(n_pts, zbins):
    """Make the array of linear interpolation points.

    Parameters
    ----------
    n_pts : int
        The number of points to use.
    zbins : array, shape (n_alpha+1, 2)
        Array of redshift ranges for each sheared slice. Always starts
        with full range (e.g., `[[0, 6], [0, 1], [1, 6]]` is two slices
        from 0 to 1 and 1 to 6).

    Returns
    -------
    pts : array, dimension (n_pts)
        The array of linear interpolation points.
    """
    return jnp.concatenate(
        [
            (jnp.arange(n_pts - 1) + 0.5) * zbins[-1][0] / (n_pts - 1),
            jnp.array([zbins[-1][0]]),
        ],
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
    model_kind="kern",
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

    Returns
    -------
    data : dict
        The model data. Pass to the functions using `**data`.
    """
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
