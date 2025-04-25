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

from des_y6_imsim_analysis.utils import (  # noqa: E402
    GMODEL_COSMOS_NZ,
    sompz_integral,
)


def _lognormal(x, mu, sigma):
    xs = (jnp.log(x) - mu) / sigma
    return jnp.exp(-0.5 * xs * xs) / sigma / x / jnp.sqrt(2.0 * jnp.pi)


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts", "model_kind"))
def model_parts_smooth(
    *,
    params,
    pts,
    n_pts,
    model_kind,
    z,
    nz=None,
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
            min_lin_f = -1.0
            max_lin_f = 1.0
            xp = jnp.array(pts[i, :], dtype=jnp.float64)
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
            gmu = jnp.log(params["gmu"] * (max_gmu - min_gmu) + min_gmu)

            min_gnrm = 1e-5
            max_gnrm = 0.1
            gnrm = params[f"gnrm_b{i}"] * (max_gnrm - min_gnrm) + min_gnrm

            min_gsigma = 0.6
            max_gsigma = 1.0
            gsigma = params["gsigma"] * (max_gsigma - min_gsigma) + min_gsigma
            gvals = gnrm * _lognormal(z, gmu, gsigma)

        elif model_kind == "fg":
            min_lin_f = -1.0
            max_lin_f = 10.0
            xp = jnp.linspace(0, 2.7, n_pts + 1)[1:]
            yp = jnp.array(
                [
                    (params[f"f{j}_b{i}"]) * (max_lin_f - min_lin_f) + min_lin_f
                    for j in range(n_pts)
                ],
                dtype=jnp.float64,
            )
            fvals = jnp.interp(z, xp, yp, left=None, right=None)

            min_lin_g = 0.0
            max_lin_g = 10.0
            xp = jnp.concatenate([jnp.linspace(0, 2.7, n_pts + 1), jnp.array([4.0])])
            yp = jnp.array(
                [0.0]
                + [
                    (params[f"g{j}"]) * (max_lin_g - min_lin_g) + min_lin_g
                    for j in range(n_pts)
                ] + [0.0],
                dtype=jnp.float64,
            )

            min_gnrm = 0.0
            max_gnrm = 0.1
            gnrm = params[f"gnrm_b{i}"] * (max_gnrm - min_gnrm) + min_gnrm
            gvals = jnp.interp(z, xp, yp, left=0.0, right=0.0)
            gvals = gnrm * gvals / jnp.sum(gvals)
        elif model_kind == "f" or model_kind == "g":
            min_lin = -1.0
            max_lin = 10.0

            if model_kind == "g":
                xp = jnp.concatenate(
                    [jnp.zeros(1), pts[i, :], jnp.array([4.0])],
                    axis=0,
                    dtype=jnp.float64,
                )
                yp = jnp.array(
                    [0.0]
                    + [
                        (params[f"a{j}_b{i}"]) * (max_lin - min_lin) + min_lin
                        for j in range(n_pts)
                    ] + [0.0],
                    dtype=jnp.float64,
                )
                fgvals = jnp.interp(z, xp, yp, left=0.0, right=0.0)
            else:
                # xp = jnp.array(pts[i, :])
                xp = jnp.linspace(0, 2.7, n_pts + 1)[1:]
                yp = jnp.array(
                    [
                        (params[f"a{j}_b{i}"]) * (max_lin - min_lin) + min_lin
                        for j in range(n_pts)
                    ],
                    dtype=jnp.float64,
                )
                fgvals = jnp.interp(z, xp, yp, left=None, right=None)

        if model_kind == "fg" or model_kind == "fgln":
            model_parts[i]["F"] = fvals
            model_parts[i]["G"] = gvals
        elif model_kind == "f":
            model_parts[i]["F"] = fgvals

            # 0 to 1
            g = params.get(f"g_b{i}", 0.0)
            model_parts[i]["G"] = g * gtemp
        else:
            model_parts[i]["G"] = fgvals

            # -0.1 to 0.1
            g = params.get(f"m_b{i}", 0.0) * 0.2
            model_parts[i]["F"] = jnp.zeros_like(fgvals) + g

    return model_parts


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts", "model_kind"))
def model_mean_smooth(
    *,
    pts,
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
        pts=pts,
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
    pts,
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
        pts=pts,
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
    pts,
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
        pts=pts,
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
    pts=None,
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
    assert pts is not None
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
            if i == 0:
                params["gmu"] = numpyro.sample("gmu", dist.Uniform(0, 1))
                params["gsigma"] = numpyro.sample("gsigma", dist.Uniform(0, 1))

        elif model_kind == "fg":
            for j in range(n_pts):
                params[f"f{j}_b{i}"] = numpyro.sample(f"f{j}_b{i}", dist.Uniform(0, 1))
                if i == 0:
                    params[f"g{j}"] = numpyro.sample(f"g{j}", dist.Uniform(0, 1))
            params[f"gnrm_b{i}"] = numpyro.sample(f"gnrm_b{i}", dist.Uniform(0, 1))
        else:
            if model_kind == "f":
                if f"g_b{i}" not in fixed_param_values:
                    params[f"g_b{i}"] = numpyro.sample(f"g_b{i}", dist.Uniform(0, 1))
            else:
                if f"m_b{i}" not in fixed_param_values:
                    params[f"m_b{i}"] = numpyro.sample(f"m_b{i}", dist.Uniform(-0.5, 0.5))

            for j in range(n_pts):
                params[f"a{j}_b{i}"] = numpyro.sample(f"a{j}_b{i}", dist.Uniform(0, 1))

    for k, v in fixed_param_values.items():
        params[k] = numpyro.deterministic(k, v)

    model_mn = model_mean(
        pts=pts,
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


def make_interpolant_pts(*, num_pts, zbins, model_kind):
    """Make the array of linear interpolation points.

    Parameters
    ----------
    num_pts : int
        The number of points to use. If not positive, then the points are set to the middle
        of the sheared ranges from the image sims. Otherwise, they are set to uniformly
        cover the range of 0 to 2.7 for num_pts-1 and then a single point in the middle
        from 2.7 to 6.01.
    zbins : array
        The shear bin edges.
    model_kind : bool
        The kind of model.

    Returns
    -------
    pts : array, dimension (4, num_pts)
        The array of linear interpolation points.
    """
    if num_pts <= 0:
        num_pts = 10

    num_bins_nominal = num_pts

    pts = []
    for bi in range(4):
        zmid = np.linspace(0.0, 2.7, num_bins_nominal)[1:-1]
        be = np.concatenate(
            [
                [0.0],
                zmid,
                [2.7],
                [6.01],
            ]
        )
        assert be.shape[0] == num_bins_nominal + 1, (be.shape, num_bins_nominal + 1)
        _pts = []
        for i in range(num_bins_nominal):
            _pts.append(be[i : i + 2])
        pts.append(_pts)

    pts = np.array(pts, dtype=np.float64)

    final_pts = []
    for i in range(4):
        _pts = []
        for zbr in pts[i]:
            print(zbr)
            _pts.append(np.mean(zbr))
        _pts[-1] = np.min(pts[i][-1])
        final_pts.append(_pts)

    pts = np.array(final_pts, dtype=np.float64)

    assert pts.shape[0] == 4
    if num_pts > 0:
        assert pts.shape[1] == num_pts

    return pts


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
        The number of points to use. If not positive, then the points are set to the middle
        of the sheared ranges from the image sims. Otherwise, they are set to uniformly
        cover the range of 0 to 2.7 for num_pts-1 and then a single point in the middle
        from 2.7 to 6.01.
    model_kind : str, optional
        The model kind. Default is "F" meaning n_eff = (1+F) * n. If "G", then
        n_eff = n + G.

    Returns
    -------
    data : dict
        The model data. Pass to the functions using `**data`.
    """
    model_kind = model_kind.lower()

    return dict(
        pts=make_interpolant_pts(num_pts=num_pts, zbins=zbins, model_kind=model_kind),
        n_pts=num_pts,
        model_kind=model_kind,
        z=z,
        nz=nzs,
        mn=mn,
        cov=cov,
        mn_pars=np.asarray(mn_pars, dtype=np.int32),
        zbins=np.asarray(zbins),
        fixed_param_values=fixed_param_values,
        extra_kwargs=("pts", "n_pts", "model_kind"),
    )
