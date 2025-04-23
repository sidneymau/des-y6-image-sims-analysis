import jax

jax.config.update("jax_enable_x64", True)

import functools  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from des_y6_imsim_analysis.utils import (  # noqa: E402
    sompz_integral,
)


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
    model_parts = {}
    for i in range(4):
        model_parts[i] = {}
        xp = jnp.concatenate(
            [jnp.zeros(1), pts[i, :], jnp.array([4.0])],
            axis=0,
            dtype=jnp.float64,
        )
        yp = jnp.array(
            [0.0] + [params[f"a{j}_b{i}"] for j in range(n_pts)] + [0.0],
            dtype=jnp.float64,
        )

        fvals = jnp.interp(z, xp, yp, left=0.0, right=0.0)

        if model_kind.lower() == "f":
            model_parts[i]["F"] = fvals
            model_parts[i]["G"] = jnp.zeros_like(fvals)
        else:
            model_parts[i]["G"] = fvals
            model_parts[i]["F"] = jnp.zeros_like(fvals)

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
        for j in range(pts.shape[1]):
            params[f"a{j}_b{i}"] = numpyro.sample(
                f"a{j}_b{i}", dist.Uniform(-1.0, 10.0)
            )

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


def make_interpolant_pts(*, num_pts, zbins):
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

    Returns
    -------
    pts : array, dimension (4, num_pts)
        The array of linear interpolation points.
    """
    if num_pts <= 0:
        pts = []
        for i in range(4):
            pts.append(zbins[1:, :].copy())
    else:
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
    return dict(
        pts=make_interpolant_pts(num_pts=num_pts, zbins=zbins),
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
