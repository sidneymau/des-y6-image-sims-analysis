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


@jax.jit
def _bump(z, a, b, w):
    # always use jax.nn.sigmoid here to ensure stable autodiff
    return sigmoid((z - a) / w) * (1 - sigmoid((z - b) / w))


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts"))
def model_parts_smooth(
    *,
    params,
    n_pts,
    pts,
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
        fvals = jnp.zeros_like(z)
        for j in range(n_pts):
            fvals += params[f"a{j}_b{i}"] * _bump(
                z, pts[i, j, 0], pts[i, j, 1], params["w"]
            )
        model_parts[i]["F"] = fvals

        g = params.get(f"g_b{i}", 0.0)
        model_parts[i]["G"] = g * gtemp

    return model_parts


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts"))
def model_mean_smooth(
    *,
    pts,
    n_pts,
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


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts"))
def model_mean(
    *,
    pts,
    n_pts,
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


@functools.partial(jax.jit, static_argnames=("extra_kwargs", "n_pts"))
def model_mean_smooth_tomobin(
    *,
    pts,
    n_pts,
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
    assert nz is not None
    assert mn is not None
    assert cov is not None
    assert mn_pars is not None
    assert zbins is not None
    assert z is not None
    assert extra_kwargs is not None

    fixed_param_values = fixed_param_values or {}

    params = {}
    if "w" not in fixed_param_values:
        params["w"] = numpyro.sample("w", dist.LogNormal(np.log(0.1), 0.1))
    for i in range(4):
        if f"g_b{i}" not in fixed_param_values:
            # std of softlaplace is pi/2 * scale, so to set std to a value V, we need
            # to set the scale to V * 2/pi
            # we use V = 1
            params[f"g_b{i}"] = numpyro.sample(
                f"g_b{i}", dist.SoftLaplace(0.0, 1 * 2.0 / jnp.pi)
            )
        for j in range(pts.shape[1]):
            params[f"a{j}_b{i}"] = numpyro.sample(f"a{j}_b{i}", dist.Uniform(-10, 10))

    for k, v in fixed_param_values.items():
        params[k] = numpyro.deterministic(k, v)

    model_mn = model_mean(
        pts=pts,
        n_pts=n_pts,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )
    numpyro.sample(
        "model", dist.MultivariateNormal(loc=model_mn, covariance_matrix=cov), obs=mn
    )


def make_bump_pts(*, num_pts, zbins):
    """Make the array of bump start end end points.

    Parameters
    ----------
    num_pts : int
        The number of bins to use. If not positive, then the bins are set to the sheared
        ranges from the image sims. Otherwise, they are set to uniformly cover the range
        of 0 to 2.7 for num_bins-1 and then a single bin from 2.7 to 6.01.
    zbins : array
        The shear bin edges.

    Returns
    -------
    pts : array, dimension (4, num_bins, 2)
        The array of bump start and end points. The first index is the tomographic bin,
        the second index is the bin number, and the third index is the start and end
        points of the bump.
    """
    if num_pts <= 0:
        pts = []
        for i in range(4):
            pts.append(zbins[1:, :].copy())
    else:
        pts = []
        for bi in range(4):
            zmid = np.linspace(0.0, 2.7, num_pts)[1:-1]
            be = np.concatenate(
                [
                    [0.0],
                    zmid,
                    [2.7],
                    [6.01],
                ]
            )
            assert be.shape[0] == num_pts + 1
            _pts = []
            for i in range(num_pts):
                _pts.append(be[i : i + 2])
            pts.append(_pts)

    pts = np.array(pts, dtype=np.float64)

    assert pts.shape[0] == 4
    if num_pts > 0:
        assert pts.shape[1] == num_pts
    assert pts.shape[2] == 2

    return pts


def make_model_data(
    *, z, nzs, mn, cov, mn_pars, zbins, fixed_param_values=None, num_pts=-1
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
        The number of bins to use. If not positive, then the bins are set to the sheared
        ranges from the image sims. Otherwise, they are set to uniformly cover the range
        of 0 to 2.7 for num_bins-1 and then a single bin from 2.7 to 6.01. The default is -1.

    Returns
    -------
    data : dict
        The model data. Pass to the functions using `**data`.
    """
    return dict(
        pts=make_bump_pts(num_pts=num_pts, zbins=zbins),
        n_pts=num_pts,
        z=z,
        nz=nzs,
        mn=mn,
        cov=cov,
        mn_pars=np.asarray(mn_pars, dtype=np.int32),
        zbins=np.asarray(zbins),
        fixed_param_values=fixed_param_values,
        extra_kwargs=("pts", "n_pts"),
    )
