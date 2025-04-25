import jax

jax.config.update("jax_enable_x64", True)

import functools  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from jax.lax import erf as jax_erf  # noqa: E402
from jax.nn import sigmoid  # noqa E402
from jax.scipy.signal import convolve  # noqa: E402

from des_y6_imsim_analysis.utils import (  # noqa: E402
    lin_interp_integral,
    sompz_integral,
)


@jax.jit
def _norm_cdf(x):
    return 0.5 * (1.0 + jax_erf(x / jnp.sqrt(2)))


@jax.jit
def _norm(x):
    return jnp.exp(-0.5 * x**2) / np.sqrt(2 * jnp.pi)


@jax.jit
def _skew_normal(x, loc, scale, alpha):
    xs = (x - loc) / scale
    return 2.0 / scale * _norm(xs) * _norm_cdf(alpha * xs)


@functools.partial(jax.jit, static_argnames=("extra_kwargs",))
def model_parts_smooth(
    *,
    params,
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

        loc = z[z.shape[0] // 2]

        # amplitude
        amp = (params[f"n_b{i}"] + 0.5) * 1

        # prior is [0, 1]
        scale = (params[f"s_b{i}"] + 0.5)

        # prior is [-10, 10]
        max_a = -1
        min_a = -50
        alpha = (params[f"a_b{i}"] + 0.5) * (max_a - min_a) + min_a

        skn = _skew_normal(z, loc, scale, alpha)
        gvals = amp * _skew_normal(z, loc, scale, alpha) / lin_interp_integral(skn, z, 0, 6)
        model_parts[i]["G"] = gvals

        min_m = -0.2
        max_m = 0.2
        ml = (params[f"ml_b{i}"] + 0.5) * (max_m - min_m) + min_m
        mh = (params[f"mh_b{i}"] + 0.5) * (max_m - min_m) + min_m
        mloc = (params[f"mloc_b{i}"] + 0.5) * 3.0
        mwidth = (params[f"mwidth_b{i}"] + 0.5) * 1.9 + 0.1

        fvals = ml + (mh - ml) * sigmoid((z - mloc) / mwidth)
        model_parts[i]["F"] = fvals

    return model_parts


@functools.partial(jax.jit, static_argnames=("extra_kwargs",))
def model_mean_smooth(
    *,
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
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )
    ngammas = []
    for i in range(4):
        ngamma = (1.0 + model_parts[i]["F"]) * nz[i] + convolve(nz[i], model_parts[i]["G"], mode="same")
        ngammas.append(ngamma)

    return jnp.stack(ngammas)


@functools.partial(jax.jit, static_argnames=("extra_kwargs",))
def model_mean(
    *,
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


@functools.partial(jax.jit, static_argnames=("extra_kwargs",))
def model_mean_smooth_tomobin(
    *,
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
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )
    return jnp.asarray(model_mn)[tbind]


def model(
    z=None,
    nz=None,
    mn=None,
    cov=None,
    mn_pars=None,
    zbins=None,
    fixed_param_values=None,
    extra_kwargs=None,
):
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
        for key in ["n", "s", "a", "ml", "mh", "mwidth", "mloc"]:
            params[f"{key}_b{i}"] = numpyro.sample(f"{key}_b{i}", dist.Uniform(-0.5, 0.5))

    for k, v in fixed_param_values.items():
        params[k] = numpyro.deterministic(k, v)

    model_mn = model_mean(
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )
    numpyro.sample(
        "model", dist.MultivariateNormal(loc=model_mn, covariance_matrix=cov), obs=mn
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

    Returns
    -------
    data : dict
        The model data. Pass to the functions using `**data`.
    """
    return dict(
        z=z,
        nz=nzs,
        mn=mn,
        cov=cov,
        mn_pars=np.asarray(mn_pars, dtype=np.int32),
        zbins=np.asarray(zbins),
        fixed_param_values=fixed_param_values,
        extra_kwargs=tuple(),
    )
