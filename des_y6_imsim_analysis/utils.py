from collections import namedtuple
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import ultraplot as uplt  # noqa: E402

# made in notebook des-y6-nz-fits-gmodel-tests.ipynb
# fmt: off
GMODEL_COSMOS_NZ = np.array([
       0.0267978 , 0.04771043, 0.06783591, 0.08088843, 0.08331847,  # noqa: E203
       0.07791406, 0.06977548, 0.05927706, 0.04697653, 0.03691436,
       0.03124399, 0.02921629, 0.02922861, 0.02894708, 0.02700823,
       0.02454354, 0.02286044, 0.0214819 , 0.01930751, 0.01603413,  # noqa: E203
       0.01262013, 0.01036409, 0.00936232, 0.00871881, 0.00777421,
       0.00656203, 0.00552846, 0.00492145, 0.00451879, 0.00403703,
       0.00350145, 0.0030605 , 0.00272324, 0.00246635, 0.0023123 ,  # noqa: E203
       0.00220467, 0.002067  , 0.00193315, 0.00184168, 0.00173158,  # noqa: E203
       0.00155971, 0.00138213, 0.00125449, 0.00117071, 0.00111203,
       0.00106572, 0.00101297, 0.0009539 , 0.00091665, 0.00091018,  # noqa: E203
       0.00090625, 0.00089362, 0.00089426, 0.00091336, 0.00092634,
       0.00090787, 0.00085459, 0.00079051, 0.00073491, 0.00067841,
       0.00061723, 0.00057064, 0.00054269, 0.00051525, 0.00048735,
       0.00047453, 0.00048129, 0.00053498, 0.00084401, 0.00200914,
       0.00451926, 0.00712873, 0.00725519, 0.00469057, 0.00197107,
       0.0006241 , 0.00026623, 0.00026291, 0.00037085, 0.00046587,  # noqa: E203
])
# we remove a small peak in the cosmos n(z) past z = 3 that did not matter in Y3
GMODEL_COSMOS_NZ[np.where((GMODEL_COSMOS_NZ > 0.00048) & (np.arange(GMODEL_COSMOS_NZ.shape[0]) > 60))] = 0.00048

# fmt: on
DZ = 0.05
Z0 = 0.035
ZBIN_LOW = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])
ZBIN_HIGH = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 6.0])


ModelData = namedtuple(
    "ModelData",
    [
        "z",
        "nzs",
        "mn_pars",
        "zbins",
        "mn",
        "cov",
    ],
)


def read_data(filename):
    """Read the data from the given filename.

    Parameters
    ----------
    filename : str
        The name of the file to read.

    Returns
    -------
    ModelData
        The data read from the file as a `ModelData` named tuple.
    """
    with h5py.File(filename) as d:
        mn = d["shear/mean"][:].astype(np.float64)
        cov = d["shear/cov"][:].astype(np.float64)
        mn_pars = tuple(
            tuple(v) for v in d["shear/mean_params"][:].astype(np.int64).tolist()
        )
        zbins = []
        for zbin in range(-1, 10):
            zbins.append(d[f"alpha/bin{zbin}"][:].astype(np.float64))
        zbins = np.array(zbins)

        z = d["redshift/zbinsc"][:].astype(np.float64)
        if np.allclose(z[0], 0.0):
            cutind = 1
        else:
            cutind = 0
        z = z[cutind:]

        n_tomo = len(list(d["redshift"].keys()))
        nzs = []
        for _bin in range(n_tomo):
            nzs.append(d[f"redshift/bin{_bin}"][:].astype(np.float64))
            nzs[-1] = nzs[-1][cutind:] / np.sum(nzs[-1][cutind:])
        nzs = np.array(nzs, dtype=np.float64)

    return ModelData(z=z, nzs=nzs, mn_pars=mn_pars, zbins=zbins, mn=mn, cov=cov)


def read_data_one_tomo_bin(filename):
    """Read the data from the given filename.

    Parameters
    ----------
    filename : str
        The name of the file to read.

    Returns
    -------
    ModelData
        The data read from the file as a `ModelData` named tuple.
    """
    with h5py.File(filename) as d:
        mn = d["shear/mean"][:].astype(np.float64)
        cov = d["shear/cov"][:].astype(np.float64)
        mn_pars = d["shear/mean_params"][:].astype(np.int64)
        mn_pars[:, 1] = 0
        mn_pars = tuple(
            tuple(v) for v in mn_pars.tolist()
        )

        zbins = []
        for zbin in range(-1, 10):
            zbins.append(d[f"alpha/bin{zbin}"][:].astype(np.float64))
        zbins = np.array(zbins)

        z = d["redshift/zbinsc"][:].astype(np.float64)
        if np.allclose(z[0], 0.0):
            cutind = 1
        else:
            cutind = 0
        z = z[cutind:]

        nzs = []
        for _bin in [-1]:
            nzs.append(d[f"redshift/bin{_bin}"][:].astype(np.float64))
            nzs[-1] = nzs[-1][cutind:] / np.sum(nzs[-1][cutind:])
        nzs = np.array(nzs, dtype=np.float64).reshape((1, -1))

    return ModelData(z=z, nzs=nzs, mn_pars=mn_pars, zbins=zbins, mn=mn, cov=cov)


def nz_binned_to_interp(nz, dz=DZ, z0=Z0):
    """Convert the binned n(z) to the linearly interpolated n(z).

    The total integral value of the n(z) (i.e., `np.sum(nz)`)
    is preserved.

    Parameters
    ----------
    nz : array
        The binned n(z) values (i.e., each value is the integral
        of the underlying n(z) over the bin from -dz/2 to +dz/2
        about the center of the bin).
    dz : float
        The bin width.
    z0 : float
        The center of the first bin.

    Returns
    -------
    z : array, shape (nz.size + 2,)
        The redshift values for the linearly interpolated dndz.
        The two extra values are at the ends of the interpolation
        for the first and last bins.
    dndz : array, shape (nz.size + 2,)
        The linearly interpolated dn(z)/dz values. The first and
        last values will be zero.
    """
    # the first bin's interpolation kernel is truncated to end at zero
    # if it goes below zero
    # an untruncated kernel goes from -dz to dz about each bin's center
    first_zval = jnp.maximum(z0 - dz, 0.0)
    fbin_dist_left_to_peak = (
        z0 - first_zval
        # ^ this factor is the location of the first bin's left most influence
        # if it is less than zero, we truncate
    )
    dndz = jnp.concatenate(
        [
            jnp.zeros(1),
            jnp.array([nz[0] / ((fbin_dist_left_to_peak + dz) / 2)]),
            nz[1:] / dz,
            jnp.zeros(1),
        ]
    )
    z = jnp.concatenate(
        [
            jnp.ones(1) * first_zval,
            jnp.arange(nz.shape[0] + 1) * dz + z0,
        ]
    )
    return z, dndz


GMODEL_COSMOS_Z, GMODEL_COSMOS_DNDZ = nz_binned_to_interp(GMODEL_COSMOS_NZ, DZ, Z0)


@jax.jit
def gmodel_template_cosmos(z):
    return jnp.interp(z, GMODEL_COSMOS_Z, GMODEL_COSMOS_DNDZ, left=0.0, right=0.0)


@jax.jit
def compute_lin_interp_mean(z, dndz):
    """Compute the mean of a linearly interpolated function.

    Parameters
    ----------
    z : array
        The grid points of the linearly interpolated function.
    dndz : array
        The values of the linearly interpolated function at the grid points.

    Returns
    -------
    float
        The mean of the function.
    """
    x1 = z[1:]
    x0 = z[:-1]
    y1 = dndz[1:]
    y0 = dndz[:-1]
    numer = jnp.sum(1 / 6 * (x1 - x0) * (x0 * (2 * y0 + y1) + x1 * (y0 + 2 * y1)))
    denom = jnp.sum((y0 + y1) * (x1 - x0) / 2)
    return numer / denom


@partial(jax.jit, static_argnames=("dz", "z0"))
def compute_nz_binned_mean(nz, dz=DZ, z0=Z0):
    """Compute the mean redshift of a binned n(z).

    Parameters
    ----------
    nz : array
        The binned n(z) values (i.e., each value is the integral
        of the underlying n(z) over the bin from -dz/2 to +dz/2
        about the center of the bin).
    dz : float, optional
        The bin width.
    z0 : float, optional
        The center of the first bin.

    Returns
    -------
    float
        The mean redshift of the binned n(z).
    """
    z, dndz = nz_binned_to_interp(nz, dz, z0)
    return compute_lin_interp_mean(z, dndz)


def sompz_integral_nojit(nz, low, high, dz=DZ, z0=Z0):
    """Integrate a binned n(z) over a range transforming it to
    a linearly interpolated n(z) in the process.

    Parameters
    ----------
    nz : array
        The binned n(z) values (i.e., each value is the integral
        of the underlying n(z) over the bin from -dz/2 to +dz/2
        about the center of the bin).
    low : float
        The lower bound of the integration range.
    high : float
        The upper bound of the integration range.
    dz : float
        The bin width.
    z0 : float
        The center of the first bin.
    """
    z, dndz = nz_binned_to_interp(nz, dz, z0)
    return lin_interp_integral(dndz, z, low, high)


sompz_integral = jax.jit(sompz_integral_nojit, static_argnames=("dz", "z0"))


def lin_interp_integral_nojit(y, x, low, high):
    """Integrate a linearly interpolated set of values
    in a range (low, high).

    Parameters
    ----------
    y : array
        The values to integrate.
    x : array
        The grid points. They must be sorted, but need not
        be evenly spaced.
    low : float
        The lower bound of the integration range.
    high : float
        The upper bound of the integration range.

    Returns
    -------
    float
        The integral of the values in the range (low, high).
    """
    # ensure integration bounds are ordered
    low, high, sign = jax.lax.cond(
        low < high,
        lambda low, high: (low, high, 1.0),
        lambda low, high: (high, low, -1.0),
        jnp.array(low).astype(jnp.float64),
        jnp.array(high).astype(jnp.float64),
    )

    low = jnp.minimum(x[-1], jnp.maximum(low, x[0]))
    high = jnp.minimum(x[-1], jnp.maximum(high, x[0]))
    low_ind = jnp.digitize(low, x)
    # for the lower index we do not use right=True, but
    # we still want to clip to a valid index of x
    low_ind = jnp.minimum(low_ind, x.shape[0] - 1)
    high_ind = jnp.digitize(high, x, right=True)
    dx = x[1:] - x[:-1]
    m = (y[1:] - y[:-1]) / dx

    # high point not in same bin as low point
    not_in_single_bin = high_ind > low_ind

    ileft = jax.lax.select(
        not_in_single_bin,
        # if not in single bin, this is the fractional bit on the left
        ((y[low_ind - 1] + m[low_ind - 1] * (low - x[low_ind - 1])) + y[low_ind])
        / 2.0
        * (x[low_ind] - low),
        # otherwise this is the whole value
        (
            (y[low_ind - 1] + m[low_ind - 1] * (low - x[low_ind - 1]))
            + (y[low_ind - 1] + m[low_ind - 1] * (high - x[low_ind - 1]))
        )
        / 2.0
        * (high - low),
    )

    # fractional bit on the right
    iright = jax.lax.select(
        not_in_single_bin,
        # if not in single bin, this is the fractional bit on the right
        (
            y[high_ind - 1]
            + (y[high_ind - 1] + m[high_ind - 1] * (high - x[high_ind - 1]))
        )
        / 2.0
        * (high - x[high_ind - 1]),
        # optherwise return 0
        0.0,
    )

    # central bits, if any
    yint = (y[1:] + y[:-1]) / 2.0 * dx
    yind = jnp.arange(yint.shape[0])
    msk = (yind >= low_ind) & (yind < high_ind - 1)
    icen = jax.lax.select(
        jnp.any(msk),
        jnp.sum(
            jnp.where(
                msk,
                yint,
                jnp.zeros_like(yint),
            )
        ),
        0.0,
    )

    return sign * (ileft + icen + iright)


lin_interp_integral = jax.jit(lin_interp_integral_nojit)


def plot_results_nz(
    *, model_module, model_data, samples=None, map_params=None, symlog=True
):
    mn_pars = tuple(tuple(mnp.tolist()) for mnp in model_data["mn_pars"])
    z = model_data["z"]
    nzs = model_data["nz"]
    mn = model_data["mn"]
    cov = model_data["cov"]
    zbins = model_data["zbins"]

    n_tomo = nzs.shape[0]

    # fmt: off
    array = [
        [1, 3,],
        [1, 3,],
        [1, 3,],
        [1, 3,],
        [2, 4,],
        [5, 7,],
        [5, 7,],
        [5, 7,],
        [5, 7,],
        [6, 8,],
    ]
    # fmt: on

    fig, axs = uplt.subplots(
        array,
        figsize=(8, 6),
        sharex=4,
        sharey=0,
        wspace=None,
        hspace=[0] * 4 + [None] + [0] * 4,
    )

    for bi in range(n_tomo):
        # first extract the stats from fit
        if samples is not None:
            ngammas = []
            for i in range(1000):
                _params = {}
                for k, v in samples.items():
                    _params[k] = v[i]
                ngamma = model_module.model_mean_smooth_tomobin(
                    **model_data, tbind=bi, params=_params
                )
                ngammas.append(ngamma)

            ngammas = np.array(ngammas)
            ngamma_mn = np.mean(ngammas, axis=0)
        elif map_params is not None:
            ngammas = np.array(
                [
                    model_module.model_mean_smooth_tomobin(
                        **model_data, tbind=bi, params=map_params
                    )
                ]
            )
            ngamma_mn = ngammas[0]
        else:
            raise ValueError("Either samples or map_params must be provided.")

        ngamma_ints = []
        for ngamma in ngammas:
            ngamma_int = []
            for j in range(zbins.shape[0] - 1):
                nind = mn_pars.index((j, bi))
                bin_zmin, bin_zmax = zbins[j + 1]
                bin_dz = bin_zmax - bin_zmin
                ngamma_int.append(sompz_integral(ngamma, bin_zmin, bin_zmax) / bin_dz)
            ngamma_ints.append(ngamma_int)
        ngamma_ints = np.array(ngamma_ints)

        # get the axes
        bihist = bi * 2
        axhist = axs[bihist]
        bidiff = bi * 2 + 1
        axdiff = axs[bidiff]

        # plot the stuff
        axhist.axhline(0.0, color="black", linestyle="dotted")
        axhist.grid(False)
        if symlog:
            axhist.set_yscale("symlog", linthresh=1e-3)
        axhist.format(
            xlim=(0, 4.19),
            title=f"bin {bi}",
            titleloc="ul",
            xlabel="redshift",
            ylabel=r"redshift density" if bi % 2 == 0 else None,
            yticklabels=[] if bi % 2 == 1 else None,
        )

        axdiff.grid(False)
        axdiff.axhline(0.0, color="black", linestyle="dotted")
        axdiff.format(
            ylim=(-3, 3),
            ylabel=r"(data - model)/error" if bi % 2 == 0 else None,
            yticklabels=[] if bi % 2 == 1 else None,
        )

        axhist.plot(
            z,
            nzs[bi] / DZ,
            drawstyle="steps-mid",
            label=r"$n_{\rm phot}(z)$",
            color="purple",
            linestyle="dashed",
        )
        axhist.plot(
            z,
            ngamma_mn / DZ,
            drawstyle="steps-mid",
            color="black",
            label=r"$n_\gamma(z)$",
        )
        for i in range(zbins.shape[0] - 1):
            nind = mn_pars.index((i, bi))
            bin_zmin, bin_zmax = zbins[i + 1]
            bin_dz = bin_zmax - bin_zmin

            nmcal_val = sompz_integral(nzs[bi], bin_zmin, bin_zmax) / bin_dz
            axhist.hlines(
                nmcal_val,
                bin_zmin,
                bin_zmax,
                color="purple",
                linestyle="dashed",
            )

            nga_val = mn[nind] / bin_dz
            nga_err = np.sqrt(cov[nind, nind]) / bin_dz
            axhist.fill_between(
                [bin_zmin, bin_zmax],
                np.ones(2) * nga_val - nga_err,
                np.ones(2) * nga_val + nga_err,
                color="blue",
                alpha=0.5,
            )
            axhist.hlines(
                nga_val,
                bin_zmin,
                bin_zmax,
                color="blue",
                label=r"$N_{\gamma}^{\alpha}$" if i == 0 else None,
            )

            ng_val = np.mean(ngamma_ints, axis=0)[i]
            axhist.hlines(ng_val, bin_zmin, bin_zmax, color="black")
            if ngamma_ints.shape[0] > 1:
                ng_err = np.std(ngamma_ints, axis=0)[i]
                axhist.fill_between(
                    [bin_zmin, bin_zmax],
                    np.ones(2) * ng_val - ng_err,
                    np.ones(2) * ng_val + ng_err,
                    color="black",
                    alpha=0.5,
                )
            axhist.legend(loc="ur", frameon=False, ncols=1)

            axdiff.fill_between(
                [bin_zmin, bin_zmax],
                (np.ones(2) * nga_val - ng_val - nga_err) / nga_err,
                (np.ones(2) * nga_val - ng_val + nga_err) / nga_err,
                color="blue",
                alpha=0.5,
            )
            # axdiff.hlines(
            #     (nga_val - ng_val) / nga_err,
            #     bin_zmin,
            #     bin_zmax,
            #     color="blue",
            # )
            # if ngamma_ints.shape[0] > 1:
            #     ng_err = np.std(ngamma_ints, axis=0)[i]
            #     axdiff.fill_between(
            #         [bin_zmin, bin_zmax],
            #         (np.ones(2) * nga_val - ng_val - ng_err) / ng_err,
            #         (np.ones(2) * nga_val - ng_val + ng_err) / ng_err,
            #         color="black",
            #         alpha=0.5,
            #     )

    return fig


def plot_results_delta_nz(
    *, model_module, model_data, samples=None, map_params=None, symlog=True
):
    mn_pars = tuple(tuple(mnp.tolist()) for mnp in model_data["mn_pars"])
    z = model_data["z"]
    nzs = model_data["nz"]
    mn = model_data["mn"]
    cov = model_data["cov"]
    zbins = model_data["zbins"]

    n_tomo = nzs.shape[0]

    fig, axs = uplt.subplots(
        nrows=2,
        ncols=2,
        figsize=(8, 6),
    )

    for bi in range(n_tomo):
        # first extract the stats from fit
        if samples is not None:
            ngammas = []
            for i in range(1000):
                _params = {}
                for k, v in samples.items():
                    _params[k] = v[i]
                ngamma = model_module.model_mean_smooth_tomobin(
                    **model_data, tbind=bi, params=_params
                )
                ngammas.append(ngamma)

            ngammas = np.array(ngammas)
            ngamma_mn = np.mean(ngammas, axis=0)
        elif map_params is not None:
            ngammas = np.array(
                [
                    model_module.model_mean_smooth_tomobin(
                        **model_data, tbind=bi, params=map_params
                    )
                ]
            )
            ngamma_mn = ngammas[0]
        else:
            raise ValueError("Either samples or map_params must be provided.")

        ngamma_ints = []
        for ngamma in ngammas:
            ngamma_int = []
            for j in range(zbins.shape[0] - 1):
                nind = mn_pars.index((j, bi))
                bin_zmin, bin_zmax = zbins[j + 1]
                bin_dz = bin_zmax - bin_zmin
                ngamma_int.append(sompz_integral(ngamma, bin_zmin, bin_zmax))
            ngamma_ints.append(ngamma_int)
        ngamma_ints = np.array(ngamma_ints)

        # get the axes
        bi_row = bi // 2
        bi_col = bi % 2
        ax = axs[bi_row, bi_col]

        ax.axhline(0.0, color="black", linestyle="dotted")
        if symlog:
            ax.set_yscale("symlog", linthresh=1e-3)
        ax.format(
            xlim=(0, 4.19),
            title=f"bin {bi}",
            titleloc="ur",
            xlabel="redshift",
            ylabel=r"$\Delta n(z)$" if bi % 2 == 0 else None,
        )

        ax.plot(
            z,
            (ngamma_mn - nzs[bi]) / DZ,
            drawstyle="steps-mid",
            color="black",
            linestyle="solid",
        )
        if ngammas.shape[0] > 1:
            ngamma_sd = np.std(ngammas, axis=0)
            ax.fill_between(
                z,
                (ngamma_mn - nzs[bi] - ngamma_sd) / DZ,
                (ngamma_mn - nzs[bi] + ngamma_sd) / DZ,
                color="black",
                alpha=0.5,
                label="model",
                step="mid",
            )
        for i in range(zbins.shape[0] - 1):
            nind = mn_pars.index((i, bi))
            bin_zmin, bin_zmax = zbins[i + 1]
            bin_dz = bin_zmax - bin_zmin

            nmcal_val = sompz_integral(nzs[bi], bin_zmin, bin_zmax)

            nga_val = (mn[nind] - nmcal_val) / bin_dz
            nga_err = np.sqrt(cov[nind, nind]) / bin_dz
            ax.fill_between(
                [bin_zmin, bin_zmax],
                np.ones(2) * nga_val - nga_err,
                np.ones(2) * nga_val + nga_err,
                color="blue",
                alpha=0.5,
                label=r"$N_{\gamma}^{\alpha}$" if i == 0 else None,
            )
            ax.hlines(
                nga_val,
                bin_zmin,
                bin_zmax,
                color="blue",
            )

            ng_val = (np.mean(ngamma_ints, axis=0)[i] - nmcal_val) / bin_dz
            ax.hlines(
                ng_val,
                bin_zmin,
                bin_zmax,
                color="black",
                label="model integral" if i == 0 else None,
            )
            ax.legend(loc="lr", frameon=False, ncols=1)

    return fig


def plot_results_fg_model(*, model_module, model_data, map_params=None, samples=None):
    n_tomo = model_data["nz"].shape[0]

    kwargs = {k: model_data[k] for k in model_data["extra_kwargs"]}
    if samples is None:
        model_parts_mn = model_module.model_parts_smooth(
            params=map_params,
            z=model_data["z"],
            nz=model_data["nz"],
            mn_pars=None,
            zbins=model_data["zbins"],
            mn=None,
            cov=None,
            **kwargs,
        )
        model_parts_sd = None
    else:
        model_parts = {bi: {"F": [], "G": []} for bi in range(n_tomo)}
        for si in range(1000):
            si_params = {}
            for k, v in samples.items():
                si_params[k] = v[si]

            si_model_parts = model_module.model_parts_smooth(
                params=si_params,
                z=model_data["z"],
                nz=model_data["nz"],
                mn_pars=None,
                zbins=model_data["zbins"],
                mn=None,
                cov=None,
                **kwargs,
            )
            for bi in range(n_tomo):
                model_parts[bi]["F"].append(si_model_parts[bi]["F"])
                model_parts[bi]["G"].append(si_model_parts[bi]["G"])

        model_parts_mn = {}
        model_parts_sd = {}
        for bi in range(n_tomo):
            model_parts_mn[bi] = {
                "F": np.mean(model_parts[bi]["F"], axis=0),
                "G": np.mean(model_parts[bi]["G"], axis=0),
            }
            model_parts_sd[bi] = {
                "F": np.std(model_parts[bi]["F"], axis=0),
                "G": np.std(model_parts[bi]["G"], axis=0),
            }

    all_g_zero = True
    for bi in range(n_tomo):
        if np.any(model_parts_mn[bi]["G"] != 0.0):
            all_g_zero = False
            break
    plot_g = not all_g_zero

    all_f_zero = True
    for bi in range(n_tomo):
        if np.any(model_parts_mn[bi]["F"] != 0.0):
            all_f_zero = False
            break
    plot_f = not all_f_zero

    if plot_f and plot_g:
        ncols = 2
    else:
        ncols = 1

    colors = uplt.Cycle("default", N=4).by_key()["color"]

    fig, axs = uplt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(8, 4) if ncols == 2 else (5, 5),
        sharex=False,
        sharey=False,
    )

    if ncols == 2:
        axf = axs[0, 0]
        axg = axs[0, 1]
    else:
        if plot_f:
            axf = axs[0]
            axg = None
        else:
            axf = None
            axg = axs[0]

    if axf is not None:
        ax = axf
        for bi in range(n_tomo):
            ax.plot(
                model_data["z"],
                model_parts_mn[bi]["F"],
                label=f"bin {bi}",
                color=colors[bi],
            )
            if model_parts_sd is not None:
                ax.fill_between(
                    model_data["z"],
                    model_parts_mn[bi]["F"] - model_parts_sd[bi]["F"],
                    model_parts_mn[bi]["F"] + model_parts_sd[bi]["F"],
                    alpha=0.2,
                    color=colors[bi],
                )

        ax.legend(loc="ur", frameon=False, ncols=1)
        ax.format(
            xlim=(0, 4.19),
            xlabel="redshift",
            ylabel=r"$F(z)$",
        )

    if axg is not None:
        ax = axg
        for bi in range(n_tomo):
            ax.plot(
                model_data["z"],
                model_parts_mn[bi]["G"],
                label=f"bin {bi}",
                color=colors[bi],
            )
            if model_parts_sd is not None:
                ax.fill_between(
                    model_data["z"],
                    model_parts_mn[bi]["G"] - model_parts_sd[bi]["G"],
                    model_parts_mn[bi]["G"] + model_parts_sd[bi]["G"],
                    alpha=0.2,
                    color=colors[bi],
                )
        ax.format(
            xlim=(0, 4.19),
            xlabel="redshift",
            ylabel=r"$G(z)$",
        )

    return fig


def measure_m_dz(*, model_module, model_data, samples, return_dict=False, shift_negative=False):
    nzs = model_data["nz"]
    n_tomo = nzs.shape[0]

    n_samples = 1000
    data = np.zeros((8, n_samples))
    for bi in range(n_tomo):
        z_nz = compute_nz_binned_mean(nzs[bi])
        assert np.allclose(sompz_integral(nzs[bi], 0.0, 6.0), 1.0)
        for i in range(n_samples):
            _params = {}
            for k, v in samples.items():
                _params[k] = v[i]
            ngamma = model_module.model_mean_smooth_tomobin(
                **model_data, tbind=bi, params=_params
            )

            if shift_negative:
                ngamma = shift_negative_nz_values(np.asarray(ngamma).copy())

            m = sompz_integral(ngamma, 0.0, 6.0) - 1.0
            z_ngamma = compute_nz_binned_mean(ngamma)

            dz = z_ngamma - z_nz
            data[bi * 2, i] = m
            data[bi * 2 + 1, i] = dz

    if return_dict:
        data = dict(
            m_b0=data[0],
            dz_b0=data[1],
            m_b1=data[2],
            dz_b1=data[3],
            m_b2=data[4],
            dz_b2=data[5],
            m_b3=data[6],
            dz_b3=data[7],
        )
    else:
        data = data.T

    return data


def shift_negative_nz_values(nz):
    """Shift negative values in an n(z) to adjacent positive bins.

    Parameters
    ----------
    nz : array
        The n(z) values.

    Returns
    -------
    array
        The n(z) values with negative values shifted to adjacent positive bins.
    """
    msk = nz < 0.0
    if np.any(msk):
        for nind in np.where(msk)[0]:
            if nind == 0:
                shifts = [1]
            elif nind == nz.shape[0] - 1:
                shifts = [-1]
            else:
                shifts = [-1, 1]
            lim_val = np.abs(nz[nind])

            shift_inds = []
            for shift in shifts:
                done = False
                shift_ind = nind + shift
                while not done and shift_ind >= 0 and shift_ind < nz.shape[0]:
                    if nz[shift_ind] >= lim_val:
                        shift_inds.append(shift_ind)
                        done = True

                    shift_ind += shift

                if not done:
                    shift_inds.append(None)

            shift_inds = [si for si in shift_inds if si is not None]
            assert len(shift_inds) > 0

            half_neg = nz[nind] / len(shift_inds)
            for shift_ind in shift_inds:
                nz[shift_ind] = nz[shift_ind] + half_neg

            nz[nind] = 0.0

        assert np.all(nz >= 0.0), "negative n(z) values remain!"

    return nz


def compute_eff_nz_from_data(
    *,
    model_module,
    mcmc_samples,
    model_data,
    input_nz,
    rng,
    clip_zero=False,
    shift_negative=False,
    progress_bar=False,
    input_nz_mean_only=False,
):
    """Compute the effective nz for a given set of input n(z) values and MCMC samples
    for the model parameters.

    Parameters
    ----------
    model_module : module
        The module containing the model functions.
    mcmc_samples : dict
        The MCMC samples for the model parameters. Dict is keyed on parameter name
        with an array of samples for each parameter.
    model_data : dict
        The model data containing the necessary information for the model. Produced from
        `model_module.make_model_data`.
    input_nz : array
        The input n(z) values. Shape is (# of input nzs, # of tomo bins, nz dimension).
    rng : np.random.RandomState
        Random number generator for sampling.
    clip_zero : bool, optional
        If True, clip the output finalnzs to be strictly non-negative. Default is False.
    shift_negative : bool, optional
        If True, attempt to shift negative n(z) values to adjacent positive bins.
        Default is False.
    progress_bar : bool, optional
        If True, show a progress bar for the computation. Default is False.
    input_nz_mean_only : bool, optional
        If True, only apply the simulation model to the mean of the input n(z) values.
        Default is False.

    Returns
    -------
    mvals : array
        The values of m for each input n(z) and model parameter sample. Shape is
        (# of input nzs, # of tomo bins).
    dzvals : array
        The values of dz for each input n(z) and model parameter sample. Shape is
        (# of input nzs, # of tomo bins).
    finalnzs : array
        The final n(z) values for each input n(z) and model parameter sample. Shape is
        (# of input nzs, # of tomo bins, nz dimension).
    """
    if input_nz_mean_only:
        ns = input_nz.shape[0]
        mn_input_nz = np.mean(input_nz, axis=0, keepdims=True)
        input_nz = np.tile(mn_input_nz, (ns, 1, 1))
        assert np.allclose(input_nz, mn_input_nz), (
            "input_nz_mean_only is not working as expected!"
        )

    if progress_bar:
        import tqdm

        range_gen = tqdm.trange(input_nz.shape[0])
    else:
        range_gen = range(input_nz.shape[0])

    if clip_zero and shift_negative:
        raise ValueError(
            "Cannot use both clip_zero and shift_negative at the same time!"
        )

    kwargs = {k: model_data[k] for k in model_data["extra_kwargs"]}

    test_key = list(mcmc_samples.keys())[0]
    n_tomo = input_nz.shape[1]
    assert n_tomo == 4

    assert input_nz.shape[1] == 4
    assert input_nz.shape[2] == model_data["z"].shape[0]

    key_mvals = []
    key_dzvals = []
    key_finalnzs = []
    for i in range_gen:
        rind = rng.choice(mcmc_samples[test_key].shape[0])

        params = {k: mcmc_samples[k][rind] for k in mcmc_samples.keys()}
        nz = input_nz[i, :, :].copy()
        for _i in range(n_tomo):
            nz[_i, :] = nz[_i, :] / np.sum(nz[_i, :])
        model_nz = model_module.model_mean_smooth(
            z=model_data["z"],
            nz=nz,
            mn_pars=model_data["mn_pars"],
            zbins=model_data["zbins"],
            params=params,
            mn=None,
            cov=None,
            **kwargs,
        )

        model_nz = np.array(model_nz)
        assert model_nz.shape == (n_tomo, model_data["z"].shape[0])

        if clip_zero:
            msk = model_nz < 0.0
            if np.any(msk):
                model_nz[msk] = 0.0

        # find the nearest bins above and below with amplitude > negative value / 2
        # and add the negative value there
        if shift_negative:
            for bi in range(n_tomo):
                model_nz[bi, :] = shift_negative_nz_values(model_nz[bi, :])

        key_mvals.append(
            [float(sompz_integral(model_nz[_i, :], 0, 6) - 1) for _i in range(n_tomo)]
        )

        for _i in range(n_tomo):
            model_nz[_i, :] = model_nz[_i, :] / np.sum(model_nz[_i, :])

        key_finalnzs.append(model_nz)
        key_dzvals.append(
            [
                float(
                    compute_nz_binned_mean(model_nz[_i, :])
                    - compute_nz_binned_mean(nz[_i, :])
                )
                for _i in range(n_tomo)
            ]
        )

    mvals = np.array(key_mvals)
    dzvals = np.array(key_dzvals)
    finalnzs = np.array(key_finalnzs)

    assert mvals.shape == (input_nz.shape[0], n_tomo)
    assert dzvals.shape == (input_nz.shape[0], n_tomo)
    assert finalnzs.shape == (input_nz.shape[0], n_tomo, input_nz.shape[-1])

    return mvals, dzvals, finalnzs


def rebin_data(data, new_bin_ranges):
    """Rebin the sim data into new bin ranges.

    Parameters
    ----------
    data : ModelData
        The original simulation data.
    new_bin_ranges : list of 2-tuples
        The new bins expressed as index ranges into the old bins.
        For example, the value `[(0, 1), (1, -1)]` would indicate
        that there are two new bins composed of the old data's 0th
        bin and all of the other bins of the old data combined.

    Returns
    -------
    rebinned_data : ModelData
        The new simulation data rebinned.
    """
    nzs = np.array(data.nzs)
    n_tomo = nzs.shape[0]

    # wgts = []
    # for ti in range(nzs.shape[0]):
    #     nz = nzs[ti, :] / np.sum(nzs[ti, :])
    #     ti_wgts = []
    #     for ai in range(data.zbins.shape[0] - 1):
    #         ti_wgts.append(
    #             sompz_integral(nz, data.zbins[ai + 1][0], data.zbins[ai + 1][1])
    #         )
    #     wgts.append(ti_wgts)

    # wgts = np.array(wgts)
    # assert np.allclose(np.sum(wgts, axis=1), 1.0)

    new_zbins = np.array(
        [[data.zbins[0][0], data.zbins[0][1]]]
        + [[data.zbins[br[0] + 1][0], data.zbins[br[1]][1]] for br in new_bin_ranges]
    )

    new_mn_pars = [(-1, i) for i in range(n_tomo)]
    for bi in range(len(new_bin_ranges)):
        new_mn_pars += [(bi, i) for i in range(nzs.shape[0])]

    proj_mat = np.zeros(
        (data.mn.shape[0], nzs.shape[0] + nzs.shape[0] * len(new_bin_ranges))
    )
    for i in range(n_tomo):
        proj_mat[i, i] = 1.0

    loc = n_tomo
    for bi in range(len(new_bin_ranges)):
        for ti in range(nzs.shape[0]):
            br = new_bin_ranges[bi]
            for bri in range(br[0], br[1] if br[1] != -1 else len(data.zbins) - 1):
                old_bi = data.mn_pars.index((bri, ti))
                proj_mat[old_bi, loc] = 1.0  # wgts[ti, bri]
            loc += 1

    assert np.allclose(np.sum(proj_mat), data.mn.shape[0])
    assert np.allclose(np.sum(proj_mat > 0, axis=1), 1)
    # proj_mat /= np.sum(proj_mat, axis=0, keepdims=True)
    new_mn = data.mn @ proj_mat
    new_cov = proj_mat.T @ data.cov @ proj_mat

    return ModelData(
        z=data.z,
        nzs=data.nzs,
        mn_pars=new_mn_pars,
        zbins=new_zbins,
        mn=new_mn,
        cov=new_cov,
    )
