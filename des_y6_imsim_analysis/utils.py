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


def read_data(filename, redshift_type="true"):
    """Read the data from the given filename.

    Parameters
    ----------
    filename : str
        The name of the file to read.
    redshift_type : str
        The type of redshift to read. Either "true" or "sompz".
        Default is "true".

    Returns
    -------
    ModelData
        The data read from the file as a `ModelData` named tuple.
    """
    with h5py.File(filename) as d:
        mn = d["shear/mean"][:].astype(np.float64)
        cov = d["shear/cov"][:].astype(np.float64)
        mn_pars = tuple(tuple(v) for v in d["shear/mean_params"][:].astype(np.int64).tolist())
        zbins = []
        for zbin in range(-1, 10):
            zbins.append(d[f"alpha/bin{zbin}"][:].astype(np.float64))
        zbins = np.array(zbins)

        z = d[f"redshift/{redshift_type}/zbinsc"][:].astype(np.float64)
        if np.allclose(z[0], 0.0):
            cutind = 1
        else:
            cutind = 0
        z = z[cutind:]

        nzs = {}
        for _bin in range(4):
            nzs[_bin] = d[f"redshift/{redshift_type}/bin{_bin}"][:].astype(np.float64)
            nzs[_bin] = nzs[_bin][cutind:] / np.sum(nzs[_bin][cutind:])

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


def plot_results(*, model_module, model_data, samples=None, map_params=None):
    mn_pars = tuple(tuple(mnp.tolist()) for mnp in model_data["mn_pars"])
    z = model_data["z"]
    nzs = model_data["nz"]
    mn = model_data["mn"]
    cov = model_data["cov"]
    zbins = model_data["zbins"]

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

    for bi in range(4):
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
            for j in range(10):
                nind = mn_pars.index((j, bi))
                bin_zmin, bin_zmax = zbins[j + 1]
                bin_dz = bin_zmax - bin_zmin
                ngamma_int.append(
                    sompz_integral(ngamma, bin_zmin, bin_zmax) / bin_dz
                )
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
        axhist.set_yscale("symlog", linthresh=0.2)
        axhist.format(
            xlim=(0, 4.1),
            ylim=(-0.02, 10.0),
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
            ylabel=r"(model - data)/error" if bi % 2 == 0 else None,
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
            z, ngamma_mn / DZ, drawstyle="steps-mid", color="black", label=r"$n_\gamma(z)$"
        )
        for i in range(10):
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


def measure_m_dz(*, model_module, model_data, samples=None, return_dict=False):
    nzs = model_data["nz"]
    n_samples = 1000
    data = np.zeros((8, n_samples))
    for bi in range(4):
        z_nz = compute_nz_binned_mean(nzs[bi])
        assert np.allclose(sompz_integral(nzs[bi], 0.0, 6.0), 1.0)
        for i in range(n_samples):
            _params = {}
            for k, v in samples.items():
                _params[k] = v[i]
            ngamma = model_module.model_mean_smooth_tomobin(
                **model_data, tbind=bi, params=_params
            )
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
