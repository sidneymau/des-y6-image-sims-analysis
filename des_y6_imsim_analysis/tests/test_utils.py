import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402
from scipy.integrate import tanhsinh  # noqa: E402
from scipy.interpolate import InterpolatedUnivariateSpline  # noqa: E402

from des_y6_imsim_analysis.garys_Tz import Tz  # noqa: E402
from des_y6_imsim_analysis.utils import (  # noqa: E402
    compute_lin_interp_mean,
    gmodel_template_cosmos,
    lin_interp_integral,
    lin_interp_integral_nojit,
    nz_binned_to_interp,
    sompz_integral,
    sompz_integral_nojit,
)


def test_gmodel_template_cosmos():
    assert gmodel_template_cosmos(0.0) == 0.0


def test_compute_lin_interp_mean():
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 1, 100))
    y = np.abs(np.sin(x * 10))

    spl = InterpolatedUnivariateSpline(x, y, k=1, ext=1)

    mn = compute_lin_interp_mean(x, y)

    numer = tanhsinh(lambda z: z * spl(z), x[0], x[-1], atol=1e-12, rtol=0).integral
    denom = tanhsinh(lambda z: spl(z), x[0], x[-1], atol=1e-12, rtol=0).integral

    assert_allclose(numer / denom, mn, rtol=0, atol=1e-7)


@pytest.mark.parametrize("dz", [0.01, 0.025, 0.04, 0.05])
@pytest.mark.parametrize("z0_offset", [0.0, 0.01, 0.025, 0.04, 0.05, 0.06])
def test_nz_binned_to_interp(dz, z0_offset):
    z0 = z0_offset + dz / 2
    x = np.arange(z0_offset, dz * 17 + z0_offset - dz / 2, dz) + dz / 2
    y = np.abs(np.sin(x))
    y = y / np.sum(y)
    tz = Tz(dz, y.shape[0], z0=z0)
    xspl, yspl = nz_binned_to_interp(y, dz, z0)

    rng = np.random.default_rng(42)
    for xrnd in [
        rng.uniform(-1, 5, 1000000),
        rng.uniform(tz.zmin, tz.zmax, 1000000),
    ]:
        yrnd_spl = np.interp(xrnd, xspl, yspl, left=0, right=0)
        yrnd_tz = tz.dndz(y, xrnd)
        assert_allclose(
            yrnd_spl,
            yrnd_tz,
            rtol=0,
            atol=1e-12,
            err_msg="yrnd_spl vs yrnd_tz",
        )


@pytest.mark.parametrize(
    "func", [sompz_integral, sompz_integral_nojit], ids=("jit", "nojit")
)
@pytest.mark.parametrize("dz", [0.01, 0.025, 0.04, 0.05])
@pytest.mark.parametrize("z0_offset", [0.0, 0.01, 0.025, 0.04, 0.05, 0.06])
def test_sompz_integral(func, z0_offset, dz):
    z0 = z0_offset + dz / 2
    x = np.arange(z0_offset, dz * 17 + z0_offset - dz / 2, dz) + dz / 2
    assert_allclose(x[0], z0)
    y = np.abs(np.sin(x))
    y = y / np.sum(y)

    tz = Tz(dz, y.shape[0], z0=z0)
    assert_allclose(tz.zmax, x.max() + dz)
    eps = 1e-6
    if z0_offset == 0.01 and dz == 0.05:
        eps = 1e-8

    def _quad(a, b):
        res = tanhsinh(
            lambda z: tz.dndz(y, z),
            a,
            b,
            atol=eps / 1e4,
            rtol=0,
            maxlevel=20,
            minlevel=8,
        )
        if res.status != 0:
            print(
                f"\n           a: {a}\n           b: {b}\n{res}",
            )
        intg = res.integral
        return intg

    # first test whole range
    assert_allclose(
        _quad(0, x.max() + dz),
        func(y, 0, x.max() + dz, dz, z0),
        rtol=0,
        atol=eps,
    )
    assert_allclose(
        _quad(-10, 10),
        func(y, -10, 10, dz, z0),
        rtol=0,
        atol=eps,
    )

    # now try out side the range on both sides
    for low, high in [(-10, -1), (-3.4, 0), (1, 10), (50, 100)]:
        assert_allclose(
            0.0,
            func(y, low, high, dz, z0),
            rtol=0,
            atol=eps,
            err_msg=f"low={low}, high={high}",
        )

    # now try random ranges in the middle
    for low, high in [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7)]:
        assert_allclose(
            _quad(low, high),
            func(y, low, high, dz, z0),
            rtol=0,
            atol=eps,
            err_msg=f"low={low}, high={high}",
        )

    # now try inside bins
    for ind in [0, 4, 15]:
        dx = x[ind + 1] - x[ind]
        low = x[ind]
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            high = x[ind] + dx * fac
            assert_allclose(
                _quad(low, high),
                func(y, low, high, dz, z0),
                rtol=0,
                atol=eps,
                err_msg=f"low={low}, high={high}",
            )

        low = x[ind] + 0.05 * dx
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            high = x[ind] + dx * fac
            assert_allclose(
                _quad(low, high),
                func(y, low, high, dz, z0),
                rtol=0,
                atol=eps,
                err_msg=f"low={low}, high={high}",
            )

        high = x[ind + 1]
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            low = x[ind + 1] - dx * fac
            assert_allclose(
                _quad(low, high),
                func(y, low, high, dz, z0),
                rtol=0,
                atol=eps,
                err_msg=f"low={low}, high={high}",
            )


def test_garys_sompz_integral():
    dz = 0.05
    z0 = 0.01 + dz / 2
    x = np.arange(0.01, 0.81 - dz / 2, dz) + dz / 2
    x = np.concatenate([[0.0], x])
    y = np.abs(np.sin(x))
    ytz = y / np.sum(y)

    yspl = y.copy()
    ynrm = np.sum(yspl)
    yspl[1] = yspl[1] / ((z0 + dz) / 2) / ynrm
    yspl[2:] = yspl[2:] / 0.05 / ynrm
    yspl = np.concatenate([yspl, [0.0]])
    xspl = np.concatenate([x, [x.max() + dz]])

    spl = InterpolatedUnivariateSpline(xspl, yspl, k=1, ext=1)
    tz = Tz(dz, ytz[1:].shape[0], z0=z0)

    rng = np.random.default_rng(42)
    xrnd = rng.uniform(0.1, 0.3, 100000)
    yrnd_spl = spl(xrnd)
    yrnd_tz = tz.dndz(ytz[1:], xrnd)
    assert_allclose(
        yrnd_spl,
        yrnd_tz,
        rtol=0,
        atol=1e-12,
        err_msg="yrnd_spl vs yrnd_tz",
    )

    def _quad_spl(a, b):
        return spl.integral(a, b)

    tz = Tz(dz, ytz[1:].shape[0], z0=z0)

    def _quad_tz(a, b):
        a = np.maximum(a, tz.zmin)
        b = np.minimum(b, tz.zmax)
        res = tanhsinh(
            lambda z: tz.dndz(ytz[1:], z),
            a,
            b,
            atol=1e-10,
            rtol=0,
            maxlevel=15,
            minlevel=5,
        )
        intg = res.integral
        return intg

    # first test whole range
    assert_allclose(
        _quad_spl(-10, 10),
        _quad_tz(-10, 10),
        rtol=0,
        atol=1e-7,
    )
    assert_allclose(
        _quad_spl(0, 0.835),
        _quad_tz(0, 0.835),
        rtol=0,
        atol=1e-7,
    )

    # now try out side the range on both sides
    for low, high in [(-10, -1), (-3.4, -0.1), (1, 10), (50, 100)]:
        assert_allclose(
            0.0,
            _quad_spl(low, high),
            rtol=0,
            atol=1e-7,
            err_msg=f"low={low}, high={high}",
        )
        assert_allclose(
            0.0,
            _quad_tz(low, high),
            rtol=0,
            atol=1e-7,
            err_msg=f"low={low}, high={high}",
        )

    # now try random ranges in the middle
    for low, high in [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7)]:
        assert_allclose(
            _quad_spl(low, high),
            _quad_tz(low, high),
            rtol=0,
            atol=1e-7,
            err_msg=f"low={low}, high={high}",
        )

    # now try inside bins
    for ind in [0, 4, 15]:
        dx = x[ind + 1] - x[ind]
        low = x[ind]
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            high = x[ind] + dx * fac
            assert_allclose(
                _quad_spl(low, high),
                _quad_tz(low, high),
                rtol=0,
                atol=1e-7,
                err_msg=f"low={low}, high={high}",
            )

        low = x[ind] + 0.05 * dx
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            high = x[ind] + dx * fac
            assert_allclose(
                _quad_spl(low, high),
                _quad_tz(low, high),
                rtol=0,
                atol=1e-7,
                err_msg=f"low={low}, high={high}",
            )

        high = x[ind + 1]
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            low = x[ind + 1] - dx * fac
            assert_allclose(
                _quad_spl(low, high),
                _quad_tz(low, high),
                rtol=0,
                atol=1e-8,
                err_msg=f"low={low}, high={high}",
            )


@pytest.mark.parametrize(
    "func", [lin_interp_integral, lin_interp_integral_nojit], ids=("jit", "nojit")
)
def test_lin_interp_integral(func):
    rng = np.random.default_rng()
    x = np.sort(rng.uniform(0, 1, 100))
    y = np.abs(np.sin(x))

    spl = InterpolatedUnivariateSpline(x, y, k=1, ext=1)

    def _quad_spl(a, b):
        return spl.integral(a, b)

    def _quad(a, b):
        return func(y, x, a, b)

    # first test whole range
    assert_allclose(
        _quad_spl(-10, 10),
        _quad(-10, 10),
        rtol=0,
        atol=1e-7,
    )
    assert_allclose(
        _quad_spl(0, 1),
        _quad(0, 1),
        rtol=0,
        atol=1e-7,
    )

    # now try out side the range on both sides
    for low, high in [(-10, -1), (-3.4, -0.1), (1, 10), (50, 100)]:
        assert_allclose(
            0.0,
            _quad_spl(low, high),
            rtol=0,
            atol=1e-7,
            err_msg=f"low={low}, high={high}",
        )
        assert_allclose(
            0.0,
            _quad(low, high),
            rtol=0,
            atol=1e-7,
            err_msg=f"low={low}, high={high}",
        )

    # now try random ranges in the middle
    for low, high in [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7)]:
        assert_allclose(
            _quad_spl(low, high),
            _quad(low, high),
            rtol=0,
            atol=1e-7,
            err_msg=f"low={low}, high={high}",
        )

    # now try inside bins
    for ind in [0, 4, 98]:
        dx = x[ind + 1] - x[ind]
        low = x[ind]
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            high = x[ind] + dx * fac
            assert_allclose(
                _quad_spl(low, high),
                _quad(low, high),
                rtol=0,
                atol=1e-7,
                err_msg=f"low={low}, high={high}",
            )

        low = x[ind] + 0.05 * dx
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            high = x[ind] + dx * fac
            assert_allclose(
                _quad_spl(low, high),
                _quad(low, high),
                rtol=0,
                atol=1e-7,
                err_msg=f"low={low}, high={high}",
            )

        high = x[ind + 1]
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            low = x[ind + 1] - dx * fac
            assert_allclose(
                _quad_spl(low, high),
                _quad(low, high),
                rtol=0,
                atol=1e-8,
                err_msg=f"low={low}, high={high}",
            )
