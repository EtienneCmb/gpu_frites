"""Benchmarking CPU and GPU codes."""
import xarray as xr
from mne.utils import ProgressBar
from time import time as tst

# profiling function
def tmt(method, n_loops=100):
    def timed(*args, **kw):
        # dry run
        method(*args, **kw)
        # timing run
        ts = tst()
        for n_l in range(n_loops):
            method(*args, **kw)
        te = tst()
        result = (te - ts) / n_loops
        return result
    return timed



###############################################################################
###############################################################################
#                               I(C; C)
###############################################################################
###############################################################################


def test_mi_1d_gg_equals():
    import numpy as np
    import cupy as cp
    from frites.core import mi_1d_gg
    from gpu_frites.core import mi_model_1d_gpu_gd

    x_dims = [(1, 20), (10, 100), (10, 100), (1, 100)]
    y_dims = [(1, 20), (10, 100), (1, 100), (10, 100)]

    for x_dim, y_dim in zip(x_dims, y_dims):
        x = np.random.rand(*x_dim)
        y = np.random.rand(*y_dim)
        # mi on cpu
        mi_cpu = mi_1d_gg(x, y)
        # mi on gpu
        mi_gpu = cp.asnumpy(mi_model_1d_gpu_gd(cp.asarray(x), cp.asarray(y)))
        # testing equality
        np.testing.assert_array_almost_equal(mi_cpu, mi_gpu)


def test_mi_gg_timing(target='cpu', ndims='1d', n_loops=100, n_trials=600):
    # get cpu / gpu ressources
    import xfrites
    import numpy as np
    _, cp = xfrites.utils.get_cupy(target=target)
    if (target == 'cpu') and (ndims == '1d'):
        from frites.core import mi_1d_gg
        fcn = mi_1d_gg
    elif (target == 'cpu') and (ndims == 'nd'):
        from frites.core import mi_nd_gg
        fcn = mi_nd_gg
    elif (target == 'gpu') and (ndims == '1d'):
        from gpu_frites.core import mi_1d_gpu_gg
        fcn = mi_1d_gpu_gg
    elif (target == 'gpu') and (ndims == 'nd'):
        from gpu_frites.core import mi_nd_gpu_gg
        fcn = mi_nd_gpu_gg
    mesg = (f"Profiling I(C; C) (fcn={fcn.__name__}; target={target}; "
            f"ndims={ndims})")

    n_times = np.arange(1500, 4000, 100)
    n_mv = np.arange(1, 20, 1)

    # generate the data
    x = cp.random.rand(int(n_times[-1]), int(n_mv[-1]), n_trials)
    y = cp.random.rand(int(n_times[-1]), int(n_mv[-1]), n_trials)

    # function to time
    def _time_loop(a, b):
        if ndims == '1d':
            for n_t in range(a.shape[0]):
                fcn(a[n_t, ...], b[n_t, ...])
        elif ndims == 'nd':
            fcn(a, b, mvaxis=-2, traxis=-1, shape_checking=False)
    fcn_tmt = tmt(_time_loop, n_loops=n_loops)

    pbar = ProgressBar(range(int(len(n_times) * len(n_mv))), mesg=mesg)
    esti = xr.DataArray(np.zeros((len(n_mv), len(n_times))),
                        dims=('mv', 'times'), coords=(n_mv, n_times))
    for n_m in range(len(n_mv)):
        for n_t in range(len(n_times)):
            esti[n_m, n_t] = fcn_tmt(
                x[0:n_t + 1, 0:n_m + 1, :], y[0:n_t + 1, 0:n_m + 1, :])
            pbar.update_with_increment_value(1)

    esti.attrs['method'] = fcn.__name__
    esti.attrs['target'] = target
    esti.attrs['ndims'] = ndims
    esti.attrs['n_loops'] = n_loops
    esti.attrs['n_trials'] = n_trials

    return esti



###############################################################################
###############################################################################
#                               I(C; D)
###############################################################################
###############################################################################


def test_mi_gd_timing(target='cpu', ndims='1d', n_loops=100, n_trials=600):
    # get cpu / gpu ressources
    import xfrites
    import numpy as np
    _, cp = xfrites.utils.get_cupy(target=target)
    if (target == 'cpu') and (ndims == '1d'):
        from frites.core import mi_model_1d_gd
        fcn = mi_model_1d_gd
    elif (target == 'cpu') and (ndims == 'nd'):
        from frites.core import mi_model_nd_gd
        fcn = mi_model_nd_gd
    elif (target == 'gpu') and (ndims == '1d'):
        from gpu_frites.core import mi_model_1d_gpu_gd
        fcn = mi_model_1d_gpu_gd
    elif (target == 'gpu') and (ndims == 'nd'):
        from gpu_frites.core import mi_model_nd_gpu_gd
        fcn = mi_model_nd_gpu_gd
    mesg = (f"Profiling I(C; D) (fcn={fcn.__name__}; target={target}; "
            f"ndims={ndims})")

    n_times = np.arange(1500, 4000, 100)
    n_mv = np.arange(1, 20, 1)

    # generate the data
    x = cp.random.rand(int(n_times[-1]), int(n_mv[-1]), n_trials)
    y = cp.random.randint(0, 3, size=(n_trials,))

    # function to time
    def _time_loop(a, b):
        if ndims == '1d':
            for n_t in range(a.shape[0]):
                fcn(a[n_t, ...], b)
        elif ndims == 'nd':
            fcn(a, b, mvaxis=-2, traxis=-1, shape_checking=False)
    fcn_tmt = tmt(_time_loop, n_loops=n_loops)

    pbar = ProgressBar(range(int(len(n_times) * len(n_mv))), mesg=mesg)
    esti = xr.DataArray(np.zeros((len(n_mv), len(n_times))),
                        dims=('mv', 'times'), coords=(n_mv, n_times))
    for n_m in range(len(n_mv)):
        for n_t in range(len(n_times)):
            esti[n_m, n_t] = fcn_tmt(x[0:n_t + 1, 0:n_m + 1, :], y)
            pbar.update_with_increment_value(1)

    esti.attrs['method'] = fcn.__name__
    esti.attrs['target'] = target
    esti.attrs['ndims'] = ndims
    esti.attrs['n_loops'] = n_loops
    esti.attrs['n_trials'] = n_trials

    return esti



###############################################################################
###############################################################################
#                               I(C; D)
###############################################################################
###############################################################################


def test_mi_ggg_timing(target='cpu', ndims='1d', n_loops=100, n_trials=600):
    # get cpu / gpu ressources
    import xfrites
    import numpy as np
    _, cp = xfrites.utils.get_cupy(target=target)
    if (target == 'cpu') and (ndims == '1d'):
        from frites.core import cmi_1d_ggg
        fcn = cmi_1d_ggg
    elif (target == 'cpu') and (ndims == 'nd'):
        from frites.core import cmi_nd_ggg
        fcn = cmi_nd_ggg
    elif (target == 'gpu') and (ndims == '1d'):
        from gpu_frites.core import cmi_1d_gpu_ggg
        fcn = cmi_1d_gpu_ggg
    elif (target == 'gpu') and (ndims == 'nd'):
        from gpu_frites.core import cmi_nd_gpu_ggg
        fcn = cmi_nd_gpu_ggg
    mesg = (f"Profiling I(C; C | C) (fcn={fcn.__name__}; target={target}; "
            f"ndims={ndims})")

    n_times = np.arange(1500, 4000, 100)
    n_mv = np.arange(1, 20, 1)

    # generate the data
    x = cp.random.rand(int(n_times[-1]), int(n_mv[-1]), n_trials)
    y = cp.random.rand(int(n_times[-1]), int(n_mv[-1]), n_trials)
    z = cp.random.rand(int(n_times[-1]), int(n_mv[-1]), n_trials)

    # function to time
    def _time_loop(a, b, c):
        if ndims == '1d':
            for n_t in range(a.shape[0]):
                fcn(a[n_t, ...], b[n_t, ...], c[n_t, ...])
        elif ndims == 'nd':
            fcn(a, b, c, mvaxis=-2, traxis=-1, shape_checking=False)
    fcn_tmt = tmt(_time_loop, n_loops=n_loops)

    pbar = ProgressBar(range(int(len(n_times) * len(n_mv))), mesg=mesg)
    esti = xr.DataArray(np.zeros((len(n_mv), len(n_times))),
                        dims=('mv', 'times'), coords=(n_mv, n_times))
    for n_m in range(len(n_mv)):
        for n_t in range(len(n_times)):
            esti[n_m, n_t] = fcn_tmt(
                x[0:n_t + 1, 0:n_m + 1, :], y[0:n_t + 1, 0:n_m + 1, :],
                z[0:n_t + 1, 0:n_m + 1, :])
            pbar.update_with_increment_value(1)

    esti.attrs['method'] = fcn.__name__
    esti.attrs['target'] = target
    esti.attrs['ndims'] = ndims
    esti.attrs['n_loops'] = n_loops
    esti.attrs['n_trials'] = n_trials

    return esti




def run_benchmark(save_to=None, n_trials=600, n_loops=100):
    bmk = {}
    kw = dict(n_loops=n_loops, n_trials=n_trials)
    for target in ['cpu', 'gpu']:
        for ndim in ['1d', 'nd']:
            kw_b ={'target': target, 'ndims': ndim, **kw}
            bmk[f'gg_{target}_{ndim}'] = test_mi_gg_timing(**kw_b)
            bmk[f'gd_{target}_{ndim}'] = test_mi_gd_timing(**kw_b)
            bmk[f'ggg_{target}_{ndim}'] = test_mi_ggg_timing(**kw_b)

    # final xarray conversion
    bmk = xr.Dataset(bmk)

    if isinstance(save_to, str):
        from datetime import datetime
        import os

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%Hh_%Mmin_%Ss.nc")
        save_as = os.path.join(save_to, dt_string)
        bmk.to_netcdf(save_as)

    return bmk

