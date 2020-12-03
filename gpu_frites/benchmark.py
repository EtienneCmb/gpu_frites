"""Benchmarking CPU and GPU codes."""



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
    _, np = xfrites.utils.get_cupy(target=target)
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
    x = np.random.rand(n_times[-1], n_mv[-1], n_trials)
    y = np.random.rand(n_times[-1], n_mv[-1], n_trials)
    
    # function to time
    def _time_loop(a, b):
        if ndims == '1d':
            for n_t in range(a.shape[0]):
                fcn(a[n_t, ...], b[n_t, ...])
        if ndims == 'nd':
            fcn(a, b, mvaxis=-2, traxis=-1, shape_checking=False)
    fcn_tmt = tmt(_time_loop, n_loops=n_loops)
    
    pbar = ProgressBar(range(int(len(n_times) * len(n_mv))), mesg=mesg)
    esti = xr.DataArray(np.zeros((len(n_mv), len(n_times))),
                        dims=('mv', 'times'), coords=(n_mv, n_times))
    for n_m, m in enumerate(n_mv):
        for n_t, t in enumerate(n_times):
            esti[n_m, n_t] = fcn_tmt(x[0:n_t, 0:n_m, :], y[0:n_t, 0:n_m, :])
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
    _, np = xfrites.utils.get_cupy(target=target)
    if (target == 'cpu') and (dim == '1d'):
        from frites.core import mi_model_1d_gd
        fcn = mi_model_1d_gd
    elif (target == 'cpu') and (dim == 'nd'):
        from frites.core import mi_model_nd_gd
        fcn = mi_model_nd_gd
    elif (target == 'gpu') and (dim == '1d'):
        from gpu_frites.core import mi_model_1d_gpu_gd
        fcn = mi_model_1d_gpu_gd
    elif (target == 'gpu') and (dim == 'nd'):
        from gpu_frites.core import mi_model_nd_gpu_gd
        fcn = mi_model_nd_gpu_gd
    mesg = (f"Profiling I(C; C) (fcn={fcn.__name__}; target={target}; "
            f"ndims={ndims})")

    n_times = np.arange(1500, 4000, 100)
    n_mv = np.arange(1, 20, 1)
    
    # generate the data
    x = np.random.rand(n_times[-1], n_mv[-1], n_trials)
    y = np.random.randint(0, 3, size=(n_trials,))
    
    # function to time
    def _time_loop(a, b):
        if ndims == '1d':
            for n_t in range(a.shape[0]):
                fcn(a[n_t, ...], b)
        if ndims == 'nd':
            fcn(a, b, mvaxis=-2, traxis=-1, shape_checking=False)
    fcn_tmt = tmt(_time_loop, n_loops=n_loops)
    
    pbar = ProgressBar(range(int(len(n_times) * len(n_mv))), mesg=mesg)
    esti = xr.DataArray(np.zeros((len(n_mv), len(n_times))),
                        dims=('mv', 'times'), coords=(n_mv, n_times))
    for n_m, m in enumerate(n_mv):
        for n_t, t in enumerate(n_times):
            esti[n_m, n_t] = fcn_tmt(x[0:n_t, 0:n_m, :], y)
            pbar.update_with_increment_value(1)

    esti.attrs['method'] = fcn.__name__
    esti.attrs['target'] = target
    esti.attrs['ndims'] = ndims
    esti.attrs['n_loops'] = n_loops
    esti.attrs['n_trials'] = n_trials
    
    return esti
