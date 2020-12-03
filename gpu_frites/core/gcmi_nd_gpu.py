"""GPU implementations of Nd MI functions."""

# I don't know if this should go here
def nd_reshape_gpu(x, mvaxis=None, traxis=-1):
    """Multi-dimentional reshaping.
    This function is used to be sure that an nd array has a correct shape
    of (..., mvaxis, traxis).
    Parameters
    ----------
    x : array_like
        Multi-dimentional array
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    Returns
    -------
    x_rsh : array_like
        The reshaped multi-dimentional array of shape (..., mvaxis, traxis)
    """
    assert isinstance(traxis, int)
    traxis = cp.arange(x.ndim)[traxis]

    # Create an empty mvaxis axis
    if not isinstance(mvaxis, int):
        x = x[..., cp.newaxis]
        mvaxis = -1
    assert isinstance(mvaxis, int)
    mvaxis = cp.arange(x.ndim)[mvaxis]

    # move the multi-variate and trial axis
    x = cp.moveaxis(x, (mvaxis, traxis), (-2, -1))

    return x
########################################################################################

def mi_nd_gpu_gg():
    pass


def mi_model_nd_gpu_gd():
    """Multi-dimentional reshaping.
    This function is used to be sure that an nd array has a correct shape
    of (..., mvaxis, traxis).
    Parameters
    ----------
    x : array_like
        Multi-dimentional array
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    Returns
    -------
    x_rsh : array_like
        The reshaped multi-dimentional array of shape (..., mvaxis, traxis)
    """
    assert isinstance(traxis, int)
    traxis = cp.arange(x.ndim)[traxis]

    # Create an empty mvaxis axis
    if not isinstance(mvaxis, int):
        x = x[..., cp.newaxis]
        mvaxis = -1
    assert isinstance(mvaxis, int)
    mvaxis = cp.arange(x.ndim)[mvaxis]

    # move the multi-variate and trial axis
    x = cp.moveaxis(x, (mvaxis, traxis), (-2, -1))

    return x

def mi_model_nd_gd_gpu(x, y, mvaxis=None, traxis=-1, biascorrect=True,
                   demeaned=False, shape_checking=True):
    """Multi-dimentional MI between a Gaussian and a discret variables in bits.
    This function is based on ANOVA style model comparison.
    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have the same shape except on the mvaxis
        (if needed).
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False
    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x and y, without the
        mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape_gpu(x, mvaxis=mvaxis, traxis=traxis)
    assert isinstance(y, cp.ndarray) and (y.ndim == 1)
    assert x.shape[-1] == len(y)

    # x.shape (..., x_mvaxis, traxis)
    nvarx, ntrl = x.shape[-2], x.shape[-1]
    u_y = cp.unique(y)
    sh = x.shape[:-2]
    zm_shape = list(sh) + [len(u_y)]

    # joint variable along the mvaxis
    if not demeaned:
        x = x - x.mean(axis=-1, keepdims=True)

    # class-conditional entropies
    ntrl_y = cp.zeros((len(u_y),), dtype=int)
    hcond = cp.zeros(zm_shape, dtype=float)
    # c = .5 * (np.log(2. * np.pi) + 1)
    for num, yi in enumerate(u_y):
        idx = y == yi
        xm = x[..., idx]
        ntrl_y[num] = idx.sum()
        xm = xm - xm.mean(axis=-1, keepdims=True)
        cm = cp.einsum('...ij, ...kj->...ik', xm, xm) / float(ntrl_y[num] - 1.)
        chcm = cp.linalg.cholesky(cm)
        hcond[..., num] = cp.log(cp.einsum('...ii->...i', chcm)).sum(-1)

    # class weights
    w = ntrl_y / float(ntrl)

    # unconditional entropy from unconditional Gaussian fit
    cx = cp.einsum('...ij, ...kj->...ik', x, x) / float(ntrl - 1.)
    chc = cp.linalg.cholesky(cx)
    hunc = cp.log(cp.einsum('...ii->...i', chc)).sum(-1)

    ln2 = cp.log(2)
    if biascorrect:
        vars = cp.arange(1, nvarx + 1)

        psiterms = psi((ntrl - vars).astype(float) / 2.) / 2.
        dterm = (ln2 - cp.log(float(ntrl - 1))) / 2.
        hunc = hunc - nvarx * dterm - psiterms.sum()

        dterm = (ln2 - cp.log((ntrl_y - 1).astype(float))) / 2.
        psiterms = cp.zeros_like(ntrl_y, dtype=float)
        for vi in vars:
            idx = ntrl_y - vi
            psiterms = psiterms + psi(idx.astype(cp.float) / 2.)
        hcond = hcond - nvarx * dterm - (psiterms / 2.)

    # MI in bits
    i = (hunc - cp.einsum('i, ...i', w, hcond)) / ln2
    # Clean GPU memory
    cp._default_memory_pool.free_all_blocks()
    return i

def cmi_nd_gpu_ggg():
    pass
