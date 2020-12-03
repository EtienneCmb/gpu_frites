"""GPU implementations of Nd MI functions."""
import cupy as cp
import cupyx
from cupyx.scipy.special import digamma as psi

###############################################################################
###############################################################################
#                                    1D
###############################################################################
###############################################################################



def mi_1d_gpu_gg():
    pass


def mi_model_1d_gpu_gd(x, y, biascorrect=False, demeaned=False):
    """Mutual information between a Gaussian and a discrete variable in bits.
    This method is based on ANOVA style model comparison.
    I = mi_model_gd(x,y) returns the MI between the (possibly multidimensional)
    Gaussian variable x and the discrete variable y.
    Parameters
    ----------
    x, y : array_like
        Gaussian arrays of shape (n_epochs,) or (n_dimensions, n_epochs). y
        must be an array of integers
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)
    Returns
    -------
    i : float
        Information shared by x and y (in bits)
    """
    # Converting to cupy array
    #x, y = cp.array(x), cp.array(y)
    x, y = cp.atleast_2d(x), cp.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not cp.issubdtype(y.dtype, cp.integer):
        raise ValueError("y should be an integer array")

    nvarx, ntrl = x.shape
    ym = cp.unique(y)

    if y.size != ntrl:
        raise ValueError("number of trials do not match")

    if not demeaned:
        x = x - x.mean(axis=1)[:, cp.newaxis]

    # class-conditional entropies
    ntrl_y = cp.zeros(len(ym))
    hcond = cp.zeros(len(ym))
    for n_yi, yi in enumerate(ym):
        idx = y == yi
        xm = x[:, idx]
        ntrl_y[n_yi] = xm.shape[1]
        xm = xm - xm.mean(axis=1)[:, cp.newaxis]
        cm = cp.dot(xm, xm.T) / float(ntrl_y[n_yi] - 1)
        chcm = cp.linalg.cholesky(cm)
        hcond[n_yi] = cp.sum(cp.log(cp.diagonal(chcm)))

    # class weights
    w = ntrl_y / float(ntrl)

    # unconditional entropy from unconditional Gaussian fit
    cx = cp.dot(x, x.T) / float(ntrl - 1)
    chc = cp.linalg.cholesky(cx)
    hunc = cp.sum(cp.log(cp.diagonal(chc)))  # + c*nvarx

    ln2 = cp.log(2)
    if biascorrect:
        vars = cp.arange(1, nvarx + 1)

        psiterms = psi((ntrl - vars).astype(cp.float) / 2.) / 2.
        dterm = (ln2 - cp.log(float(ntrl - 1))) / 2.
        hunc = hunc - nvarx * dterm - psiterms.sum()

        dterm = (ln2 - cp.log((ntrl_y - 1).astype(cp.float))) / 2.0
        psiterms = cp.zeros(len(ym))
        for vi in vars:
            idx = ntrl_y - vi
            psiterms = psiterms + psi(idx.astype(cp.float) / 2.)
        hcond = hcond - nvarx * dterm - (psiterms / 2.)

    # MI in bits
    i = (hunc - cp.sum(w * hcond)) / ln2
    return i

def cmi_1d_gpu_ggg():
    pass
