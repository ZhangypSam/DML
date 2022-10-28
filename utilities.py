import numpy as np
from econml.utilities import check_input_arrays, shape, ndim

def expand_treatments(X=None, *Ts):
    X, *Ts = check_input_arrays(X, *Ts)
    n_rows = 1 if X is None else shape(X)[0]
    outTs = []
    for T in Ts:
        if ndim(T) == 0:
            T = np.full((n_rows,) + np.shape(T)[1:], T)
        outTs.append(T)

    return (X,) + tuple(outTs)