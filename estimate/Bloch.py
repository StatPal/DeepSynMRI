import numpy as np

def Bloch(W_i, TE_vec, TR_vec):
    return (W_i[0] * (1 - W_i[1]**TR_vec) * W_i[2]**TE_vec)

# TE_vals = np.array([0.5, 0.4, 0.45])
# TR_vals = np.array([0.3, 0.35, 0.25])
# print(Bloch(np.array([1.4, 0.2, 0.5]), 0.5, 0.3))
# print(Bloch(np.array([1.4, 0.2, 0.5]), 0.4, 0.35))
# print(Bloch(np.array([1.4, 0.2, 0.5]), TE_vals, TR_vals))

# i = 1
# train_im = np.array([200, 120.5, 150.4])
# W_i = np.array([1.4, 0.2, 0.5])
# print( sum( (Bloch(np.array([1.4, 0.2, 0.5]), TE_vals, TR_vals) - train_im) ** 2) )


def obj_fn(W_i, TE_vec, TR_vec, train_i):
    pred = Bloch(W_i, TE_vec, TR_vec)
    return sum((pred - train_i)**2)

# obj_fn(W_i, TE_vals, TR_vals, train_im)

def grad_fn(W_i, TE_vec, TR_vec, train_i):
    



from scipy.optimize import minimize
# x0 = W_i

# bnds = ((0, 450), (0, 1), (0, 1))
# additional = (TE_vals, TR_vals, train_im)
# abc = minimize(obj_fn, x0, args=additional, method='L-BFGS-B', bounds = bnds)
# print(abc)

from math import exp


def LS_est(TE_vec, TR_vec, train_mat, TE_scale, TR_scale):
    bnds = ((0.0001, 450), (exp(-1/(0.01*TR_scale)), exp(-1/(4*TR_scale))), (exp(-1/(0.001*TE_scale)), exp(-1/(0.2*TE_scale))))
    print(bnds)
    x0 = np.array([np.mean(train_mat[0]), exp(-1/(2*TR_scale)), exp(-1/(0.1*TE_scale))])

    n, m = train_mat.shape
    print(n)
    W = np.empty(shape=[n, 3])
    for i in range(n):
        if i % 10000 == 0:
            print(i)
        additional = (TE_vec, TR_vec, train_mat[i])
        x0[0] = np.mean(train_mat[i])
        abc = minimize(obj_fn, x0, args=additional, method='L-BFGS-B', bounds = bnds)
        #print(abc)
        W[i,] = abc.x
    
    return W




from joblib import Parallel, delayed

def LS_est_i(i, TE_vec, TR_vec, train_mat, x0, bnds):
    additional = (TE_vec, TR_vec, train_mat[i])
    x0[0] = np.mean(train_mat[i])
    abc = minimize(obj_fn, x0, args=additional, method='L-BFGS-B', bounds = bnds)
    return abc.x


def LS_est_par(TE_vec, TR_vec, train_mat, TE_scale, TR_scale):
    bnds = ((0.0001, 450), (exp(-1/(0.01*TR_scale)), exp(-1/(4*TR_scale))), (exp(-1/(0.001*TE_scale)), exp(-1/(0.2*TE_scale))))
    x0 = np.array([np.mean(train_mat[0]), exp(-1/(2*TR_scale)), exp(-1/(0.1*TE_scale))])

    n, m = train_mat.shape
    print(n)
    W = np.empty(shape=[n, 3])
    
    W = Parallel(n_jobs=2)(
            delayed(LS_est_i)(i, TE_vec, TR_vec, train_mat, x0, bnds) for i in range(n))
    return W

