import numpy as np
from estimate.Bloch import *

debug = 0;


from math import exp, log


def obj_fn(W_i, TE_vec, TR_vec, train_i):
    pred = Bloch(W_i, TE_vec, TR_vec)
    return sum((pred - train_i)**2)

if debug:
    print(obj_fn(W_i, TE_vals, TR_vals, train_im))
    print(train_im.shape)

def grad_fn(W_i, TE_vec, TR_vec, train_i):
    grad = np.zeros(3);
    pred = Bloch(W_i, TE_vec, TR_vec)
    m = train_i.shape;
    for j in range(3):
        grad[0] = grad[0] - 2 * (train_i[j] - pred[j]) * (1 - W_i[1] ** TR_vec[j]) * (W_i[2] ** TE_vec[j])
        grad[1] = grad[1] + 2 * (train_i[j] - pred[j]) * W_i[0] * (TR_vec[j] * W_i[1] ** (TR_vec[j] - 1)) * (W_i[2] ** TE_vec[j])
        grad[2] = grad[2] - 2 * (train_i[j] - pred[j]) * W_i[0] * (1 - W_i[1] ** TR_vec[j]) * (TE_vec[j] * W_i[2] ** (TE_vec[j] - 1))

    return grad

if debug:
    grad = grad_fn(W_i, TE_vals, TR_vals, train_im)
    num_grad = np.zeros(3);
    tol_val = 0.0001
    tmp_W_i = np.zeros(3)
    for i in range(3):
        tmp_W_i = np.copy(W_i)
        tmp_W_i[i] = W_i[i] + tol_val
        num_grad[i] = (obj_fn(tmp_W_i, TE_vals, TR_vals, train_im) - obj_fn(W_i, TE_vals, TR_vals, train_im))
    num_grad = num_grad/tol_val
    print(grad)
    print(num_grad)




from scipy.optimize import minimize

if debug:
    x0 = W_i
    bnds = ((0, 450), (0, 1), (0, 1))
    additional = (TE_vals, TR_vals, train_im)
    abc = minimize(obj_fn, x0, args=additional, method='L-BFGS-B', bounds = bnds)
    print(abc)
    abc2 = minimize(obj_fn, x0, jac = grad_fn, args=additional, method='L-BFGS-B', bounds = bnds)
    print(abc2)
    print(obj_fn(x0, TE_vals, TR_vals, train_im))
    print(obj_fn(abc.x, TE_vals, TR_vals, train_im))
    print(obj_fn(abc2.x, TE_vals, TR_vals, train_im))



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
    if i % 10000 == 0:
        print(i)
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


