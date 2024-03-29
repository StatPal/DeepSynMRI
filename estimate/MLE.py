import numpy as np
from estimate.Bloch import *
# from Bloch import *

debug = 0

if debug:
    np.set_printoptions(precision=6, suppress=True)
    TE_vals = np.array([0.01, 0.03, 0.04, 0.01])
    TR_vals = np.array([0.6, 0.6, 1, 0.8])
    # print(Bloch(np.array([1.4, 0.4, 0.005]), 0.4, 0.35))
    # print(Bloch(np.array([145.4, 0.2, 0.005]), TE_vals, TR_vals))

    # i = 1
    train_im = np.array([62.116354, 55.622071, 67.995199, 72.626893])
    # W_i = np.array([106.0 , 0.2, 0.004])  ## Best LS value
    W_i = np.array([56.0 , 0.1, 0.001])
    # print( sum( (Bloch(W_i, TE_vals, TR_vals) - train_im) ** 2) )
    sigma_train = np.array([1.3, 1.1, 1.5, 1.1])


from math import exp, log
from scipy.special import i0, i0e

def obj_fn(W_i, TE_vec, TR_vec, train_i, sigma):
    m = train_i.size
    pred = Bloch(W_i, TE_vec, TR_vec)
    likeli_sum = 0
    for j in range(m):
        tmp2 = train_i[j]/(sigma[j] ** 2)
        tmp3 = (train_i[j] ** 2 + pred[j] ** 2)/(2 * (sigma[j] ** 2))
        # tmp1 = log(i0(tmp2*pred[j]));  ## Possibly creating bugs
        tmp1 = log(i0e(tmp2*pred[j])) + abs(tmp2*pred[j]);   ## i0e(x) = exp(-abs(x)) * i0(x)  =>  log(i0e(x)) = -abs(x) + log(i0(x))
        likeli_sum = likeli_sum + (log(tmp2) + tmp1 - tmp3);   ## BUG - there is already a half here, also, abs() was not included before (Though it's obvious)
    
    return (-likeli_sum)     

## There was a BUG - no minus sign in likelihood (negative log likelihood) - CHECK TODO




if debug:
    print('\n\nFirst value\n')
    print(W_i)
    print(obj_fn(W_i, TE_vals, TR_vals, train_im, sigma_train))
    # print(train_im.shape)



from scipy.optimize import minimize

if debug:
    x0 = W_i
    # bnds = ((0, 450), (0, 1), (0, 1))
    bnds = ((0, 450), (1e-16, 0.77), (0, 0.006))
    additional = (TE_vals, TR_vals, train_im, sigma_train)
    abc = minimize(obj_fn, x0, args=additional, method='L-BFGS-B', bounds = bnds)
    # print(abc)
    print('\nSecond value\n')
    print(abc.x)
    print('\nFirst obj fn value\n')
    print(obj_fn(x0, TE_vals, TR_vals, train_im, sigma_train))
    print('\nSecond obj fn value\n')
    print(obj_fn(abc.x, TE_vals, TR_vals, train_im, sigma_train))



def MLE_est(W_init, TE_vec, TR_vec, train_mat, TE_scale, TR_scale, sigma_train, mask):
    bnds = ((0.0001, 450), (exp(-1/(0.01*TR_scale)), exp(-1/(4*TR_scale))), (exp(-1/(0.001*TE_scale)), exp(-1/(0.2*TE_scale))))
    print(bnds)

    n, m = train_mat.shape
    print(n)
    W = W_init
    for i in range(n):
        if i % 10000 == 0:
            print(i)
        if mask[i] == 0:
            additional = (TE_vec, TR_vec, train_mat[i], sigma_train)
            x0 = W_init[i]
            abc = minimize(obj_fn, x0, args=additional, method='L-BFGS-B', bounds = bnds)
            W[i,] = abc.x
    
    return W




from joblib import Parallel, delayed

def MLE_est_i(i, W_init, TE_vec, TR_vec, train_mat, bnds, sigma_train, mask):
    if i % 10000 == 0:
        print(i)
    if mask[i] == 0:
        additional = (TE_vec, TR_vec, train_mat[i], sigma_train)
        x0 = W_init[i]
        abc = minimize(obj_fn, x0, args=additional, method='L-BFGS-B', bounds = bnds)
        return abc.x
    else:
        return W_init[i]


def MLE_est_par(W_init, TE_vec, TR_vec, train_mat, TE_scale, TR_scale, sigma_train, mask):
    bnds = ((0.0001, 450), (exp(-1/(0.01*TR_scale)), exp(-1/(4*TR_scale))), (exp(-1/(0.001*TE_scale)), exp(-1/(0.2*TE_scale))))
    x0 = np.array([np.mean(train_mat[0]), exp(-1/(2*TR_scale)), exp(-1/(0.1*TE_scale))])

    n, m = train_mat.shape
    print(n)
    W = W_init
    
    W = Parallel(n_jobs=2)(
            delayed(MLE_est_i)(i, W_init, TE_vec, TR_vec, train_mat, bnds, sigma_train, mask) for i in range(n))
    return W


