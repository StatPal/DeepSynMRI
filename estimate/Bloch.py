import numpy as np

def Bloch(W_i, TE_vec, TR_vec):
    return (W_i[0] * (1 - W_i[1]**TR_vec) * W_i[2]**TE_vec)

debug = 0;

if debug:
    TE_vals = np.array([0.5, 0.4, 0.45])
    TR_vals = np.array([0.3, 0.35, 0.25])
    # print(Bloch(np.array([1.4, 0.2, 0.5]), 0.5, 0.3))
    # print(Bloch(np.array([1.4, 0.2, 0.5]), 0.4, 0.35))
    # print(Bloch(np.array([1.4, 0.2, 0.5]), TE_vals, TR_vals))

    # i = 1
    train_im = np.array([200, 120.5, 150.4])
    W_i = np.array([120.4, 0.2, 0.5])
    # print( sum( (Bloch(np.array([1.4, 0.2, 0.5]), TE_vals, TR_vals) - train_im) ** 2) )


from joblib import Parallel, delayed

def Bloch_i(i, W, TE_vec, TR_vec):
    return (W[i, 0] * (1 - W[i, 1]**TR_vec) * W[i, 2]**TE_vec)

def predict_image(W, TE_vec, TR_vec):
    n, k = W.shape
    pred = Parallel(n_jobs=2)(
            delayed(Bloch_i)(i, W, TE_vec, TR_vec) for i in range(n))
    return pred

