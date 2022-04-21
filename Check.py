
import os
import numpy as np
import nibabel as nib
import pandas as pd
import datetime
from matplotlib import pyplot as plt

np.set_printoptions(precision=6, suppress=True)


from estimate.Bloch import *



## Checking bloch
TE_vals = np.array([0.5, 0.45, 0.4, 0.35])
TR_vals = np.array([0.3, 0.25, 0.2, 0.24])
W_i = np.array([120.4, 0.2, 0.5])

train_im = np.array([200, 150.4, 175, 220])

print(Bloch(W_i, 0.5, 0.3))
print(Bloch(W_i, 0.4, 0.35))
print(Bloch(W_i, TE_vals, TR_vals))
# print(Bloch(W_i, TE_vals, TR_vals, 40))

# print( sum( (Bloch(W_i, TE_vals, TR_vals, 20) - train_im) ** 2) )
print( sum( (Bloch(W_i, TE_vals, TR_vals) - train_im) ** 2) )



## LS: 
from estimate.LS import *

train_im = np.transpose(train_im[:,None])
mask_vec = np.array([0])

print( LS_est(TE_vals, TR_vals, train_im, 1, 1, mask_vec, angle=90) )



## Performance:

print('\n\n\n Many voxels\n\n')

W = np.array([[50, 0.01, 0.003], [36, 0.02, 0.04], [56, 0.02, 0.04], [106, 0.2, 0.04]])
# W = np.array([[50, 0.01, 0.003], [36, 0.02, 0.04]])
TE = np.array([0.01, 0.03, 0.04, 0.01])
TR = np.array([0.6, 0.6, 1, 0.8])
print(W)


# print( predict_image(W, TE, TR) )
print( predict_image_par(W, TE, TR))  ## Same





## Now check more voxels:
train_new = predict_image(W, TE, TR)

sigma_val = np.array([0,0,0,0])
mask_vec = np.array([0,0,0,0])
est_W = np.asarray( LS_est_par(TE, TR, train_new, 1, 1, mask_vec, angle=90) )

print(est_W)


## Predict again from estimated:

print( predict_image(est_W, TE, TR) )

print('\n\n\n Measure\n\n')
print(   np.mean( abs( predict_image(est_W, TE, TR) - train_new) , axis=1)  )
