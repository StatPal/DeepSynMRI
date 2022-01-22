import os
import numpy as np
import nibabel as nib
import pandas as pd
import datetime


print(datetime.datetime.now(), flush=True)


from nibabel.testing import data_path
img = nib.load("../data/ZHRTS2.nii")
print(img.shape)
print(datetime.datetime.now(), flush=True)


img_new = img.get_fdata()
print(img_new.shape)


image_vec = np.ones([256*256*20, 12])
for i in range(12):
    image_vec[:, i] = img_new[:,:,:,i].transpose([2,1,0]).reshape(-1)  ## Transpose to match with that of R 

print(image_vec.shape)
print(image_vec[1200,:])


image_vec[image_vec == 0.0] = 0.5 ## Pre-processing to remove the -Inf issue in likelihood.
n, m = image_vec.shape

## Other input parameters:
TE_values = np.array([0.01, 0.015, 0.02, 0.01, 0.03, 0.04, 0.01, 0.04, 0.08, 0.01, 0.06, 0.1])
TR_values = np.array([0.6, 0.6, 0.6, 1, 1, 1, 2, 2, 2, 3, 3, 3])
sigma_values = np.array([1.99146, 1.81265, 1.82837, 2.30221, 1.63414, 1.71876, 3.13695, 1.77141, 1.55651, 2.72191, 1.63068, 1.4359])
TE_scale = 2.01 / min(TE_values)
TR_scale = 2.01 / min(TR_values)
r_scale = 1.0
TE_values = TE_values * TE_scale
TR_values = TR_values * TR_scale
image_vec = image_vec / r_scale
sigma_values = sigma_values / r_scale

## Make a mask or supply the mask:
mask = 1 - (np.sum(image_vec > 50, axis=1) > 0)


## Divide into train and test with 3 train images:
train_ind = [0, 8, 9]
test_ind = np.setdiff1d(range(12), train_ind)
train = image_vec[:, train_ind]
sigma_train = sigma_values[train_ind]
TE_train = TE_values[train_ind]
TR_train = TR_values[train_ind]
test = image_vec[:, test_ind]
sigma_test = sigma_values[test_ind]
TE_test = TE_values[test_ind]
TR_test = TR_values[test_ind]






### LS and MLE estimates


from estimate.Bloch import *

print(datetime.datetime.now(), flush=True)
# W_LS_par = LS_est_par(TE_train, TR_train, train, TE_scale, TR_scale)
# print(datetime.datetime.now(), flush=True)
# pd.DataFrame(W_LS_par).to_csv("intermed/W_LS_par.csv", header=None, index=None)
# # print(W_LS_par[0:10,])

W_LS_par = pd.read_csv("intermed/W_LS_par.csv", header=None).to_numpy()



from estimate.Bloch_MLE import *
# print(datetime.datetime.now(), flush=True)
# W_MLE_par = MLE_est_par(W_LS_par, TE_train, TR_train, train, TE_scale, TR_scale, sigma_train, mask)
# print(datetime.datetime.now(), flush=True)

# pd.DataFrame(W_MLE_par).to_csv("intermed/W_MLE_par.csv", header=None, index=None)


W_MLE_par = pd.read_csv("intermed/W_MLE_par.csv", header=None).to_numpy()



### DL from W
from W_DL import DL_W
DL_W(W_LS_par, [20, 256, 256, 3], [1, 2, 0, 3], 256, 256, 20, 5000)

