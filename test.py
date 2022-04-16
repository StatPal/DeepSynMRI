import os
import numpy as np
import nibabel as nib
import pandas as pd
import datetime


print(datetime.datetime.now(), flush=True)


from nibabel.testing import data_path
img = nib.load("../data/ZHRTS2.nii")    ## Load data file
print(img.shape)
print(datetime.datetime.now(), flush=True)


img_new = img.get_fdata()
print(img_new.shape)


image_vec = np.ones([256*256*20, 12])
for i in range(12):
    image_vec[:, i] = img_new[:,:,:,i].transpose([2,1,0]).reshape(-1)  ## Transpose to match with shape of R 

print(image_vec.shape)


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

from estimate.LS import *

# pd.DataFrame(LS_pred).to_csv("intermed/LS_pred.csv", header=None, index=None)

LS_pred = pd.read_csv("intermed/LS_pred.csv", header=None);

(abs(LS_pred - test)).mean(axis=0)

this_bnd = np.asarray([2.432390, 2.549109, 4.123066, 3.326344, 3.260466, 10.301354, 3.957978, 4.030147, 3.509603])
other_bnd = np.asarray([2.432421, 2.549141, 4.123078, 3.326372, 3.260465, 10.301330, 3.957997, 4.030103, 3.509567])

sum(other_bnd / this_bnd)/9
