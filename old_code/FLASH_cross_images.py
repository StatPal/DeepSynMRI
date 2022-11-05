## Two flash images - test images
## 3000, 100    - angle 15
## 600, 20      - angle 15



import os
import numpy as np
import nibabel as nib
import pandas as pd
import datetime
from matplotlib import pyplot as plt


print(datetime.datetime.now(), flush=True)




## Make a mask or supply the mask:
mask_all = nib.load('../data/mask/subject47_crisp_v.mnc.gz')
mask_all = mask_all.get_fdata()
mask = (mask_all == 0)

mask_reshaped_mid = mask.transpose([2,1,0])
mask_reshaped = mask_reshaped_mid[1::2, 1::2, 1::10]
mask_reshaped.shape   # Matches R size

image_vec = np.ones([181*217*36, 12+2])
train_ind = [0, 8, 9]
test_ind = np.setdiff1d(range(12+2), train_ind)
FLASH_ind = [12,13]

for i in range(2):
    img = nib.load('../data/noise-5-INU-20/brainweb_'+str(i)+'.mnc.gz')
    data = img.get_fdata()
    data_reshaped = data.transpose([2,1,0])
    image_vec[:, train_ind[i]] = data_reshaped.reshape(-1)

for i in range(8):
    img = nib.load('../data/test-noise-0-check/brainweb_'+str(i)+'.mnc.gz')
    data = img.get_fdata()
    data_reshaped = data.transpose([2,1,0])
    image_vec[:, test_ind[i]] = data_reshaped.reshape(-1)

## FLASH cases:
for i in range(2):
    img = nib.load('../data/FLASH/brainweb_'+str(i)+'.mnc.gz')
    data = img.get_fdata()
    data_reshaped = data.transpose([2,1,0])
    image_vec[:, FLASH_ind[i]] = data_reshaped.reshape(-1)

np.mean(image_vec, axis=0)
np.max(image_vec, axis=0)

# print(dat_iter.shape)
# n_x = 181; n_y = 217; n_z = 36
# dat_2 = dat_iter.reshape(n_x, n_y, n_z, 12+2)
# plt.imsave("tmp.pdf", dat_2[:,:,18,1])




scale_value = 400 / np.max(image_vec[:,range(12)]);
image_vec = image_vec * 400 / np.max(image_vec[:,range(12)])
n, m = image_vec.shape



## Other input parameters:
TE_values = np.array([0.01, 0.015, 0.02, 0.01, 0.03, 0.04, 0.01, 0.04, 0.08, 0.01, 0.06, 0.1,    0.1, 0.02])
TR_values = np.array([0.6, 0.6, 0.6, 1, 1, 1, 2, 2, 2, 3, 3, 3,     3, 0.6])
sigma_values = np.array([1.99146, 1.81265, 1.82837, 2.30221, 1.63414, 1.71876, 3.13695, 1.77141, 1.55651, 2.72191, 1.63068, 1.4359, 0, 0])
min_TE = min(TE_values[range(12)])
min_TR = min(TR_values[range(12)])
TE_scale = 2.01 / min_TE
TR_scale = 2.01 / min_TR
r_scale = 1.0
TE_values = TE_values * TE_scale
TR_values = TR_values * TR_scale
image_vec = image_vec / r_scale
sigma_values = sigma_values / r_scale


## Divide into train and test with 3 train images:
train = image_vec[:, train_ind]
sigma_train = sigma_values[train_ind]
TE_train = TE_values[train_ind]
TR_train = TR_values[train_ind]
test = image_vec[:, test_ind]
sigma_test = sigma_values[test_ind]
TE_test = TE_values[test_ind]
TR_test = TR_values[test_ind]



n_x = 181; n_y = 217; n_z = 36



### LS and MLE estimates

from estimate.Bloch import *
from estimate.LS import *

W_LS_par = pd.read_csv("intermed/W_LS_par.csv", header=None).to_numpy()



## Predict
LS_pred_old  = predict_image_par(W_LS_par, TE_test[range(9)], TR_test[range(9)])
LS_pred = np.asarray(LS_pred_old)
tmp_diff = abs(LS_pred - test[:,0:9])
print( np.mean(tmp_diff, axis=0) )
print( np.mean(tmp_diff / (test[:,0:9] + LS_pred), axis=0) )


LS_pred_FLASH_old = predict_image_par(W_LS_par, TE_test[9:11], TR_test[9:11], angle=15)
LS_pred_FLASH = np.asarray(LS_pred_FLASH_old)
tmp_diff_FLASH = abs(LS_pred_FLASH - test[:,9:11])
np.mean(tmp_diff_FLASH, axis=0)
print( np.mean(tmp_diff_FLASH / (test[:,9:11] + LS_pred_FLASH), axis=0) )


## What if just the angle was assumed to be 90??
np.mean(  abs(LS_pred[:,[1,8]] - test[:,9:11]) , axis=0) 
np.mean(  abs(LS_pred[:,[1,8]] - test[:,9:11]) / (LS_pred[:,[1,8]] + test[:,9:11]) , axis=0) 

# np.mean(test, axis=0)
# np.mean(LS_pred, axis=0)