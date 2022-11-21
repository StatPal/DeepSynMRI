## Clean everything etc

import os
import numpy as np
import nibabel as nib
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim



## Inputs
img1 = nib.load('../data/noise-1-INU-00/brainweb_0.mnc.gz')
n_x = img1.shape[0]; n_y = img1.shape[1]; n_z = img1.shape[2]

image_vec = np.ones([n_x*n_y*n_z, 12])
train_ind = [0, 8, 9]
test_ind = np.setdiff1d(range(12), train_ind)


for i in range(3):
    img = nib.load('../data/noise-1-INU-00/brainweb_'+str(i)+'.mnc.gz')
    data = img.get_fdata()
    data_reshaped = data
    image_vec[:, train_ind[i]] = data_reshaped.reshape(-1)


for i in range(9):
    img = nib.load('../data/test-noise-0-INU-00/brainweb_'+str(i)+'.mnc.gz')
    data = img.get_fdata()
    data_reshaped = data
    image_vec[:, test_ind[i]] = data_reshaped.reshape(-1)


dat_2 = image_vec.reshape(n_x, n_y, n_z, 12)
plt.imsave("tmp.pdf", dat_2[18,:,:,1])
# plt.imsave("tmp.pdf", dat_2[:,109,:,1])
# plt.imsave("tmp.pdf", dat_2[:,:,81,1])



## SCALING: - It is not same as the previous scaling as possibly then every error had the same scaling I think 
train_scale_factor = 400 / np.max(image_vec)
image_vec = image_vec * train_scale_factor
n, m = image_vec.shape
print(np.max(image_vec, axis=0))







## Make a mask or supply the mask:
mask_all = nib.load('../data/mask/subject47_crisp_v.mnc.gz')
mask_all = mask_all.get_fdata()
mask = (mask_all == 0)
mask_reshaped = mask[5::10, 1::2, 1::2]
mask_reshaped.shape
# mask_vec = mask_reshaped.reshape((-1,1))
mask_vec = mask_reshaped.reshape(-1)
mask_vec = mask_vec[:,None]  # shape (*,1)


# dat_2 = image_vec.reshape(n_x, n_y, n_z, 12)
# plt.imsave("tmp.pdf", dat_2[18,:,:,1])
# plt.imsave("tmp.pdf", dat_2[:,109,:,1])
# plt.imsave("tmp.pdf", dat_2[:,:,81,1])

# masking = mask_vec.reshape(n_x, n_y, n_z)
# plt.imsave("tmp.pdf", masking[18,:,:])
# plt.imsave("tmp.pdf", masking[:,109,:])
# plt.imsave("tmp.pdf", masking[:,:,81])







## Other input parameters:
#                       x                                                   x     x
TE_values = np.array([0.01, 0.015,  0.02, 0.01, 0.03,  0.04, 0.01, 0.04, 0.08, 0.01, 0.06, 0.1])
TR_values = np.array([ 0.6,   0.6,   0.6,    1,     1,    1,    2,    2,    2,    3,    3,   3])
sigma_values = np.array([1.99146, 1.81265, 1.82837, 2.30221, 1.63414, 1.71876, 3.13695, 1.77141, 1.55651, 2.72191, 1.63068, 1.4359])
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



# dat_2 = train.reshape(n_x, n_y, n_z, 3)
# plt.imsave("tmp-DL.pdf", dat_2[:,:,10, 1])
# # Check:
# dat_2.reshape((-1,3)).shape
# train.shape
# np.array_equal(train, dat_2.reshape((-1,3)))

# train_3D = train.reshape(n_x, n_y, n_z, 3)
# pd.DataFrame(train_3D[:,:,18, 0]).to_csv("figs/new/DL-LS-train-noise-1-INU-00-img-0-axis-1.csv", header=None, index=None)




### DL from training images
from W_DL import DL_W

num_iter = 150
train_DL = DL_W(train, n_x, n_y, n_z, num_iter)




### LS estimates
from estimate.Bloch import *
from estimate.LS import *

print(datetime.datetime.now(), flush=True)
W_LS_par = LS_est_par(TE_train, TR_train, train_DL, TE_scale, TR_scale, mask_vec, 90)  ## 1-INU-00 - angle was not specified - spotted
pd.DataFrame(W_LS_par).to_csv("intermed/DL-W_LS_par-noise-1-INU-00.csv", header=None, index=None)
print(datetime.datetime.now(), flush=True)                  ## Takes about 18 min in my laptop
W_LS_par = pd.read_csv("intermed/DL-W_LS_par-noise-1-INU-00.csv.gz", header=None).to_numpy()

dat_2 = W_LS_par.reshape(n_x, n_y, n_z, 3)
plt.imsave("tmp-DL.pdf", dat_2[:,:,10, 1])







## Predict
LS_pred_old  = predict_image_par(W_LS_par, TE_test, TR_test, 90)
LS_pred = np.asarray(LS_pred_old)

dat_2 = LS_pred.reshape(n_x, n_y, n_z, 9)
plt.imsave("tmp-DL.pdf", dat_2[:,:,10, 1])


LS_pred[mask_vec[:,0],:] = 0
test[mask_vec[:,0],:] = 0

LS_pred_3D = LS_pred.reshape(n_x, n_y, n_z, 9)
test_3D = test.reshape(n_x, n_y, n_z, 9)

SSIM_vals = np.zeros(9)
for i in range(9):
    SSIM_vals[i] = ssim(LS_pred_3D[:,:,:,i], test_3D[:,:,:,i])

tmp_diff = abs(LS_pred - test)
print( np.mean(tmp_diff, axis=0) )
print( np.sqrt( np.mean(tmp_diff **2, axis=0) ) )
print(SSIM_vals * 100)
perf_LS = np.asarray([np.mean(tmp_diff, axis=0), np.sqrt( np.mean(tmp_diff **2, axis=0) ), SSIM_vals * 100, 
                        np.nanmean(2 * tmp_diff / (test + LS_pred), axis=0), np.sqrt( np.nanmean(2 * tmp_diff ** 2 / (test ** 2 + LS_pred ** 2), axis=0)),
                        np.mean(tmp_diff[~mask_vec[:,0],:], axis=0) / np.mean(abs(test[~mask_vec[:,0],:] - np.median(test[~mask_vec[:,0],:]))),
                        np.sqrt( np.mean((tmp_diff[~mask_vec[:,0],:])**2, axis=0) ) / np.std(test[~mask_vec[:,0],:], axis=0)])

