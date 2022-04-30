## Training real image cases

import os
import numpy as np
import nibabel as nib
import pandas as pd
import datetime
from matplotlib import pyplot as plt

np.set_printoptions(precision=6, suppress=True)
print(datetime.datetime.now(), flush=True)


phantom = nib.load("../data/ZHRTS2.nii").get_fdata().transpose([2,1,0,3])
image_vec = phantom.reshape((256*256*20,-1))  ## Check
image_vec.shape

# phantom[0,0,0,0:12] = np.array([0.01, 0.015,  0.02, 0.01, 0.03,  0.04, 0.01, 0.04, 0.08, 0.01, 0.06, 0.1])
# image_vec = phantom.reshape((256*256*20,-1))  ## Check
# phantom[0,0,0,0:12]
# image_vec[0,0:12]

image_vec[image_vec == 0.0] = 0.5




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


mask_vec = np.ones(256*256*20) - ( np.sum(image_vec > 50, axis=1) > 0)
mask_vec = mask_vec.astype(int)

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



train_already_saved = pd.read_csv("../whole_new/real_new/train_mask.csv", header=None).to_numpy()




train[900,0:12]
train_already_saved[900,0:12]

train[1910,0:12]
train_already_saved[1910,0:12]

## I guess working
n_x = 256; n_y = 256; n_z = 20;

for i in range(3):
    tmp = (pd.read_csv("../whole_new/real_new/3D/intermed/intermed_"+str(i)+"_noisy_train-seed-1.csv.gz", header=None).to_numpy())[:,0]
    train_2 = tmp.reshape(n_z, n_y, n_x)
    # train_2.shape
    # plt.imsave("tmp-DL.pdf", train_2[:,:,10])
    train_already_saved[:,i] = train_2.transpose([0,2,1]).reshape(-1)

# dat_2 = train_already_saved.reshape(n_z, n_y, n_x, 3)
# dat_2.shape
# plt.imsave("tmp-DL.pdf", dat_2[10,:,:, 1])
# plt.imsave("tmp-DL.pdf", dat_2[:,:,128, 1])


# dat_2 = train.reshape(n_z, n_y, n_x, 3)
# plt.imsave("tmp.pdf", dat_2[10,:,:, 1])
# plt.imsave("tmp.pdf", dat_2[:,:,128, 1])




### LS and MLE estimates
from estimate.Bloch import *
from estimate.LS import *

print(datetime.datetime.now(), flush=True)
# W_LS_par = LS_est_par(TE_train, TR_train, train, TE_scale, TR_scale, mask_vec, 90)  ## BUG - angle was not specified - spotted
# pd.DataFrame(W_LS_par).to_csv("intermed/real_W_LS_par.csv", header=None, index=None)
print(datetime.datetime.now(), flush=True)                  ## Takes about 40 min in my laptop
W_LS_par = pd.read_csv("intermed/real_W_LS_par.csv", header=None).to_numpy()


# dat_2 = W_LS_par.reshape(n_x, n_y, n_z, 3)
# plt.imsave("tmp.pdf", dat_2[:,:,10, 1])




## Predict
LS_pred_old  = predict_image_par(W_LS_par, TE_test, TR_test, 90)  ## BUG 2: There would be angle too
LS_pred = np.asarray(LS_pred_old)

# dat_2 = LS_pred.reshape(n_x, n_y, n_z, 9)
# plt.imsave("tmp.pdf", dat_2[:,:,10, 1])



LS_pred[mask_vec,:] = 0
test[mask_vec,:] = 0


tmp_diff = abs(LS_pred - test)
print('\n\nLS\n\n')
print( np.mean(tmp_diff, axis=0) )
print( np.mean(tmp_diff) )
print( np.mean(tmp_diff ** 2, axis=0) )
print( np.nanmean(2 * tmp_diff / (test + LS_pred), axis=0) )









### MLE estimates
from estimate.Bloch import *
from estimate.MLE import *

# print(datetime.datetime.now(), flush=True)
# W_MLE_par = MLE_est_par(W_LS_par, TE_train, TR_train, train, TE_scale, TR_scale, sigma_train, mask_vec)
# pd.DataFrame(W_MLE_par).to_csv("intermed/real_W_MLE_par.csv", header=None, index=None)
# print(datetime.datetime.now(), flush=True)
W_MLE_par = pd.read_csv("intermed/real_W_MLE_par.csv", header=None).to_numpy()

## Predict
MLE_pred_old  = predict_image_par(W_MLE_par, TE_test, TR_test, 90)  ## BUG 2: There would be angle too
MLE_pred = np.asarray(MLE_pred_old)

MLE_pred[mask_vec,:] = 0
test[mask_vec,:] = 0

tmp_diff = abs(MLE_pred - test)
print('\n\nMLE\n\n')
print( np.mean(tmp_diff, axis=0) )
print( np.mean(tmp_diff) )
print( np.mean(tmp_diff ** 2, axis=0) )
print( np.nanmean(2 * tmp_diff / (test + MLE_pred), axis=0) )














print(datetime.datetime.now(), flush=True)
# W_LS_par = LS_est_par(TE_train, TR_train, train_already_saved, TE_scale, TR_scale, mask_vec, 90)  ## BUG - angle was not specified - spotted
# pd.DataFrame(W_LS_par).to_csv("intermed/real_W_LS_par-DL.csv", header=None, index=None)
# print(datetime.datetime.now(), flush=True)                  ## Takes about 40 min in my laptop
W_LS_par = pd.read_csv("intermed/real_W_LS_par-DL.csv", header=None).to_numpy()

## Predict
LS_pred_old  = predict_image_par(W_LS_par, TE_test, TR_test, 90)  ## BUG 2: There would be angle too
LS_pred = np.asarray(LS_pred_old)

LS_pred[mask_vec,:] = 0
test[mask_vec,:] = 0


tmp_diff = abs(LS_pred - test)
print('\n\nDL LS\n\n')
print( np.mean(tmp_diff, axis=0) )
print( np.mean(tmp_diff) )
print( np.mean(tmp_diff ** 2, axis=0) )
print( np.nanmean(2 * tmp_diff / (test + LS_pred), axis=0) )


