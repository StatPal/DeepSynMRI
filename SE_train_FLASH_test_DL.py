## Training flash images - test images - do later
# all train angle 90

import os
import numpy as np
import nibabel as nib
import pandas as pd
import datetime
from matplotlib import pyplot as plt

np.set_printoptions(precision=6, suppress=True)

print(datetime.datetime.now(), flush=True)




## Make a mask or supply the mask:
mask_all = nib.load('../data/mask/subject47_crisp_v.mnc.gz')
mask_all = mask_all.get_fdata()
mask = (mask_all == 0)

mask_reshaped_mid = mask.transpose([2,1,0])
# mask_reshaped = mask_reshaped_mid[1::2, 1::2, 1::10]
# mask_reshaped = mask_reshaped_mid[1::2, 1::2, 9::10]   ## Here 4-5 would be a better option
mask_reshaped = mask_reshaped_mid[1::2, 1::2, 5::10]
mask_reshaped.shape   # Matches R size
mask_vec = mask_reshaped.reshape((-1,1))

n_x = 181; n_y = 217; n_z = 36
dat_2 = mask_reshaped.reshape(n_x, n_y, n_z)
plt.imsave("tmp.pdf", dat_2[:,:,10])


image_vec = np.ones([181*217*36, 12])
train_ind = [0, 8, 9]
test_ind = np.setdiff1d(range(12), train_ind)


data = pd.read_csv('../whole_new/LS_with_deep_slices/train_noisy-5-INU-00.csv.gz', header=None).to_numpy()
tmp = data[:,0].reshape(n_z, n_y, n_x).transpose([2,1,0]).reshape(-1)
img = nib.load('../data/noise-5-INU-00/brainweb_0.mnc.gz')
data = img.get_fdata()
tmp2 = data.transpose([2,1,0]).reshape(-1)
train_scale_factor = np.max(tmp)/np.max(tmp2)
print(train_scale_factor)  ## Needed for DL






for i in range(3):     ## BUG - was range(2)
    img = nib.load('../data/noise-5-INU-00/brainweb_'+str(i)+'.mnc.gz')
    data = img.get_fdata()
    data_reshaped = data.transpose([2,1,0])
    print(train_ind[i])
    image_vec[:, train_ind[i]] = data_reshaped.reshape(-1)

dat_2 = image_vec.reshape(n_x, n_y, n_z, 12)
plt.imsave("tmp-DL.pdf", dat_2[:,:,10, 0])

for i in range(9):
    img = nib.load('../data/FLASH-test-noise-0-INU-00/brainweb_'+str(i)+'.mnc.gz')
    data = img.get_fdata()
    data_reshaped = data.transpose([2,1,0])
    image_vec[:, test_ind[i]] = data_reshaped.reshape(-1)
print(np.max(image_vec, axis=0))

image_vec = image_vec * train_scale_factor
n, m = image_vec.shape
print(np.max(image_vec, axis=0))


## DL images: 
for i in range(3):
    data = pd.read_csv('../whole_new/LS_with_deep_slices/3D/intermed/train_noisy-5-INU-00.csv.gzintermed_'+str(i)+'_noisy_train-seed-1.csv.gz', header=None).to_numpy()
    data_2 = data[:,0].reshape(n_z, n_y, n_x)
    data_reshaped = data_2.transpose([2,1,0])
    image_vec[:, train_ind[i]] = data_reshaped.reshape(-1)

print(np.max(image_vec, axis=0))



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



dat_2 = train.reshape(n_x, n_y, n_z, 3)
plt.imsave("tmp-DL.pdf", dat_2[:,:,10, 1])
# Check:
dat_2.reshape((-1,3)).shape
train.shape
np.array_equal(train, dat_2.reshape((-1,3)))




### LS and MLE estimates
from estimate.Bloch import *
from estimate.LS import *

# print(datetime.datetime.now(), flush=True)
# W_LS_par = LS_est_par(TE_train, TR_train, train, TE_scale, TR_scale, mask_vec, 90)  ## BUG - angle was not specified - spotted
# pd.DataFrame(W_LS_par).to_csv("intermed/DL-W_LS_par-INU-00.csv", header=None, index=None)
print(datetime.datetime.now(), flush=True)                  ## Takes about 18 min in my laptop
W_LS_par = pd.read_csv("intermed/DL-W_LS_par-INU-00.csv", header=None).to_numpy()


dat_2 = W_LS_par.reshape(n_x, n_y, n_z, 3)
plt.imsave("tmp-DL.pdf", dat_2[:,:,10, 1])




## Predict
LS_pred_old  = predict_image_par(W_LS_par, TE_test, TR_test, 15)  ## BUG 2: There would be angle too
LS_pred = np.asarray(LS_pred_old)

dat_2 = LS_pred.reshape(n_x, n_y, n_z, 9)
plt.imsave("tmp-DL.pdf", dat_2[:,:,10, 1])



LS_pred[mask_vec[:,0],:] = 0
test[mask_vec[:,0],:] = 0


tmp_diff = abs(LS_pred - test)
print( np.mean(tmp_diff, axis=0) )
print( np.nanmean(2 * tmp_diff / (test + LS_pred), axis=0) )

