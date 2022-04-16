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
mask_all.shape
mask = (mask_all == 0)

mask_reshaped_mid = mask.transpose([2,1,0])
# plt.imsave("tmp.pdf", mask_reshaped_mid[:,:,181])
mask_reshaped = mask_reshaped_mid[1::2, 1::2, 1::10]
mask_reshaped.shape   # Matches R size


image_vec = np.ones([181*217*36, 12])
train_ind = [0, 8, 9]
test_ind = np.setdiff1d(range(12), train_ind)


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

print(image_vec.shape)
image_vec = image_vec * 400 / np.max(image_vec)
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



## Divide into train and test with 3 train images:
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



dat_iter = train
print(dat_iter.shape)
n_x = 181; n_y = 217; n_z = 36
dat_2 = dat_iter.reshape(n_x, n_y, n_z, 3)
plt.imsave("tmp.pdf", dat_2[:,:,18,1])
# plt.imsave("tmp.pdf", dat_2[:,100,:,1])
# plt.imsave("tmp.pdf", dat_2[90,:,:,1])

# # dat_2 = dat_iter.reshape(n_z, n_y, n_x, 3)
# # plt.imsave("tmp.pdf", dat_2[:,:,18,1])




### DL from parametric maps
from W_DL import DL_W

num_iter = 4
W_DL_LS = DL_W(train, [36, 217, 181, 3], [1, 2, 0, 3], 181, 217, 36, num_iter)

pd.DataFrame(W_DL_LS).to_csv("intermed/W_DL_LS.csv.gz", header=None, index=None)

print(W_DL_LS.shape)

dat_2 = W_DL_LS.reshape(n_x, n_y, n_z, 3)
plt.imsave("tmp.pdf", dat_2[:,:,18,1])
