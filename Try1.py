import os
import numpy as np
import nibabel as nib


from nibabel.testing import data_path
# example_filename = os.path.join(data_path, 'example4d.nii.gz')
# example_filename = os.path.join(data_path, 'example4d.nii.gz')
# img = nib.load(example_filename)
# img_new = img.get_fdata()
# # print(img_new[1,])


img = nib.load("../data/ZHRTS2_small.nii")
print(img.shape)



img_new = img.get_fdata()
print(img_new.shape)
#print(img_new[127,127,0,:])

#print(img_new[127:130,127:130,0:2,0])

# print(img_new[27:30,27:30,0,0])
# print("\n")
# print(img_new[27:30,27:30,0:2,0])
# print("\n")
# print(img_new[27:30,27:30,0:2,0].transpose([0,2,1]))
# #print(img_new[127:130,127:130,0:2,0].shape)
# print(img_new[27:30,27:30,0:2,0].transpose([2,1,0]).reshape(-1))


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









from estimate.Bloch import *

W = LS_est(TE_train, TR_train, train, TE_scale, TR_scale)

print(W[0:10,])