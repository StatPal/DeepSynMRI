# DeepSynMRI: Personalized synthetic MR imaging with deep learning enhancements

This is a python package for deep learning ehancements to synthetic Magnetic Resonance Imaging (MRI) via a Deep Image Prior, that is especially effective in noisier situations. The repository provides the code for the implementation in the paper ["Personalized synthetic MR imaging with deep learning enhancements" by Pal, Dutta and Maitra, Magnetic Resonance in Medicine, 2022](http://doi.org/10.1002/mrm.29527).

### Workflow
The main workflow is:
The training images in 3 (or more) settings is DL enhanced first using the fucntion `DL_smooth_3`. Then LS estimate of the underlying parametric maps (W, reparametrized rho, T1, T2) is calculated using the function `LS_est_par`. And then `predict_image_par` uses the `W` to predict MR images in new settings.  

An complete example is provided with example in the file `SE_example_DL.py`. 
The dataset is in the folder `data` with three sub-folders, one sub-folder with the mask, one sub-folder with the three training volumetric images, and one sub-folder with nine test volumetric images. 

### Python Environment
The environment is created by conda with commands:
```bash
conda create --name DeepSynMRI
conda activate DeepSynMRI
conda install numpy
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib
conda install scikit-image
conda install nibabel -c conda-forge
conda install joblib
conda install pandas
```
For GPU the pytorch line should be replaced by `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

The environment should be first activated by something like
```bash
conda activate DeepSynMRI
```


## Details
All the images are vectorized and the training images are positioned in 0th, 8th and 9th position of a whole training and testing matrix named `image_vec`. From this matrix, we do subsequent calculations. All the images are scaled so that the maximmmum value is ~400. 

<!-- TE values are `0.01, 0.015,  0.02, 0.01, 0.03,  0.04, 0.01, 0.04, 0.08, 0.01, 0.06, 0.1`
TR values are `0.6,   0.6,   0.6,    1,     1,    1,    2,    2,    2,    3,    3,   3` -->


### The functions:
The three main functions for a bare minimum of the whole process are: 
- The main DL fuinction used here is, `DL_smooth_3(train, n_x, n_y, n_z, iter_num)`. Here `train` is the training data of size `(n, 3)` where `n = n_x * n_y * n_z` and `n_x, n_y, n_z` are the size of each 3D original MR images, and `iter_num` is the iteration number of the DL method for each image. This can be imported as `from DL import DL_smooth_3`. (This function can be also used to smooth the `W` images after they are estimated instead of smoothing the training images first.) 

- `LS_est_par(TE_train, TR_train, train, TE_scale, TR_scale, mask_vec, flip_angle)` is the function to do the least square estimate of `W`, i.e., the paramteric maps of rho, T1, T2 (reparametrized, see the paper), which is the output of size `n x m`, where m is the number of training images (usually `m = 3`). `TE_train` and `TR_train` are the TE and TR values of the corresponnding training images. `mask_vec` is the mask vector of size `(n, 1)` where 1 means the voxel is masked. `flip_angle` is the flip angle in degrees, which is 90 for Spin Echo acquisitions(default). `TE_scale` and `TR_scale` should be `1` if the TE and TR values are not scaled from the original values[^1].

- `predict_image_par(W, TE_test, TR_test, flip_angle)` is the function to get the predicted synthetic image from the paramteric maps `W` with `TE_test` and `TR_test` as the TE and TR values for the test settings and `flip_angle` as the flip angle of the test images. 

Last two functions can be called as: 
```py
from estimate.Bloch import *
from estimate.LS import *
```


Other functions include: 
- `MLE_est_par(W_init, TE_vec, TR_vec, train_mat, TE_scale, TR_scale, sigma_train, mask)` which calculates the MLE for the `W` according to Rice distribution instead of using Least Square method (which assumes simplified Normal distribution). `W_init` is a initial value of `W` provided for this process. `sigma_train` is an estimate of the standard deviation parameters, usually estimated using some other means. 
- `DL_single(train, n_x, n_y, n_z, iter_num)` does a similar thing as `DL_smooth_3` but with one image at a time. 
- `DL_smooth_m(train, n_x, n_y, n_z, m, iter_num)` does a similar thing as `DL_smooth_3` but with `m` image at a time. This can be used to smooth the images after they are predicted. 

Most of the functions are paralleized. 

[^1]: (These are the scale by which all the original TE and TR values are multiplied first. This was done to mitigate some numerical problems for calculation of the gradient and hessian corresponding to the Bloch equation, so that the minimum TE/TR values become 2.01. However, for this current implementation, this scaling is not needed.)

### Structure of the package
The structure of the whole package is as follows:
```{bash}
.
|-- Check
|   |-- Check.py
|   |-- Check.R
|   |-- compare_rho_etc.R
|   |-- SE_real.py
|   `-- test.py
|-- data
|   |-- mask
|   |   `-- subject47_crisp_v.mnc.gz
|   |-- noise-1-INU-00
|   |   |-- brainweb_0.mnc.gz
|   |   |-- brainweb_1.mnc.gz
|   |   `-- brainweb_2.mnc.gz
|   `-- test-noise-0-INU-00
|       |-- brainweb_0.mnc.gz
|       |-- brainweb_1.mnc.gz
|       |-- brainweb_2.mnc.gz
|       |-- brainweb_3.mnc.gz
|       |-- brainweb_4.mnc.gz
|       |-- brainweb_5.mnc.gz
|       |-- brainweb_6.mnc.gz
|       |-- brainweb_7.mnc.gz
|       `-- brainweb_8.mnc.gz
|-- DL.py
|-- estimate
|   |-- Bloch.py
|   |-- LS.py
|   `-- MLE.py
|-- intermed
|-- LICENSE
|-- models
|   |-- common.py
|   |-- downsampler.py
|   |-- __init__.py
|   `-- skip.py
|-- pred_DL.py
|-- README.md
|-- SE_example_DL.py
`-- utils
    |-- common_utils.py
    |-- denoising_utils.py
    `-- __init__.py
```

### Notes
If you think you have cuda enabled GPU which is large enough so that it can accomodate the large 3D network, you can change the lines as follows:
`dtype = torch.FloatTensor` to `dtype = torch.cuda.FloatTensor` and 
`net_param.data.copy_(new_param.cpu())` to `net_param.data.copy_(new_param.cuda())`
in the `DL.py` file.


### Citation
You can create a citation file from `Cite this repository` at the right side of the online github repository. Or you can copy from the following.
#### APA
```
Pal, S., Dutta, S., & Maitra, R. Personalized synthetic MR imaging with deep learning enhancements. Magnetic Resonance in Medicine. https://doi.org/10.1002/mrm.29527
```
#### BibTeX
```
@article{https://doi.org/10.1002/mrm.29527,
    author = {Pal, Subrata and Dutta, Somak and Maitra, Ranjan},
    title = {Personalized synthetic MR imaging with deep learning enhancements},
    journal = {Magnetic Resonance in Medicine},
    doi = {https://doi.org/10.1002/mrm.29527},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29527},
    eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/mrm.29527}
}
```
