# DeepSynMRI

This is a package which uses deep learning method, Deep Image Prior, to enahance Synthetic MRI, especially effective for noisy MRI. 


The environment created by:
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

For GPU
```bash
conda create --name DeepSynMRI
conda activate DeepSynMRI
conda install numpy
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install matplotlib
conda install scikit-image
conda install nibabel -c conda-forge
conda install joblib
conda install pandas
```



```bash
conda activate DeepSynMRI
```