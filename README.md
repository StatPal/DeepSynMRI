# DeepSynMRI

Created by:
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



```
conda activate DeepSynMRI

conda activate ~/anaconda3/envs/DeepSynMRI
```