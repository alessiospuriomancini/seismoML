# SeismoML: generative models for deep learning-accelerated Bayesian inference of seismic events

This is the repository for the paper [`"Accelerating Bayesian microseismic event location with deep learning"`](https://se.copernicus.org/articles/12/1683/2021/)` by A. Spurio Mancini et al. The repository is a collection of deep generative models for hyper-fast generation of syntethic microseismic traces given their coordinates within a 3D simulated domain. The generative models described in the paper are:

1) `NN_direct`: direct emulation of the seismic traces from the coordinates using a Neural Network (NN)
2) `PCA + NN`: NN emulation of Principal Components of the seismic traces retained from Principal Component Analysis (PCA) 
3) `PCA + GP`: emulation via Gaussian Processes of the seismic traces PCA components
4) `AE + NN`: NN emulation of the encoded features of the seismic traces through an AutoEncoder (AE)
5) `AE + GP`: GP emulation of the AE encoded features
6) `CVAE`: Conditional Variational AutoeEncoder
7) `WGAN_GP`: Wasserstein Generative Adversarial Network - Gradient Penalty
8) `Das18`: a GP - based emulation framework following [Das et al., 2018](https://academic.oup.com/gji/article/215/2/1257/5056164)

There should be one Jupyter notebook for each of these cases. Moreover, this repository contains the possible preprocessing steps, and the raw data obtained ith some Matlab code.
All of the other code is in Python3. 


# Installation

Create your own virtual environment with `conda` and `pip install` the required libraries.

    conda create -n yourenvname
    pip install -r requirements.txt


# Training

Each generative model is implemented in a Jupyter notebook which can be run to train the deep learning framework. 

# Saved models

The models used for the paper `"Accelerating Bayesian microseismic event location with deep learning"` can be found in their respective `saved_models_iso_{name of generative model}` folder. The generation of seismic traces used for training and testing can be repeated starting from the 3D density and velocity model stored in `mar_mdl1_segment1.mat`. We include the seismic traces used for training and testing in the paper, contained in the file `seismograms_4000seismo_ISO.npy`, along with their respective coordinates on the simulated 3D grid, stored in `coordinates_4000seismo_ISO.npy`.

# Citation

If you use byproducts of this analysis, please cite the relevant paper:
   
  @Article{SpurioMancini211,
  AUTHOR = {Spurio Mancini, A. and Piras, D. and Ferreira, A. M. G. and Hobson, M. P. and Joachimi, B.},
  TITLE = {Accelerating Bayesian microseismic event location with deep learning},
  JOURNAL = {Solid Earth},
  VOLUME = {12},
  YEAR = {2021},
  NUMBER = {7},
  PAGES = {1683--1705},
  URL = {https://se.copernicus.org/articles/12/1683/2021/},
  DOI = {10.5194/se-12-1683-2021}
  }
