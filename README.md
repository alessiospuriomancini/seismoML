# Paper number 1: a set of data science models for isotropic seismogram generation

In this repository, we show our work with 2000 isotropic sources, with the goal of developing 9 different approaches for the forward modelling of such data.

We always use 2000 data for training, 1000 for validation and quote all the results on a test set of 1000 additional seismograms.

The 9 approaches are as follows:

1) Saptarshi, as in [Das et al., 2018](https://academic.oup.com/gji/article/215/2/1257/5056164)
2) Mancini-Saptarshi  # this maybe needs to me removed?
3) AE + GP
4) PCA + GP
5) AE + NN_direct
6) PCA + NN_direct
7) CVAE
8) WGAN_GP
9) NN_direct

There should be one Jupyter notebook for each of these cases. Moreover, this repository contains the possible preprocessing steps, and the raw data obtained ith some Matlab code.
All of the other code is in Python3. 
