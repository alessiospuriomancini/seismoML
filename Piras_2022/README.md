# SeismoML: generative models for deep learning-accelerated Bayesian inference of seismic events

This is the repository for the paper [`"Towards fast machine-learning-assisted Bayesian posterior inference of microseismic event location and source mechanism"`](https://academic.oup.com/gji/advance-article-abstract/doi/10.1093/gji/ggac385/6750231?utm_source=advanceaccess&utm_campaign=gji&utm_medium=email) by Piras et al. The repository contains most material to reproduce the results of the paper, or descriptions on how to do it. If anything is unclear, or if you need access to any dataset that is not available here, do not hesitate to send an email to [Davide Piras](mailto:dr.davide.piras@gmail.com).

## Installation
We recommend creating a fresh `conda` environment, and installing all packages listed in `requirements.txt`. The work presented in the paper has been done with `python 3.6.10` (now outdated), but it should work seamlessly even with more recent versions.

    conda create -n seismoML python=3.6.10
    conda activate seismoML
    git clone https://github.com/alessiospuriomancini/seismoML.git
    cd seismoML/Piras_2022
    pip install -r requirements.txt
   
At this point you should be able to run the Jupyter notebooks. If you have trouble installing `pymultinest`, have a look at its [installation page](https://johannesbuchner.github.io/PyMultiNest/install.html).

## Content description
We provide three Jupyter notebooks, to deal with the three different source mechanisms:

- [`isotropic`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/isotropic.ipynb): contains the model training and inference for isotropic (ISO) sources. 
- [`dc`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/dc.ipynb): contains the model training and inference for double couple (DC) sources.
- [`clvd`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/clvd.ipynb): contains the model training and inference for compensated linear vector dipole (CLVD) sources.

All these analyses are done on subsets of the original data used for the paper, considering only 5 receivers and 1000 seismograms for each receiver (shared in [`data`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/data/)); we only consider synthetic Gaussian noise. We also provide the [`saved_models`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/saved_models/) folder, with the saved models trained on the entire dataset, or on this smaller version of it (indicated with the suffix `smaller`). The folder [`inference`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/inference/) contains the inference results for each source mechanism with the models trained on this smaller dataset (suffix: `smaller`), and the observation taken from it as well; change the name of the folder within the notebooks (cells 12 and 15) if you want to re-run the MCMC chains, otherwise these saved results will be retrieved to save you time. In the same folder, you will find the paper results, with the models trained on the entire dataset, and the observation taken from the larger dataset. The [`utils`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/utils.py) script contains a few utilities used in the notebooks.

Note that we do not deal with arrival time techniques here, and that we only consider synthetic Gaussian noise; for the full analysis, including realistic noise, density and velocity models, and the whole dataset of seismograms, reach out to [Davide Piras](mailto:dr.davide.piras@gmail.com).

## Contributing and contacts

Feel free to [fork](https://github.com/alessiospuriomancini/seismoML/fork) this repository to work on it; otherwise, please [raise an issue](https://github.com/alessiospuriomancini/seismoML/issues) or contact [Davide Piras](mailto:dr.davide.piras@gmail.com).

## Citation

If you work with parts of this paper or its data byproducts, please cite the corresponding paper:

     @article{Piras22,
              author = {Piras, D and Mancini, A Spurio and Ferreira, A M G and Joachimi, B and Hobson, M P},
              title = "{Towards fast machine-learning-assisted Bayesian posterior inference of microseismic event location and source mechanism}",
              journal = {Geophysical Journal International},
              year = {2022},
              month = {10},
              issn = {0956-540X},
              doi = {10.1093/gji/ggac385},
              url = {https://doi.org/10.1093/gji/ggac385},
              note = {ggac385},
              eprint = {https://academic.oup.com/gji/advance-article-pdf/doi/10.1093/gji/ggac385/46356491/ggac385.pdf},
             }


