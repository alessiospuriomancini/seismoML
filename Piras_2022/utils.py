import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def preprocess_coord(y, coords_receiver, split, test_valid, std=False):
    """ Preprocess the coordinates. This includes adding the 4th coordinate, 
    and offsetting by the position of the receivers.
    
    Parameters
    ----------
    y : array-like
        The coordinates; first dimension should represent the number of samples.
    coords_receiver : array-like
        The coordinates of the receivers.
    split : int
        The point at which training and validation+testing data are separated.
    std : bool, default='False'
        Whether the coordinates should be standardised or not.

    Returns
    -------
    new_coords : array-like
        The entire dataset (training and not) of preprocessed coordinates.
    """    
    N = y.shape[0]
    shifted = y - coords_receiver
    distances = np.linalg.norm(shifted, axis = 1)
    new_coords = np.zeros((N,4))
    new_coords[:, :3] = shifted
    new_coords[:, -1] = distances
    training_coords = new_coords[:split]
    validation_coords = new_coords[split:split+test_valid]
    testing_coords = new_coords[split+test_valid:]
    if std:
        coord_mean, coord_std = np.mean(training_coords, axis=0), np.std(training_coords, axis=0)
        print('Mean: {}, std dev: {}'.format(coord_mean, coord_std))
        new_coords_std = (new_coords - coord_mean) / coord_std
        new_coords = new_coords_std
    return new_coords


def preprocess_power(x, split, log=False, std=False):
    """ Preprocess the power spectra.
    
    Parameters
    ----------
    power_spectra : array-like
        The power spectra; first dimension should represent the number of samples.
    split : int
        The point at which training and validation+testing data are separated.
        Only training data are considered.
    log : bool, default='False'
        Whether the decimal logarithm of the power spectra should be considered or not.
    std : bool, default='False'
        Whether the power spectra should be standardised or not.

    Returns
    -------
    preprocessed_spectra : array-like
        The entire dataset (training and not) of preprocessed spectra.
    """
    train_power = x[:split]
    if log:
        log_power = np.log10(x)
        if std:
            train_log_power = log_power[:split]
            power_mean, power_std = train_log_power.mean(), train_log_power.std()
            std_log_power = (log_power - power_mean) / power_std
            print('Mean: {}, std dev: {}'.format(power_mean, power_std))
            preprocessed_spectra = std_log_power
        else:
            preprocessed_spectra = log_power
    elif std:
        power_mean, power_std = train_power.mean(), train_power.std()
        std_power = (x - power_mean) / power_std
        print('Mean: {}, std dev: {}'.format(power_mean, power_std))
        preprocessed_spectra = std_power
    else:
        print('No seismo preprocessing step done.')
        preprocessed_spectra = x
    
    return preprocessed_spectra


def PCA_compression(power_spectra, split, number_PCA_comp):
    """ Perform PCA compression on some given input power spectra.
    
    Parameters
    ----------
    power_spectra : array-like
        The power spectra; first dimension should represent the number of samples.
    split : int
        The point at which training and validation+testing data are separated.
    number_PCA_comp : int
        Number of PCA components to keep.
    
    Returns
    -------
    meanseismo : float
        The mean of all power spectra along the first axis (the number of samples)
    stdseismo : float
        The mean of all power spectra along the first axis (the number of samples)    
    dominantseismo_all : array-like
        All compressed spectra.
    basis : array-like, shape (number of PCA components, number of spectra components)
        The PCA basis to perform the compression (or decompression).
    """
    # this normalization is needed by PCA
    power_train = power_spectra[:split]
    meanseismo = np.mean(power_train, axis=0)
    stdseismo = np.std(power_train, axis=0)
    power_train_stand = (power_train - meanseismo)/stdseismo
    power_spectra_stand = (power_spectra - meanseismo)/stdseismo
    
    pca = PCA(n_components=number_PCA_comp)
    pca.fit(power_train_stand)
    basis = pca.components_
    dominantseismo_train = pca.transform(power_train_stand)
    dominantseismo_all = pca.transform(power_spectra_stand)
    
    reconstruction_train = np.matmul(dominantseismo_train, basis)

    return meanseismo, stdseismo, dominantseismo_all, basis


def create_observable(observed_seismogram, power_start, power_cut, scale=0):
    """ Produce the observable on which inference is going to be performed. 
    The comparison is done in log10 space, so a sesimogram is taken, noise is added,
    the power spectrum is calculated, and its log is taken.
    
    Parameters 
    ----------
    observed_seismogram : array-like
        The observed seismogram, unpreprocessed.
    power_start : int
        The initial frequency component of the power spectrum to consider
    power_cut : int
        The final frequency component of the power spectrum to consider.
    scale : float, default=0
        The Gaussian standard deviation that represents the noise added to the seismogram.
        
    Returns
    -------
    observable : array-like
        The observable on which inference is going to be performed.
    """
    noisy_seismo = observed_seismogram + np.random.normal(scale=scale, size=(observed_seismogram.shape[-1],))
    rfft_data = np.real(np.fft.rfft(noisy_seismo))
    ifft_data = np.imag(np.fft.rfft(noisy_seismo))
    observable = rfft_data**2. + ifft_data**2
    observable = observable[power_start:power_cut]
    observable = np.log10(observable)
    return observable


def create_cov_matrix(training_data, power_start, power_cut, scale=0):
    """ Produce the covariance matrix used during the inference.
    
    Parameters 
    ----------
    training_data : array-like
        All unpreprocessed training data.
    power_start : int
        The initial frequency component of the power spectrum to consider
    power_cut : int
        The final frequency component of the power spectrum to consider.
    scale : float, default=0
        The Gaussian standard deviation that represents the noise added to the seismogram.
        
    Returns
    -------
    invcovmat : array-like
        The inverse of the covariance matrix used for inference.
    """
    noisy_seismo = training_data + np.random.normal(scale=scale, size=(training_data.shape[0], training_data.shape[-1]))
    rfft_tr = np.real(np.fft.rfft(noisy_seismo))
    ifft_tr = np.imag(np.fft.rfft(noisy_seismo))
    power_tr = rfft_tr**2. + ifft_tr**2
    power_tr = power_tr[:, power_start:power_cut]
    power_tr = np.log10(power_tr)
    cov = np.cov(power_tr, rowvar=False)
    invcovmat = np.linalg.inv(cov)
    return invcovmat