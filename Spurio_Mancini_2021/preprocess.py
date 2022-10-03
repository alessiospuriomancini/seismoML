import numpy as np

def preprocess_seismo(x, split, log=False, std=False, rescale=False, rescale_onlyamp=False):
    train_seismo = x[:split]
    if rescale:
        X_dim = x.shape[-1]
        from scipy.ndimage.interpolation import shift
        reference_seismogram = x[0]
        Ai_refseismo = np.max(reference_seismogram)
        ti_refseismo = reference_seismogram.argmax()
        print("Amplitude of first peak in reference seismo", Ai_refseismo)
        print("Time index of first peak in reference seismo", ti_refseismo)
        # obtaining and saving all peaks and corresponding indices
        Ai_allseismo = np.max(x, axis=1)
        ti_allseismo = np.argmax(x, axis=1)
        amplitude_rescale = Ai_refseismo/Ai_allseismo
        shift_index = ti_refseismo - ti_allseismo
        print('Saving amplitude ratios and shift indices')
        np.savetxt('./amplitude_rescale_NOTsorted.txt', amplitude_rescale)
        np.savetxt('./shift_index_NOTsorted.txt', shift_index)
        seismograms_rescaled = np.multiply(x, np.repeat(amplitude_rescale, X_dim).reshape(-1, X_dim))
        for index in range(seismograms_rescaled.shape[0]):
            seismograms_rescaled[index] = shift(seismograms_rescaled[index], shift_index[index])
        if std:
            print('Rescaling and standardizing')
            train_rescaled_seismo = seismograms_rescaled[:split]
            seismograms_mean, seismograms_std = train_rescaled_seismo.mean(), train_rescaled_seismo.std()
            std_rescaled_seismo = (seismograms_rescaled - seismograms_mean) / seismograms_std
            print('Mean: {}, std dev: {}'.format(seismograms_mean, seismograms_std))
            return std_rescaled_seismo
        else:
            print('Rescaling only.')
            return seismograms_rescaled

    elif rescale_onlyamp:
        X_dim = x.shape[-1]
        from scipy.ndimage.interpolation import shift
        reference_seismogram = x[0]
        Ai_refseismo = np.max(reference_seismogram)
        print("Amplitude of first peak in reference seismo", Ai_refseismo)
        # obtaining and saving all peaks and corresponding indices
        Ai_allseismo = np.max(x, axis=1)
        amplitude_rescale = Ai_refseismo/Ai_allseismo
        print('Saving amplitude ratios')
        np.savetxt('./amplitude_rescale.txt', amplitude_rescale)
        seismograms_rescaled = np.multiply(x, np.repeat(amplitude_rescale, X_dim).reshape(-1, X_dim))
        if std:
            print('Rescaling and standardizing')
            train_rescaled_seismo = seismograms_rescaled[:split]
            seismograms_mean, seismograms_std = train_rescaled_seismo.mean(), train_rescaled_seismo.std()
            std_rescaled_seismo = (seismograms_rescaled - seismograms_mean) / seismograms_std
            print('Mean: {}, std dev: {}'.format(seismograms_mean, seismograms_std))
            return std_rescaled_seismo
        else:
            print('Rescaling only.')
            return seismograms_rescaled

    elif log:
        log_seismo = np.sign(x) * np.log10(1 + np.abs(x/(1./np.log(10))))
        if std:
            print('Log and standardizing')
            train_log_seismo = log_seismo[:split]
            seismograms_mean, seismograms_std = train_log_seismo.mean(), train_log_seismo.std()
            std_log_seismo = (log_seismo - seismograms_mean) / seismograms_std
            print('Mean: {}, std dev: {}'.format(seismograms_mean, seismograms_std))
            return std_log_seismo
        else:
            print('Log only')
            return log_seismo
    elif std:
        print('Standardizing only')
        seismograms_mean, seismograms_std = train_seismo.mean(), train_seismo.std()
        std_seismo = (x - seismograms_mean) / seismograms_std
        print('Mean: {}, std dev: {}'.format(seismograms_mean, seismograms_std))
        return std_seismo
    else:
        print('No preprocessing step done.')
        return x

def preprocess_coord(y, split, test_valid, sort=False, std=False):
        N = y.shape[0]
        ref = [41,41,244]
        shifted = y - ref
        distances = np.linalg.norm(shifted, axis = 1)
        new_coords = np.zeros((N,4))
        new_coords[:, :3] = shifted
        new_coords[:, -1] = distances
        training_coords = new_coords[:split]
#        validation_coords = new_coords[split:split+test_valid]
        testing_coords = new_coords[split+test_valid:]
        if sort:
            distances_training = distances[:split]
            sorted_dist_training_indices = np.argsort(distances_training)
#            distances_validation = distances[split:split+test_valid]
#            sorted_dist_validation_indices = np.argsort(distances_validation)
            distances_testing = distances[split+test_valid:]
            sorted_dist_testing_indices = np.argsort(distances_testing)
            training_coords = training_coords[sorted_dist_training_indices]
#            validation_coords = validation_coords[sorted_dist_validation_indices]
            testing_coords = testing_coords[sorted_dist_testing_indices]
            new_coords[:split] = training_coords
#            new_coords[split:split+test_valid] = validation_coords
            new_coords[split+test_valid:] = testing_coords
            if std:
                print('Shifted and standardized; sorted')
                coord_mean, coord_std = np.mean(training_coords, axis=0), np.std(training_coords, axis=0)
                print('Mean: {}, std dev: {}'.format(coord_mean, coord_std))
                new_coords_std = (new_coords - coord_mean) / coord_std
#                return new_coords_std, sorted_dist_training_indices, sorted_dist_validation_indices, sorted_dist_testing_indices
                return new_coords_std, sorted_dist_training_indices, sorted_dist_testing_indices
            else:
#                return new_coords, sorted_dist_training_indices, sorted_dist_validation_indices, sorted_dist_testing_indices
                return new_coords, sorted_dist_training_indices, sorted_dist_testing_indices
        else:
            if std:
                print('Shifted and standardized; not sorted')
                coord_mean, coord_std = np.mean(training_coords, axis=0), np.std(training_coords, axis=0)
                print('Mean: {}, std dev: {}'.format(coord_mean, coord_std))
                new_coords_std = (new_coords - coord_mean) / coord_std
                return new_coords_std, coord_mean, coord_std
            else:
                print('Shifted but not standardized; not sorted')
                return new_coords
