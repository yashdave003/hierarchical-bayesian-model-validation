import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pywt
import librosa
from scipy import stats
import zipfile
import io
import shutil
from tqdm.notebook import tqdm


USE_MATLAB = False # required for Erblet transforms
if USE_MATLAB:
    import matlab.engine 
    eng = matlab.engine.start_matlab()
    # use utilities/ltfat_path.txt to store path to local installation (under .gitignore)
    # may want to use a .env file in the future if more environment variables are required
    with open('ltfat_path.txt') as f:
        eng.addpath(f.read())
    eng.ltfatstart(nargout=0)
else:
    eng = None

def erblet_file(file_path, verify_reconstruction=False, visualize=False):
    if not USE_MATLAB:
        raise NotImplementedError('MATLAB is required to perform ERBlet transforms')
    
    c, fc = eng.erblet(file_path, verify_reconstruction, visualize, nargout=2)
    coefs = np.array([np.array(ci)[:, 0] for ci in c], dtype=object)
    freqs = np.array(fc)[:, 0]

    return coefs, freqs
erblet_file.affix = 'erb'

def cwt_file(file_path, wavelet='cmor1.5-1.0', low_freq=80, high_freq=24000, num_scales=100, 
             visualize=False, title='CWT with Morlet Wavelet'):
    rate, signal = wavfile.read(file_path)
    frequencies = np.geomspace(low_freq, high_freq, num_scales)
    scales = pywt.frequency2scale(wavelet, frequencies / rate)
    coefs, freqs = pywt.cwt(signal.astype('float32'), scales, wavelet, 1/rate)

    if visualize:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(np.arange(len(signal)) / rate, freqs, np.abs(coefs), norm='log', cmap='inferno', vmin=100)
        plt.colorbar(label='Magnitude')
        plt.yscale('log')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(title)
        plt.show()

    return coefs, freqs
cwt_file.affix = 'cwt'

def stft_file(file_path, n_fft=1024, visualize=False, title='Log-Frequency Spectrogram (STFT Magnitude)'):
    signal, rate = librosa.load(file_path, sr=None)
    coefs = librosa.stft(signal, n_fft=n_fft)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)

    if visualize:
        coefs_db = librosa.amplitude_to_db(np.abs(coefs), ref=np.max)
        librosa.display.specshow(coefs_db, y_axis='log', x_axis='time', sr=rate, n_fft=n_fft)
        plt.colorbar(format="%+2.f dB")
        plt.title(title)
        plt.show()

    return coefs, freqs
stft_file.affix = 'stft'

def fft_file(file_path, n=None, discard_negative=True, visualize=False, title='Fourier Spectrum'):
    rate, signal = wavfile.read(file_path)
    coefs = np.fft.fft(signal, n)
    freqs = np.fft.fftfreq(len(signal) if n is None else n, 1/rate)
    if discard_negative:
        negative_idx = len(freqs) // 2
        coefs = coefs[:negative_idx]
        freqs = freqs[:negative_idx]

    if visualize:
        plt.plot(freqs, np.abs(coefs), lw=.1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.yscale('log')
        plt.title(title)
        plt.show()
    
    return coefs, freqs
fft_file.affix = 'fft'

def transform_list(transform_file, file_list, file_names, *args, 
                   progress_bar=True, affix=None, compress=False, **kwargs):
    '''warning: compressing exports will make downstream reading substantially slower'''
    first_freqs = None
    def check_freqs(transform):
        nonlocal first_freqs
        coefs, freqs = transform
        if first_freqs is None:
            first_freqs = freqs
        else:
            assert np.array_equal(freqs, first_freqs), 'frequency scales are not consistent across files'
        return coefs
    
    if affix is None:
        affix = getattr(transform_file, 'affix', 'xform')

    filenames_out = (f"{name.rsplit('.', 1)[0]}_{affix}.npy" for name in file_names)
    coefs_gen = (check_freqs(transform_file(file_path, *args, **kwargs)) for file_path in file_list)
    pairs_gen = zip(filenames_out, coefs_gen)
    if progress_bar:
        pairs_gen = tqdm(pairs_gen, 'Computing and exporting coefficients', total=len(file_list))

    zip_compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(affix + '_coefs.npz', 'w', zip_compression) as zipf:
        for filename, coefs in pairs_gen:
            buffer = io.BytesIO()
            np.save(buffer, coefs)
            buffer.seek(0)
            with zipf.open(filename, 'w', force_zip64=True) as zipf_entry:
                shutil.copyfileobj(buffer, zipf_entry)

    np.save(affix + '_freqs', first_freqs)


def load_coefs_by_freq(coefs_npz_path, batch_size=None, subsample_every=1, cache=False, debug=False):
    '''if batch_size is None, load all files into memory at once'''
    cache_match = False
    if cache:
        cache_args = (coefs_npz_path, subsample_every)
        if (cache_match := getattr(load_coefs_by_freq, 'cache_args', None) == cache_args):
            coefs_by_freq = load_coefs_by_freq.cached_coefs

    if not cache_match:
        with np.load(coefs_npz_path, allow_pickle=True) as coefs_npz:
            n_freqs = len(test_file := coefs_npz[coefs_npz.files[0]])
            combine = np.concat if isinstance(test_file[0], np.ndarray) else np.array

            if isinstance(subsample_every, (np.ndarray, list, tuple)):
                assert len(subsample_every) == n_freqs, (
                    'subsample_every should either be a single number or have the same length as freqs'
                )
            else:
                subsample_every = type('IndexableConstant', (), {
                    '__getitem__': (lambda c: lambda self, idx: c)(subsample_every)
                    })()
            
            coefs_by_freq = [None] * n_freqs
            if batch_size is None:
                batch_size = n_freqs
            for j in range(-(-n_freqs // batch_size)):
                batches = [coefs_npz[file][j*batch_size:(j+1)*batch_size].copy() for file in coefs_npz]
                for k in range(len(batches[0])):
                    complex_coefs = combine([batch[k] for batch in batches])
                    coef_components = np.concat([np.real(complex_coefs), np.imag(complex_coefs)])
                    i = j * batch_size + k
                    if (s := subsample_every[i]) == 1:
                        coefs_by_freq[i] = coef_components
                    else:
                        coefs_by_freq[i] = np.sort(coef_components)[::s].copy()
                    if debug:
                        print(f'Compiled freq. {i + 1}/{n_freqs}', end='\r')

        if cache:
            load_coefs_by_freq.cache_args = cache_args
            load_coefs_by_freq.cached_coefs = coefs_by_freq

    return coefs_by_freq

# current potential issues:
#   - groups when max_depth is reached; maybe should assume no grouping by default
def freq_band_groupings(coefs_npz_path, ks_threshold=.05, batch_size=None, subsample_every=1, 
                        presplit_depth=1, max_depth=None, cache=False, debug=False):
    '''if batch_size is None, load all files into memory at once'''
    coefs_by_freq = load_coefs_by_freq(coefs_npz_path, batch_size, subsample_every, cache, debug)
    n_freqs = len(coefs_by_freq)
    
    def freq_band_helper(left_endpoint, right_endpoint, depth):
        if left_endpoint + 1 == right_endpoint or depth == max_depth:
            return [(left_endpoint, right_endpoint)]
        
        midpoint = (left_endpoint + right_endpoint) // 2
        if debug:
            print(f'{"  " * depth}[{left_endpoint}, {midpoint}) ~ [{midpoint}, {right_endpoint}): ', end='')
        if depth >= presplit_depth:
            coefs_left = np.concat(coefs_by_freq[left_endpoint:midpoint])
            coefs_right = np.concat(coefs_by_freq[midpoint:right_endpoint])
            ks_res = stats.ks_2samp(coefs_left, coefs_right)
            if debug:
                print(f'{ks_res.statistic:.5f}, {ks_res.pvalue}')
            if ks_res.statistic < ks_threshold:
                return [(left_endpoint, right_endpoint)]
        elif debug:
            print('presplit')
            
        return freq_band_helper(left_endpoint, midpoint, depth + 1) \
            + freq_band_helper(midpoint, right_endpoint, depth + 1)
    
    return freq_band_helper(0, n_freqs, 0)

def group_coefs_by_band(coefs_by_freq, bands):
    return [np.concat(coefs_by_freq[band[0]:band[1]]) for band in bands]

def geometric_count_bands(bands, visualize=False):
    '''bands must have been computed for a continuous, linear frequency scale (fft, stft)'''
    counts = np.array([band[1] - band[0] for band in bands])

    indices = np.arange(len(counts))
    log_counts = np.log(counts)

    m = np.corrcoef(indices, log_counts)[0, 1] * np.std(log_counts) / np.std(indices)
    b = np.mean(log_counts) - m * np.mean(indices)
    pred_counts = np.exp(m * indices + b)

    new_band_indices = np.round(np.cumsum(np.append(0, pred_counts)))
    new_band_indices[-1] = bands[-1][1]
    new_bands = [(int(start), int(end)) for start, end in zip(new_band_indices, new_band_indices[1:])]

    if visualize:
        plt.plot(counts, label='True counts')
        plt.plot(pred_counts, label='Predicted (smooth)')
        plt.plot([band[1] - band[0] for band in new_bands], label='Predicted (actual)', ls='--', c='r')
        plt.yscale('log')
        plt.xlabel('Band index')
        plt.ylabel('Band count (log)')
        plt.title('Frequency band counts after grouping')
        plt.legend()
        plt.show()
    
    return new_bands

def geometric_endpoint_bands(bands, freqs, visualize=False):
    '''bands may have been computed for a nonlinear frequency scale, still preferably continuous'''
    endpoints = np.array([freqs[band[0]] for band in bands])

    indices = np.arange(len(endpoints))
    log_endpoints = np.log(endpoints)
    m = np.corrcoef(indices, log_endpoints)[0, 1] * np.std(log_endpoints) / np.std(indices)
    b = np.mean(log_endpoints) - m * np.mean(indices)

    new_endpoints = np.exp(m * indices + b)
    new_band_indices = list(map(int, np.searchsorted(freqs, new_endpoints)))
    new_band_indices.append(len(freqs))
    new_bands = [(start, end) for start, end in zip(new_band_indices[:-1], new_band_indices[1:]) if start != end]

    if visualize:
        plt.plot(endpoints, label='True')
        plt.plot(new_endpoints, label='Predicted (smooth)')
        plt.plot([freqs[band[0]] for band in new_bands], label='Predicted (actual)', ls='--', c='r')
        plt.axhline(freqs[-1], c='k', alpha=.5)
        plt.yscale('log')
        plt.xticks(range(0, len(endpoints), 2))
        plt.xlabel('Band index')
        plt.ylabel('Left endpoint frequency (log)')
        plt.legend()
        plt.show()

    return new_bands


####################################################
# previous implementation(s), not currently in use #
####################################################

def frequency_band_convergence(import_files, glossary_df, transform, partition_basis=None, pval_threshold=.05, max_depth=None, macro_processing=True): 
    
    def converge_freq(glossary_df): 
        transform_coefs_file = import_files[0]
        associated_freqs_file = import_files[1]

        npz_loaded = np.load(transform_coefs_file, allow_pickle=True)
        associated_freqs = np.load(associated_freqs_file, allow_pickle=True)

        naming_convention = "_" + transform
        coeffs_ = []
        for file in glossary_df["filename"]: 
            transform_file = file[:-4] + naming_convention
            file_coeffs = npz_loaded[transform_file]
            coeffs_.extend(file_coeffs)

        high_indx = len(associated_freqs)-1
        low_indx = 0
        mid_indx = int((high_indx + low_indx) / 2)

        if macro_processing: 
            collective_coeffs_ = [[] for f in range(len(associated_freqs))]

            for file_coeffs_ in coeffs_:
                for freq in file_coeffs_:
                    collective_coeffs_[freq].extend(file_coeffs_[freq])
            collective_coeffs_ = pd.Series(collective_coeffs_)

            macro_intervals, total_cuts = converge(collective_coeffs_, associated_freqs, [low_indx, mid_indx], [mid_indx+1, high_indx], pval_threshold, 0, max_depth)
            return macro_intervals
        else: 
            micro_intervals = []
            for cl in coeffs_: 
                freqs_copy = associated_freqs[:]
                freq_interval, total_cuts = converge(cl, freqs_copy, [low_indx, mid_indx], [mid_indx+1, high_indx], pval_threshold, 0, max_depth)

                micro_intervals.extend(freq_interval)
            return micro_intervals
    
    partitions = None 
    if partition_basis: 
        partitions = glossary_df.groupby(partition_basis)

    if partitions: 
        freq_bands = []
        for part_df in partitions: 
             freq_bands.append(list(converge_freq(part_df[1]).index))
    else: 
        freq_bands = list(converge_freq(glossary_df).index)

    return freq_bands

def converge(coef_list, freqs, slit1_inv, slit2_inv, pval_threshold, cuts, max_depth=None):
   """
    frequency converge that tests if two slits frequencies has disimilar enough distributions to
    be considered separate by a ks-threshold, and if so, parses them apart in the coef list 
    and recrusively does so until two slits are considered similar enough 

    Args:
        coef_list: pd.series where index indicates frequency (range) and assocaited coef list
        freq: an array indicating the matching frequenices to our coefficents 
        slit1_inv: array with 2 entries, first entry mapping to index of lower freq of interval, 
                   second entry mapping to index of upper frequency of interval
        slit2_inv: same format as slit1_inv
        pval_threshold: metric of dsimiliarity
        cuts: how many cuts prior have been made
        max_depth: threshold limit for number of cuts

    Returns:
        converged_freq: an array of converged frequencies, 
        number of cuts made (for recrusive purposes )
    """
   if (max_depth and cuts >= max_depth) or slit1_inv[0] >= slit1_inv[1] or slit2_inv[0] >= slit2_inv[1]: 
      ## base case
      sing_freqs = []
      if slit1_inv[0] == slit1_inv[1]: 
          sing_freqs.append[freq[slit1_inv[0]]]
      if slit2_inv[0] == slit2_inv[1]: 
          sing_freqs.append[freq[slit2_inv[0]]]
      return sing_freqs, 0 

   slit1_ = []
   slit2_ = []

   for freq in np.arange(slit1_inv[0], slit1_inv[1]+1):
      slit1_.extend(coef_list[freq])
   for freq in np.arange(slit2_inv[0], slit2_inv[1]+1):
      slit2_.extend(coef_list[freq])

   pval_ = stats.ks_2samp(slit1_, slit2_).pvalue
   if pval_ > pval_threshold: 
      ## base case
      ## bands are similar enough, merge them
      lower_freq = freqs[slit1_inv[0]]
      upper_freq = freqs[slit2_inv[1]]
      return [[lower_freq, upper_freq]], 0
   
   else: 
      ## recrusive case
      ## bands are disimilar enough,
      ## keep them seperate 

      lower_freqs, cuts_1 = converge(coef_list, freqs, [slit1_inv[0], int((slit1_inv[0] + slit1_inv[1]) /2)], [int((slit1_inv[0] + slit1_inv[1]) /2 + 1), slit1_inv[1]], pval_threshold, cuts + 1, max_depth)
      upper_freqs, cuts_2 = converge(coef_list, freqs, [slit2_inv[0], int((slit2_inv[0] + slit2_inv[1]) /2)], [int((slit2_inv[0] + slit2_inv[1]) /2 + 1), slit2_inv[1]], pval_threshold, cuts_1, max_depth)

      return lower_freqs + upper_freqs, cuts_1 + cuts_2 
   
# Functions for KS Banding Template Notebook
def preprocess_and_save_by_band(coefs_npz_path, unified_bands_indices, temp_dir, group_id, group_map):
    """
    Shards the NPZ file into binary files. 
    """
    print(f"  Pre-processing and sharding data by band to: {temp_dir}")
    
    current_group_names = group_map[group_id]
    
    # Open file handles
    file_writers = {}
    for i, _ in enumerate(unified_bands_indices):
        file_writers[i] = {}
        for group_name in current_group_names:
            temp_file_path = temp_dir / f"band_{i}_{group_name}.bin"
            file_writers[i][group_name] = open(temp_file_path, 'wb')

    with np.load(coefs_npz_path, allow_pickle=True) as data:
        total_files = len(data.files)
        print(f"  - Found {total_files} files in NPZ archive. Starting sharding...")
        
        for i, item in enumerate(data.files):
            if (i + 1) % 200 == 0:
                print(f"    - Sharding file {i+1}/{total_files}...")
        
            try:
                base_filename = item.split('_')[0]
                actor_id = int(base_filename.split('-')[group_id])
                group_label = current_group_names[1] if actor_id % 2 == 0 else current_group_names[0]
                
                coeffs = data[item]
                if coeffs.size == 0:
                    continue

                num_freqs = coeffs.shape[0]

                for band_idx, (start_idx, end_idx) in enumerate(unified_bands_indices):
                    if start_idx < num_freqs:
                        # Handle slicing based on dimensions
                        if coeffs.ndim == 2:
                            # STFT: (Freq, Time) -> Slice rows
                            band_coeffs = coeffs[start_idx:end_idx, :]
                        elif coeffs.ndim == 1:
                            # FFT: (Freq,) -> Slice elements
                            band_coeffs = coeffs[start_idx:end_idx]
                        else:
                            continue # Skip >2D or 0D

                        # Check if slice is empty
                        if band_coeffs.size > 0:
                            # Ensure complex64 for consistency
                            file_writers[band_idx][group_label].write(band_coeffs.astype(np.complex64).tobytes())
            
            except Exception:
                continue

    # Close handles
    for band_idx in file_writers:
        for group_name in file_writers[band_idx]:
            file_writers[band_idx][group_name].close()


def compare_distributions_from_files(temp_dir, unified_bands_indices, group_names):
    print(f"  Performing KS test band by band from sharded files...")
    
    group1_name, group2_name = group_names[0], group_names[1]
    results = []

    for i, (start_idx, end_idx) in enumerate(unified_bands_indices):
        path_g1 = temp_dir / f"band_{i}_{group1_name}.bin"
        path_g2 = temp_dir / f"band_{i}_{group2_name}.bin"

        if not path_g1.exists() or not path_g2.exists() or path_g1.stat().st_size == 0 or path_g2.stat().st_size == 0:
            continue

        group1_coeffs = np.fromfile(path_g1, dtype=np.complex64)
        group2_coeffs = np.fromfile(path_g2, dtype=np.complex64)
        
        ks_stat_real, _ = stats.ks_2samp(np.real(group1_coeffs), np.real(group2_coeffs))
        ks_stat_imag, _ = stats.ks_2samp(np.imag(group1_coeffs), np.imag(group2_coeffs))
        
        results.append({
            'band_indices': f'{start_idx}-{end_idx - 1}' if end_idx > start_idx + 1 else str(start_idx),
            'ks_stat_real': ks_stat_real,
            'ks_stat_imag': ks_stat_imag,
        })
    return pd.DataFrame(results)


def load_single_group_by_band(coefs_npz_path: str,
                              unified_bands_indices: list,
                              group_id: int,
                              group_map: dict,
                              category_map: dict,
                              target_group: str):
    print(f"  - Loading data for target group '{target_group}' from: {os.path.basename(coefs_npz_path)}")
    group_names = group_map[group_id]

    processed_data = {
        target_group: {
            (f'{start}-{end - 1}' if end > start + 1 else str(start)): {'real': [], 'imag': []}
            for start, end in unified_bands_indices
        }
    }

    freq_to_band_map = {}
    for start, end in unified_bands_indices:
        band_label = f'{start}-{end - 1}' if end > start + 1 else str(start)
        for freq_idx in range(start, end):
            freq_to_band_map[freq_idx] = band_label

    with np.load(coefs_npz_path, allow_pickle=True) as data:
        total_files = len(data.files)
        print(f"  - {total_files} files, collecting for '{target_group}'...")
        processed_count = 0
        for i, item in enumerate(data.files):
            try:
                base_filename = item.split('_')[0]
                actor_id = int(base_filename.split('-')[group_id])
                group_label = group_names[1] if actor_id % 2 == 0 else group_names[0]

                if group_label != target_group:
                    continue
                
                processed_count += 1
                if processed_count % 200 == 0:
                    print(f"    - Found and processed {processed_count} files for '{target_group}'...")

                coeffs = data[item]

                if coeffs.ndim == 2:  # 2D array (e.g., STFT)
                    for freq_idx in range(coeffs.shape[0]):
                        band_label = freq_to_band_map.get(freq_idx)
                        if band_label:
                            target_list_real = processed_data[target_group][band_label]['real']
                            target_list_imag = processed_data[target_group][band_label]['imag']
                            target_list_real.extend(np.real(coeffs[freq_idx, :]))
                            target_list_imag.extend(np.imag(coeffs[freq_idx, :]))

                elif coeffs.ndim == 1: # 1D array
                    if coeffs.dtype == object:  # Jagged array (e.g., CWT/Erblet)
                        for freq_idx, inner_array in enumerate(coeffs):
                            band_label = freq_to_band_map.get(freq_idx)
                            if band_label and hasattr(inner_array, '__iter__'):
                                target_list_real = processed_data[target_group][band_label]['real']
                                target_list_imag = processed_data[target_group][band_label]['imag']
                                target_list_real.extend(np.real(inner_array))
                                target_list_imag.extend(np.imag(inner_array))
                    else: # Standard 1D array (e.g., FFT)
                         for freq_idx, coeff_val in enumerate(coeffs):
                            band_label = freq_to_band_map.get(freq_idx)
                            if band_label:
                                processed_data[target_group][band_label]['real'].append(np.real(coeff_val))
                                processed_data[target_group][band_label]['imag'].append(np.imag(coeff_val))

            except (ValueError, IndexError) as e:
                print(f"  - Warning: Could not parse filename '{item}', skipped. Error: {e}")
                continue

    print(f"  - Data collection complete for '{target_group}'. Converting lists to NumPy arrays...")
    for band_label, data_dict in processed_data[target_group].items():
        if data_dict['real']:
            data_dict['real'] = np.array(data_dict['real'])
            data_dict['imag'] = np.array(data_dict['imag'])
        else:
            data_dict['real'] = np.array([])
            data_dict['imag'] = np.array([])

    return processed_data