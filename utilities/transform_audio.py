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


USE_MATLAB = True # required for Erblet transforms
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
        raise NotImplementedError('MATLAB is required to perform Erblet transforms')
    
    c, fc = eng.erblet(file_path, verify_reconstruction, visualize, nargout=2)
    coefs = np.array([np.array(ci)[:, 0] for ci in c], dtype=object)
    freqs = np.array(fc)[:, 0]

    return coefs, freqs
erblet_file.affix = 'erb'

def cwt_file(file_path, wavelet='cmor1.5-1.0', low_freq=80, high_freq=20000, num_scales=100, 
             visualize=False, title='CWT with Morlet Wavelet'):
    rate, signal = wavfile.read(file_path)
    frequencies = np.logspace(np.log10(low_freq), np.log10(high_freq), num_scales)
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


def frequency_band_convergence(import_files, glossary_df, transform, partition_basis=None, pval_threshold=.05, max_depth=None, macro_processing=True): 
    
    def converge_freq(glossary_df): 
        transform_coefs_file = import_files[0]
        associated_freqs_file = import_files[1]

        npz_loaded = np.load(transform_coefs_file, allow_pickle=True)
        associated_freqs = np.load(associated_freqs_file, allow_pickle=True)
        tick_freq = [0 for freq in range(len(associated_freqs))]

        naming_convention = "_" + transform
        coeffs_ = []
        for file in glossary_df["filename"]: 
            transform_file = file[:-4] + naming_convention
            file_coeffs = npz_loaded[transform_file]
            coeffs_.extend(file_coeffs)

        high_indx = len(associated_freqs)-1
        low_indx = 0
        mid_indx = (high_indx + low_indx) / 2

        if macro_processing: 
            collective_coeffs_ = [[] for f in range(len(associated_freqs))]

            for file_coeffs_ in coeffs_: ## 1440 times
                for freq in file_coeffs_:
                    collective_coeffs_[freq].extend(file_coeffs_[freq])
            collective_coeffs_ = pd.Series(collective_coeffs_)

            tick_freq, total_cuts = converge(collective_coeffs_, tick_freq, [low_indx, mid_indx], [mid_indx+1, high_indx], pval_threshold, 0, max_depth)

            lower = None 
            upper = None 

            f_indx = 0
            macro_intervals = []
            ## [0 1 1 1 0 0 1 1]
            while f_indx < len(tick_freq): 
                if tick_freq[f_indx] == 0: ## meant to keep seperate
                    macro_intervals.extend(associated_freqs[f_indx])
                else: ## means we encountered a one
                    if not lower: 
                        lower = associated_freqs[f_indx]
                    elif (f_indx + 1 < len(tick_freq) and tick_freq[f_indx]+1 == 0) or (f_indx + 1 == len(tick_freq) and tick_freq[f_indx] ==1):
                        upper = associated_freqs[f_indx]
                        macro_intervals.extend[(lower, upper)]

                        lower = None
                        upper = None
                f_indx = f_indx + 1
            return macro_intervals
        else: 
            micro_intervals = []
            for cl in coeffs_: 
                tick_freq, total_cuts = converge(cl, tick_freq, [low_indx, mid_indx], [mid_indx+1, high_indx], pval_threshold, 0, max_depth)

                lower = None 
                upper = None 

                f_indx = 0
                cur_interval = []
                while f_indx < len(tick_freq): 
                    if tick_freq[f_indx] == 0: ## meant to keep seperate
                        cur_interval.extend(associated_freqs[f_indx])
                    else: ## means we encountered a one
                        if not lower: 
                            lower = associated_freqs[f_indx]
                        elif (f_indx + 1 < len(tick_freq) and tick_freq[f_indx]+1 == 0) or (f_indx + 1 == len(tick_freq) and tick_freq[f_indx] ==1):
                            upper = associated_freqs[f_indx]
                            cur_interval.extend[(lower, upper)]

                            lower = None
                            upper = None
                    f_indx = f_indx + 1
                micro_intervals.extend(cur_interval)
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

def converge(coef_list, tick_freq, slit1_inv, slit2_inv, pval_threshold, cuts, max_depth=None):
   """
    frequency converge that tests if two slits frequencies has disimilar enough distributions to
    be considered separate by a ks-threshold, and if so, parses them apart in the coef list 
    and recrusively does so until two slits are considered similar enough 

    Args:
        coef_list: pd.series where index indicates frequency (range) and assocaited coef list
        tick_freq: an array with # of indeces as number of freqs, a 1 indicates it should be merged with adjacent ones
        slit1_inv: array with 2 entries, first entry mapping to index of lower freq of interval, 
                   second entry mapping to index of upper frequency of interval
        slit2_inv: same format as slit1_inv
        pval_threshold: metric of dsimiliarity
        cuts: how many cuts prior have been made
        max_depth: threshold limit for number of cuts

    Returns:
        coef_list modified if more cuts have been made, indeces may be tuples if frequences are banded together
        number of cuts made (for recrusive purposes )
    """
   if (max_depth and cuts >= max_depth) or slit1_inv[0] >= slit1_inv[1] or slit2_inv[0] >= slit2_inv[1]: 
      ## base case
      return coef_list, 0 

   slit1_ = []
   slit2_ = []

   for freq in np.arange(slit1_inv[0], slit1_inv[1]+1):
      slit1_.extend(coef_list[freq])
   for freq in np.arange(slit2_inv[0], slit2_inv[1]+1):
      slit2_.extend(coef_list[freq])

   pval_ = stats.ks_2samp(slit1_, slit2_).pvalue
   if pval_ > pval_threshold: 
      ## base case
      ## bands are similar enough,
      new_tick_freq = tick_freq[:]
      for i in np.arange(slit1_[0], slit2_inv[1]+1):
          new_tick_freq[i] = 1
      
      return new_tick_freq, 0
   
   else: 
      ## recrusive case
      ## bands are disimialr enough, repeat procress in their new intervals

      tick_freq, cuts_1 = converge(coef_list, tick_freq, [slit1_inv[0], (slit1_inv[0] + slit1_inv[1]) /2], [(slit1_inv[0] + slit1_inv[1]) /2 + 1, slit1_inv[1]], pval_threshold, cuts + 1, max_depth)
      tick_freq, cuts_2 = converge(coef_list, tick_freq, [slit2_inv[0], (slit2_inv[0] + slit2_inv[1]) /2], [(slit2_inv[0] + slit2_inv[1]) /2 + 1, slit2_inv[1]], pval_threshold, cuts_1, max_depth)

      return tick_freq, cuts_1 + cuts_2 