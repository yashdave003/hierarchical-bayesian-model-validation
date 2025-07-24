import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pywt
import librosa
from scipy import stats
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

def cwt_file(file_path, wavelet='cmor1.5-1.0', low_freq=80, high_freq=8000, num_scales=100, 
             visualize=False, title='CWT with Morlet Wavelet'):
    rate, signal = wavfile.read(file_path)
    frequencies = np.logspace(np.log10(low_freq), np.log10(high_freq), num_scales)
    scales = pywt.frequency2scale(wavelet, frequencies / rate)
    coefs, freqs = pywt.cwt(signal, scales, wavelet, 1/rate)

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

def transform_list(transform_file, file_list, file_names, *args, progress_bar=True, affix=None, **kwargs):
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
    transformed_names = [name.rsplit('.', 1)[0] + '_' + affix for name in file_names]

    coefs_gen = (check_freqs(transform_file(file_path, *args, **kwargs)) for file_path in file_list)
    if progress_bar:
        coefs_gen = tqdm(coefs_gen, total=len(file_list))

    np.savez(affix + '_coefs', **dict(zip(transformed_names, coefs_gen)))
    np.save(affix + '_freqs', first_freqs)

def frequency_band_convergence(IMPORT_DIR, glossary_df, transform, partition_basis=None, ks_threshold=.05, max_depth=None, macro_processing=True): 
    
    def converge_freq(glossary_df): 
        naming_convention = "_" + transform + ".npz"
        coeffs_ = []
        for file in glossary_df["filename"]: 
            transform_file = os.path.join(IMPORT_DIR, file[:-4] + naming_convention)
            with np.load(transform_file) as npz_representation:
                file_coeffs = [npz_representation[key] for key in sorted(npz_representation.files)] 
            coeffs_.append[file_coeffs]

        if macro_processing: 
            ## across all files, concaneate all coeffs associated with like freq.
            ## i.e file i has coeffs Ci_200-400 and file j has coeffs Cj_200-400, store them in one array
            ## for all  i != j < len(total_files)
            ##, peform else process once 
            macro_intervals, total_cuts = converge(coeffs_, [40, 4040], [4041, 8000], ks_threshold, 0, max_depth)
            return macro_intervals
        else: 
            micro_intervals = []
            for cl in coeffs_: 
                interval, total_cuts = converge(cl, [40, 4040], [4041, 8000], ks_threshold, 0, max_depth)
                micro_intervals.extend(interval)
            return micro_intervals
    
    partitions = None 
    if partition_basis: 
        partitions = glossary_df.groupby(partition_basis)

    if partitions: 
        freq_bands = []
        for part_df in partitions: 
             freq_bands.append(converge_freq(part_df[1]))
    else: 
        freq_bands = converge_freq(glossary_df)

    return freq_bands

def converge(coef_list, slit1_inv, slit2_inv, ks_threshold, cuts, max_depth=None):
   """
    frequency converge that tests if two slits frequencies has disimilar enough distributions to
    be considered separate by a ks-threshold, and if so, parses them apart in the coef list 
    and recrusively does so until two slits are considered similar enough 

    Args:
        coef_list: pd.series where index indicates frequency (range) and assocaited coef list
        slit1_inv: array with 2 entries, first entry mapping to lower freq of interval, second entry mapping upper
                   frequency of interval
        slit2_inv: same format as slit1_inv
        ks_threshold: metric of dsimiliarity
        cuts: how many cuts prior have been made
        max_depth: threshold limit for number of cuts

    Returns:
        coef_list modified if more cuts have been made, indeces may be tuples if frequences banded together
        number of cuts made (for recrusive purposes )
    """
   if max_depth and cuts >= max_depth: 
      ## base case
      return coef_list, 0 

   slit1_ = []
   slit2_ = []

   for freq in np.arange(slit1_inv[0], slit1_inv[1]+1):
      slit1_.extend(coef_list[freq])
   for freq in np.arange(slit2_inv[0], slit2_inv[1]+1):
      slit2_.extend(coef_list[freq])

   ks_ = stats.ks_2samp(slit1_, slit2_).statistic
   if ks_ > ks_threshold: 
      ## base case
      ## bands are similar enough, merge them
      new_slit = slit1_.extend(slit2_)
      for freq in np.arange(slit1_inv[0], slit2_inv[1]+1):
         coef_list.drop(freq, axis="index")

      new_lower = slit1_inv[0]
      new_upper = slit2_inv[1]
      to_concat = pd.Series(new_slit, index=[(new_lower, new_upper)]) ## stored as tuple to ensure unique indexing

      coef_list = pd.concat(coef_list, to_concat)
      return coef_list, 0  
   
   else: 
      ## recrusive case
      ## bands are disimialr enough, repeat procress in their new intervals

      coef_list, cuts_1 = converge(coef_list, [slit1_inv[0], (slit1_inv[0] + slit1_inv[1]) /2], [(slit1_inv[0] + slit1_inv[1]) /2 + 1, slit1_inv[1]], ks_threshold, cuts + 1, max_depth)
      coef_list, cuts_2 = converge(coef_list, [slit2_inv[0], (slit2_inv[0] + slit2_inv[1]) /2], [(slit2_inv[0] + slit2_inv[1]) /2 + 1, slit2_inv[1]], ks_threshold, cuts_1, max_depth)

      return coef_list, cuts_1 + cuts_2 