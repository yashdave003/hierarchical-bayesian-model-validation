import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pywt
import librosa
from tqdm.notebook import tqdm


USE_MATLAB = True # required for Erblet transforms
if USE_MATLAB:
    import matlab.engine 
    eng = matlab.engine.start_matlab()
else:
    eng = None

def erblet_file(file_path, visualize=False):
    if not USE_MATLAB:
        raise NotImplementedError('MATLAB is required to perform Erblet transforms')
    
    c, fc = eng.erblet(file_path, visualize, nargout=2)
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
        plt.pcolormesh(np.arange(len(signal)) / rate, freqs, np.abs(coefs), cmap='jet', shading='auto')
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

def transform_list(transform_file, file_list, file_names, *args, progress=True, affix=None, **kwargs):
    first_freqs = None
    def check_freq(transform):
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

    coefs_gen = (check_freq(transform_file(f, *args, **kwargs)) for f in file_list)
    if progress:
        coefs_gen = tqdm(coefs_gen, total=len(file_list))

    np.savez(affix + '_coefs', **dict(zip(transformed_names, coefs_gen)))
    np.save(affix + '_freqs', first_freqs)