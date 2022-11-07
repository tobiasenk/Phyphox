import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def cm2inch(value):
    """cm to inch"""
    return value/2.54

def timeseries_plot(x, y, x_label, x_unit, y_label, y_unit):
    """plot of a preprocessed signal as time series with individual unit as ylabel and meter based xaxis"""
    size = [16.1, 4]
    plt.figure(figsize=(cm2inch(size[0]), cm2inch(size[1])))
    plt.plot(x, y, 'o-', linewidth=0.5)
    plt.ylabel(str(y_label)+' '+str(y_unit))
    plt.xlabel(str(y_label)+' '+str(x_unit))
    plt.xlim(x.iloc[0], x.iloc[-1])
    plt.grid()
    plt.tight_layout()
    return plt

def fft(x, y):
    """windowed fft of a timeseries signal"""
    # https://www.cbcity.de/die-fft-mit-python-einfach-erklaert
    # figsize
    size = [16.1, 4]
    # Real Signal
    window_width = len(y)
    # window function to get periodic signals from real function
    window = np.hanning(window_width)
    # amplitude spectra
    FFTY = np.fft.fft(window * y)  # axis=-1) # Discrete Fourier Transformation with fft and hann window
    N = int(len(FFTY) / 2 + 1)  # half fft signal length
    FFTY = np.abs(FFTY[:N])  # only real
    # Set real physical axis
    dt = x[1] - x[0]
    fa = 1.0 / dt  # scan frequency
    #print('dt=%.5fs (Sample Time)' % dt)
    #print('fa=%.2fHz (Frequency)' % fa)
    # X vector starts with 0 ends to the maximum frequency, which can be reconstructed fa/2, filled with N (length of half FFT)
    # maximal frequency which can be constructed = half of max sampling frequency fa (Nyquist-Frequency)
    FFTX = np.linspace(0, fa/2, N, endpoint=True)  # maximale X-Achse
    # real physical y-axis with 2/N
    FFTY = FFTY * 2 / N
    # write results in dict
    FFT = {}
    FFT['x'] = FFTX
    FFT['y'] = FFTY
    return FFT

def stft(x, y, window_width=1, mode='stft'):
    """spectrogram of a timeseries signal"""
    # constants
    dt = x[1] - x[0]
    fa = 1.0 / dt  # scan frequency
    window_width = window_width  # (native x_axis unit)
    window = np.hanning(window_width)
    movement = window_width//16
    nperseg = int(window_width / dt)  # window width of FFT in steps. Example: 100m/0.1m = 1000 steps as window width for 100m
    noverlap = nperseg - int(movement / dt)  # overlap between two windows
    # Spectrogram
    if mode == 'spectrogram':
        f, t, Sxx = signal.spectrogram(
            x=y,
            fs=dt,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=None,
            detrend='constant',
            return_onesided=True,
            scaling='density', axis=-1,
            mode='psd')
    elif mode == 'stft':
        f, t, Sxx = signal.stft(
            x=y,
            fs=dt,
            nperseg=nperseg,
            return_onesided=True,
            scaling='psd')
    t_new = np.linspace(start=x[0], stop=x.iloc[-1], num=len(t), endpoint=True)
    # write results in dict
    STFT = {}
    STFT['t'] = t
    STFT['t_new'] = t_new
    STFT['f'] = f
    STFT['Sxx'] = Sxx
    return STFT
