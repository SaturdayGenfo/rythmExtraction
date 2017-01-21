import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def tfd(s, Fe, N, show=True, k=1000):
    tf = 1.0/N*fft(s)
    axefreq = np.linspace(0, 0.5* Fe, N//2)
    if show:
        plt.subplot(2, 1, 1)
        plt.plot(s[:k])
        plt.ylabel("Amplitude")
        plt.xlabel("Temps")
        plt.subplot(2, 1, 2)
        plt.plot(axefreq, np.abs(tf[:N//2]), 'r')
        plt.xlabel("Frequences")
        plt.ylabel("Module")
        plt.show()
    return tf


