import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def tfd(s, Fe, padd = True, show=True):
    N = s.shape[0]
    if padd:
        p = np.ceil(np.log2(N))
        s = np.concatenate([s, np.zeros(2**(p) - N)])
        N = 2**p
        
    tf = fft(s)
    axefreq = np.linspace(0, 0.5* Fe, N//2)
    if show:
        plt.subplot(2, 1, 1)
        plt.xlabel("Temps")
        plt.plot(s, 'g')
        plt.ylabel("Amplitude")
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticklabels([])
        cur_axes.axes.get_yaxis().set_ticklabels([])
        plt.subplot(2, 1, 2)
        plt.plot(axefreq, np.abs(tf[:N//2]), 'r')
        cur_axes = plt.gca()
        cur_axes.axes.get_yaxis().set_ticklabels([])
        plt.xlabel("Frequences")
        plt.ylabel("Module")
        plt.savefig('filt2.png', dpi=300)
        plt.show()
    return tf


