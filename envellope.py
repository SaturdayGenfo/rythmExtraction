 # -*- coding: utf-8 -*-
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

def hhanning(fe):
    N = 0.1*fe
    X = np.arange(N//2, N)
    return 0.5*(1-np.cos(2.0*np.pi*X/N))

def envelope(s, fe, show=True, k=1000):

    s2 = rectify(s[:k]) #mettre au carré toutes les données
    h = np.concatenate([hhanning(fe), np.zeros(k-0.05*fe)])  
    tfs2 = fft(s2) #transformée de Fourrier du signal au carré
    tfh = fft(h) #transformée de Fourier de h
    env = ifft(np.multiply(tfh, tfs2)) 
    denv = np.zeros(k//4)

    for i in range(k//4):
        denv[i] = env[4*i]
        
    d = halfRectify(differentiate(env, 5))


    if show:
        
        plt.subplot(3,1,1)
        plt.plot(s2[:k], 'b')
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticklabels([])
        cur_axes.axes.get_yaxis().set_ticklabels([])
        plt.ylabel("Signal")
        plt.subplot(3,1,2)
        plt.plot(env[:k], 'r')
        plt.ylabel("Enveloppe")
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticklabels([])
        cur_axes.axes.get_yaxis().set_ticklabels([])
        plt.subplot(3,1,3)
        plt.plot(d[:k], 'y')
        plt.ylabel("Derivée")
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticklabels([])
        cur_axes.axes.get_yaxis().set_ticklabels([])
        plt.savefig('result.png', dpi=300)
        plt.show()

    return d

def differentiate(env, n):

    N = env.shape[0]
    W = np.concatenate([np.ones(n), -1*np.ones(n)], 0)
    WEtendu = np.concatenate([W, np.zeros(N-2*n)], 0)

    derivee = ifft(np.multiply(fft(WEtendu), fft(env)))

    return derivee
    
def combFilter(x, delay, fe):
    alpha = 0.8
    y = np.concatenate([np.zeros(delay), np.zeros_like(x)])
    for i in range(0, x.shape[0]):
        y[i+delay] = alpha*y[i] + (1-alpha)*x[i]
    filt = ifft(np.multiply(fft(y[delay:]), fft(x)))
    return filt

def energy(s):
    return np.sum(np.abs(np.multiply(s, s)))
    
def combBank(s, fe):
    combs = []
    for bpm in range(30, 300):
        f = bpm/60.0 
        delay = int(np.ceil(fe/f))
        combs.append(energy(combFilter(s, delay, fe)))
    return combs
         
def rectify(signal):
    return np.abs(signal)
def halfRectify(signal):
    return np.array([max(s, 0) for s in signal])
