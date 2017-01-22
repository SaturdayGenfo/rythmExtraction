 # -*- coding: utf-8 -*-
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

def hhanning(fe):
    N = 0.1*fe
    X = np.arange(N//2, N)
    return 0.5*(1-np.cos(2.0*np.pi*X/N))

def envelope(s, a, fe, show=True, k=1000):


    square = np.vectorize(lambda x: x**2)  #mettre au carré toute un vecteur
    power = np.vectorize(lambda x: a**x) # idem mais a puissance x


    s2 = rectify(s[:k]) #mettre au carré toutes les données

    h = np.concatenate([hhanning(fe), np.zeros(k-0.05*fe)])  # vecteur contenant toutes les puissances de a

    tfs2 = fft(s2) #transformée de Fourrier du signal au carré
    tfh = fft(h) #transformée de Fourier de h


    env = ifft(np.multiply(tfh, tfs2)) 
    denv = np.zeros(k//4)

    for i in range(k//4):
        denv[i] = env[4*i]
        
    d = halfRectify(differentiate(env, 50))
    
    smoothd = peaks(d, 100)


    if show:
        plt.subplot(4,1,1)
        plt.plot(s2[:k], 'b')
        plt.subplot(4,1,2)
        plt.plot(env[:k], 'r')
        plt.subplot(4,1,3)
        plt.plot(d[:k], 'y')
        plt.subplot(4,1,4)
        plt.plot(smoothd[:k], 'g')
        plt.show()

    return env

def passbas(signal, a, k):
    power = np.vectorize(lambda x: a**x)
    h = power(np.arange(k))
    tfh = fft(h)
    return ifft(np.multiply(tfh, fft(signal)))

def differentiate(env, n):

    N = env.shape[0]
    W = np.concatenate([np.ones(n), -1*np.ones(n)], 0)
    WEtendu = np.concatenate([W, np.zeros(N-2*n)], 0)

    derivee = ifft(np.multiply(fft(WEtendu), fft(env)))

    return derivee

def theta(signal, k, i):
    minIndex = max(k-i, 0)
    maxIndex= min(k+i, signal.shape[0])
    return np.median(signal[minIndex:maxIndex])

def peaks(signal, i):
    N = signal.shape[0]
    p = []
    highest = 0
    for k in range(N):
        if signal[k] > theta(signal, k, i):
            p.append(signal[k])
            highest = max(highest, signal[k])
        else:
            p.append(0)
    for k in range(N):
        if p[k] < 0.85*highest :
            p[k] = 0
    return np.array(p)
def rectify(signal):
    return np.abs(signal)
def halfRectify(signal):
    return np.array([max(s, 0) for s in signal])
