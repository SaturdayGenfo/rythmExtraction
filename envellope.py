 # -*- coding: utf-8 -*-
from scipy.fftpack import fft, ifft
from scipy.signal import decimate
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def band(f1, f2, fe, N):
    p1 = 1.0*f1/fe * N
    p2 = 1.0*f2/fe *N
    return int(p1), int(p2)
    
def chopUp(s, fe):
    N = s.shape[0]
    bandes = [(0,200), (200,400), (400,800), (800, 1600), (1600,3200), (3200, fe//2)]
    signaux = []
    for (f1, f2) in bandes :
        b1, b2 = band(f1, f2, fe, N)
        filt = []
        for i in range(N//2):
            if i >=b1 and i< b2:
                filt.append(1)
            else :
                filt.append(0)
        filt = np.concatenate([filt, filt[::-1]])
        plt.plot(filt)
        signaux.append(ifft(np.multiply(fft(s), filt)))
    return signaux
    
def sommation(s, fe, dfact):   
    l = chopUp(s, fe)
    suum = np.zeros(270)
    for k in l:
        d = envelope(k, fe, dfact, False)
        print("Env done")
        onband = np.array(combBank(d, fe//dfact))
        suum = suum + 1.0/np.max(onband)*onband
        print("Combed")
    b = monotoneminimum(suum)
    suum = np.array(suum) - np.array(b)
    beat = np.argmax(suum)+ 30
    plt.xlim(30, 300)
    plt.plot(np.linspace(30,300,270),  suum)
    plt.title("Tempo éstimé : " + str(beat))
    plt.savefig("CombSum.png", dpi = 300)
    return suum


def maxfft(s, fe):
    N = s.shape[0]
    p = np.ceil(np.log2(N)) +1
    padd = np.concatenate([s, np.zeros(2**(p) - N)])
    tf= fft(padd)
    axfreq = np.linspace(0, 0.5*fe, 2**(p-1))
    plt.plot(axfreq, np.abs(tf[:tf.shape[0]//2]))
    plt.xlim(0, 5)
    mf = np.argmax(tf[2**p * 0.2/fe:2**p * 5.0/fe]) *fe*1.0/2**p + 0.2
    plt.title("BPM : " + str(mf*60))
    plt.show()
    
    

def autocorr(s, fe, dfact):
    l = chopUp(s, fe)
    bpm = []
    for k in l:
        d = envelope(k, fe, dfact, False)
        bpm.append(determinate_bpm(d, fe//dfact))
    return bpm
        

def determinate_bpm(s,fe, show=True) :
    autocorr = signal.fftconvolve(s, s[::-1], mode='full')
    N=s.shape[0]
    m = np.argmax(autocorr[N + 0.2*fe:N + 5*fe]) * (1.0/fe) + 0.2 
    if show:
        X = np.linspace((1-N)*1.0/fe,1.0*N/float(fe),2*N-1)
        plt.xlim(0.2, 5)
        plt.title("BPM: "+ str(1.0/m * 60))
        plt.plot(X,autocorr)
        plt.show()
    return  1.0/m * 60
    
def hhanning(fe):
    N = 0.2*fe
    X = np.arange(N//2, N)
    return 0.5*(1-np.cos(2.0*np.pi*X/N))

'''
def decimate(signal, k):
    N = signal.shape[0]
    d = np.zeros(N)
    for i in range(N//k):
        d[i] = signal[k*i]
    return d
'''
def envelope(s, fe, dfact, show=True, k=1000):

    N = s.shape[0]
    s2 = rectify(s) 
    h = np.concatenate([hhanning(fe), np.zeros(N-0.1*fe)])  
    tfs2 = fft(s2) 
    tfh = fft(h) 
    env = ifft(np.multiply(tfh, tfs2)) 
    denv = decimate(real(env), dfact)
    d = halfRectify(differentiate(denv, 50))
    

    if show:
        
        plt.subplot(3,1,1)
        plt.plot(np.arange(0, k/float(fe), 1.0/fe), s2[:k], 'b')
        plt.ylabel("Signal")
        cur_axes = plt.gca()
        cur_axes.axes.get_yaxis().set_ticklabels([])
        plt.subplot(3,1,2)
        plt.plot(np.arange(0, k/float(fe), dfact*1.0/fe), denv[:k], 'r')
        plt.ylabel("Enveloppe")
        cur_axes = plt.gca()
        cur_axes.axes.get_yaxis().set_ticklabels([])
        plt.subplot(3,1,3)
        plt.plot(np.arange(0, k/float(fe), dfact*1.0/fe),d[:k], 'y')
        plt.ylabel("Derivée")
        cur_axes = plt.gca()
        cur_axes.axes.get_yaxis().set_ticklabels([])
        plt.savefig('result.png', dpi=300)
        plt.show()

    return d
    

def monotoneminimum(s):
    d = []
    base = []
    for i in range(269):
        d.append(s[i+1]-s[i])
    d.append(d[-1])
    a = 0
    while a < 269:
        if d[a] <= 0:
            b = a+1
            while b < 269 and d[b] < 0:
                base.append(s[b-1])
                b += 1
                
            a = b
            base.append(s[a])
        else:
            b = a+1
            while b < 269 and s[b] > s[a]:
                base.append(s[a])
                b+=1
            a = b
            base.append(s[a])
    base.append(base[-1])
    return base
        

def real(s):
    r = np.vectorize(lambda x : x.real)
    return r(s)
        
def differentiate(env, n):

    N = env.shape[0]
    W = np.concatenate([np.ones(n), -1*np.ones(n)], 0)
    WEtendu = np.concatenate([W, np.zeros(N-2*n)], 0)

    derivee = ifft(np.multiply(fft(WEtendu), fft(env)))

    return real(derivee)
    
def combFilter(x, delay, fe):
    alpha = 0.5**(delay*1.75/fe)
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
