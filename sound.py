from scipy.io.wavfile import read
from fourrier import tfd

def audioIn(show=True, k=1000):
    filename = raw_input("File name : ")
    audio = read(filename)
    try :
        s = audio[1][:,0]
    except IndexError:
        s = audio[1]
    Fe = audio[0]
    N = s.shape[-1]
    return s, tfd(s, Fe, N, show, k), Fe
