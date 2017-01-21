import numpy as np
import matplotlib.pyplot as plt

f = open("input.txt")
data = []
T = []
for line in f:
    s = line.split('|')
    if len(s) > 3:
        T.append(float(s[-1]) - 1481546370000)
        data.append(map(float, s[-2][1:-1].split(',')))

X = [x for (x,y,z) in data]

N = len(X)
Te = 1.0/N
plt.subplot(2, 1, 1)
plt.plot(X)
print(X)

tfd = np.fft.fft(X)
k = np.arange(N)
axefreq = np.linspace(0.0, 1.0/(2.0*Te), N//2)
plt.subplot(2, 1, 2)
plt.plot(axefreq, np.abs(tfd[:N//2]))

plt.show()
