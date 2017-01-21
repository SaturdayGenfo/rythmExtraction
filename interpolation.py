


def trouver_le_pas(T) :
    i = 0
    freq={}
    for i in range(1,len(T)) :
        pas = round(T[i]-T[i-1],2)
        if (pas in freq) :
            freq[pas] +=1
        else :
            freq[pas] =1
    ranking = sorted(freq.items(),key=lambda x : x[1])
    print(ranking)
    return(ranking[-1][0])


def interpolation(T,data) :
    pas = trouver_le_pas(T)  
    tmax=T[-1]
    tmin=T[0]
    time_interval=tmax-tmin
    n=int(time_interval/pas)
    I=[data[0]]
    i=1
    j=1
    while(i<n) :
        if (T[j]>=(i*pas+tmin)) :
            while (T[j]>=(i*pas+tmin)) :
                I.append((i*pas+tmin-T[j-1])*(data[j]-data[j-1])/(T[j]-T[j-1])+data[j-1])
                i=i+1
        j=j+1
    return I
        

import matplotlib.pyplot as plt
import numpy as np

T = [0, 0.5, 0.9, 1.5, 2.0, 2.6, 3.1, 3.5, 4.0, 4.4, 4.9, 5.5, 6.0, 7.0, 7.5]

data = [i**4 for i in T]

I = interpolation(T, data)

plt.plot(np.linspace(0, 8, 1000), [i**4 for i in np.linspace(0, 8, 1000)] , color='r')

plt.plot([0.5*i for i in range(15)], I, marker='o', color='b')

plt.show()
