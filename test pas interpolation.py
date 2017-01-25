


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
    return I, pas
        

