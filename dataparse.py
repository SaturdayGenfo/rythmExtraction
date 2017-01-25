import matplotlib.pyplot as plt


def read(show=True, k= 1000):
    filename = raw_input("File name : ")
    f = open(filename)
    data = []
    T = []
    for line in f:
        s = line.split('|')
        print(s)
        T.append(float(s[-1]))
        data.append(map(float, s[-2][1:-1].split(',')))
    if show:
        draw(data)
    return data, T



def draw(data) :
    plt.subplot(3, 1, 1)
    plt.plot([x for (x,y,z) in data])
    plt.ylabel('Axe x')

    plt.subplot(3, 1, 2)
    plt.plot([y for (x,y,z) in data])
    plt.ylabel('Axe y')

    plt.subplot(3, 1, 3)
    plt.plot([z for (x,y,z) in data])
    plt.ylabel('Axe z')

    plt.show()

def trouver_le_pas(T) :
    i = 0
    freq={}
    for i in range(1,len(T)) :
        pas = round(T[i]-T[i-1],2)
        if pas < 0:
            print("WTF", i)
        if (pas in freq) :
            freq[pas] +=1
        else :
            freq[pas] =1
    ranking = sorted(freq.items(),key=lambda x : x[1])
    print(ranking)
    return(ranking[-1][0])


def interpolation(data, T) :
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