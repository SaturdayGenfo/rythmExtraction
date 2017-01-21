import matplotlib.pyplot as plt


def read(show=True, k= 1000):
    filename = raw_input("File name : ")
    f = open(filename)
    data = []
    T = []
    for line in f:
        s = line.split('|')
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


def read(show=True, k= 1000):
    filename = raw_input("File name : ")
    f = open(filename)
    data = []
    T = []
    for line in f:
        s = line.split('|')
        T.append(float(s[-1]))
        data.append(map(float, s[-2][1:-1].split(',')))
    if show:
        draw(data)
    return data, T
