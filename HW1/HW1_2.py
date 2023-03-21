import csv
import matplotlib.pyplot as plt
import numpy as np

def read_csv(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        x = []
        t = []
        for row in csv_reader:
            x.append(row[0])
            t.append(row[1])
    return np.float_(x[1:]), np.float_(t[1:])

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def phi(X, M=10):
    design = []
    for xi in X:
        row = [1]
        for j in range(1,M):
            mu = 3*j / M
            row.append(sigmoid((xi-mu)/0.1))
        design.append(row)
    return design
# def phi(X, M=10):
#     design = []
#     for xi in X:
#         row = [1]
#         for j in range(1,M):
#             row.append(xi**j)
#         design.append(row)
#     return design


def SN(phi_x):
    SNinv = (10^-6)*np.identity(10) + np.dot(np.transpose(phi_x), phi_x)
    return np.linalg.inv(SNinv)

def mN(S, phi_x, t):
    return np.dot(S, np.dot(np.transpose(phi_x),t))

def plot_curve(x, t, y, N, std):
    plt.scatter(x, t)
    plt.plot(np.linspace(0, 3, 50), y)
    plt.plot(np.linspace(0, 3, 50), y+std)
    plt.plot(np.linspace(0, 3, 50), y-std)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("N = " + str(N))
    plt.show()

def var(phi_x, S):
    return 1 + np.dot(np.dot(phi_x, S), np.transpose(phi_x))

def std_dev(v):
    result = []
    for i in range(len(v)):
        result.append(np.sqrt(np.absolute(v[i][i])))
    return result

def main():

    #read data and split
    x, t = read_csv('HW1.csv')
    x_train, t_train = np.array(x[:50]), np.array(t[:50])
    N_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    for N in N_list:
        x_in, t_in = x_train[:N], t_train[:N]
        S = SN(phi(x_in))
        post_mean = mN(S, phi(x_in), t_in)
        v = var(phi(np.linspace(0, 3, 50)), S)
        std = std_dev(v)
        y = np.dot(phi(np.linspace(0, 3, 50)), post_mean)
        plot_curve(x_in, t_in, y, N, std)
        

if __name__ == '__main__':
    main()