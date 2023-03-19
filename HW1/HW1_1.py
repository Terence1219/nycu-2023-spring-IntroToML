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

def phi(X, M):
    design = []
    for xi in X:
        row = [1]
        for j in range(1,M):
            mu = 3*j / M
            row.append(sigmoid((xi-mu)/0.1))
        design.append(row)
    return design

def main():
    x, t = read_csv('HW1.csv')
    x_train, t_train = np.array(x[:50]), np.array(t[:50])
    x_test, t_test = np.array(x[50:]), np.array(t[50:])
    M_list = [1, 3, 5, 10, 20, 30]
    for M in range(2,3):
        phi_M = phi(x_train, M)
        W_ML = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(phi_M), phi_M)), np.transpose(phi_M)), t_train)
        y_hat = np.dot(phi_M, W_ML)
    #print(np.shape(phi(x_train,3)))
    plt.plot(x_train,t_train)
    #plt.plot(x_train,y_hat)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    main()