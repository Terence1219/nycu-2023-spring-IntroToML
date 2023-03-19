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

def y_hat(phi_M,t):
    W_ML = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(phi_M), phi_M)), np.transpose(phi_M)), t)
    y = np.dot(phi_M, W_ML)
    return y

def y_hat_reg(phi_M, t, M , l=0.1):
    W_ML = np.dot(np.dot(np.linalg.inv(l*np.identity(M) + np.dot(np.transpose(phi_M), phi_M)), np.transpose(phi_M)), t)
    y = np.dot(phi_M, W_ML)
    return y, W_ML

def plot_curve(x, t, y, M):
    plt.scatter(x,t)
    plt.plot(x,y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("M = " + str(M))
    plt.show()

def MSE(t, y):
    err = np.subtract(t, y)
    square_err = np.power(err,2)
    return np.sum(square_err) / 2

def MSE_reg(t, y, W, l=0.1):
    err = np.subtract(t, y)
    square_err = np.power(err,2)
    data_term =  np.sum(square_err) / 2
    reg_term = np.sum(np.power(W,2)) * l / 2
    return data_term + reg_term


def plot_MSE(MSE_list):
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1,31),MSE_list)
    plt.xticks(np.arange(1,31))
    plt.yticks(np.arange(0,401,50))
    plt.xlabel('M')
    plt.ylabel('MSE')
    plt.title('Mean Square Error')
    plt.show()

def main():

    #read data and split
    x, t = read_csv('HW1.csv')
    x_train, t_train = np.array(x[:50]), np.array(t[:50])
    x_test, t_test = np.array(x[50:]), np.array(t[50:])
    M_list = [1, 3, 5, 10, 20, 30]

    #part1
    # for M in M_list:
    #     y = y_hat(phi(x_train, M), t_train)
    #     plot_curve(x_train, t_train, y, M)
    
    #print(np.shape(phi(x_train,3)))

    #part2
    # MSE_list = []
    # for M in range(1,31):
    #     y = y_hat(phi(x_train, M), t_train)
    #     MSE_list.append(MSE(t_train, y))
    # plot_MSE(MSE_list)

    #part4-1
    # for M in M_list:
    #     y = y_hat_reg(phi(x_train, M), t_train, M)
    #     plot_curve(x_train, t_train, y, M)

    #part4-2
    # MSE_list = []
    # for M in range(1,31):
    #     y, W = y_hat_reg(phi(x_train, M), t_train, M)
    #     MSE_list.append(MSE_reg(t_train, y, W))
    # plot_MSE(MSE_list)

if __name__ == '__main__':
    main()