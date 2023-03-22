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

def y_hat(phi_M, t, phi_in=None):
    W_ML = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(phi_M), phi_M)), np.transpose(phi_M)), t)
    if phi_in is not None:
        y = np.dot(phi_in, W_ML)
    else:
        y = np.dot(phi_M, W_ML)
    return y

def y_hat_reg(phi_M, t, M, phi_in=None, l=0.1):
    W_ML = np.dot(np.dot(np.linalg.inv(l*np.identity(M) + np.dot(np.transpose(phi_M), phi_M)), np.transpose(phi_M)), t)
    if phi_in is not None:
        y = np.dot(phi_in, W_ML)
    else:
        y = np.dot(phi_M, W_ML)
    return y, W_ML

def plot_curve(x, t, y, M):
    plt.scatter(x, t)
    plt.plot(np.linspace(0, 3, 50), y)
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


def plot_MSE(MSE_list, dataset):
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1,31),MSE_list)
    plt.xlabel('M')
    plt.ylabel('MSE')
    plt.title('Mean Square Error on ' + dataset)
    plt.show()
    

def kfold(data_x, data_t):
    train_x = [data_x[10:50], np.concatenate([data_x[0:10], data_x[20:50]]), np.concatenate([data_x[0:20], data_x[30:50]]), np.concatenate([data_x[0:30], data_x[40:50]]), data_x[0:40]]
    val_x = [data_x[0:10], data_x[10:20], data_x[20:30], data_x[30:40], data_x[40:50]]
    train_t = [data_t[10:50], np.concatenate([data_t[0:10], data_t[20:50]]), np.concatenate([data_t[0:20], data_t[30:50]]), np.concatenate([data_t[0:30], data_t[40:50]]), data_t[0:40]]
    val_t = [data_t[0:10], data_t[10:20], data_t[20:30], data_t[30:40], data_t[40:50]]
    return train_x, val_x, train_t, val_t

def main():
    #read data and split
    x, t = read_csv('HW1.csv')
    x_train, t_train = np.array(x[:50]), np.array(t[:50])
    x_test, t_test = np.array(x[50:]), np.array(t[50:])
    M_list = [1, 3, 5, 10, 20, 30]

    #part1
    # for M in M_list:
    #     y = y_hat(phi(x_train, M), t_train, phi(np.linspace(0, 3, 50),M))
    #     plot_curve(x_train, t_train, y, M)
    

    # #part2
    # MSE_train = []
    # for M in range(1,31):
    #     y = y_hat(phi(x_train, M), t_train)
    #     MSE_train.append(MSE(t_train, y))
    # plot_MSE(MSE_train, "Training Set")
    # MSE_test = []
    # for M in range(1,31):
    #     y = y_hat(phi(x_train, M), t_train, phi(x_test, M))
    #     MSE_test.append(MSE(t_test, y))
    # plot_MSE(MSE_test, "Testing Set")
    

    # #part3
    # train_x, val_x, train_t, val_t = kfold(x_train, t_train)
    # MSE_min = 1000
    # bestM = 0
    # for M in range(1,21):
    #     CV_error = 0
    #     for i in range(5):
    #         y = y_hat(phi(train_x[i], M), train_t[i], phi(val_x[i], M))
    #         CV_error += (MSE(val_t[i], y))
    #     CV_error = CV_error/5
    #     #print(M, ":", CV_error)
    #     if CV_error < MSE_min:
    #         bestM = M
    #         MSE_min = CV_error

    # y = y_hat(phi(x_train, bestM), t_train, phi(x_test, bestM))
    # print(MSE(t_test, y))
    # y = y_hat(phi(x_train, bestM), t_train, phi(np.linspace(0, 3, 50), bestM))
    # plot_curve(x_train, t_train, y, bestM)


    # #part4-1
    # for M in M_list:
    #     y, W = y_hat_reg(phi(x_train, M), t_train, M, phi(np.linspace(0, 3, 50),M))
    #     plot_curve(x_train, t_train, y, M)

    #part4-2
    MSE_train = []
    for M in range(1,31):
        y, W = y_hat_reg(phi(x_train, M), t_train, M)
        MSE_train.append(MSE_reg(t_train, y, W))
    plot_MSE(MSE_train, "Training Set")
    MSE_test = []
    for M in range(1,31):
        y, W = y_hat_reg(phi(x_train, M), t_train, M, phi(x_test, M))
        MSE_test.append(MSE_reg(t_test, y, W))
    plot_MSE(MSE_test, "Testing Set")

if __name__ == '__main__':
    main()