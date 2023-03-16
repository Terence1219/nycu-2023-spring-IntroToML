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

def basis(x):
    a = (x-mu)/0.1
    sigmoid = 1 / (1 + np.exp(-a))
    return a

def main():
    x, t = read_csv('HW1.csv')
    x_train, t_train = x[:50], t[:50]
    x_test, t_test = x[50:], t[50:]
    M_list = [1, 3, 5, 10, 20, 30]
    # plt.scatter(x,t)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('2D Scatter Plot')
    # plt.show()

if __name__ == '__main__':
    main()