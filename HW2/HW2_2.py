import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def read_xlsx(filename): #data index+1=class data[class][i][x1(0) or x2(1)] 

    df = pd.read_excel(filename)
    data = [[[], []], [[], []], [[], []]]
    for i in range(len(df)):
        index = int(df.iloc[i:i+1,3:4].to_numpy()[0][0] - 1)
        if index == 3:
            data[0][0].append(df.iloc[i:i+1,1].to_numpy()[0])
            data[0][1].append(df.iloc[i:i+1,2].to_numpy()[0])
        else:
            data[index][0].append(df.iloc[i:i+1,1].to_numpy()[0])
            data[index][1].append(df.iloc[i:i+1,2].to_numpy()[0])
    for d in data:
        np.array(d[0])
        np.array(d[1])
    return data

def gen_model(data):   
    sigma = np.cov(data[0])
    sigma_inv = np.linalg.inv(sigma)
    mu, p = mean_and_p(data)
    weight_v = []
    for i in range(len(mu)):
        wk = np.dot(sigma_inv, mu[i])
        wk0 = -0.5*(np.dot(np.dot(np.transpose(mu[i]), sigma_inv), mu[i])) + np.log(p[i])
        weight_v.append([wk, wk0])
    gen_predict(weight_v)

def mean_and_p(data):
    result = []
    p = []
    total = 0
    for d in data:
        result.append([np.mean(d[0]),np.mean(d[1])])
        p.append(len(d[0]))
        total += len(d[0])
    p = np.array(p)/total
    return result, p

def gen_predict(weight):
    result = [[[], []], [[], []], [[], []]]
    for x1 in range(101):
        for x2 in range(101):
            a_v = []
            for w in weight:
                x = [x1, x2]
                a = np.dot(np.transpose(w[0]), x) + w[1]
                a_v.append(a)
            i = np.argmax(a_v)
            result[i][0].append(x1)
            result[i][1].append(x2)
    gen_plot(result)
    plt.title('Generative model decision boundaries')
    plt.show()

def gen_plot(result):
    plt.scatter(result[0][0], result[0][1], c='b')
    plt.scatter(result[1][0], result[1][1], c='g')
    plt.scatter(result[2][0], result[2][1], c='r')  

def dis_model(data): #3 classes(K) 4 basis(M)
    phi = basis_v(data) #3*400*4 K*data*M
    weight = np.array([[0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01]]) #3*4 K*M
    for i in range(2):
        a_v = gen_a(weight, phi) #4*400*4 K*data*K
        y_v = softmax(a_v) #4*400 K*data
        gradient(y_v, phi)
        dis_predict(weight, data)
        print(weight)
        weight -= gradient(y_v, phi) * 0.001
        
def basis_v(data):
    result = [[],[],[]]
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            result[i].append(basis(data[i][0][j], data[i][1][j], data))
    return result

def basis(x, y, data):
    xy = [x,y]
    result = []
    result.append(multivariate_normal.pdf(xy, mean=[0,100], cov=np.cov(data[0])))
    result.append(multivariate_normal.pdf(xy, mean=[100,100], cov=np.cov(data[0])))
    result.append(multivariate_normal.pdf(xy, mean=[0,0], cov=np.cov(data[0])))
    result.append(multivariate_normal.pdf(xy, mean=[100,0], cov=np.cov(data[0])))
    np.array(result)
    return result

def gen_a(weight, phi):
    result = [[],[],[]]
    for i in range(len(phi)):
        for d in phi[i]:
            result[i].append(np.dot(weight, d))
        result[i] = np.array(result[i])
    return result

def softmax(a_v):
    result = [[],[],[]]
    for i in range(len(a_v)):
        for a in a_v[i]:
            denominator = np.sum(np.exp(a))
            result[i].append(np.exp(a[i])/denominator)
        result[i] = np.array(result[i])
    return result

def gradient(y_v, phi):
    result = []
    for i in range(len(y_v)):
        result.append(np.transpose(np.dot(y_v[i]-1, phi[i]))) #M
    result = np.array(result)
    return result #K*M
    
def dis_predict(weight, data):
    result = [[[], []], [[], []], [[], []]]
    for x1 in range(101):
        for x2 in range(101):
            phi = basis(x1, x2, data)
            a_v = np.dot(weight, phi)
            i = np.argmax(a_v)
            result[i][0].append(x1)
            result[i][1].append(x2)
    gen_plot(result)
    plt.title('Discriminative model decision boundaries')
    plt.show()

def main():
    data = read_xlsx('HW2.xlsx')
    gen_model(data)
    dis_model(data)
    

if __name__ == '__main__':
    main()