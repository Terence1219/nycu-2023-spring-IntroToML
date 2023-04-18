import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def read_xlsx(filename): #data index+1=class data[class][i][x1(0) or x2(1)] 

    df = pd.read_excel(filename)
    data = [[[], []], [[], []], [[], []], [[], []]]
    for i in range(len(df)):
        data[int(df.iloc[i:i+1,3:4].to_numpy()[0][0] - 1)][0].append(df.iloc[i:i+1,1].to_numpy()[0])
        data[int(df.iloc[i:i+1,3:4].to_numpy()[0][0] - 1)][1].append(df.iloc[i:i+1,2].to_numpy()[0])
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
    result = [[[], []], [[], []], [[], []], [[], []]]
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

def gen_plot(result):
    plt.scatter(result[0][0], result[0][1], c='b')
    plt.scatter(result[1][0], result[1][1], c='g')
    plt.scatter(result[2][0], result[2][1], c='r')
    plt.scatter(result[3][0], result[3][1], c='k')
    plt.title('Generative model decision boundaries')
    plt.show()

def dis_model(data):
    phi, m, c = basis(data)
    weight = np.array([0.001, 0.01, 0.01, 0.1])
    for i in range(100):
        a_v = gen_a(weight, phi)
        y_v = softmax(a_v)
        
        weight -= gradient(y_v, phi) * 10
        print(weight)
    dis_predict(weight, m, c)

def basis(data):
    result = [[],[],[],[]]
    m = [np.mean(data[0][0]),np.mean(data[0][1])]
    c = np.cov(data[0])
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            #result[i].append(multivariate_normal.pdf([data[i][0][j] ,data[i][1][j]], mean=[np.mean(data[i][0]),np.mean(data[i][1])], cov=np.cov(data[i])))
            result[i].append(multivariate_normal.pdf([data[i][0][j] ,data[i][1][j]], mean=m, cov=c))
    return result, m, c

def gen_a(weight, phi):
    result = [[],[],[],[]]
    for i in range(len(phi)):
        for d in phi[i]:
            result[i].append(weight*d)
        result[i] = np.array(result[i])
    return result

def softmax(a_v):
    result = [[],[],[],[]]
    for i in range(len(a_v)):
        for a in a_v[i]:
            denominator = np.sum(np.exp(a))
            # ans = []
            # for ak in a:
            #     ans.append(np.exp(ak)/denominator)
            result[i].append(np.exp(a[i])/denominator)
        result[i] = np.array(result[i])
    return result

def dis_predict(weight, m, c):
    result = [[[], []], [[], []], [[], []], [[], []]]
    for x1 in range(101):
        for x2 in range(101):
            phi = multivariate_normal.pdf([x1 ,x2], mean=m, cov=c)
            a_v = []
            for w in weight:
                a = w*phi
                a_v.append(a)
            #print(weight)
            i = np.argmax(a_v)
            result[i][0].append(x1)
            result[i][1].append(x2)
    gen_plot(result)

def gradient(y_v, phi):
    result = []
    for i in range(len(y_v)):
        result.append(np.dot(y_v[i]-1, np.transpose(phi[i])))
    result = np.array(result)
    return result

def main():
    data = read_xlsx('HW2.xlsx')
    #gen_model(data)
    dis_model(data)
    
    

if __name__ == '__main__':
    main()