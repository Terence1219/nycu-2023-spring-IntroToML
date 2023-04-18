import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print(p)
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

def gen_plot(result):
    plt.scatter(result[0][0], result[0][1], c='b')
    plt.scatter(result[1][0], result[1][1], c='g')
    plt.scatter(result[2][0], result[2][1], c='r')
    #plt.scatter(result[3][0], result[3][1], c='k')
    plt.title('Generative model decision boundaries')
    plt.show()

def main():
    data = read_xlsx('HW2.xlsx')
    gen_model(data)
    
    

if __name__ == '__main__':
    main()