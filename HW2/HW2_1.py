import numpy as np
import pandas as pd

def read_xlsx(filename): #data index+1=class data[class][i] = [x1,x2]

    df = pd.read_excel(filename)

    data = [[], [], [], []]
    for i in range(len(df)):
        data[df.iloc[i:i+1,3:4].to_numpy()[0][0] - 1].append(df.iloc[i:i+1,1:3].to_numpy()[0])
    
    return data
    

def main():
    data = read_xlsx('HW2.xlsx')

if __name__ == '__main__':
    main()