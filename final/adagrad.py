import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rate = 0.2


def da(y,y_p,x):
    return (y-y_p)*(-x)


def db(y,y_p):
    return (y-y_p)*(-1)


def calc_loss(a,b,x,y):
    tmp = y - (a * x + b)
    tmp = tmp ** 2
    SSE = sum(tmp) / (2 * len(x))
    return SSE


def draw_hill(x,y):
    a = np.linspace(-20,20,100)
    print(a)
    b = np.linspace(-20,20,100)
    x = np.array(x)
    y = np.array(y)

    allSSE = np.zeros(shape=(len(a),len(b)))
    for ai in range(0,len(a)):
        for bi in range(0,len(b)):
            a0 = a[ai]
            b0 = b[bi]
            SSE = calc_loss(a0,b0,x,y)
            allSSE[ai][bi] = SSE
    a,b = np.meshgrid(a,b)
    return [a,b,allSSE]


df = pd.read_csv('incomeN.csv')
for i in range(1,df.shape[1]):
    x = max(df[str(i)])
    y = min(df[str(i)])
    for j in range(0,df.shape[0]-1):
        tmp = df.loc[j,str(i)]
        df.loc[j,str(i)]=(tmp - y)/(x - y)
df.to_csv("incomePre.csv",index=False)



