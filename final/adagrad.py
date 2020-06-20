import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def shuffle(X,Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize],Y[randomize])


def sigmoid(z):
    return np.clip(1.0/(1.0+np.exp(-z)),1e-6,1-1e-6)


def check(X,w,b):
    return sigmoid(np.add(np.dot(X,w),b))


def loss(yP,yL):
    return -np.dot(yL,np.log(yP))-np.dot((1-yL),np.log(1-yP))


def accuracy(yPred,yLabel):
    return np.sum(yPred == yLabel)/len(yPred)


def train(trainData,trainLabel,devData,devLabel,rate,max_iter,batch_size,num_train,num_dev):
    w = np.zeros(trainData.shape[1])
    b = np.zeros(1,)
    loss_train = []  # 训练集损失
    loss_validation = []  # 测试集损失
    train_acc = []  # 训练集准确率
    test_acc = []  # 测试集准确率
    eps = 1e-6
    times = max_iter + 1
    lrB = 0
    lrW = 0
    wGrad = []
    bGrad = []
    for i in range(1,max_iter+1):
        if i%1000 == 0:
            print('run ',i)
        x,y=shuffle(trainData,trainLabel)
        for j in range(int(np.floor(num_train/batch_size))):
            xS = x[j*batch_size:(j+1)*batch_size]
            yS = y[j*batch_size:(j+1)*batch_size]

            yP = check(xS,w,b)
            pError = yS - yP

            wGrad = -np.mean(np.multiply(pError,xS.T),1)
            bGrad = -np.mean(pError)

            lrW = lrW + wGrad ** 2
            lrB = lrB + bGrad ** 2

            w = w - (rate / np.sqrt(lrW+eps)) * wGrad
            b = b - (rate / np.sqrt(lrB+eps)) * bGrad
        yP = check(x,w,b)
        yh = np.round(yP)
        train_acc.append(accuracy(yh,y))
        loss_train.append(loss(yP,y)/num_train)

        yTP = check(devData,w,b)
        yTH = np.round(yTP)
        test_acc.append(accuracy(yTH,devLabel))
        loss_validation.append(loss(yTP,devLabel)/num_dev)

        if np.linalg.norm(wGrad) <= eps and np.linalg.norm(bGrad) <= eps :
            times = i
            break
    return w,b,times,loss_train,loss_validation,train_acc,test_acc

"""
df = pd.read_csv('incomeN.csv')
for i in range(1,df.shape[1]): #初始化 参数归一化
    x = df[str(i)].mean( axis = 0)
    y = df[str(i)].std()
    for j in range(0,df.shape[0]):
        tmp = df.loc[j,str(i)]
        df.loc[j,str(i)]=(tmp - x)/y
df.to_csv("incomePreMean.csv",index=False)

df = pd.read_csv('incomeN.csv')
for i in range(1,df.shape[1]): #初始化 参数归一化
    x = max(df[str(i)])
    y = min(df[str(i)])
    for j in range(0,df.shape[0]):
        tmp = df.loc[j,str(i)]
        df.loc[j,str(i)]=(tmp - y)/(x - y)
df.to_csv("incomePre.csv",index=False)
"""

"""
df = pd.read_csv('incomePre.csv')
#切分数据集
trainData = df.iloc[0:3000,0:57]
devData = df.iloc[3000:4000,0:57]
trainLabel = df.iloc[0:3000,57]
devLabel = df.iloc[3000:4000,57]


max_iter = 100000 #迭代次数
num_train = 3000
num_dev =1000
batch_size = 128
rateChoose = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0]
for rate in rateChoose:
    print('rate =',rate)
    filename = 'output/result' + '_' + str(rate) + '_01.txt'
    f = open(filename, 'w')
    print('01 rate =', rate,file = f)
    w,b,times,loss_train,loss_val,train_acc,test_acc = train(trainData.values,trainLabel.values,devData.values,devLabel.values,rate,max_iter,batch_size,num_train,num_dev)
    print(w,b,times,file=f)
    print(loss_train,loss_val,file=f)
    print(train_acc,test_acc,file=f)
    f.close()
"""
df = pd.read_csv('incomePreMean.csv')
#切分数据集
trainData = df.iloc[0:3000,0:57]
devData = df.iloc[3000:4000,0:57]
trainLabel = df.iloc[0:3000,57]
devLabel = df.iloc[3000:4000,57]


max_iter = 100000 #迭代次数
num_train = 3000
num_dev =1000
batch_size = 128
rateChoose = [0.1, 0.2, 0.4, 0.5, 1.0]
for rate in rateChoose:
    print('Mean rate =',rate)
    filename = 'output/result' + '_' + str(rate) + '_Mean.txt'
    f = open(filename, 'w')
    print('rate =', rate,file = f)
    w,b,times,loss_train,loss_val,train_acc,test_acc = train(trainData.values,trainLabel.values,devData.values,devLabel.values,rate,max_iter,batch_size,num_train,num_dev)
    print(w,b,times,file=f)
    print(loss_train,loss_val,file=f)
    print(train_acc,test_acc,file=f)
    f.close()






