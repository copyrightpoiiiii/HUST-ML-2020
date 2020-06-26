import matplotlib.pyplot as plt
import numpy as np

plt.title('loss function', fontsize = 14, fontweight='bold')

f = open('output/mean-01.txt','r')
data = f.read()
arr = data.strip('[]').split(',')
y = []
x = []
y_stick = []
x_stick = range(1,len(arr),20000)
for i in range (1,2000,10):
    y.append(float(format(float(arr[i]),'.5f')))
    x.append(i)
x = np.array(x)
y = np.array(y)
plt.plot(x,y,label = 'training data loss function')

f = open('output/mean-2.txt','r')
data = f.read()
arr = data.strip('[]').split(',')
y = []
x = []
y_stick = []
x_stick = range(1,len(arr),20000)
for i in range (1,2000,10):
    y.append(float(format(float(arr[i]),'.5f')))
    x.append(i)
x = np.array(x)
y = np.array(y)
plt.plot(x,y,label = 'val data loss function')

plt.legend()
plt.show()

"""
f = open('output/result_0.01_Mean.txt','r')
data = f.readlines()
for i in range(0,len(data)):
    data[i] = data[i].rstrip('\n')
print(data[3])
"""
