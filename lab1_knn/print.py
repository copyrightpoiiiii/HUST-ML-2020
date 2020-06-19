import re
import matplotlib.pyplot as plt
file = open("output/result.txt")
sum = []
for line in file :
    tmp = re.findall(r"\d+\.?\d*",line)
    sum.append((int(tmp[0]),float(tmp[3])))
sum.sort(key = lambda x:x[0])
x = []
y = []
for poi in sum:
    if poi[0] <= 100 :
        x.append(poi[0])
        y.append(poi[1])
fig  = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,y)
fig.suptitle('misclassification rate', fontsize = 14, fontweight='bold')
ax.set_xlabel("k")
ax.set_ylabel("correct rate")
plt.show()



