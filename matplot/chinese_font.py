#coding=utf8
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

#case 1
x = np.linspace(0, 10)
line, = plt.plot(x, np.sin(x), '--', linewidth=2)
plt.savefig('images/plot1.png', format='png')

#case 2
x=[]
y=[]
num=0.0
while num < math.pi * 4:  
    y.append(math.sin(num))  
    x.append(num)  
    num += 0.1  
plt.clf()
plt.plot(x, y, 'b')
plt.savefig('images/plot2.png', format='png')

#case 3

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"./msyh.ttf") 

#matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  #设置缺省字体
plt.clf()
xData = np.arange(0, 12, 1)
yData1 = xData.__pow__(2.0)
yData2 = np.arange(15, 71, 5)
plt.figure(num=2, figsize=(8, 6))
font.set_size(14)
plt.title(u'Plot 1中国',  fontproperties=font)
font.set_size(10)
plt.xlabel(u'x-axis测试', fontproperties=font)
plt.ylabel(u'y-轴s', fontproperties=font)
s=u'平方和'
line, = plt.plot(xData, yData1, color='b', linestyle='--', marker='o', label=s)
#line.set_label(s)
s=u'直线5'
plt.plot(xData, yData2, color='r', linestyle='-', label=s)

font.set_size(10)
plt.legend(loc='upper left', prop=font)
plt.savefig('images/plot3.png', format='png')
