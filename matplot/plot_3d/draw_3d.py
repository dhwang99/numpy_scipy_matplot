# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:17:13 2015

@author: Eddy_zheng
"""

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio 

import pdb

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

pdb.set_trace()

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.savefig('images/3d1.png', format='png')

plt.clf()

mat1 = '4a.mat' #这是存放数据点的文件，需要它才可以画出来。上面有下载地址
data = sio.loadmat(mat1)
m = data['data']

x,y,z = m[0],m[1],m[2]
ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程

#将数据点分成三部分画，在颜色上有区分度
ax.scatter(x[:1000],y[:1000],z[:1000],c='y') #绘制数据点
ax.scatter(x[1000:3000],y[1000:3000],z[1000:3000],c='r')
ax.scatter(x[3000:],y[3000:],z[3000:],c='g')

ax.set_zlabel('Z') #坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')

plt.savefig('images/3d2.png', format='png')

plt.clf()

def f(x,y):
    z = (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
    return z
 
n = 256
 
#均匀生成-3到3之间的n个值
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
#生成网格数据
X,Y = np.meshgrid(x,y)
#pdb.set_trace()
 
fig = plt.figure()
#2行2列的子图中的第一个，第一行的第一列
subfig1 = fig.add_subplot(2,2,1)
#画等值线云图
surf1 = plt.contourf(X, Y, f(X,Y))
#添加色标
fig.colorbar(surf1)
#添加标题
plt.title('contourf+colorbar')
 
#d第二个子图，第一行的第二列
subfig2 = fig.add_subplot(2,2,2)
#画等值线
surf2 = plt.contour(X, Y, f(X,Y))
#等值线上添加标记
plt.clabel(surf2, inline=1, fontsize=10, cmap='jet')
#添加标题
plt.title('contour+clabel')
 
#第三个子图，第二行的第一列
subfig3 = fig.add_subplot(2,2,3,projection='3d')
#画三维边框
surf3 = subfig3.plot_wireframe(X, Y, f(X,Y), rstride=10, cstride=10, color = 'b')
#画等值线
#plt.contour(X, Y, f(X,Y))
#设置标题
plt.title('plot_wireframe+contour')
 
#第四个子图，第二行的第二列
subfig4 = fig.add_subplot(2,2,4,projection='3d')
#画三维图
surf4 = subfig4.plot_surface(X, Y, f(X,Y), rstride=1, cstride=1, cmap='jet',
        linewidth=0, antialiased=False)
#设置色标
fig.colorbar(surf4)
#设置标题
plt.title('plot_surface+colorbar')
plt.savefig('images/3d3.png', format='png')
