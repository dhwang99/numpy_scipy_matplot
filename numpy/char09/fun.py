#encoding=utf8

import numpy as np
import matplotlib.pyplot as plt

func = np.poly1d(np.array([1,2,3,4]).astype(float))

x = np.linspace(-10, 10, 30)

#
y = func(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.savefig('images/polyld.png', format='png')

func1 = func.deriv(m=1)
y1 = func1(x)

#多子图
plt.clf()
#plt.plot(x, y, 'r', x, y1, 'g')
plt.plot(x, y, 'ro', x, y1, 'g--')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.savefig('images/polyldwithderiv.png', format='png')

func2 = func.deriv(m=2)
y2 = func2(x)
plt.clf()
plt.subplot(311)
plt.plot(x, y, 'r-')
plt.title('Polynomial')

plt.subplot(312)
plt.plot(x, y1, 'b^')
plt.title('First Derivative')

plt.subplot(313)
plt.plot(x, y2, 'go')
plt.title('Second Derivative')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('images/multiimg.png', format='png')

from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
from matplotlib.dates import MonthLocator
from matplotlib.finance import quotes_historical_yahoo_ochl 
from matplotlib.finance import candlestick_ochl 
import sys
from datetime import date

today = date.today()
start = (today.year -1 , today.month, today.day)
alldays = DayLocator()
months = MonthLocator()

month_formatter = DateFormatter("%b %Y")

symbol = "DISH"
quotes = quotes_historical_yahoo_ochl(symbol, start, today)

#k线图
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)

ax.xaxis.set_major_formatter(month_formatter)
candlestick_ochl(ax, quotes)
fig.autofmt_xdate()
plt.savefig('images/kline_cadlestick.png', format='png')

#直方图. 需要了解下算法和意义
#等区间内的频次分布，一般类似概率分布图。如正态
import pdb
quotes = np.array(quotes)
#pdb.set_trace()
close = quotes.T[4]
plt.clf()
plt.hist(close, int(np.sqrt(len(close))))
plt.savefig('images/hist.png', format='png')

#对数图
dates = quotes.T[0]
volume = quotes.T[5]
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
#对数坐标画图
plt.semilogy(dates, volume)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)

ax.xaxis.set_major_formatter(month_formatter)
fig.autofmt_xdate()
plt.savefig('images/semilogy.png', format='png')

#散点图. 这个图的作用还不太理解，有时间了解下
#股票收益率和成交量的散点图
ret = np.diff(close)/close[:-1]
volchange = np.diff(volume)/volume[:-1]
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(ret, volchange, c=ret * 100, s=volchange*100, alpha=0.5)
ax.set_title('close and volume returns')
ax.grid(True)
plt.savefig('images/scatter.png', format='png')

#着色
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(dates, close)
plt.fill_between(dates, close.min(), close, where=close>close.mean(), facecolor="green", alpha=0.4)
plt.fill_between(dates, close.min(), close, where=close<close.mean(), facecolor="red", alpha=0.4)

ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)

ax.xaxis.set_major_formatter(month_formatter)
ax.grid(True)
fig.autofmt_xdate()
plt.savefig('images/fillbetween.png', format='png')

#使用图例和注释
plt.clf()
fig=plt.figure()
ax=fig.add_subplot(111)
emas = []
for i in range(9, 18, 3):
    weights = np.exp(np.linspace(-1.0, 0., i))
    weights /= weights.sum()
    ema = np.convolve(weights, close)[i-1:-i+1]
    idx = (i - 6)/3
    ax.plot(dates[i-1:], ema, lw=idx, label="EMA(%s)"%i)
    data = np.column_stack((dates[i-1:], ema))
    emas.append(np.rec.fromrecords(data, names=["dates", "ema"]))

first = emas[0]["ema"].flatten()
second = emas[1]["ema"].flatten()
bools = np.abs(first[-len(second):] - second)/second < 0.0001
xpoints = np.compress(bools, emas[1])

for xpoint in xpoints:
    ax.annotate('x', xy=xpoint, textcoords='offset points', xytext=(-50, 30), arrowprops=dict(arrowstyle="->"))

leg = ax.legend(loc='best', fancybox=True)
leg.get_frame().set_alpha(0.5)
ax.plot(dates, close, lw=1.0, label="Close")
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)

ax.xaxis.set_major_formatter(month_formatter)
ax.grid(True)
fig.autofmt_xdate()
plt.savefig('images/legend.png', format='png')

#三维绘图
#y=x*x + y*y
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u=np.linspace(-1, 1, 100)
x, y = np.meshgrid(u, u)
z = x ** 2 + y ** 2
ax.plot_surface(x, y, z, rstride=4, cstride=4, cmap=cm.YlGnBu_r)
plt.savefig('images/3d.png', format='png')

#等高线
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(x, y, z)
plt.savefig('images/contour.png', format='png')

#动画
