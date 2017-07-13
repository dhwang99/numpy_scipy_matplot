#coding: utf8

from datetime import datetime
import numpy as np

import pdb
import matplotlib.pyplot as plt

def datestr2week(s):
    return datetime.strptime(s, '%Y/%m/%d').date().weekday()

def summarize(a, o, h, l, c):
    week_open = o[a[4]]
    week_close = c[a[0]]
    week_high = np.max(np.take(h, a))
    week_low = np.min(np.take(l, a))

    return ('apple', week_open, week_low, week_high, week_close)

#dates, close=np.loadtxt('data.csv', delimiter=',', usecols=(0,4), converters={1:datestr2week}, skiprows=1, unpack=True)
dates, open, high, low, close=np.loadtxt('data3.csv', delimiter=',', usecols=(0, 1, 2, 3, 4), skiprows=1, converters={0:datestr2week}, unpack=True)

print "Dates:", dates
print "close:", close

#计算每周报表，最高、最低、开、收盘价
dd=dates[:16]

first_friday = np.ravel(np.where(dd == 4))[0]
last_monday = np.ravel(np.where(dd == 0))[-1]

week_indices = np.arange(first_friday, last_monday + 1)

week_indices = np.split(week_indices, 3)

week_summary = np.apply_along_axis(summarize, 1, week_indices, open, high, low, close)

print week_summary

np.savetxt('weeksummary.csv', week_summary, delimiter=",", fmt="%s")

#计算20天股价真实波动幅度均值(average true range)
N = 20
h = high[-N:]
l = low[-N:]
preivous_close = close[-1-N:-1]

true_range = np.maximum(h-l, h - preivous_close, preivous_close - l)
print "true_range:", true_range
atr = np.zeros(N)
atr[0] = 0.0

for i in range(1, N):
    atr[i] = (N - 1) * atr[i-1] + true_range[i]
    print atr[i]
    atr[i] /= N

print "ATR:", atr

#计算20天内的简单平移
weights = np.ones(N) / N
#pdb.set_trace()
sma = np.convolve(weights, close)
sma = sma[N-1:-N + 1]
t = np.arange(N - 1, len(close))

plt.figure(num=2, figsize=(8, 6))
plt.plot(t, close[N-1:], lw=1.0)
plt.plot(t, sma, lw=2.0)
plt.savefig('images/sma.png', format='png')

#计算20天内的指数平移
weights = np.exp(np.linspace(-1, 0, N)) 
weights /= weights.sum()
ema = np.convolve(weights, close)[N-1:-N + 1]

plt.clf()
plt.figure(num=2, figsize=(8, 6))
plt.plot(t, close[N-1:], lw=1.0)
plt.plot(t, ema, lw=2.0)
plt.savefig('images/ema.png', format='png')

#绘制布林带
deviation = []
C = len(close)

for i in range(N-1, C):
    if i+N < C:
        dev = close[i:i+N]
    else:
        dev = close[-N:]

    averages = np.zeros(N)
    averages.fill(sma[i - N - 1])
    dev = dev - averages
    dev = dev ** 2
    dev = np.sqrt(np.mean(dev))
    deviation.append(dev)

deviation = 2 * np.array(deviation)

print len(deviation), len(sma)
upperBB = sma + deviation
lowerBB = sma - deviation

c_slice = close[N-1:]
between_bands = np.where((c_slice < upperBB) & (c_slice > lowerBB))

print lowerBB[between_bands]
print close[between_bands]
print upperBB[between_bands]

between_bands = len(np.ravel(between_bands))
print "Ratio between bands", float(between_bands) / len(c_slice)

t = np.arange(N-1, C)

plt.clf()
plt.figure(num=2, figsize=(8, 6))
plt.plot(t, c_slice, lw=1.0)
plt.plot(t, sma, lw=2.0)
plt.plot(t, upperBB, lw=3.0)
plt.plot(t, lowerBB, lw=4.0)
plt.savefig('images/boll_band.png', format='png')

#line model
#前20天股价，线性回归预测当前股价（最小二乘法）
b = close[-N:]
b = b[::-1]
print "b:", b

A = np.zeros((N, N), float)
print "Zeros by N", A

for i in range(N):
    A[i,] = close[-N-1-i:-1-i]

print "A", A

(x, residuals, rank, s) = np.linalg.lstsq(A, b)
print x, residuals, rank, s

print np.dot(b, x)

#line model 2
#前20天股价，线性回归预测当前股价（最小二乘法）
b = close[-1-N:-1]
print "b:", b
K=5
A = np.zeros((N, K), float)
A1 = np.zeros((N, K+1), float)
print "Zeros by N", A

for i in range(N):
    A[i,] = close[-2-K-i:-2-i]
    A1[i,] = np.hstack([close[-2-K-i:-2-i], [1]])

print "A", A

(x, residuals, rank, s) = np.linalg.lstsq(A, b)
(x1, residuals1, rank1, s1) = np.linalg.lstsq(A1, b)
#pdb.set_trace()
print x, residuals, rank, s
print x1, residuals1, rank1, s1

print "real b:", b[-1]
print np.dot(b[-K-1:-1], x)
print np.dot(b[-K-1:-1], x1[:-1]) + x1[-1]

M=10
l1=np.zeros(N, float)
l2=np.zeros(N, float)
l3=np.zeros(N, float)
A = np.zeros((M, K), float)
A1 = np.zeros((M, K+1), float)

#计算前几天的预测
#用二十天的值拟合比5天的值拟合要靠谱点
for j in range(N):
    c=close[j+1:]
    b=close[j+1:j+1+M]
    #for i in range(K):
    for i in range(M):
        A[i,] = c[i+1:i+K+1]
        A1[i,] = np.hstack([c[i+1:i+K+1], [1]])

    (x, residuals, rank, s) = np.linalg.lstsq(A, b)
    (x1, residuals1, rank1, s1) = np.linalg.lstsq(A1, b)

    l1[j] = np.dot(b[:K], x)
    l2[j] = np.dot(b[:K], x1[:-1]) + x1[-1]
    l3[j] = close[j]
    print "x, residuals, rank, s:", x, residuals, rank, s
    print "x1, residuals1, rank1, s1:", x1, residuals1, rank1, s1

#pdb.set_trace()
t=np.arange(N)
plt.clf()
plt.plot(t, l1)
#plt.plot(t, l2)
plt.plot(t, l3)
plt.savefig('images/gujia_lstsq.png', format='png')

#趋势线（用最高、低、收盘价均值作为枢轴点)

pivits = (high + low + close) / 3
def fit_line(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    rst = np.linalg.lstsq(A, y)
    return rst[0]

t =np.arange(len(close))
#pdb.set_trace()
sa, sb = fit_line(t, pivits - (high - low))
ra, rb = fit_line(t, pivits + (high - low))

support = sa * t + sb
resistance = ra * t + rb

condition = (close > support) & (close < resistance)

print "Condition:", condition
between_bands = np.where(condition)

print support[between_bands]
print close[between_bands]
print resistance[between_bands]

between_bands = len(np.ravel(between_bands))
print "Number points between bands", between_bands
print "Ratio between bands", float(between_bands) / len(close)

print "Tomorrow support:", sa * (t[-1] + 1) + sb
print "Tomorrow resistance", ra * (t[-1] + 1) + rb

#另一种计算支撑位和阻力位之间数据点个数的方法
a1 = close[close > support]
a2 = close[close < resistance]
print "Number of points between bands 2nd approach", len(np.intersect1d(a1, a2))

plt.clf()
plt.plot(t, close)
plt.plot(t, support)
plt.plot(t, resistance)
plt.savefig('images/support_resistance.png', format='png')

#数组修剪和压缩
a = np.arange(5)
print "a=", a
print "Clipped", a.clip(1, 2) # get element values in [1,2]
print "Comporessed", a.compress(a>2) #value greater than 2

#阶乘
b = np.arange(1,9)
print "b = ", b
print "Factorial ", b.prod()
print "Factorial2 ", b.cumprod()
