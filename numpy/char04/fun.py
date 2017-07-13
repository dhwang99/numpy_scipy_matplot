#coding: utf8

from datetime import datetime
import numpy as np

import pdb
import matplotlib.pyplot as plt

def datestr2week(s):
    return datetime.strptime(s, '%Y-%m-%d').date().weekday()

td, to, th, tl, tc, tv, ta = np.loadtxt('../data/tencent2.csv', delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6), skiprows=1, converters={0:datestr2week}, unpack=True)

bd, bo, bh, bl, bc, bv, ba = np.loadtxt('../data/bidu2.csv', delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6), skiprows=1, converters={0:datestr2week}, unpack=True)

#简单收益率
tencent_returns = np.diff(tc) / tc[:-1]
bidu_returns = np.diff(bc) / bc[:-1]

#协方差
covariance = np.cov(bidu_returns, tencent_returns)

print "covariance:" , covariance
print "covariance diagonal:" , covariance.diagonal()
print "covariance trace:" , covariance.trace()

#相关系数。后一个是正式定义
print "Correlation coefficient", covariance / (bidu_returns.std() * tencent_returns.std())
print "Correlation coefficient", np.corrcoef(bidu_returns, tencent_returns)

#判断股价是否同步：上一交易日差值 - 平均差值  超过 差值标准差2倍
difference = bc - tc
avg = np.mean(difference)
dev = np.std(difference)

print "Out of ysnc"< np.abs(difference[-1] - avg) > 2 * dev

t = np.arange(len(bidu_returns))
plt.clf()
plt.plot(t, bidu_returns, lw=1, label='bidu')
plt.plot(t, tencent_returns, lw=2, label='tencent')
plt.legend(loc='upper left')
plt.savefig('images/price_diff.png', format='png')

#多项式拟合（本例用三次）
t =np.arange(len(bc))
#pdb.set_trace()
poly = np.polyfit(t, bc - tc, 3)
print "Polynomial fit:", poly

print "Next value ", np.polyval(poly, t[-1] + 1)
#求多项式的根
print "Roots:", np.roots(poly)

#求多项式的导数
der = np.polyder(poly)
print "Derivative:", der
#求多项式的导数的根.即极值点
print np.roots(der)
#用argmax,argmin求极值
vals = np.polyval(poly, t)
print "max:", np.argmax(vals), " min:", np.argmin(vals)

plt.clf()
plt.plot(t, bc - tc)
plt.plot(t, vals)
plt.savefig('images/poly.png', format='png')

#用简单平移后的数据进行多项式拟合（本例用三次）
N=5
weights = np.ones(N) / N
sma_bc = np.convolve(weights, bc)[N-1:-N-1]
sma_tc = np.convolve(weights, tc)[N-1:-N-1]
t =np.arange(len(sma_bc))
poly = np.polyfit(t, sma_bc - sma_tc, 3)
print "Polynomial fit:", poly

print "Next value ", np.polyval(poly, t[-1] + 1)
#求多项式的根
print "Roots:", np.roots(poly)

#求多项式的导数
der = np.polyder(poly)
print "Derivative:", der
#求多项式的导数的根.即极值点
print np.roots(der)
#用argmax,argmin求极值
vals = np.polyval(poly, t)
print "max:", np.argmax(vals), " min:", np.argmin(vals)

plt.clf()
plt.plot(t, sma_bc - sma_tc)
plt.plot(t, vals)
plt.savefig('images/sma_poly.png', format='png')

#用指数平移后的数据进行多项式拟合（本例用三次）
N=5
weights = np.ones(N) / N
weights = np.exp(np.linspace(-1.0, 0., N))
weights /= weights.sum()
sma_bc = np.convolve(weights, bc)[N-1:-N-1]
sma_tc = np.convolve(weights, tc)[N-1:-N-1]
t =np.arange(len(sma_bc))
poly = np.polyfit(t, sma_bc - sma_tc, 3)
print "Polynomial fit:", poly

print "Next value ", np.polyval(poly, t[-1] + 1)
#求多项式的根
print "Roots:", np.roots(poly)

#求多项式的导数
der = np.polyder(poly)
print "Derivative:", der
#求多项式的导数的根.即极值点
print np.roots(der)
#用argmax,argmin求极值
vals = np.polyval(poly, t)
print "max:", np.argmax(vals), " min:", np.argmin(vals)

plt.clf()
plt.plot(t, sma_bc - sma_tc)
plt.plot(t, vals)
plt.savefig('images/ema_poly.png', format='png')

#计算OBV
bchange=np.diff(bc)
print "bidu change:", bchange

signs = np.sign(bchange)
print "Signs:", signs

pieces = np.piecewise(bchange, [bchange < 0, bchange > 0], [-1, 1]) #好巧啊
print "Pieces:", pieces

print "Arrays equal?", np.array_equal(signs, pieces)
print "On balance volume:", bv[1:] * signs

def calc_profit(open, high, low, close):
    br = 0.95
    buy = open * float(0.95)
    if low < buy < high:
        return (close - buy) / buy
    else:
        return 0.

func = np.vectorize(calc_profit)

profits = func(bo, bh, bl, bc)
print "Profits:", profits

real_trades = profits[profits != 0]
#pdb.set_trace()
print "Number of trades", len(real_trades), round(100.0 * len(real_trades) / len(bc), 3), "%"
print "Average profit/loss %", round(100.0 * np.mean(real_trades), 2), "%"

winning_trades = profits[profits > 0]
print "Number of trades", len(winning_trades), round(100.0 * len(winning_trades) / len(bc), 3), "%"
print "Average profit %", round(100.0 * np.mean(winning_trades), 2), "%"

losing_trades = profits[profits < 0]
print "Number of trades", len(losing_trades), round(100.0 * len(losing_trades) / len(bc), 2), "%"
print "Average loss %", round(100.0 * np.mean(losing_trades), 2), "%"

#hanning 平滑.
#有其它平滑函数：hamming,blackman, bartlett, kaiser
N=8
weights = np.hanning(N)
print "Weights", weights

smooth_bidu = np.convolve(weights, bidu_returns)[N-1:-N+1]
smooth_tencent = np.convolve(weights, tencent_returns)[N-1:-N+1]

t=np.arange(N-1, len(bidu_returns))
plt.clf()
plt.plot(t, bidu_returns[N-1:], lw=1.0, label='bidu_returns')
plt.plot(t, smooth_bidu, lw=2.0, label='bidu_smooth')
plt.plot(t, tencent_returns[N-1:], lw=1.0, label='tencent_returns')
plt.plot(t, smooth_tencent, lw=2.0, label='tencent_smooth')

plt.legend(loc='upper left')
plt.savefig('images/hanning_smooth.png', format='png')

K=3
t=np.arange(N-1, len(tencent_returns))
poly_bidu = np.polyfit(t, smooth_bidu, K)
poly_tencent = np.polyfit(t, smooth_tencent, K)

poly_sub = np.polysub(poly_bidu, poly_tencent)
xpoints = np.roots(poly_sub)
print "Intersection points:", xpoints

reals = np.isreal(xpoints)
print "Real number?", reals

xpoints = np.select([reals], [xpoints])
xpoints = xpoints.real

print "Real intersection points", xpoints

print "Sans 0s", np.trim_zeros(xpoints)
