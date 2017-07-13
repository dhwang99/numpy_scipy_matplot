#encoding=utf8

import numpy as np
import pdb
from scipy import io
from scipy  import stats
from scipy import signal
from scipy import fftpack
from scipy import optimize
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt

from datetime import date
from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
from matplotlib.dates import MonthLocator
from matplotlib.finance import quotes_historical_yahoo_ochl 
from matplotlib.finance import candlestick_ochl 

a = np.arange(8)
io.savemat('a.mat', {"array":a})

b = io.loadmat('a.mat')
print a
print b['array']

#分析随机数

#按正态生成随机数
generated = stats.norm.rvs(size=900)
#用正态分布fit数据
print "Mean", "Std", stats.norm.fit(generated)

#计算偏度：概率分布的偏斜（非对称）程度。即观察到的数据集服从正态分布的概率
print "Skewtest", "pvalue", stats.skewtest(generated)

#峰度：陡峭程度
print "Kurtosistest", "pvalue", stats.kurtosistest(generated)

#正态性检验
print "Normaltest", "pvalue", stats.normaltest(generated)

#该区段里，95%处的值
print "95 percentile", stats.scoreatpercentile(generated, 95)

#数值1所在的百分比
print "Percent at 1", stats.percentileofscore(generated, 1)

#该图的直方图
plt.hist(generated, np.sqrt(len(generated)))
plt.savefig("images/hist.png", format='png')


from scipy  import stats
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import jarque_bera

from matplotlib.finance import quotes_historical_yahoo_ochl 
from datetime import date

def get_close(symbol):
    today = date.today()
    start = (today.year - 1, today.month, today.day)
    quotes = quotes_historical_yahoo_ochl(symbol, start, today)
    quotes = np.array(quotes)

    return quotes.T

#计算对数收益率
spy = np.diff(np.log(get_close('SPY')[4]))
dia = np.diff(np.log(get_close('DIA')[4]))

#均值检验
print "Means comparison:", stats.ttest_ind(spy, dia)

#Kolmogorov-Smirnov检验。两组样本同分布的可能性
print "Kolmogorov smirnov test", stats.ks_2samp(spy, dia)

#Jarque bera正态性检验
print "Jarque Bera test", jarque_bera(spy - dia)[1]

#对数收益率和差值直方图
plt.clf()
plt.hist(spy, histtype='step', lw=1, label='SPY')
plt.hist(dia, histtype='step', lw=2, label='DIA')
plt.hist(spy - dia, histtype='step', lw=3, label='Delta')
plt.legend()
plt.savefig('images/pair.png', format='png')


quotes = get_close('QQQ')
qqq = quotes[4]
dates = quotes[0]
#pdb.set_trace()
y = signal.detrend(qqq)

alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(dates, qqq, 'o', dates, qqq - y, '-') 

ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(month_formatter)
fig.autofmt_xdate()
plt.savefig('images/signal.png', format='png')


#傅立叶分析
plt.clf()
fig = plt.figure()
fig.subplots_adjust(hspace=.3)
ax = fig.add_subplot(211)

ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(month_formatter)
#调大字号
#ax.tick_params(axis='both', which='major', labelsize='x-large')
ax.tick_params(axis='both', which='major', labelsize='x-small')
amps = np.abs(fftpack.fftshift(fftpack.rfft(y)))
amps[amps < 0.1 * amps.max()] = 0
filtered = -fftpack.irfft(fftpack.ifftshift(amps))

plt.plot(dates, y, 'o', label='detrended') 
plt.plot(dates, filtered, label='filtered')

fig.autofmt_xdate()
#plt.legend(prop={'size':'x-large'})
plt.legend(prop={'size':'x-small'})

ax2 = fig.add_subplot(212)
ax2.tick_params(axis='both', which='major', labelsize='x-large')
N = len(qqq)
f = np.linspace(-N/2, N/2, N)
plt.plot(f, amps, label='transformed')
#plt.legend(prop={'size':'x-large'})
plt.legend(prop={'size':'x-small'})

plt.savefig('images/fft.png', format='png')

#数学优化
#最小二乘法，sin拟合
plt.clf()
fig = plt.figure()
fig.subplots_adjust(hspace=.3)
ax = fig.add_subplot(211)

ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(month_formatter)
ax.tick_params(axis='both', which='major', labelsize='x-small')

def residuals(p, y, x):
    A,k,theta,b = p
    err = y - A * np.sin(2*np.pi*k*x + theta) + b
    return err

p0 = [filtered.max(), f[amps.argmax()]/(2*N), 0, 0]
print "P0", p0

plsq = optimize.leastsq(residuals, p0, args = (filtered, dates))
p = plsq[0]
plt.plot(dates, y, 'o', label='detrended') 
plt.plot(dates, filtered, label='filtered')
plt.plot(dates, p[0] * np.sin(2*np.pi*dates*p[1] + p[2]) + p[3], '^', label='fit')
fig.autofmt_xdate()
#plt.legend(prop={'size':'x-large'})
plt.legend(prop={'size':'x-small'})

ax2 = fig.add_subplot(212)
ax2.tick_params(axis='both', which='major', labelsize='x-large')
N = len(qqq)
f = np.linspace(-N/2, N/2, N)
plt.plot(f, amps, label='transformed')
#plt.legend(prop={'size':'x-large'})
plt.legend(prop={'size':'x-small'})

plt.savefig('images/leastsq.png', format='png')

#积分。梯形法计算
#计算高斯积分
print "Gaussian integral", np.sqrt(np.pi), integrate.quad(lambda x:np.exp(-x**2), -np.inf, np.inf)

#插值
x = np.linspace(-18, 18, 36)
noise = 0.1 * np.random.random(len(x))
signal = np.sinc(x) + noise

x2 = np.linspace(-18, 18, 180)
interpreted = interpolate.interp1d(x, signal)
cubic = interpolate.interp1d(x, signal, kind='cubic')

y = interpreted(x2)
y2 = cubic(x2)

plt.clf()
plt.plot(x, signal, 'o', label='data')
plt.plot(x2, y, '-', label='linear')
plt.plot(x2, y2, '-', lw=2, label='cubic')
plt.savefig('images/interpreted.png', format='png')

#图像处理

