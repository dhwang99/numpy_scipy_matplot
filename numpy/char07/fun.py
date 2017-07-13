#encoding=utf8

import datetime
import numpy as np

def datestr2num(s):
    return datetime.datetime.strptime(s, '%Y/%m/%d').toordinal()

#按字典排序
#Date,Open,High,Low,Close,Volume,Adj Close
#2015/11/27,155.2,155.5,152.3,153.2,16407000,153.2

dates,closes=np.loadtxt('../char3/0700.hk.csv', delimiter=',', usecols=(0,4), skiprows=1,converters={0:datestr2num}, unpack=True)

indices = np.lexsort((dates, closes))

print "Indices", indices
print ["%s %s" % (datetime.date.fromordinal(int(dates[i])), closes[i]) for i in indices] 

#复数排序
np.random.seed(42)
complex_number = np.random.random(5) + 1j * np.random.random(5)
print "Complex numbers:\n", complex_number
print "Sorted:\n", np.sort_complex(complex_number)

#搜索
a = np.array([2,4,8,6])
np.argmax(a)
np.argwhere(a<=4)

#
a = np.arange(10)
#找到插入位置
indices = np.searchsorted(a, [-2, 7, 10])
print indices
print "The full array:", np.insert(a, indices, [-2, 7, 10])

#抽取元素. 和where类似
a = np.arange(7)
condition = (a % 2) == 0
print a
#返回元素值
print "Even numbers:", np.extract(condition, a)
print "Even numbers indicates:", np.where(condition)[0]
#返回元素索引
print "Non zero:", np.nonzero(a)

a = np.arange(7) + 1
condition = (a % 2) == 0
print a
print "Even numbers:", np.extract(condition, a)
print "Even numbers indicates:", np.where(condition)[0]
print "Non zero:", np.nonzero(a)

#以下为金融函数
#计算终值： future value
#以利率3%、每季度付金额10，存5年，现值1000的终值
fval = np.fv(0.03/4, 5*4, -10, -1000)
import matplotlib.pyplot as plt
print "Future value", fval

#改变年数
fvals = []

for i in xrange(1, 10):
    fvals.append(np.fv(0.03/4, i*4, -10, -1000))

plt.plot(fvals, 'bo')
plt.savefig('images/fv.png', format='png')

#计算现值
#注：因为是计算支出的现金流，结果前为负号
print "Present value", np.pv(0.03/4, 5*4, -10, fval)

#计算净现值。不明白？
#按折现率计算的净现金流之和
cashflows = np.random.randint(100, size=5)
cashflows = np.insert(cashflows, 0, -100)
print "Cashflows:", cashflows
print "Net present value:", np.npv(0.03, cashflows)

#内部收益率
#净现值为0时的有效利率
print "Internal rate of return", np.irr(cashflows)

#分期付款
#借30年，年利率为10%. 等额本息
print "Payment:", np.pmt(0.10/12, 12 * 30, 1000000)

#计算付款期数
print "Number of payments:", np.nper(0.10/12, -100, 9000)

#等额本金计算。麻烦点，每个月要算利息

#窗函数

#巴特利特窗
window = np.bartlett(42)
plt.clf()
plt.plot(window)
plt.savefig('images/bartlett.png', format='png')

#布莱克曼窗
#三项余弦的和： w(n) = 0.42 - 0.5*cos(2*pi*n/M) + 0.08*cos(4*pi*n/M)
#使用布莱克曼窗平滑股价
#用到了卷积。忘了，需要了解下
N = 5
window = np.blackman(N)
closes_i = closes[:30]
smoothed = np.convolve(window/window.sum(), closes_i, mode='same')
plt.clf()
plt.plot(smoothed[N:-N], lw=2, label="smoothed")
plt.plot(closes_i[N:-N], label='closes')
plt.legend(loc='best')
plt.savefig('images/blackman.png', format='png')

#汉明窗
#加权余弦函数 w(n) = 0.54 + 0.46(cos(2*pi*n/(M-1)), 0<=n<=M-1
window = np.hamming(42)
plt.clf()
plt.plot(window)
plt.savefig('images/hamming.png', format='png')

#凯泽窗
#w(n) = Io(beta * sqrt(1-4*n**2/(M-1)**2))/Io(beta). 其中Io为零阶贝塞尔函数
window = np.kaiser(42, 14)
plt.clf()
plt.plot(window)
plt.savefig('images/kaiser.png', format='png')

#专用函数
#修正的贝塞尔函数
x = np.linspace(0, 4, 100)
vals = np.i0(x)
plt.clf()
plt.plot(x, vals)
plt.savefig('images/bessel_function.png', format='png')

#sinc function
vals = np.sinc(x)
plt.clf()
plt.plot(x, vals)
plt.savefig('images/sinc_function.png', format='png')

xx = np.outer(x, x)
vals = np.sinc(xx)

implot = plt.imshow(vals)
implot.write_png('images/sinc_2d_function.png')
