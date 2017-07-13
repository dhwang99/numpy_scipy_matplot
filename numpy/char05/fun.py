#coding: utf8

from datetime import datetime
import numpy as np

import pdb
import matplotlib.pyplot as plt

#矩阵
A = np.mat('1 2 3;4 5 6;7 8 9')
print "Creation from a string:\n", A

print "transpose A:\n", A.T

print '#'*50
print 'create new A. the up is not inversable mat'

A = np.mat('1 2;3 4')
print "A:\n", A
print "Inverse A:\n", A.I
B = A * A.I
print "mat dot:\n", B
print "(A.I).I:\n", (A.I).I

A = np.mat(np.arange(9).reshape(3,3))
print "Creation from array:\n", A

#创建复合矩阵
A = np.eye(2)
B = 2 * A

print "A:\n", A
print "B:\n", B

print "Compound matrix (A B; B A):\n", np.bmat("A B;B A")

#通用函数
def ultimate_answer(a, b):
    result = np.zeros_like(a)
    result.flat = 42 * b * a
    print "a,b", a, b
    #pdb.set_trace()
    return result

ufunc = np.frompyfunc(ultimate_answer, 2, 1)
print "The answer:", ufunc(np.arange(4), 10)

print "The answer:", ufunc(np.arange(4).reshape(2,2), 10)

#通用函数方法
#ulimate function: add

a = np.arange(9)
print "Reduce:", np.add.reduce(a)

#like cumsum
print "Accumulate:", np.add.accumulate(a)

print "Reduceat:", np.add.reduceat(a, [0,5,2,7])

print "Outer:", np.add.outer(np.arange(3), a)

#数组除法
a = np.array([2, 6, 5])
b = np.array([1, 2, 3])

print "Divide:", np.divide(a,b), np.divide(b, a)
print "True Divide:", np.true_divide(a,b), np.true_divide(b, a)
#向下取整， 先divide, 再floor
print "Floor Divide", np.floor_divide(a,b), np.floor_divide(b,a)
c = 3.14 * b
print "Floor Divide 2", np.floor_divide(c,b), np.floor_divide(b,c)
print "floor Divide:", np.true_divide(a,b), np.true_divide(b, a)

print "divide:", a / b, b / (1.0 * a)
print "divide:", a // (1.0 * b), b // (1.0 * a)

#模运算
a = np.arange(-4, 4)
print "Remainder:", np.remainder(a, 2)
print "Mod:", np.mod(a, 2)
print "Mod:", a % 2 
print "Fmode:", np.fmod(a, 2)


#Fibonacci数列
F = np.matrix([[1,1],[1,0]])
print "F:", F

print "8th Fibonacci:", (F**7)[0,0]

#比奈特公式. Binet's Formula
n = np.arange(1,9)
sqrt5 = np.sqrt(5)
phi = (1 + sqrt5) / 2
fibonacci = np.rint((phi**n - (-1/phi)**n)/sqrt5)
print "Fibonacci:", fibonacci

#利萨茹曲线
#X = A*sina(at + n/2)
#Y = B*sin(bt)

a = 9
b = 8
a = float(a)
b = float(b)
t = np.linspace(-np.pi, np.pi, 201)
x = np.sin(a * t + np.pi/2)
y = np.sin(b * t)
plt.plot(x, y)
plt.savefig('images/lissajous.png', format='png')

#方波
#sum(4sin((2k -1)t)/((2k-1)pi)), k from 1 to limited

t = np.linspace(-np.pi, np.pi, 201)
k = np.arange(1, 99)
k = 2 * k - 1
f = np.zeros_like(t)

for i in range(len(t)):
    f[i] = np.sum(np.sin(k * t[i])/k)

plt.clf()
f = (4/np.pi) * f
plt.plot(t, f)
plt.savefig('images/fourier.png', format='png')

#锯齿波
t = np.linspace(-np.pi, np.pi, 201)
k = np.arange(1, 99)
f = np.zeros_like(t)
for i in range(len(t)):
    f[i] = np.sum(np.sin(2 * np.pi * k * t[i])/k)

f = (-2/np.pi) * f
plt.clf()
plt.plot(t, f, lw=1.0)
plt.plot(t, np.abs(f), lw=2.0)
plt.savefig('images/juchi.png', format='png')

#二进制
x=np.arange(-9,9)
y = -x
#求异或。操作数符号不等时，xor返回负值
print "Sign diffenent?", (x^y) < 0
print "Sign different?", np.less(np.bitwise_xor(x,y), 0)

#判断是否为2的幂数
print "Power of 2?\n", x, "\n", (x&(x-1)) == 0
print "Power of 2?\n", x, "\n", np.equal(np.bitwise_and(x, (x-1)), 0)

#求余(2幂数的余数)
print "Modules 4\n", x, '\n', x & ((1<<2) -1)
print "Modules 4\n", x, '\n', np.bitwise_and(x, np.left_shift(1,2) -1)

