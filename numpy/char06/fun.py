#coding=utf8
import numpy as np
import pdb

A = np.mat("0 1 2;1 0 3; 4 -3 8")
print "A\n", A

inverse = np.linalg.inv(A)
print "inverse\n", inverse

print "check\n", A * inverse
print "check\n", inverse * A


#解方程组
A = np.mat("1 -2 1; 0 2 -8;-4 5 9")
print "A\n", A
b = np.array([0, 8, -9])
print "b\n", b

x = np.linalg.solve(A, b)
print "solution", x

print "Check", np.dot(A, x)

#特征值&特征向量
A = np.mat('3 -2; 1 0')
print 'A\n', A

print "Eigenvalues", np.linalg.eigvals(A)

#第一个是特征值数组，第二个是对应的特征向量矩阵
eigenvalues, eigenvectors = np.linalg.eig(A)
pdb.set_trace()
print "First tuple of eig", eigenvalues
print "Second tuple of eig", eigenvectors

for i in range(len(eigenvalues)):
    print "Left", np.dot(A, eigenvectors[:, i])
    print "Right", eigenvalues[i] * eigenvectors[:, i]
    print ""

#SVD
A = np.mat("4 11 14;8 7 -2")
print "A\n", A

U, Sigma, V = np.linalg.svd(A, full_matrices=False)
print "U\n", U
print "Sigma\n", Sigma
print "V\n", V
print 

print "Product\n", U * np.diag(Sigma) * V

#广义逆矩阵
A = np.mat("4 11 14; 8 7 -2")
print "A\n", A

pseudoinv = np.linalg.pinv(A)
print "Pseudo inverse\n", pseudoinv

print "Check", A * pseudoinv

#determinant(行列式）, 与方阵相关的一个标量值

A = np.mat('3 4;5 6')
print "A\n", A

print "Determinant", np.linalg.det(A)

#快速傅里叶变换
x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)

transformed = np.fft.fft(wave)
print np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9)

from matplotlib import pyplot as plt

plt.plot(transformed)
plt.savefig('images/fft_transformed.png', format='png')

#移频
shifted = np.fft.fftshift(transformed)
print np.all(np.abs(np.fft.ifftshift(shifted) - transformed) < 10 ** -9)

plt.clf()
plt.plot(transformed, lw=2)
plt.plot(shifted, lw=3)
plt.savefig('images/fft_shifted.png', format='png')

#随机数。模拟随机游走
cash = np.zeros(10000)
cash[0] = 1000
outcome = np.random.binomial(9, 0.5, size=len(cash))

for i in range(1, len(cash)):
    if outcome[i] < 5:
        cash[i] = cash[i-1] - 1
    elif outcome[i] < 10:
        cash[i] = cash[i-1] + 1
    else:
        raise AssertionError("Unexpected outcome " + outcome)

print "outcome min and max:", outcome.min(), outcome.max()

plt.clf()
plt.plot(np.arange(len(cash)), cash)
plt.savefig('images/cash_random_walk.png', format='png')

#超几何分布
points = np.zeros(100)
outcomes = np.random.hypergeometric(25, 1, 3, size=len(points))

for i in range(len(points)):
    if outcomes[i] == 3:
        points[i] = points[i-1] + 1
    elif outcomes[i] == 2:
        points[i] = points[i-1] - 6
    else:
        print outcomes[i]

plt.clf()
plt.plot(np.arange(len(points)), points)
plt.savefig('images/hypergeometric.png', format='png')

#绘制正态分布
plt.clf()

N = 10000
normal_values = np.random.normal(size=N)
dummy, bins, dummy = plt.hist(normal_values, np.sqrt(N), normed=True, lw=1)
sigma = 1
mu = 0
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma**2)), lw = 2)
plt.savefig('images/normal.png', format='png')
plt.clf()

lognormal_values = np.random.lognormal(size=N)
dummy, bins, dummy = plt.hist(lognormal_values, np.sqrt(N), normed=True, lw=1)
sigma = 1
mu = 0

x = np.linspace(min(bins), max(bins), len(bins))
pdf = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

plt.plot(x, pdf, lw=3)
plt.savefig('images/log_normal.png', format='png')

