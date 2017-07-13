#encoding=utf8
import numpy as np

a = np.arange(5)
print a.dtype
print a

#数组shape(维度)
m=np.array([np.arange(2), np.arange(2)])
print m.shape

#数据索引&下标
a = np.array([[1,2],[3,4]])
print a
print a[0,0],a[0,1],a[1,0],a[1,1]

#数据类型
print np.arange(7, dtype='f')
print np.arange(7, dtype='D')

#数据切片
a = np.arange(9)
print a[3:7]
#以上长为2选取
print a[:7:2]
#负向选取
print a[::-1]

#多维数组的切片和索引
b = np.arange(24).reshape(2,3,4)
print b.shape
print b
print b[:,0,0]
print b[0]
print b[0,:,:]
print b[0,...]
print b[0,1,::2]
print b[:,1]
print b[0,:,1]
print b[0,:,-1]
print b[0,::-1,-1]
print b[0,::2,-1]

#改变数组维度
print b.ravel()
#以下重新分配了内存
print b.flatten()  

b.shape = (6,4)
print b

#转置 
print b.transpose()

b.resize((2,12))

#数组组合
a = np.arange(9).reshape(3,3)
print a
b = a * 2
print b
#水平组合
print np.hstack((a,b))
print np.concatenate((a,b), axis=1) 
#垂直组合
print np.vstack((a,b))
print np.concatenate((a,b), axis=0) 
#深度组合
print np.dstack((a,b))

#列组合
#对一维有效, 把一维当成一列。二维和hstack相同
oned = np.arange(3)
twice_oned = 2 * oned

print np.column_stack((oned, twice_oned))
print np.column_stack((oned, a))

#行组合
print np.row_stack((oned, twice_oned))
print np.row_stack((oned, a))

#数组分割
print np.hsplit(a, 3)
print np.split(a, 3, axis=1)

print np.vsplit(a, 3)
print np.split(a, 3, axis=0)

c = np.arange(27).reshape(3,3,3)
print c
print np.dsplit(c, 3)

#数组属性
print b.dtype
print b.shape
print b.size, b.ndim, b.itemsize
b.resize(6,4)
print b
print b.T
#一维数组，T属性为原数组
d=np.array([1.j + 1, 2.j + 3])
print d.real
print d.imag
print d.dtype
print d.dtype.str

#扁平迭代器
f = b.flat
for item in f: print item
print b.flat[2]
print b.flat[[1,3]]

b.flat[[1,3]] = 1
print b

#数组转换
print b
print b.tolist()
