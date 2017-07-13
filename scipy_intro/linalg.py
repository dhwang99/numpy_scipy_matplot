import numpy as np

#solve function

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([1,2,3])

X = np.linalg.solve(A, B)
print X
print np.dot(A, X) - B
