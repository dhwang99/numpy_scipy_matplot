
import sys
from datetime import datetime
import numpy as np

def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    
    return c

def pythonsum(n):
    a = range(n)
    b = range(n)
    c = []

    for i in range(n):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])


    return c

size = int(sys.argv[1])

start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print "last 2 val:", c[-2:]
print "time used as microseconds:", delta.microseconds

start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print "last 2 val:", c[-2:]
print "time used as microseconds:", delta.microseconds
