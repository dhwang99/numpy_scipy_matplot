import numpy as np
import matplotlib.pyplot as plt
import math


#case 2
x=[]
y=[]
num=0.0
while num < math.pi * 4:  
    y.append(math.sin(num))  
    x.append(num)  
    num += 0.1  
plt.plot(x, y, 'b')
plt.savefig('images/plot2.png', format='png')

