import numpy as np
import matplotlib.pyplot as plt
 
#case 1
x = np.linspace(0, 10)
line, = plt.plot(x, np.sin(x), '--', linewidth=2)
plt.savefig('images/plot1.png', format='png')
