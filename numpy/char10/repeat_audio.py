#encoding=utf8

from scipy.io import wavfile
import matplotlib.pyplot as plt
import urllib2
import numpy as np
import sys

WAV_FILE='smashingbaby.wav'
'''
response = urllib2.urlopen('http://www.thesoundarchive.com/austinpowers/smashingbaby.wav')
print response.info()
fd = open(WAV_FILE, 'w')
fd.write(response.read())
fd.close()
'''

sample_rate, data = wavfile.read(WAV_FILE)
print "Data type", data.dtype, 'Shape', data.shape

plt.subplot(2, 1, 1)
plt.title('Original')
plt.plot(data)

plt.subplot(2, 1, 2)
repeated = np.tile(data, 4)
plt.title('Repeated')
plt.plot(repeated)
wavfile.write('repeated_yababy.wav', sample_rate, repeated)
plt.savefig('images/repeated_wav.png', format='png')
