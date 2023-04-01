import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('out_ngp.txt', delimiter=' ', unpack=True)
data = np.log(data)
plt.imshow(data, cmap='plasma', interpolation='nearest')
plt.savefig('ngp.png')


data = np.loadtxt('out_cic.txt', delimiter=' ', unpack=True)
data = np.log(data)
plt.imshow(data, cmap='plasma', interpolation='nearest')
plt.savefig('cic.png')

data = np.loadtxt('out_tsc.txt', delimiter=' ', unpack=True)
data = np.log(data)
plt.imshow(data, cmap='plasma', interpolation='nearest')
plt.savefig('tsc.png')


data = np.loadtxt('out_psc.txt', delimiter=' ', unpack=True)
data = np.log(data)
plt.imshow(data, cmap='plasma', interpolation='nearest')
plt.savefig('psc.png')

