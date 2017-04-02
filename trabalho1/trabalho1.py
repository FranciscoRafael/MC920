from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt


img = misc.imread('peppers.png')




def bin(img, m, n, canal):
	count = 0
	vet_bin = np.zeros((256))
	for i in range (m):
		for j in range (n):
			vet_bin[img[i][j][canal]] = vet_bin[img[i][j][canal]] + 1
			count = count + 1
	return vet_bin


def hist(vet):
	y_axis = vet
	x_axis = range(len(y_axis))
	width_n = 0.4
	bar_color = 'black'

	plt.bar(x_axis, y_axis, width=width_n, color=bar_color, align='center')
	plt.show()


def escala_bin (vet, num_bin):
	bins = 256/num_bin
	reescala = np.zeros(num_bin)

	for i in range (len(reescala)):
		k = 0
		while (k < bins):
			reescala[i] = reescala[i] + vet[bins*i + k]
			k = k + 1

	return reescala


tam = img.shape
m = tam[0]
n = tam[1]

vet = bin(img, m, n, 0)
print vet[0:8]
vet2 = escala_bin(vet, 4)
print vet2
hist(vet2)