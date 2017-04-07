from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

img_peppers = misc.imread('peppers.png')
img_baboon = misc.imread('baboon.png')


def bin_image(img, m, n):
	r = np.zeros((256))
	g = np.zeros((256))
	b = np.zeros((256))

	for i in range (m):
		for j in range (n):
			r[img[i][j][0]] = r[img[i][j][0]] + 1
			g[img[i][j][1]] = g[img[i][j][1]] + 1
			b[img[i][j][2]] = b[img[i][j][2]] + 1

	return r, g, b


def hist(vet):
	y_axis = vet
	x_axis = range(len(y_axis))
	width_n = 0.4
	bar_color = 'black'

	plt.bar(x_axis, y_axis, width=width_n, color=bar_color, align='center')
	plt.show()


def escala_bin (r_old, g_old, b_old, num_bin):
	bins = 256/num_bin
	r_new = np.zeros(num_bin)
	g_new = np.zeros(num_bin)
	b_new = np.zeros(num_bin)
	for i in range (len(r_new)):
		k = 0
		while (k < bins):
			r_new[i] = r_new[i] + r_old[bins*i + k]
			g_new[i] = g_new[i] + g_old[bins*i + k]
			b_new[i] = b_new[i] + b_old[bins*i + k]
			k = k + 1

	return r_new, g_new, b_new


def dist_euclidiana(hist1, hist2):

	soma = 0; 
	tam = len(hist1)
	for i in range (tam):
		soma = soma + ((hist1[i]- hist2[i])**2)


	return math.sqrt(soma)

def planos(gray_iamge, M, N, plan):
	planbin = np.zeros(shape = (M,N))
	for i in range(M):
		for j in range (N):
			plan_bin = bin(gray_image[i][j])[2:].zfill(8)
			planbin[i][j] = plan_bin[8-plan]

	return planbin

def entropia(plan_img, M, N):
	num_pixels = float(M*N)
	num_um = np.count_nonzero(plan_img)
	num_zero = num_pixels - num_um
	p_um = num_um/num_pixels
	p_zero = num_zero/num_pixels
	print p_um + p_zero
	entr = -1*(p_zero*np.log2(p_zero) + p_um*np.log2(p_um))

	return entr


#tam_peppers = img_peppers.shape
#m_peppers = tam_peppers[0]
#n_peppers = tam_peppers[1]

#tam_baboon = img_baboon.shape
#m_baboon = tam_baboon[0]
#n_baboon = tam_baboon[1]


#num_pixels_baboon = m_baboon*n_baboon
#num_pixels_peppers = m_peppers*n_peppers

#r_baboon, g_baboon, b_baboon = bin(img_baboon, m_baboon, n_baboon)
#r_peppers, g_peppers, b_peppers= bin(img_peppers, m_peppers, n_peppers)

#r_peppers_n, g_peppers_n, b_peppers_n = escala_bin(r_peppers, g_peppers, b_peppers, 256)
#r_baboon_n, g_baboon_n, b_baboon_n = escala_bin(r_baboon, g_baboon, b_baboon, 256)


#r_peppers_n = r_peppers_n/num_pixels_peppers
#g_peppers_n = g_peppers_n/num_pixels_peppers
#b_peppers_n = b_peppers_n/num_pixels_peppers


#r_baboon_n = r_baboon_n/num_pixels_baboon
#g_baboon_n = g_baboon_n/num_pixels_baboon
#b_baboon_n = b_baboon_n/num_pixels_baboon

#dist_r = dist_euclidiana(r_peppers_n, r_baboon_n)
#dist_g = dist_euclidiana(g_peppers_n, g_baboon_n)
#dist_b = dist_euclidiana(b_peppers_n, b_baboon_n)

#dist_total = (dist_r + dist_g + dist_b)/3



gray_image = cv2.cvtColor(img_peppers, cv2.COLOR_BGR2GRAY)


tam_gray = gray_image.shape
m_gray = tam_gray[0]
n_gray = tam_gray[1]


#for i in range (1, 9):
	#a = planos(gray_image, m_gray, n_gray, i)
	#plt.imshow(a, cmap='gray')
	#plt.show()

for i in range (1,9):
	a = planos(gray_image, m_gray, n_gray, i)
	plt.imshow(a, cmap='gray')
	plt.show()
	k = entropia(a, m_gray, n_gray)
	print k 


