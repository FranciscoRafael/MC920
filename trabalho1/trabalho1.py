from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

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


def hist(vet, titulo):
	y_axis = vet
	x_axis = range(len(y_axis))
	width_n = 0.4
	bar_color = 'black'

	plt.title('Grafico ' + titulo , fontsize=20)
	plt.xlabel('Bins', fontsize=20)
	plt.ylabel('Numero de Pixels', fontsize=20)

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

def planos(img, M, N, plan):
	planbin = np.zeros(shape = (M,N))
	for i in range(M):
		for j in range (N):
			plan_bin = bin(img[i][j])[2:].zfill(8)
			planbin[i][j] = plan_bin[8-plan]

	return planbin

def entropia(plan_img, M, N):
	num_pixels = float(M*N)
	num_um = np.count_nonzero(plan_img)
	num_zero = num_pixels - num_um
	p_um = num_um/num_pixels
	p_zero = num_zero/num_pixels
	entr = -1*(p_zero*np.log2(p_zero) + p_um*np.log2(p_um))

	return entr

def plot_plan_entr(image, titulo):

	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	tam_gray = gray_image.shape
	m_gray = tam_gray[0]
	n_gray = tam_gray[1]
	print "Para a image", titulo
	for i in range (1,9):
		a = planos(gray_image, m_gray, n_gray, i)
		a = a*255
		plt.title('Plano' + str(i) , fontsize=20)
		plt.imshow(a, cmap='gray')
		plt.show()
		k = entropia(a, m_gray, n_gray)
		print "Entropia do plano", i, ":", k


#Leitura das imagens utilizadas como teste.
img_peppers = misc.imread('peppers.png')
img_baboon = misc.imread('baboon.png')
img_watch = misc.imread('watch.png')

#tamanho da imagem Peppers usada no comparativo
tam_peppers = img_peppers.shape
m_peppers = tam_peppers[0]
n_peppers = tam_peppers[1]

#tamanho da imagem BAboon usada no comparativo
tam_baboon = img_baboon.shape
m_baboon = tam_baboon[0]
n_baboon = tam_baboon[1]

#encontrando a quantidade de pixels de cada imagem
num_pixels_baboon = m_baboon*n_baboon
num_pixels_peppers = m_peppers*n_peppers

#encontrando um vetor de 256 bins para cada imagem
r_baboon, g_baboon, b_baboon = bin_image(img_baboon, m_baboon, n_baboon)
r_peppers, g_peppers, b_peppers= bin_image(img_peppers, m_peppers, n_peppers)



#Redimensionando o vetor de bins para a quantidade de bins desejada 
v = [4, 32, 128, 256]
for i in range (len(v)):
	r_peppers_n, g_peppers_n, b_peppers_n = escala_bin(r_peppers, g_peppers, b_peppers, v[i])
	r_baboon_n, g_baboon_n, b_baboon_n = escala_bin(r_baboon, g_baboon, b_baboon, v[i])

	#Mostra dos Histogramas na tela  
	hist(r_peppers_n, "R_Peppers"+ str(v[i]))
	hist(r_baboon_n, "R_Baboon" + str(v[i]))
	hist(g_peppers_n, "G_Peppers" + str(v[i]))
	hist(g_baboon_n, "G_Baboon" + str(v[i]))
	hist(b_peppers_n, "B_Peppers" + str(v[i]))
	hist(b_baboon_n, "B_Baboon" + str(v[i]))

	#normalizacao dos vetores em probabilidades. A soma das probabilidades de todos os pixels sejam 1.
	#feito nas duas imagens

	r_peppers_n = r_peppers_n/num_pixels_peppers
	g_peppers_n = g_peppers_n/num_pixels_peppers
	b_peppers_n = b_peppers_n/num_pixels_peppers
	r_baboon_n = r_baboon_n/num_pixels_baboon
	g_baboon_n = g_baboon_n/num_pixels_baboon
	b_baboon_n = b_baboon_n/num_pixels_baboon

	#calculo da distancia euclidiana para R G e B
	dist_r = dist_euclidiana(r_peppers_n, r_baboon_n)
	dist_g = dist_euclidiana(g_peppers_n, g_baboon_n)
	dist_b = dist_euclidiana(b_peppers_n, b_baboon_n)

	#Media entre as distancias
	dist_total = (dist_r + dist_g + dist_b)/3

	print "A distancia euclidiana para bins", v[i], ":", dist_total


#geracao dos planos de bis e da entropia para a imagem solicitada

plot_plan_entr(img_watch, "watch")
plot_plan_entr(img_peppers, "peppers")