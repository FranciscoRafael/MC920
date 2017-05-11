#
# Nome: Francisco Rafael Capiteli Carneiro
# RA: 157888	
# Trabalho 2 - MC920 - Introducao ao Processamento e Analise de Imagens
#
#
#
#
#
from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import skvideo.io


def read_video(nome):
	videodata = skvideo.io.vread(nome)
	return videodata

def convert_gray(video):
	frames = video.shape[0]
	x = video.shape[1]
	y = video.shape[2]
	new_video = np.empty((frames, x, y))
	for i in range((video.shape[0])):
		new_video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
	return new_video


def pixels_t1(video):
	frames = video.shape[0]
	vect = np.zeros(frames - 1)
	for k in range (frames - 1):
		vect[k] = np.count_nonzero((abs(video[k] - video[k + 1]) > 115))

	return vect 


def square_sum_diff(video, div):
	frames = video.shape[0]
	m = video.shape[1]
	n = video.shape[2]
	xbloco = m/div
	ybloco = n/div
	count = 0
	vect = np.zeros(frames -1)
	for k in range(frames -1):
		for i in range(div):
			for j in range (div):
				x = video[k][i*xbloco:(i+1)*xbloco, j*ybloco:(j+1)*ybloco]
				y = video[k+1][i*xbloco:(i+1)*xbloco, j*ybloco:(j+1)*ybloco]
				B = np.sum((x-y)**2)
				if(B > 150000):
					count = count + 1
		vect[k] = count
		count = 0
	

	return vect




video = read_video("umn.mp4")
video = convert_gray(video)
print(video.shape)
dif = square_sum_diff(video, 16)
#dif = pixels_t1(video)

t = np.arange(len(dif))
s = dif
plt.plot(t, s)
plt.show()

