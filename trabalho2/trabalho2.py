#
# Nome: Francisco Rafael Capiteli Carneiro
# RA: 157888	
# Trabalho 2 - MC920 - Introducao ao Processamento e Analise de Imagens
#
#

from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import skvideo.io
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt




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
	   vect[k] = np.count_nonzero((abs(video[k] - video[k + 1]) > 128))

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

def edge_det(video):

    frames = video.shape[0]
    vect = np.zeros(frames-1)
    for k in range(frames -1):
        edge = feature.canny(video[k], sigma = 0)
        edge1 = feature.canny(video[k+1], sigma = 0)
        x = np.count_nonzero(edge != edge1)
        vect[k] = x

    return vect


def find_t(vect, T):
    tam = len(vect)
    new = np.zeros(tam)

    for i in range (tam):
        if vect[i] > T:
            if (i >= 0 or i< (tam-1) and vect[i-1] < 0.3*vect[i]):
                new[i] = i

    k = np.count_nonzero(new)
    adr = np.zeros(k)
    j = 0; 
    for i in range(tam):
        if(new[i] != 0):
            adr[j] = new[i]
            j = j+1   

    return adr


def video_frames (video, adr):
    frames = video.shape[0]
    m = video.shape[1]
    n = video.shape[2]
    tam = len(adr);
    final = np.zeros((tam, m, n, 3));
    for k in range (tam):
        t = int(adr[k])
        print t
        final[k] = np.copy(video[t])

    return final      


video = read_video("toy.mp4")
videog = convert_gray(video)


dif = pixels_t1(videog)

j = find_t(dif, 10)
x = video_frames(video, j)
print x.shape[0]



skvideo.io.vwrite("outputvideo.mp4", x)

t = np.arange(len(dif))
s = dif
plt.plot(t, s)
plt.show()

