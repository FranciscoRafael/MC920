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


def read_video(nome):
	cap = cv2.VideoCapture(nome)
	i = 0;

	num_frame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	num_width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	num_height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

	buf = np.empty((num_frame, num_height, num_width, 3), np.dtype('uint8'))

	i = 0
	ret = True

	while (i < num_frame  and ret):
	    ret, buf[i] = cap.read()
	    i += 1

	cap.release()
	return buf

def convert_gray(video):
	frames = video.shape[0]
	x = video.shape[1]
	y = video.shape[2]
	new_video = np.empty((frames, x, y))
	print(new_video.shape)
	for i in range((video.shape[0])):
		arr = np.asarray(video[i])
		plt.imshow(arr)
		plt.show()
		new_video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
		arr = np.asarray(new_video[i])
		plt.imshow(arr, cmap='gray')
		plt.show()

	return new_video

video = read_video("umn.mp4")
video = convert_gray(video)
