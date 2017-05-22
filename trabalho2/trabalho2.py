#
# Nome: Francisco Rafael Capiteli Carneiro
# RA: 157888	
# Trabalho 2 - MC920 - Introducao ao Processamento e Analise de Imagens
#
#

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
		new_video[i] = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
	return new_video


def pixels_t1(video, T):
	frames = video.shape[0]
	vect = np.zeros(frames - 1)
	for k in range (frames - 1):
	   vect[k] = np.count_nonzero((abs(video[k] - video[k + 1]) > T))

	return vect 


def square_sum_diff(video, div, T):
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
				if(B > T):
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
        if vect[i] >= T:
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
        final[k] = np.copy(video[t])

    return final      

def hist_dif(video):
    frames = video.shape[0]
    vect = np.zeros(frames-1)
    for k in range(frames-1):
        sum1 = 0
        hist1,bins = np.histogram(video[k].ravel(),256,[0,256])
        hist2,bins = np.histogram(video[k+1].ravel(),256,[0,256])
        a = abs(hist1 - hist2)
        sum1 = np.sum(a)
        vect[k] = sum1

    return vect    


def plot_var(vect, nome_video, metodo, T):
    t = np.arange(len(vect))
    s = vect
    plt.axhline(y=T, color='r', linestyle='-')
    plt.title('Grafico do video ' + nome_video , fontsize=14)
    plt.xlabel('Frames', fontsize=14)
    plt.ylabel('Metodo: ' + metodo, fontsize=14)
    plt.plot(t, s)
    plt.show()


def find_T_hist(vect, val):
    m = np.average(vect)
    d = np.std(vect)
    return (m + val*d)


toy = read_video("toy.mp4")
toyg = convert_gray(toy)

lisa = read_video("lisa.mpg")
lisag = convert_gray(lisa)

umn = read_video("umn.mp4")
umng = convert_gray(umn)

xylophone = read_video("xylophone.mp4")
xylophoneg = convert_gray(xylophone)


#-------------------------------------------------------------------------------
#primeiro dados para o toy.mp4

LIMIAR_PIXELS_TOY = 115
LIMIAR_SSD_TOY = 150000
toy_p = pixels_t1(toyg, LIMIAR_PIXELS_TOY)
toy_sqr8 = square_sum_diff(toyg, 8, LIMIAR_SSD_TOY)
toy_sqr16 = square_sum_diff(toyg, 16, LIMIAR_SSD_TOY)
toy_hist = hist_dif(toyg)
toy_edge = edge_det(toyg)

LIMIAR_T2_PIXELS_TOY = 85
LIMIAR_T2_SSD8_TOY = 15
LIMIAR_T2_SSD16_TOY = 16
LIMIAR_T1_HIST_TOY = find_T_hist(toy_hist, 1)
LIMIAR_T1_DIF_BOARD = 33000


plot_var(toy_p, "toy.mp4", "Diferenca de Pixels", LIMIAR_T2_PIXELS_TOY)
plot_var(toy_sqr8, "toy.mp4", "Diferenca entre Blocos 8", LIMIAR_T2_SSD8_TOY)
plot_var(toy_sqr16, "toy.mp4", "Diferenca entre blocos 16", LIMIAR_T2_SSD16_TOY)
plot_var(toy_hist, "toy.mp4", "Diferenca entre Histogramas", LIMIAR_T1_HIST_TOY)
plot_var(toy_edge, "toy.mp4", "Diferenca entre Mapas de borda", LIMIAR_T1_DIF_BOARD)

adr_p = find_t(toy_p, LIMIAR_T2_PIXELS_TOY)
adr_sqr8 = find_t(toy_sqr8, LIMIAR_T2_SSD8_TOY)
adr_sqr16 = find_t(toy_sqr16, LIMIAR_T2_SSD16_TOY)
adr_hist = find_t(toy_hist, LIMIAR_T1_HIST_TOY)
adr_edge = find_t(toy_edge, LIMIAR_T1_DIF_BOARD)

frames_abrut_p = video_frames(toy, adr_p)
skvideo.io.vwrite("saida_toy_Dif_pixels.mp4", frames_abrut_p)

frames_abrut_sqr8 = video_frames(toy, adr_sqr8)
skvideo.io.vwrite("saida_toy_SSD8.mp4", frames_abrut_sqr8)


frames_abrut_sqr16 = video_frames(toy, adr_sqr16)
skvideo.io.vwrite("saida_toy_SSD16.mp4", frames_abrut_sqr16)


frames_abrut_hist = video_frames(toy, adr_hist)
skvideo.io.vwrite("saida_toy_Dif_Hist.mp4", frames_abrut_hist)


frames_abrut_edge = video_frames(toy, adr_edge)
skvideo.io.vwrite("saida_toy_dif_borda.mp4", frames_abrut_edge)


#------------------------------------------------------------------------------------


#dados para o lisa.mpg

LIMIAR_PIXELS_LISA = 115
LIMIAR_SSD_LISA = 100000
lisa_p = pixels_t1(lisag, LIMIAR_PIXELS_LISA)
lisa_sqr8 = square_sum_diff(lisag, 8, LIMIAR_SSD_LISA)
lisa_sqr16 = square_sum_diff(lisag, 16, LIMIAR_SSD_LISA)
lisa_hist = hist_dif(lisag)
lisa_edge = edge_det(lisag)

LIMIAR_T2_PIXELS_LISA = 1000
LIMIAR_T2_SSD8_LISA = 45
LIMIAR_T2_SSD16_LISA = 100
LIMIAR_T1_HIST_LISA = find_T_hist(lisa_hist, 3.5)
LIMIAR_T1_DIF_BOARD_LISA = 11850

plot_var(lisa_p, "lisa.mpg", "Diferenca de Pixels", LIMIAR_T2_PIXELS_LISA)
plot_var(lisa_sqr8, "lisa.mpg", "Diferenca entre Blocos 8", LIMIAR_T2_SSD8_LISA)
plot_var(lisa_sqr16, "lisa.mpg", "Diferenca entre blocos 16", LIMIAR_T2_SSD16_LISA)
plot_var(lisa_hist, "lisa.mpg", "Diferenca entre Histogramas", LIMIAR_T1_HIST_LISA)
plot_var(lisa_edge, "lisa.mpg", "Diferenca entre Mapas de borda", LIMIAR_T1_DIF_BOARD_LISA)


adr_p = find_t(lisa_p, LIMIAR_T2_PIXELS_LISA)
adr_sqr8 = find_t(lisa_sqr8, LIMIAR_T2_SSD8_LISA)
adr_sqr16 = find_t(lisa_sqr16, LIMIAR_T2_SSD16_LISA)
adr_hist = find_t(lisa_hist, LIMIAR_T1_HIST_LISA)
adr_edge = find_t(lisa_edge, LIMIAR_T1_DIF_BOARD_LISA)

frames_abrut_p = video_frames(lisa, adr_p)
skvideo.io.vwrite("saida_lisa_Dif_pixels.mp4", frames_abrut_p)

frames_abrut_sqr8 = video_frames(lisa, adr_sqr8)
skvideo.io.vwrite("saida_lisa_SSD8.mp4", frames_abrut_sqr8)

frames_abrut_sqr16 = video_frames(lisa, adr_sqr16)
skvideo.io.vwrite("saida_lisa_SSD16.mp4", frames_abrut_sqr16)

frames_abrut_hist = video_frames(lisa, adr_hist)
skvideo.io.vwrite("saida_lisa_Dif_Hist.mp4", frames_abrut_hist)

frames_abrut_edge = video_frames(lisa, adr_edge)
skvideo.io.vwrite("saida_lisa_dif_borda.mp4", frames_abrut_edge)

#------------------------------------------------------------------------------------

# dados para o umn.mp4

LIMIAR_PIXELS_UMN = 115
LIMIAR_SSD_UMN = 100000
umn_p = pixels_t1(umng, LIMIAR_PIXELS_UMN)
umn_sqr8 = square_sum_diff(umng, 8, LIMIAR_SSD_UMN)
umn_sqr16 = square_sum_diff(umng, 16, LIMIAR_SSD_UMN)
umn_hist = hist_dif(umng)
umn_edge = edge_det(umng)

LIMIAR_T2_PIXELS_UMN = 100
LIMIAR_T2_SSD8_UMN = 23
LIMIAR_T2_SSD16_UMN = 32
LIMIAR_T1_HIST_UMN = find_T_hist(umn_hist, 3)
LIMIAR_T1_DIF_BOARD_UMN = 12000

plot_var(umn_p, "umn.mpg", "Diferenca de Pixels", LIMIAR_T2_PIXELS_UMN)
plot_var(umn_sqr8, "umn.mpg", "Diferenca entre Blocos 8", LIMIAR_T2_SSD8_UMN)
plot_var(umn_sqr16, "umn.mpg", "Diferenca entre blocos 16", LIMIAR_T2_SSD16_UMN)
plot_var(umn_hist, "umn.mpg", "Diferenca entre Histogramas", LIMIAR_T1_HIST_UMN)
plot_var(umn_edge, "umn.mpg", "Diferenca entre Mapas de borda", LIMIAR_T1_DIF_BOARD_UMN)


adr_p = find_t(umn_p, LIMIAR_T2_PIXELS_UMN)
adr_sqr8 = find_t(umn_sqr8, LIMIAR_T2_SSD8_UMN)
adr_sqr16 = find_t(umn_sqr16, LIMIAR_T2_SSD16_UMN)
adr_hist = find_t(umn_hist, LIMIAR_T1_HIST_UMN)
adr_edge = find_t(umn_edge, LIMIAR_T1_DIF_BOARD_UMN)

frames_abrut_p = video_frames(umn, adr_p)
skvideo.io.vwrite("saida_umn_Dif_pixels.mp4", frames_abrut_p)

frames_abrut_sqr8 = video_frames(umn, adr_sqr8)
skvideo.io.vwrite("saida_umn_SSD8.mp4", frames_abrut_sqr8)

frames_abrut_sqr16 = video_frames(umn, adr_sqr16)
skvideo.io.vwrite("saida_umn_SSD16.mp4", frames_abrut_sqr16)

frames_abrut_hist = video_frames(umn, adr_hist)
skvideo.io.vwrite("saida_umn_Dif_Hist.mp4", frames_abrut_hist)

frames_abrut_edge = video_frames(umn, adr_edge)
skvideo.io.vwrite("saida_umn_dif_borda.mp4", frames_abrut_edge)

#------------------------------------------------------------------------------------

#dados para o xylophone.mp4

LIMIAR_PIXELS_XYLOPHONE = 115
LIMIAR_SSD_XYLOPHONE = 100000
xylophone_p = pixels_t1(xylophoneg, LIMIAR_PIXELS_XYLOPHONE)
xylophone_sqr8 = square_sum_diff(xylophoneg, 8, LIMIAR_SSD_XYLOPHONE)
xylophone_sqr16 = square_sum_diff(xylophoneg, 16, LIMIAR_SSD_XYLOPHONE)
xylophone_hist = hist_dif(xylophoneg)
xylophone_edge = edge_det(xylophoneg)

LIMIAR_T2_PIXELS_XY = 150
LIMIAR_T2_SSD8_XY = 11
LIMIAR_T2_SSD16_XY = 17
LIMIAR_T1_HIST_XY = find_T_hist(xylophone_hist, 1)
LIMIAR_T1_DIF_BOARD_XY = 14500

plot_var(xylophone_p, "xylophone.mpg", "Diferenca de Pixels", LIMIAR_T2_PIXELS_XY)
plot_var(xylophone_sqr8, "xylophone.mpg", "Diferenca entre Blocos 8", LIMIAR_T2_SSD8_XY)
plot_var(xylophone_sqr16, "xylophone.mpg", "Diferenca entre blocos 16", LIMIAR_T2_SSD16_XY)
plot_var(xylophone_hist, "xylophone.mpg", "Diferenca entre Histogramas", LIMIAR_T1_HIST_XY)
plot_var(xylophone_edge, "xylophone.mpg", "Diferenca entre Mapas de borda", LIMIAR_T1_DIF_BOARD_XY)

adr_p = find_t(xylophone_p, LIMIAR_T2_PIXELS_XY)
adr_sqr8 = find_t(xylophone_sqr8, LIMIAR_T2_SSD8_XY)
adr_sqr16 = find_t(xylophone_sqr16, LIMIAR_T2_SSD16_XY)
adr_hist = find_t(xylophone_hist, LIMIAR_T1_HIST_XY)
adr_edge = find_t(xylophone_edge, LIMIAR_T1_DIF_BOARD_XY)

frames_abrut_p = video_frames(xylophone, adr_p)
skvideo.io.vwrite("saida_xylophone_Dif_pixels.mp4", frames_abrut_p)

frames_abrut_sqr8 = video_frames(xylophone, adr_sqr8)
skvideo.io.vwrite("saida_xylophone_SSD8.mp4", frames_abrut_sqr8)

frames_abrut_sqr16 = video_frames(xylophone, adr_sqr16)
skvideo.io.vwrite("saida_xylophone_SSD16.mp4", frames_abrut_sqr16)

frames_abrut_hist = video_frames(xylophone, adr_hist)
skvideo.io.vwrite("saida_xylophone_Dif_Hist.mp4", frames_abrut_hist)

frames_abrut_edge = video_frames(xylophone, adr_edge)
skvideo.io.vwrite("saida_xylophone_dif_borda.mp4", frames_abrut_edge)
#------------------------------------------------------------------------------------