
#
# Nome: Francisco Rafael Capiteli Carneiro
# RA: 157888	
# Trabalho 3 - MC920 - Introducao ao Processamento e Analise de Imagens
#
#

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os

def return_svd (image, k):

	U, S, V = np.linalg.svd(image[:, :, k])
	return U, S, V

def compress(image, k):

	new_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
	for i in range (new_image.shape[2]): 
		U, S, V = return_svd(image, i)
		new_U = U[:, 0:k]
		new_S = S[0:k]
		new_V = V[0:k, :]
		new_image[:, :, i] = np.dot(np.dot(new_U, np.diag(new_S)), new_V)
	return new_image


def rmse_images(image, new_image): 

	dif = np.count_nonzero((image - new_image)**2)
	m = image.shape[0]
	n = image.shape[1]
	mean = np.float(dif)/np.float(m*n)
	rmse = np.sqrt(mean)

	#rmse = np.sqrt(np.mean((image-new_image)**2))
	return rmse

def plot_ (x, rmse, ratio, nome): 

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(x, rmse, 'g-')
	ax2.plot(x, ratio, 'b-')
	plt.title('Grafico da imagem ' + nome , fontsize=14)
	ax1.set_xlabel('K')
	ax1.set_ylabel('RMSE', color='g')
	ax2.set_ylabel('Compress Ratio', color='b')
	plt.show()

def main(arq): 
	k = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 250]
	ratio = np.zeros((len(k)))
	rmse = np.zeros((len(k)))
	for i in range (len(k)):
		image = misc.imread(arq)
		new_image = compress(image, k[i])
		t = rmse_images(image, new_image)
		rmse[i] = t
		misc.imsave(str(arq +'compress'+str(k[i]) + '.png'), new_image)
		original_bytes = os.path.getsize(arq)
		compress_bytes = os.path.getsize(str(arq +'compress'+str(k[i]) + '.png'))
		p = np.float(compress_bytes)/np.float(original_bytes)
		print (arq + ": k = " + str(k[i]) + ". RMSE: " + str(t) + ". Ratio: " + str(p) + ".")
		ratio[i] = p
	plot_(k, rmse, ratio, arq)


print("----------------------------------------------------------------------------")
main("baboon.png")
print("----------------------------------------------------------------------------")
main("fire.png")
print("----------------------------------------------------------------------------")
main("peppers.png")
print("----------------------------------------------------------------------------")