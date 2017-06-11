from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os

def return_svd (image, k):

	U, S, V = np.linalg.svd(image[:, :, k])

	return U, S, V

def sum_bytes(U, S, V): 

	return U.nbytes + S.nbytes + V.nbytes


def compress(image, k):

	compress_bytes = 0 
	original_bytes = 0
	new_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

	for i in range (new_image.shape[2]): 
		U, S, V = return_svd(image, i)
		original_bytes += sum_bytes(U, S, V)
		new_U = U[:, 0:k]
		new_S = S[0:k]
		new_V = V[0:k, :]
		compress_bytes += sum_bytes(new_U[:, 0:k], new_S[0:k], new_V[0:k, :])
		new_image[:, :, i] = np.dot(np.dot(new_U, np.diag(new_S)), new_V)

	return new_image, compress_bytes, original_bytes


def rmse_images(image, new_image): 

	rmse = np.sqrt(np.mean((image-new_image)**2))

	return rmse

def plot_ (x, rmse, ratio): 

	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	ax1.plot(x, rmse, 'g-')
	ax2.plot(x, ratio, 'b-')

	ax1.set_xlabel('K')
	ax1.set_ylabel('RMSE', color='g')
	ax2.set_ylabel('Compress Ratio', color='b')

	plt.show()

def main(): 
	k = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250]
	ratio = np.zeros((len(k)))
	rmse = np.zeros((len(k)))
	for i in range (len(k)):
		image = misc.imread("monalisa.png")
		new_image, compress_bytes, original_bytes = compress(image, k[i])
		t = rmse_images(image, new_image)
		rmse[i] = t
		misc.imsave('/home/ehlocao/Unicamp/mc920/MC920/trabalho3/images_out/'+ str("monalisa.png" +'compress'+str(k[i]) + '.png'), new_image)
		original_bytes = os.path.getsize("baboon.png")
		compress_bytes = os.path.getsize(str('/home/ehlocao/Unicamp/mc920/MC920/trabalho3/images_out/'+ str("monalisa.png" +'compress'+str(k[i]) + '.png')))
		p = np.float(compress_bytes)/np.float(original_bytes)
		print "monalisa.png: k = " + str(k[i]) + ". RMSE: " + str(t) + ". Ratio: " + str(p) + "." 
		ratio[i] = p

	plot_(k, rmse, ratio)
main()