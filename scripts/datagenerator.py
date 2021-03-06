import numpy as np 
import pandas as pd 
from scipy.misc import imresize

import cv2

# generador de los datos para  alimentar el modelo 
def fix_crop_transform( image , mask , x , y , w , h ):



	H,W = image.shape[:2]

	

	if x == -1 and y == -1:
		x = (W-w) // 2
		y = (H-h ) //2 

	if (x,y,w,h) != (0,0,W,H):
		image = image[y:y+h, x:x+w]
		mask = mask[y:y+h, x:x+w  ]

	#print( image.shape )
	return image, mask


def transform_images(  img , mask , w ,  h  ):

	H , W = img.shape[:2] 
	if H!=h:
		y = np.random.choice(H-h)
	else:
		y=0

	if W!=w:
		x = np.random.choice(W-w)
	else:
		x=0


	return fix_crop_transform(img, mask, x,y,w,h)



class DataGenerator(object):

	def __init__(self , imw , imh , channels , batch_size , shuffle = True ):

		self.imw = imw 
		self.imh = imh
		self.channels = channels 
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.H = 224
		self.W = 224
		self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

	def generate(self , prefix , labels , list_IDS):


		while 1:

			indexs = self._get_exploration_order(list_IDS)
			print(self.batch_size)
			imax = int( len(indexs) // self.batch_size )
			print( imax )
			for i in range( imax ):

				list_IDS_tmps = [ list_IDS[k] for  k in indexs[i*self.batch_size:(i+1)*self.batch_size] ]

				imgs , masks = self._data_generation( prefix , labels , list_IDS_tmps )

				#print("shape output generator")
				#print("Generator mask seee")
				#print( masks.shape )
				yield imgs , masks 



	

	def _get_exploration_order(self , list_IDS):

		indexs = np.arange( len(list_IDS))

		if self.shuffle:
			np.random.shuffle(indexs)

		return indexs

	def _data_generation(self , prefix , labels , list_IDS_tmps):
		# prefix = ../data/ready/
		#  
		# image
		X = np.empty(  ( self.batch_size , self.W , self.H  , self.channels) )
		
		# mascara y pesos
		h = 220
		Y = np.empty( ( self.batch_size , h , h , 1    ) )

		for  i , idd in enumerate( list_IDS_tmps ):

			x_partial = np.load( prefix +  "/imgs/" + str(idd) +'.npy')
			y_partial = np.load( prefix + "/masks/" + str(idd) + ".npy" )

			y_partial = y_partial[: , : , 0 ]

			#print(y_partial.shape)
			x , y = transform_images( x_partial , y_partial , self.W , self.H )
			#y = y.reshape( (self.W , self.H  ) )
			# transform mask
			y = imresize(y , (220 , 220  )  )
			y = y.reshape( (h , h , 1)  )

			# Transform X 
			x_gray = cv2.cvtColor( x , cv2.COLOR_BGR2GRAY)
			x = self.clahe.apply( x_gray )
			x = cv2.cvtColor( x , cv2.COLOR_GRAY2BGR )
			
			X[ i , : , : , : ] = x

			Y[ i , :   ] = y
			#Y[ i , : , : , 1 ] = np.load(prefix  + "ws/" + idd + ".npy" )


		#X , Y = self.transform_images(X,Y  , self.W , self.H  )

		return X , Y 





