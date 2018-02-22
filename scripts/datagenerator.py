import numpy as np 
import pandas as pd 

# generador de los datos para  alimentar el modelo 

class DataGenerator(object):

	def __init__(self , imw , imh , channels , batch_size , shuffle = True ):

		self.imw = imw 
		self.imh = imh
		self.channels = channels 
		self.batch_size = batch_size
		self.shuffle - shuffle

	def generate(self , prefix , labels , list_IDS):


		while 1:

			indexs = self._get_exploration_order(list_IDS)

			imax = int( len(indexs)/ self.batch_size )

			for i in range( imax ):

				list_IDS_tmps = [ list_IDS[k] for  k in indexes[i*self.batch_size:(i+1)*self.batch_size] ]

				imgs , masks = self._data_generation( prefix , labels , list_IDS_tmps )

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
		X = np.empty(  ( self.batch_size , self.imw , self.imh  , self.channels) )
		
		# mascara y pesos
		Y = np.empty( ( self.batch_size , self.imw , self.imh , 2    ) )

		for  i , idd in enumerate( list_IDS_tmps ):
			# 
			X[ i , : , : , : ] = np.load( prefix +  "imgs/" + idd +'.npy')

			Y[ i , : , : , 0 ] = np.load( prefix + "maks/" + idd + ".npy" ) 
			Y[ i , : , : , 1 ] = np.load(prefix  + "ws/" + idd + ".npy" )


		return X , Y 





