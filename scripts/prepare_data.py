import pandas as pd 
import numpy as np 
from skimage import io 
import os
import pickle 
from scipy import ndimage

def getWs( mask ):


	transf = ndimage.distance_transform_edt( mask )
	transf[ transf > 0 ]  = 1/transf[ transf >0 ]
	ls = np.where(  np.logical_and(  transf < 0.1 , transf > 0.0 )  )
	transf[ls] = 0.2
	return 3*transf 




def prepare_data( paths , output  ):

	# 
	train_files = os.listdir( paths )
	imgs = []
	masks = []


	all_files = [  paths + x + '/images/' + x + ".png"  for x in train_files ]
	i = 0 
	for train_file , file in  zip( train_files  , all_files)  :

		img = io.imread( file )
		img = img[: , : , 0:3 ]
		print( img.shape )
		imgs.append( img )

		mask_files = os.listdir( paths + train_file + "/masks" )
		mask_files = [   mask for mask in mask_files if mask.endswith(".png")  ]


		all_masks = [  paths + train_file + '/masks/' + mask_file   for mask_file in mask_files ]
		mask_final = np.zeros( ( img.shape[0]  , img.shape[1] ) )
		ws_final = np.zeros( (img.shape[0]  , img.shape[1] )  )

		for mask in all_masks:

			mask = io.imread(  mask )
			ws = getWs( mask )

			ws_final = np.maximum( ws_final , ws )
			mask_final = np.maximum( mask_final , mask  )


		mask_and_weights = np.stack( [mask_final , ws_final] , axis = -1 )
		print( mask_and_weights.shape )
		masks.append( mask_and_weights )



	indx = 0 

	for img  , mask in zip( imgs , masks ):

		np.save( output + "/imgs/{}.npy".format(indx) , img  )
		np.save( output + "/masks/{}.npy".format(indx ) , mask )
		indx = indx + 1 

	print("Saved {} files and masks as np arrays ".format( indx ) )

	pickle.dump( imgs , open(output + "full_data.p" , "wb") )
	pickle.dump( masks , open(output + "full_data_masks.p" , "wb") )


if __name__ == "__main__":
	paths = "../../data/data/"
	output = "../../data/ready/"
	prepare_data( paths , output)



