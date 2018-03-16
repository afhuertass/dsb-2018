import numpy as np 
import pandas as pd 
import model 
import tensorflow as tf 
import os
from datagenerator import DataGenerator
from keras.callbacks import TensorBoard
from time import time
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from scipy.misc import imsave
batch_size = 2
prefix = "../data/ready"
prefix_model = "../models/"
ids = range(500)
n_folds = 4
params = {'imw': 32,
'imh': 32,
'channels': 3,
'batch_size': batch_size,
'shuffle': True}


def save_preds( preds , fold  ):

	for  i , im in enumerate(preds):
		im = im.reshape( ( 220 , 220 ) )
		imsave( "../preds/{}/{}.png".format(fold, i) , im )



def train():


	ids = np.arange(1, 604)
	

	kf = KFold( n_splits = n_folds )
	fold = 0
	epochs = 1000
	cvscores = [] 
	for train_index , test_index in kf.split(ids):

		print( train_index )  
		

		#print( train_index[[1,2,3,4,5,6]])
		callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath= prefix_model+"best_m_{}".format(fold), monitor='val_loss', save_best_only=True)]

		training_generator = DataGenerator(**params).generate( prefix , ids[train_index] , ids[train_index] )
		valid_generator = DataGenerator(**params).generate( prefix , ids[test_index] , ids[test_index] )

		
		K.clear_session()
		linknet = model.get_model2( )

	#print( linknet.summary() )
		linknet.compile(loss = model.loss , optimizer = "adam"  , metrics=['accuracy']  )
		#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

		valid_steps = len(  test_index   )/batch_size 
		steps_per_epoch =   len(train_index) / (2*batch_size)  # 2 

		linknet.fit_generator(generator = training_generator , steps_per_epoch = steps_per_epoch  , callbacks = callbacks , validation_data = valid_generator
			 , validation_steps = valid_steps , epochs = epochs
		 )

		scores = linknet.evaluate_generator( generator = valid_generator , steps = valid_steps  )
		cvscores.append( scores[0] )

		predictions =  linknet.predict_generator( generator = training_generator  , steps = 100 )
		save_preds(predictions , fold )
		fold = fold + 1 
		del linknet



	for  i ,sc in enumerate( cvscores ) :

		print("loss fold {}:{} ".format(   i  , sc ) )


if __name__ =="__main__":

	train()