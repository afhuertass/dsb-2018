


import numpy as np 
import pandas as pd 
import model 
import tensorflow as tf 
import os
from datagenerator import DataGenerator



batch_size = 2
prefix = "../data/ready"
ids = range(500)

params = {'imw': 32,
'imh': 32,
'channels': 3,
'batch_size': batch_size,
'shuffle': True}

def train():

	ids = range( 1, 600)


	kf = KFold(n_splits= 4  , random_state = seed , shuffle = True ) 

	for ids_train , ids_valid in kf.split( ids ):

		training_generator = DataGenerator(**params ).generate( prefix , ids_train , ids_train )
		eval_generator = DataGenerator(**)params.generate(prefix , ids_valid , ids_valid )
		linknet = model.get_model2()

		linknet.compile(loss = model.loss , optimizer = "adam"  , metrics=['accuracy'] )
		linknet.fit_generator(generator = training_generator , steps_per_epoch = 100  )

	#training_generator = DataGenerator(**params).generate( prefix , ids , ids )
	



if __name__ =="__main__":

	train()