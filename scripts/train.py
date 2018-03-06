


import numpy as np 
import pandas as pd 
import model 
import tensorflow as tf 
import os
from datagenerator import DataGenerator



batch_size = 2
prefix = "../data/ready"
ids = range(32)

params = {'imw': 32,
'imh': 32,
'channels': 3,
'batch_size': batch_size,
'shuffle': True}

def train():

	
	training_generator = DataGenerator(**params).generate( prefix , ids , ids )
	linknet = model.get_model2( )

	#print( linknet.summary() )
	linknet.compile(loss = model.loss , optimizer = "adam"  , metrics=['accuracy'] )

	#linknet.fit_generator(generator = training_generator , steps_per_epoch = 2   )


if __name__ =="__main__":

	train()