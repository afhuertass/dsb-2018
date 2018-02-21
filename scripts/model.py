

# file to build the keras linknet model

from keras.models import Sequential
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D , MaxPooling2D 
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K


input_shape = ( )

def loss( img , mask ):


	return 1  - bce(img , mask) + dice(img , mask )
def dice( img , mask ):

	# do something

def bce( img , mask ) :
	# cross entropy con pesos

def build_model():

	#Build the LinkNetModel

	model = Sequential()

	model.add( )

	model.add( Conv2D(  64 , kernel_size=(2,2),  input_shape = input_shape  )  )
	
	model.compile(optimizer=adam, loss=root_mean_squared_error)

	return model 

