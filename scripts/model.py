

# file to build the keras linknet model

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K


def dice_error( img , mask ):

	# do something

def bce( img , mask , ws) :
	# cross entropy con pesos

def build_model():

	#Build the LinkNetModel
	return None

