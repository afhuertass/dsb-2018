

# file to build the keras linknet model

from keras.models import Sequential
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.applications import ResNet50
from keras.layers import Conv2D , MaxPooling2D , BatchNormalization , Activation , Conv2DTranspose
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer 
from keras import backend as K 

input_shape = ( )

input_shape_resnet = ( 224 , 224 )

premodels = {
	"resnet" : ResNet50
}


def jaccard_loss( y_true , y_pred): 
	# implementation of jaccard distance loss 
	# recordar que la mascara esta contamidada con los pesos 
	smooth = 100 

	intersection = K.sum( K.abs( y_true*y_pred ) , axis = -1   )
	sum_ = K.sum( K.abs( y_true )  + K.abs( y_pred ) , axis = -1 )
	jac = (intersection + smooth)/( sum_ - intersection + smooth )

	return ( 1 - jac )*smooth 



def loss( y_true , y_pred ):


	return 1  + bce(y_true , y_pred) - dice( y_true , y_mask )

def dice( y_true , y_pred ):
	smooth = 1 
	intersection = K.sum(  K.abs( y_true*y_pred)  , axis = -1 )
	return ( 2.* intersection + smooth )/( K.sum( K.square(y_true) , -1 ) + K.sum(K.square(y_pred) , -1 )  + smooth )

	# do something

def bce( y_true ,  y_pred ) :
	# cross entropy con pesos
	ws = y_true[ : , : , : , 1]
	# return weigthed cross entropy 
	return binary_crossentropy( y_true*ws , y_pred  )


class DecoderBlock( Layer ):

	def __init__( self, in_channels , n_filters ):

		self.output_dim = output_dim
		self.in_channels = in_channels
		self.n_filters = n_filters 

		super( DecoderBlock , self )

	def build( self , input_shape ):

		## aqui va la el bloque
		self.conv1 = Conv2D( self.in_channels //4  , kernel_size = 1 , input_shape = input_shape )
		self.norm1 = BatchNormalization()
		self.relu1 = Activation("relu")


		self.deconv2 = Conv2DTranspose( self.in_channels// 4 , kernel_size = 3 , strides = 2 , )
		self.norm2 = BatchNormalization()
		self.relu2 = Activation("relu")

		self.conv3 = Conv2D( self.n_filters , kernel_size = 1  )
		self.norm3 = BatchNormalization()
		self.relu3 = Activation("relu")


		super( DecoderBlock , self).build( input_shape)

	def call( self, x ):

		x = self.conv1( x )
		x = self.norm1( x )
		x = self.relu1( x )
		x = self.deconv2( x )
		x = self.norm2( x )
		x = self.relu2(x )
		x = self.conv3(x)
		x = self.norm3(x)
		x = self.relu(3)

		return x 


		#return K.dot( x , self.kernel )


class LinkNet( Layer):

	def __init__( self , num_channels = 3  ) :

		self.num_channels = num_channels

		#
	def build(self, input_shape ):
		# aqui la red
		filters = [ 64 , 128 , 256 , 512 ]

		resnetModel = ResNet50(weights='imagenet', pooling=max, include_top = False)

		self.firstconv = resnet.conv1
		self.firstbn = resnet.bn_conv1 
		self.firstrelu = resnet.activation_1 
		self.firstmaxpool = resnet.max_pooling2d_1 

		self.encoder1 = resnet.activation_1 
		self.encoder2 = resnet.activation_2 
		self.encoder3 = resnet.activation_3
		self.encoder4 = resnet.activation_4 

		self.decoder1 = DecoderBlock(filters[0] , filters[0] )
		self.decoder2 = DecoderBlock(filters[1] , filters[0] )
		self.decoder3 = DecoderBlock(filters[2] , filters[1] )
		self.decoder4 = DecoderBlock(filters[3] , filters[2] )

		self.finaldeconv1 = Conv2DTranspose( 32 , kernel_size = 3 , strides=2 )
		self.finalrelu1 = Activation("relu")
		self.finalconv2 = Conv2D( 32 , kernel_size = 3)
		self.finalrelu2 = Activation("relu") 
		self.finalconv3 = Conv2D( 32  , num_classes , kernel_size = 2 ) 


		super( LinkNet , self).build(input_shape)

	def call( self , x ):
		# VERIFICAR X DEBE SER LA ENTRADA QUE RECIVE EL RESNET
		# EL SHAPE ADECuado
		x = self.firstconv( x )
		x = self.firstbn(x )
		x = self.firstrelu(x )
		x = self.firstmaxpool( x )

		e1 = self.encoder1( x )
		e2 = self.encoder2( e1 )
		e3 = self.encoder3( e2 )
		e4 = self.encoder4( e3 )

		d4 = self.decoder4( e4 ) + e3 
		d3 = self.decoder3( d4 ) + e2 
		d2 = self.decoder2( d3) + e1 
		d1 = self.decoder1( d2) 

		f1 = self.finaldeconv1( d1 )
		f2 = self.finalrelu1( f1 )
		f3 = self.finalconv2( f2 ) 
		f4 = self.finalrelu2( f3 )
		f5 = self.finalconv3( f4)

		return f5 

def get_model():

	model = Sequential()
	model.add( LinkNet()  )

	adam = Adam( lr = 0.0001 )
	model.compile( optimizer  = adam , loss = loss )

	return model 




