

# file to build the keras linknet model

from keras.models import Sequential
from keras.models import Model
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.applications import ResNet50
from keras.layers import Reshape , Conv2D , MaxPooling2D , BatchNormalization , Activation , Conv2DTranspose , Input , UpSampling2D , Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer 

from  encoders import *

K.set_image_dim_ordering('tf')

input_shape_resnet = ( 224 , 224  , 3)

premodels = {
	"resnet" : ResNet50
}
lencoder1 = [
"res2a_branch2a" ,
 "bn2a_branch2a" ,
 "activation_2" ,
]

lencoder2 = ["res2a_branch2b" ,
"bn2a_branch2b" ,
"activation_3" ,
]
lencoder3 = [
"res2a_branch2b" ,
 "bn2a_branch2b" , 
 "activation_3"
]
lencoder4 = [
"res2a_branch2c" , 
 "res2a_branch1" , 
 "bn2a_branch2c" ,
 "bn2a_branch1" ,
 "add_1" ,
 "activation_4" 
]
def jaccard( y_true , y_pred): 
	# implementation of jaccard distance loss 
	# recordar que la mascara esta contamidada con los pesos 
	# y_true includes de weigths 
	y_true = y_true[: , : , : , 0 ]

	smooth = 100 
	
	intersection = K.sum( K.abs( y_true*y_pred ) , axis = -1   )
	sum_ = K.sum( K.abs( y_true )  + K.abs( y_pred ) , axis = -1 )
	jac = (intersection + smooth)/( sum_ - intersection + smooth )

	return ( 1 - jac )*smooth 



def loss( y_true , y_pred   ):

	ws = 1.0 
	print("true shape")
	print( y_true.shape )
	print("pred shape")
	print(y_pred.shape)
	return 1  + bce(y_true , y_pred , ws ) - dice( y_true , y_pred )

def dice( y_true , y_pred ):
	smooth = 1 
	intersection = K.sum(  K.abs( y_true*y_pred)  , axis = -1 )
	return ( 2.* intersection + smooth )/( K.sum( K.square(y_true) , -1 ) + K.sum(K.square(y_pred) , -1 )  + smooth )

	# do something

def bce( y_true,  y_pred , ws ) :
	# cross entropy con pesos
	#ws = y_pred[ : , : , : , ]
	#y_true = y_true[: , : , : , 0]

	print("zeip of you")
	print( y_true.shape )
	print( y_pred.shape )
	
	y_pred.set_shape( (None , None , None , 1 ))
	
	# return weigthed cross entropy 
	return binary_crossentropy( y_true*ws , y_pred  )


class DecoderBlock22( Layer ):

	def __init__( self, in_channels , n_filters ):
		# 512 256 
		#self.output_dim = output_dim
		self.in_channels = in_channels
		self.n_filters = n_filters 

		super( DecoderBlock , self ).__init__()

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
		print("shape conv1 decoder")
		print(x.shape)
		x = self.norm1( x )
		print("shape norm decoder")
		print(x.shape)

		x = self.relu1( x )
		tmp_shape = x.shape 

		x = self.deconv2( x )
		x = self.norm2( x )
		x.set_shape( tmp_shape )
		print("shape norm2 decoder")
		print(x.shape )
		x = self.relu2(x )
		x = self.conv3(x)
		x = self.norm3(x)
		x = self.relu3(x)
		print( "decoder output shape ")
		print(x.shape)
		return x 


		#return K.dot( x , self.kernel )
class Encoder( Layer ):

	def __init__( self , resnet , layer_names ):
		# recieves a 
		self.resnet = resnet
		self.layers = []
		self.name = "enc"
		self.layer_names = layer_names
		super( Encoder , self).__init__()
		return
	def build(self , input_shape) :


		for layers in self.layer_names:
			# layers es una lista 
			self.layers.append( self.resnet.get_layer(layer) )


		super(Encoder , self ).build( input_shape)
		return
	def call(self , x):

		for l in self.layers:
			x = l(x)

		return x 

class LinkNet( Layer):

	def __init__( self ,  input_shape = input_shape_resnet ,num_channels = 3  ) :

		self.num_channels = num_channels
		self.num_classes = 1
		self.build(input_shape)
		super( LinkNet , self ).__init__( input_shape = input_shape_resnet)

		#
	def build(self, input_shape ):
		# aqui la red
		filters = [ 64 , 128 , 256 , 512 ]

		resnet = ResNet50(weights='imagenet' , pooling = max ,include_top = False  ) #, pooling=max, include_top = False)

		#self.firstconv = resnet. conv1
		self.firstconv = resnet.get_layer("conv1")
		print("ola que hace ")
		
		self.firstbn = resnet.get_layer("bn_conv1")
		self.firstrelu = resnet.get_layer("activation_1") 
		self.firstmaxpool = resnet.get_layer("max_pooling2d_1")
		print( self.firstconv.shape )
		#self.encoder1 = resnet.get_layer("activation_1")  #.activation_1 
		self.encoder1 = Encoder1( resnet )
		#self.encoder2 = resnet.get_layer("activation_2") #activation_2 
		self.encoder2 = Encoder2( resnet )
		#self.encoder3 = resnet.get_layer("activation_3") #activation_3
		self.encoder3 = Encoder3(resnet , self.firstmaxpool )
		#self.encoder4 = resnet.get_layer("activation_4") #activation_4 
		self.encoder4 = Encoder4( resnet  )

		self.decoder1 = DecoderBlock(filters[0] , filters[0] )

		self.decoder2 = DecoderBlock(filters[1] , filters[0] )
		self.decoder3 = DecoderBlock(filters[2] , filters[1] )
		self.decoder4 = DecoderBlock(filters[3] , filters[2] )

		self.finaldeconv1 = Conv2DTranspose( 32 , kernel_size = 3 , strides=2 )
		self.finalrelu1 = Activation("relu")
		self.finalconv2 = Conv2D( 32 , kernel_size = 3)
		self.finalrelu2 = Activation("relu") 
		self.finalconv3 = Conv2D( filters  = self.num_classes , kernel_size = 2 ) 


		super( LinkNet , self).build( input_shape )

	def call( self , x ):
		# VERIFICAR X DEBE SER LA ENTRADA QUE RECIVE EL RESNET
		# EL SHAPE ADECuado
		print( "inpuy shape")
		print(x.shape)
		x = self.firstconv( x )
		x = self.firstbn(x )
		x = self.firstrelu(x )
		x = self.firstmaxpool( x )
		print( "first layer  shape")
		print( x.shape )
		e1 = self.encoder1( x )
		e2 = self.encoder2( e1 )
		e3 = self.encoder3( e2 )
		print("good")
		print(e3.shape)
		e4 = self.encoder4( e3 )

		print("encoder shapes ")
		print(e1.shape )
		print(e2.shape )
		print(e3.shape )
		print(e4.shape )

		print(" decoder 3 shape ")

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
	model.add( LinkNet( input_shape = input_shape_resnet )  )

	adam = Adam( lr = 0.0001 )
	model.compile( optimizer  = adam , loss = loss )

	return model

def get_model2(  input_shape = input_shape_resnet , num_classes = 1  ):
	K.set_image_data_format('channels_last')
	linknet = LinkNet2( input_shape = input_shape_resnet )
	K.set_image_dim_ordering('tf')
	#inputs = linknet.get_input().output 
	#inputs.set_shape( ( None ,224,224 , 3 ))
	inputs = linknet.get_input()

	print( inputs.shape )
	#inputs = Input(shape = input_shape_resnet )

	x = linknet.firstconv(inputs)
	print("xxxxx shape")
	print(x.shape)
	x = linknet.firstbn(x)
	x = linknet.firstrelu(x)
	x = linknet.firstmaxpool(x)
	print("input shape encoders")
	print(x.shape)
	e1 = linknet.encoder1.call2( x )
	e2 = linknet.encoder2.call2( e1 )
	e3 = linknet.encoder3.call2( e2 )
	
	e4 = linknet.encoder4.call2( e3 )

	print( "encoders shapes ")
	print( e1.shape )
	print( e2.shape )
	print( e3.shape )
	print( e4.shape )

	#d4 = linknet.decoder4( e4 )

	#d4 = linknet.decoder4.call2( e4 ) + e3
	#print( "e2 upsampled ")
	#print(e2.shape)

	d4 = linknet.decoder4.call2( e4  )
	e3 = UpSampling2D((2,2))(e3)
	d4 = keras.layers.Add() ([   e3 , d4  ] )

	#d3 = linknet.decoder3.call2( d4 ) + e2
	d3 = linknet.decoder3.call2(d4  )

	#y = UpSampling2D((2,2) )( e2 )
	x = UpSampling2D((2,2) )( e2 )
	x = UpSampling2D((2,2))(x)
	d3 = keras.layers.Add() ([  x , d3 ] )
	#d2 = linknet.decoder2.call2( d3 )  + e1 

	d2 = linknet.decoder2.call2(d3  )

	y = UpSampling2D((8,8))(e1)
	d2 = keras.layers.Add() ([  y , d2 ] ) # 440 , 440 , 64
	print( "dedede")
	print( d2.shape )
	d1 = linknet.decoder1.call2( d2  )
	print( "decoders shape")

	print( d4.shape )
	print( d3.shape )
	print( d2.shape )
	print( d1.shape )

	y = Conv2D(  32 , kernel_size = (3,3) , strides = 2 , padding = "same" )( d1 )
	y = Activation("relu")( y )

	# [ None , 440 , 440 , 32]

	y = Conv2D( 16 , kernel_size =( 2,2) , strides = 2 , padding = "same") (y)
	y = Activation("relu")(y)

	# [ 220 , 220 , 16]
	y = Conv2D( 8 , kernel_size=(2,2) , strides = 2 , padding= "same")(y)
	y = Activation("relu")(y)

	# [110 , 110 , 8]
	y = Conv2D( 4 , kernel_size=(2,2) , strides = 2 , padding="same")(y)
	y = Activation("relu")(y)

	# {55 , 55 , 4} 

	rs = Reshape(   [55*55*4]  )( y )
	size = 224*224
	#flat = Flatten(  )( y )
	fc = Dense( size   )( rs )
	fc = Activation("relu")(fc)
	
	
	output = Reshape( [ 224 , 224 , 1 ])(fc)
	#fc.set_shape( ( None , 224 ,224 , 1  ) )
	print("output")

	print(output)

	# [None , 220 , 220 , 32]



	print("y fake")
	print( y.shape )
	
	#f1 = linknet.finaldeconv1( d1 )
	#f2 = Activation("relu")( f1 )
	#f3 = linknet.finalconv2( f2 )
	#f4 = linknet.finalrelu2( f3)
	#f5 = linknet.finalconv3( f4 )

	#print( f1.shape )
	#f5.set_shape( (None , 224 , 224 , 1 ))
	model = Model( inputs = inputs , outputs = output   )
	
	print( model.summary()  )
	return model 






