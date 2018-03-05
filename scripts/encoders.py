
from keras.engine.topology import Layer
from keras.applications import ResNet50
from keras.layers import Conv2D , MaxPooling2D , BatchNormalization , Activation , Conv2DTranspose , Input , UpSampling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer 
import keras
lencoder1 = [
"res2a_branch2a" ,
 "bn2a_branch2a" ,
 "activation_2" ,
]
input_shape_resnet = ( 224 , 224  , 3)

class DecoderBlock( ):

	def __init__( self, in_channels , n_filters , input_shape = ()  ):
		# 512 256 
		#self.output_dim = output_dim
		self.in_channels = in_channels
		self.n_filters = n_filters 
		self.input_shape = input_shape
		
		self.build(   )

	def build( self ):

		## aqui va la el bloque
		self.conv1 = Conv2D( self.in_channels //4  , kernel_size = 1 ) #, input_shape = self.input_shape 
		self.norm1 = BatchNormalization()
		self.relu1 = Activation("relu")


		# ( B , H , W , C/4 )

		self.deconv2 = Conv2DTranspose( self.in_channels// 4 , kernel_size = 3 , strides = 2 ,  padding = "same")
		self.norm2 = BatchNormalization()
		self.relu2 = Activation("relu")
		# ( B , H , W , C/4 )

		self.conv3 = Conv2D( self.n_filters , kernel_size = 1  )
		self.norm3 = BatchNormalization()
		self.relu3 = Activation("relu")

	def call2( self , x   ):
		print( "inputs decoded")
		print( x.shape )
		x = self.conv1( x )
		x = self.norm1( x )

		print( "inputs decoder -- first conv ")
		print( x.shape )
		x = self.relu1( x )
		tmp_shape = x.shape 

		x = self.deconv2( x )
		x = self.norm2( x )
		#x.set_shape( tmp_shape )
		
		x = self.relu2(x )
		x = self.conv3(x)
		x = self.norm3(x)
		x = self.relu3(x)

		print("output decoder")
		#x.set_shape( (None , spatial_dims[0] , spatial_dims[1] , self.n_filters ) )
		print( x.shape )

		return x 


class Encoder1(  ):
	"""docstring for ClassName"""
	def __init__(self, resnet ):

		self.resnet = resnet
		self.build()

	def build(self  ):

		self.c0 = self.resnet.get_layer("res2a_branch2a")
		self.c1 = self.resnet.get_layer( "bn2a_branch2a")
		self.c2 = self.resnet.get_layer( "activation_2" )

	def call2( self , x ):

		x = self.c0(x)
		x = self.c1(x)
		x = self.c2(x)
		return x


class Encoder2(  ):
	"""docstring for ClassName"""
	def __init__(self, resnet ):

		self.resnet = resnet

		self.build()
		
	def build(self  ):

		self.c0 = self.resnet.get_layer("res2a_branch2b")
		self.c1 = self.resnet.get_layer( "bn2a_branch2b")
		self.c2 = self.resnet.get_layer( "activation_3" )
		

	def call2( self , x ):

		x = self.c0(x)
		x = self.c1(x)
		x = self.c2(x)
		
		return x 

class Encoder3(  ):
	"""docstring for ClassName"""
	def __init__(self, resnet , max_polling  ):
		# max polling is the max_pooling2d_1 layer in resnet 
		self.resnet = resnet
		self.max_polling = max_polling

		self.build()
	def build(self  ):

		self.c0 = self.resnet.get_layer("res2a_branch2c")
		self.c1 = self.resnet.get_layer( "res2a_branch1")
		self.c2 = self.resnet.get_layer( "bn2a_branch2c" )
		self.c3 = self.resnet.get_layer( "bn2a_branch1" )
		self.c4 = self.resnet.get_layer("add_1")
		self.c5 = self.resnet.get_layer( "activation_4" )
		

	def call2(self , x ) :

		x = self.c0( x )
		p1 =  self.c1( self.max_polling.get_output_at(0) )
		p1.set_shape( x.shape )
		x = self.c2( x )
		p2 = self.c3( p1 )	
		#x = self.c4( x , p2 )
		#embedding_sum = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(embed)
		#y = self.c4 ( [ x , p2 ] ) 
		y = keras.layers.Add() ([ x , p2 ] )
		x = self.c5( y )
		#x = self.c5( x + p2 ) 
		print(" output shape encoder3 ")
		print(x.shape)
		return x

class Encoder4(  ):
	"""docstring for ClassName"""
	def __init__(self, resnet ):
		# max polling is the max_pooling2d_1 layer in resnet 
		self.resnet = resnet
		
		self.build()

	def build(self ):

		self.cx0 = self.resnet.get_layer("res2b_branch2a")
		#print( self.cx0 )
		self.cx1 = self.resnet.get_layer( "bn2b_branch2a")
		self.cx2 = self.resnet.get_layer( "activation_5")

		

	def call2(self , xx ) :
		print("encoder4")
		print(xx.shape)
		xx = self.cx0(xx)
		xx = self.cx1(xx)
		xx = self.cx2(xx )

		return xx

class LinkNet2( object ):

	def __init__( self ,  input_shape = input_shape_resnet ,num_channels = 3  ) :

		self.num_channels = num_channels
		self.num_classes = 1
		self.build(input_shape)
		

		#
	def build(self, input_shape ):
		# aqui la red
		filters = [ 64 , 128 , 256 , 512 ]

		resnet = ResNet50(weights='imagenet', pooling=max, include_top = False)

		self.input = resnet.get_layer("input_1")

		#self.firstconv = resnet. conv1
		self.firstconv = resnet.get_layer("conv1")
		print("Hola ke aze")
		print( self.firstconv.input_shape )
		self.firstbn = resnet.get_layer("bn_conv1")
		self.firstrelu = resnet.get_layer("activation_1") 
		self.firstmaxpool = resnet.get_layer("max_pooling2d_1")

		#self.encoder1 = resnet.get_layer("activation_1")  #.activation_1 
		self.encoder1 = Encoder1( resnet )
		#self.encoder2 = resnet.get_layer("activation_2") #activation_2 
		self.encoder2 = Encoder2( resnet )
		#self.encoder3 = resnet.get_layer("activation_3") #activation_3
		self.encoder3 = Encoder3(resnet , self.firstmaxpool )
		#self.encoder4 = resnet.get_layer("activation_4") #activation_4 
		self.encoder4 = Encoder4( resnet  )

		self.decoder1 = DecoderBlock(filters[0] , filters[0]  , input_shape = input_shape_resnet )
		self.decoder2 = DecoderBlock(filters[1] , filters[0]  , input_shape = [-1 , 55 , 55 , 64] )
		self.decoder3 = DecoderBlock(filters[2] , filters[0]  , input_shape = [-1 , 55 , 55 , 64] )
		self.decoder4 = DecoderBlock(filters[2] , filters[2]  , input_shape = [-1 , 55, 55 , 64]  )

		self.finaldeconv1 = Conv2DTranspose( 1 , kernel_size = 3 , strides=2 )
		
		#self.finaldeconv1.set_shape( [ None , 55 , 55 , 32 ]  )
		self.finalrelu1 = Activation("relu")
		#self.finalup1 = UpSampling2D( (2,2) )
		self.finalconv2 = Conv2D( 32 , kernel_size = 3)
		self.finalrelu2 = Activation("relu")
		#self.finalup2 = UpSampling2D((2,2))
		self.finalconv3 = Conv2D( filters  = self.num_classes , kernel_size = 2 )
		

	def get_input(self):

		return self.input 
		

