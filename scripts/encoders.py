
from keras.engine.topology import Layer 
import keras
lencoder1 = [
"res2a_branch2a" ,
 "bn2a_branch2a" ,
 "activation_2" ,
]
class Encoder1(Layer ):
	"""docstring for ClassName"""
	def __init__(self, resnet ):

		self.resnet = resnet


		super(Encoder1, self).__init__()
	def build(self , input_shape ):

		self.c0 = self.resnet.get_layer("res2a_branch2a")
		self.c1 = self.resnet.get_layer( "bn2a_branch2a")
		self.c2 = self.resnet.get_layer( "activation_2" )

		super(Encoder1 , self ).build(input_shape)
	def call(self , x ) :

		x = self.c0(x)
		x = self.c1(x)
		x = self.c2(x)

		return x 

lencoder2 = ["res2a_branch2b" ,
"bn2a_branch2b" ,
"activation_3" ,
]
class Encoder2(Layer ):
	"""docstring for ClassName"""
	def __init__(self, resnet ):

		self.resnet = resnet


		super(Encoder2, self).__init__()
	def build(self , input_shape ):

		self.c0 = self.resnet.get_layer("res2a_branch2b")
		self.c1 = self.resnet.get_layer( "bn2a_branch2b")
		self.c2 = self.resnet.get_layer( "activation_3" )

		super(Encoder2 , self ).build(input_shape)
	def call(self , x ) :

		x = self.c0(x)
		x = self.c1(x)
		x = self.c2(x)
		
		return x


class Encoder3(Layer ):
	"""docstring for ClassName"""
	def __init__(self, resnet , max_polling  ):
		# max polling is the max_pooling2d_1 layer in resnet 
		self.resnet = resnet
		self.max_polling = max_polling

		super(Encoder3, self).__init__()
	def build(self , input_shape ):

		self.c0 = self.resnet.get_layer("res2a_branch2c")
		self.c1 = self.resnet.get_layer( "res2a_branch1")
		self.c2 = self.resnet.get_layer( "bn2a_branch2c" )
		self.c3 = self.resnet.get_layer( "bn2a_branch1" )
		self.c4 = self.resnet.get_layer("add_1")
		self.c5 = self.resnet.get_layer( "activation_4" )
		
		super(Encoder3 , self ).build(input_shape)
	def call(self , x ) :

		x = self.c0( x )
		p1 = self.c1( self.max_polling.get_output_at(0) )
		p1.set_shape( x.shape )
		x = self.c2( x )
		p2 = self.c3( p1 )
		print("shape maxpooled thing")
		print( p1.shape)
		print("other")
		print(x.shape)
		#x = self.c4( x , p2 )
		x = self.c5( x + p2 ) 
		return x

class Encoder4(Layer ):
	"""docstring for ClassName"""
	def __init__(self, resnet ):
		# max polling is the max_pooling2d_1 layer in resnet 
		self.resnet = resnet
		

		super(Encoder4, self).__init__()

	def build(self , input_shape ):

		self.cx0 = self.resnet.get_layer("res2b_branch2a")
		self.cx1 = self.resnet.get_layer( "bn2b_branch2a")
		self.cx2 = self.resnet.get_layer( "activation_5")
		
		
		super(Encoder4 , self ).build(input_shape)
	def call(self , xx ) :
		print("encoder4")
		print(xx)
		xx = self.cx0(xx)
		#xx = self.c1(xx)
		#xx = self.c2(xx )

		return xx