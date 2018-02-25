
from datagenerator import DataGenerator 

import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation

ids = range(1,10)
batch_size = 2
prefix = "../data/ready"


params = {'imw': 32,
'imh': 32,
'channels': 4,
'batch_size': batch_size,
'shuffle': True}

print("")
training_generator = DataGenerator(**params).generate( prefix , ids , ids )
print("wtf")
for f in training_generator:
	print (f[0].shape)
	#break 

model = Sequential()

model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])



def test_dataloader():

	return 

if "__name__" == "__main__":

	test_dataloader()
