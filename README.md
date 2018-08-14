# dsb-2018

Code using the ideas of LinkNet, based on the winners of the Carvana Challenge on Kaggle. Written in Keras

THe code is structured as it follows:

the scripts folder contains the code.

scripts/model.py contains the model itself, at this moment the most important method is get_model2() with builds and returns the keras model.
scripts/encoders.py contains the code to build some of the blocks needed for the model.
scripts/datagenerator.py contains the code to make a generator to feed the keras model.
scripts/train.py is the main part of the program building and training the model.So far the model feeds on numpy files that are saved of data/ready/imgs, and tha are saved with the format i.png, i being an interger. The ground truth masks are in data/ready/masks/i.png. the code currently train and saves foud models and their respective dice scores. Looking forward to improve this process ad soon as I have some free time.


