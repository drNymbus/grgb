# GRGB

__Converting grayscale images into colored image, using neuralNets.__

## Tools

* All pictures used to to train the net should be in the same folder

* The main file is "converter.py" see help to get some information on how to use it. With this file you can :
  * Resize your dataset
  * Train custom Pytorch model (one previously built by converter.py), I'll try let as much option as possible for loss functions and optimizers parameters such as learning rate, closure or momentum (depending on the neural network, depending on researches advancement)
  * Apply a colorisation of a grayscale picture

## Usage

* resize width height option=value . . .
  - path : the dataset to resize, otherwise resize \"data/pictures"
  - out : the output folder for the resized images

* train res=48x48 trainset=data/paysage_48x48/ option=value . . .
  - trainset : the path to the training data, the res option should be defined, default : "data/paysage_48x48/"
  - res : the resolution of the pictures (ex:res=32x32), default : 48x48
  - save : the path to save the model after training, default : "model/net.pth"
  - model : trains an existing model
  - epoch : the number of epochs to train the model
  - cuda : a boolean to enable or disable GPU (ex: cuda=True | cuda=False)
* convert path/to/file.png (a complete folder could be given too) option=value . . .
  - model : the model to load to convert your picture(s)
  - save : the path to save the converted pictures, otherwise saves them in the same folder as the original
  - view : show the result before saving

*ex: python3 converter.py train res=32x32 trainset=data/pictures_32x32/ save=model/net.pth epoch=15*

An GUI is planned, after I've succesfully trained a model capable of colorizing images. This will maybe be done in form of a Kivy app, an executable file or a website.
