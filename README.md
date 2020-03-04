#GRGB

Converting grayscale images into colored image, using neuralNets.

## Tools

To use this model training with your own dataset, those are some useful informations :

* Tree structure :
    widthxheight
       └ mini-batch0
           ├ pic0
           ├ pic1
           ├ ...
       └ mini-batch1
           ├ pic0
           ├ pic1
           ├ ...
       └ ...

* The main file is "converter.py" see help to get some information on how to use it. With this file you can :
  * Resize your dataset
  * Train custom Pytorch model (one previously built by converter.py), I'll try let as much option as possible for loss functions and optimizers parameters such as learning rate, closure or momentum (depending on the neural network, depending on researches advancement)
  * Apply a colorisation of a grayscale picture
