#GRGB

    Converting grayscale images into colored image, using neuralNets.
    -Gatherer:
        - DataLoader : parse data/<samples>
                       files should be separate by folders (could be one folder)
                       this class load one image at a time
        - Transform : actually use to load image at transform them to grayscale
                      can be also used to resize images and save them
    -Neural:
        - Net : thanks to the torch.nn.Module, this class is the neural net as an object
                this class should be protected (in term of Java keyword).
    - Training : This is the main class to train, test or use a Neural net
                 You can load or save nets from this class, all saved models will appear in neural/models/net.pth
