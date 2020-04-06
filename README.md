# NNFramework
Framework for neural networks.

Divided in three files, hyper_parameters, layers and network.

hyper_parameters: contains activation function and loss function classes and some functions implemented.
layers: contains the layer class and two example layers.
network: network class.

network class: contains a task and layers. It already have some checks about the layers added, but you can always improve it. Also, it checks the inputs, X and y, they must be a two dimensional data.

layer class: it has a feed forward function, its derivative in respect of in features and its derivative in respect of its weights, they are both necessary to run backpropagation. You can choose which activation function it will use. Actually it does not run backpropagation, network class does, but calculate weights' delta and has a update method separately. If you want to implement a different feed forward function you just need to implement a (new) feed forward function, its derivative in respect of in features and its derivative in respect of its weights (as the examples).
