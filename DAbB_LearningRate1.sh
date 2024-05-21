### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

# python Control_CNN.py --mode combine --channel 64 --linear 20

# python [Control]UnsupervisedDAbB.py --source SVHN_BW --target MNIST_RS --batch_size 50 --lr_domain 0.01 --lr_class 0.25
# python [Control]UnsupervisedDAbB.py --source SVHN_BW --target MNIST --batch_size 50 --lr_domain 0.01 --lr_class 0.25
# python [Control]UnsupervisedDAbB.py --source SVHN --target MNIST_RS --batch_size 50 --lr_domain 0.02 --lr_class 0.2
python [Control]UnsupervisedDAbB.py --source CIFAR10 --target STL10 --batch_size 50 --lr_domain 2 --lr_class 5
python [Control]UnsupervisedDAbB.py --source STL10 --target CIFAR10 --batch_size 50 --lr_domain 0.01 --lr_class 0.1

