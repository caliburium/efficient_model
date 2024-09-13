### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

# python Control_CNN.py --mode combine --channel 64 --linear 20

python [CORAL]SVHN_MNIST.py
python [CORAL]SVHN_MNIST.py
python [CORAL]SVHN_MNIST.py

