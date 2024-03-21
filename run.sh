### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

# python Control_CNN.py --mode combine --channel 64 --linear 20

python [Control]SimplerCNN.py --mode imagenette --channel 64
python [Control]SimplerCNN.py --mode stl10 --channel 64
python [Control]SimplerCNN.py --mode svhn --channel 64
python [Control]SimplerCNN.py --mode cifar10 --channel 64
python [Control]SimplerCNN.py --mode combine --channel 64

