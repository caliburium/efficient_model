### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

# python Control_CNN.py --mode combine --channel 64 --linear 20

python [Control]SimplerCNN.py --mode combine --channel 48
python [Control]SimplerCNN.py --mode imagenette --channel 48
python [Control]SimplerCNN.py --mode stl10 --channel 48
python [Control]SimplerCNN.py --mode svhn --channel 48
python [Control]SimplerCNN.py --mode cifar10 --channel 48

python [Control]SimplerCNN.py --mode combine --channel 32
python [Control]SimplerCNN.py --mode imagenette --channel 32
python [Control]SimplerCNN.py --mode stl10 --channel 32
python [Control]SimplerCNN.py --mode svhn --channel 32
python [Control]SimplerCNN.py --mode cifar10 --channel 32

python [Control]SimplerCNN.py --mode combine --channel 96
python [Control]SimplerCNN.py --mode imagenette --channel 96
python [Control]SimplerCNN.py --mode stl10 --channel 96
python [Control]SimplerCNN.py --mode svhn --channel 96
python [Control]SimplerCNN.py --mode cifar10 --channel 96

python [Control]SimplerCNN.py --mode combine --channel 128
python [Control]SimplerCNN.py --mode imagenette --channel 128
python [Control]SimplerCNN.py --mode stl10 --channel 128
python [Control]SimplerCNN.py --mode svhn --channel 128
python [Control]SimplerCNN.py --mode cifar10 --channel 128