### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

# python Control_CNN.py --mode combine --channel 64 --linear 20

python [Prunus]MSC.py --lr 0.01
python [Prunus]MSC.py --lr 0.05

python [Prunus]MSC.py --lr 0.01 --switcher_weight 10.0
python [Prunus]MSC.py --lr 0.05 --switcher_weight 10.0

python [Prunus]MSC.py --lr 0.01 --disc_weight 10.0 --switcher_weight 10.0
python [Prunus]MSC.py --lr 0.05 --disc_weight 10.0 --switcher_weight 10.0