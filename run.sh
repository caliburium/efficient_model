### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

python [DANN]MSC.py --hidden_size 512
python [DANN]MSC.py --hidden_size 384
python [DANN]MSC.py --hidden_size 256
python [DANN]MSC.py --hidden_size 128
