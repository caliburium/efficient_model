### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

python [CNN]MSC.py --hidden_size 64
python [CNN]MSC.py --hidden_size 64
python [CNN]MSC.py --hidden_size 64
python [CNN]MSC.py --hidden_size 96
python [CNN]MSC.py --hidden_size 96
python [CNN]MSC.py --hidden_size 96
python [CNN]MSC.py --hidden_size 128
python [CNN]MSC.py --hidden_size 128
python [CNN]MSC.py --hidden_size 128
python [CNN]MSC.py --hidden_size 192
python [CNN]MSC.py --hidden_size 192
python [CNN]MSC.py --hidden_size 192
python [CNN]MSC.py --hidden_size 256
python [CNN]MSC.py --hidden_size 256
python [CNN]MSC.py --hidden_size 256
python [CNN]MSC.py --hidden_size 384
python [CNN]MSC.py --hidden_size 384
python [CNN]MSC.py --hidden_size 384
python [CNN]MSC.py --hidden_size 512
python [CNN]MSC.py --hidden_size 512
python [CNN]MSC.py --hidden_size 512

#python [Prunus]MSC.py --lr 0.01 --disc_weight 10.0 --switcher_weight 10.0
#python [Prunus]MSC.py --lr 0.05 --disc_weight 10.0 --switcher_weight 10.0