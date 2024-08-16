### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

# python Control_CNN.py --mode combine --channel 64 --linear 20

python [Control]DANN+Pretrain.py --dis_lr 0.01
python [Control]DANN+Pretrain.py --dis_lr 0.025
python [Control]DANN+Pretrain.py --dis_lr 0.05
python [Control]DANN+Pretrain.py --dis_lr 0.075
python [Control]DANN+Pretrain.py --dis_lr 0.1
python [Control]DANN+Pretrain.py --dis_lr 0.2
python [Control]DANN+Pretrain.py --dis_lr 0.5