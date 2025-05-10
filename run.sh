### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

python [DANN]MSC.py --hidden_size 128 --disc_weight 10.0 --pretrain_epoch 10
python [DANN]MSC.py --hidden_size 256 --disc_weight 10.0 --pretrain_epoch 10
python [DANN]MSC.py --hidden_size 384  --disc_weight 10.0 --pretrain_epoch 10
python [DANN]MSC.py --hidden_size 512  --disc_weight 10.0 --pretrain_epoch 10
python [DANN]MSC.py --hidden_size 128 --disc_weight 5.0 --pretrain_epoch 10
python [DANN]MSC.py --hidden_size 256 --disc_weight 5.0 --pretrain_epoch 10
python [DANN]MSC.py --hidden_size 384  --disc_weight 5.0 --pretrain_epoch 10
python [DANN]MSC.py --hidden_size 512  --disc_weight 5.0 --pretrain_epoch 10
