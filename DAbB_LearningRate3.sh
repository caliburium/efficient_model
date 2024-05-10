### mode = combine, mnist, svhn, cifar10, svhn
### channel = cnn output channel, hidden = channel/2
### linear = classifier hidden layer

# python Control_CNN.py --mode combine --channel 64 --linear 20

python [Control]UnsupervisedDAbB.py --batch_size 75 --lr_domain 0.01 --lr_class 0.1
python [Control]UnsupervisedDAbB.py --batch_size 75 --lr_domain 0.01 --lr_class 0.15
python [Control]UnsupervisedDAbB.py --batch_size 75 --lr_domain 0.01 --lr_class 0.2
python [Control]UnsupervisedDAbB.py --batch_size 75 --lr_domain 0.02 --lr_class 0.1
python [Control]UnsupervisedDAbB.py --batch_size 75 --lr_domain 0.02 --lr_class 0.15
python [Control]UnsupervisedDAbB.py --batch_size 75 --lr_domain 0.02 --lr_class 0.2
python [Control]UnsupervisedDAbB.py --batch_size 75 --lr_domain 0.005 --lr_class 0.05
python [Control]UnsupervisedDAbB.py --batch_size 75 --lr_domain 0.005 --lr_class 0.1
python [Control]UnsupervisedDAbB.py --batch_size 75 --lr_domain 0.005 --lr_class 0.15