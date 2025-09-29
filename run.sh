#python '[CNN]MSC.py'
#python '[MLP]MSC.py'
#
#python '[CNN]Single.py' --data 'MNIST' --hidden_size 4096
#python '[CNN]Single.py' --data 'SVHN' --hidden_size 4096
#python '[CNN]Single.py' --data 'CIFAR' --hidden_size 4096
#
#python '[CNN]Single.py' --data 'MNIST' --hidden_size 1024
#python '[CNN]Single.py' --data 'SVHN' --hidden_size 1024
#python '[CNN]Single.py' --data 'CIFAR' --hidden_size 1024
#
#python '[MLP]Single.py' --data 'MNIST' --hidden_size 4096
#python '[MLP]Single.py' --data 'SVHN' --hidden_size 4096
#python '[MLP]Single.py' --data 'CIFAR' --hidden_size 4096
#
#python '[MLP]Single.py' --data 'MNIST' --hidden_size 1024
#python '[MLP]Single.py' --data 'SVHN' --hidden_size 1024
#python '[MLP]Single.py' --data 'CIFAR' --hidden_size 1024

python '[Prunus]MSC.py' --reg_alpha 0.5 --reg_beta 1
python '[Prunus]MSC.py' --reg_alpha 0.5 --reg_beta 1.5
python '[Prunus]MSC.py' --reg_alpha 0.5 --reg_beta 2
python '[Prunus]MSC.py' --reg_alpha 0.5 --reg_beta 2.5
python '[Prunus]MSC.py' --reg_alpha 0.5 --reg_beta 3
python '[Prunus]MSC.py' --reg_alpha 0.5 --reg_beta 3.5
python '[Prunus]MSC.py' --reg_alpha 0.5 --reg_beta 4
python '[Prunus]MSC.py' --reg_alpha 0.5 --reg_beta 4.5
python '[Prunus]MSC.py' --reg_alpha 0.5 --reg_beta 5