# sys.path.append('../')
import torch
import argparse
import os
from clustering.domain_split import domain_split
import caffenet
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from copy import deepcopy
from clustering.domain_split import calc_mean_std
from torch import nn
import torch
from numpy.random import *
from torchvision.datasets.folder import make_dataset, default_loader
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
import torch.nn.functional as F
from data_loader import *
            
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1).mean()
        return b

class inv_lr_scheduler(_LRScheduler):
    def __init__(self, optimizer, alpha, beta, total_epoch, last_epoch=-1):
        self.alpha = alpha
        self.beta = beta
        self.total_epoch = total_epoch
        super(inv_lr_scheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [base_lr * ((1 + self.alpha * self.last_epoch / self.total_epoch) ** (-self.beta)) for base_lr in self.base_lrs]


def eval_model(model, eval_data, device, epoch, filename):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    data_num = 0
    for inputs, labels in eval_data:
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            data_num += inputs.size(0)
    epoch_loss = running_loss / len(eval_data.dataset)
    epoch_acc = running_corrects / len(eval_data.dataset)
    log = 'Eval: Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc)
    print(log)
    with open(filename, 'a') as f: 
        f.write(log + '\n')
    return epoch_acc

def get_disc_dim(name, clustering, domain_num, clustering_num):
    if name == 'deepall':
        return None
    elif name == 'general' and clustering == True:
        return clustering_num
    elif name == 'general' and clustering != True:
        return domain_num
    else:
        raise ValueError('Name of train unknown %s' % name)
    
def copy_weights(net_from, net_to):
    for m_from, m_to in zip(net_from.modules(), net_to.modules()):
        if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()
    return net_from, net_to



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-root', default='/home/hail/efficient_model/dg_mmld/data/PACS/kfold/')
    parser.add_argument('--save-root', default='/home/hail/efficient_model/dg_mmld/ckpt/')
    parser.add_argument('--result-dir', default='PACS')
    parser.add_argument('--train', default='general')
    parser.add_argument('--data', default='PACS')
    parser.add_argument('--model', default='caffenet')
    parser.add_argument('--clustering', action='store_true')
    parser.add_argument('--clustering-method', default='Kmeans')
    parser.add_argument('--num-clustering', type=int, default=3)
    parser.add_argument('--clustering-step', type=int, default=1)
    parser.add_argument('--entropy', choices=['default', 'maximum_square'], default='default')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-step', type=int, default=10) 
    parser.add_argument('--save-step', type=int, default=10)

    parser.add_argument('--exp-num', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
 
    parser.add_argument('--num-epoch', type=int, default=30)
    parser.add_argument('--scheduler', default='step')
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lr-step', type=int, default=24)
    parser.add_argument('--lr-decay-gamma', type=float, default=0.1)
    parser.add_argument('--nesterov', action='store_true')

    parser.add_argument('--fc-weight', type=float, default=10.0)
    parser.add_argument('--disc-weight', type=float, default=10.0)
    parser.add_argument('--entropy-weight', type=float, default=1.0)
    parser.add_argument('--grl-weight', type=float, default=1.0)
    parser.add_argument('--loss-disc-weight', action='store_true')

    parser.add_argument('--color-jitter', action='store_true')
    parser.add_argument('--min-scale', type=float, default=0.8)
    parser.add_argument('--instance-stat', action='store_true')

    parser.add_argument('--feature-fixed', action='store_true')

    args = parser.parse_args()

    args.momentum = 0.9
    args.weight_decay = 0.0005
    
    path = args.save_root + args.result_dir
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + '/models')
    
    with open(path+'/args.txt', 'w') as f:
        f.write(str(args))
        
    domain = get_domain(args.data)
    source_domain, target_domain = split_domain(domain, args.exp_num)

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    get_domain_label, get_cluster = True, False

    source_train, source_val, target_test = random_split_dataloader(
        data=args.data, data_root=args.data_root, source_domain=source_domain, target_domain=target_domain,
        batch_size=args.batch_size, get_domain_label=get_domain_label, get_cluster=get_cluster, num_workers=4,
        color_jitter=args.color_jitter, min_scale=args.min_scale)
        



    num_epoch = args.num_epoch
    lr_step = args.lr_step
    
    # caffenet-specific something
    disc_dim = len(source_domain)
    model = caffenet.DGcaffenet(num_classes=source_train.dataset.dataset.num_class, num_domains=disc_dim, pretrained=True).to(device)
    model_lr = [(model.base_model.features, 1.0),  (model.base_model.classifier, 1.0),
            (model.base_model.class_classifier, 1.0 *args.fc_weight), (model.discriminator, 1.0 * args.disc_weight)]
    
    optimizers = [optim.SGD(model.parameters(), lr=args.lr*alpha, momentum=args.momentum, weight_decay=args.weight_decay) for model, alpha in model_lr]

    if args.scheduler == 'inv':
        schedulers = [inv_lr_scheduler(optimizer=opt, gamma=args.lr_decay_gamma) for opt in optimizers]
    elif args.scheduler == 'step':
        schedulers = [StepLR(optimizer=opt, step_size=lr_step, gamma=args.lr_decay_gamma) for opt in optimizers]
    else:
        raise ValueError('Name of scheduler unknown %s' %args.scheduler)
            
    best_acc = 0.0
    test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epoch):

        print('Epoch: {}/{}, Lr: {:.6f}'.format(epoch, num_epoch-1, optimizers[0].param_groups[0]['lr']))
        print('Temporary Best Accuracy is {:.4f} ({:.4f} at Epoch {})'.format(test_acc, best_acc, best_epoch))
        
        dataset = source_train.dataset.dataset
    
        if args.clustering:
            if epoch % args.clustering_step == 0:
                pseudo_domain_label = domain_split(dataset, model, device=device,
                                            cluster_before=dataset.clusters,
                                            filename= path+'/nmi.txt', epoch=epoch,
                                            nmb_cluster=args.num_clustering, method=args.clustering_method,
                                            pca_dim=256, whitening=False, L2norm=False, instance_stat=args.instance_stat)
                dataset.set_cluster(np.array(pseudo_domain_label))

        if args.loss_disc_weight:
            if args.clustering:
                hist = dataset.clusters
            else:
                hist = dataset.domains

            weight = 1. / np.histogram(hist, bins=model.num_domains)[0]
            weight = weight / weight.sum() * model.num_domains
            weight = torch.from_numpy(weight).float().to(device)
            
        else:
            weight = None
            
        # training model
        class_criterion = nn.CrossEntropyLoss()
        print(weight)
        domain_criterion = nn.CrossEntropyLoss(weight=weight)
        entropy_criterion = HLoss()

        p = epoch / num_epoch
        alpha = (2. / (1. + np.exp(-10 * p)) -1) * args.grl_weight
        beta = (2. / (1. + np.exp(-10 * p)) -1) * args.entropy_weight
        model.discriminator.set_lambd(alpha)
        model.train()  # Set model to training mode
        running_loss_class = 0.0
        running_correct_class = 0
        running_loss_domain = 0.0
        running_correct_domain = 0
        running_loss_entropy = 0
        # Iterate over data.
        for inputs, labels, domains in source_train:
            inputs = inputs.to(device)
            labels = labels.to(device)
            domains = domains.to(device)
            # zero the parameter gradients
            for optimizer in optimizers:
                optimizer.zero_grad()
            # forward
            output_class, output_domain = model(inputs)

            loss_class = class_criterion(output_class, labels)
            loss_domain = domain_criterion(output_domain, domains)
            loss_entropy = entropy_criterion(output_class)
            _, pred_class = torch.max(output_class, 1)
            _, pred_domain = torch.max(output_domain, 1)

            total_loss = loss_class + loss_domain + loss_entropy * beta
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            running_loss_class += loss_class.item() * inputs.size(0)
            running_correct_class += torch.sum(pred_class == labels.data)
            running_loss_domain += loss_domain.item() * inputs.size(0)
            running_correct_domain += torch.sum(pred_domain == domains.data)
            running_loss_entropy += loss_entropy.item() * inputs.size(0)

            print('Class Loss: {:.4f} Acc: {:.4f}, Domain Loss: {:.4f} Acc: {:.4f}, Entropy Loss: {:.4f}'.format(
                loss_class.item(), (pred_class == labels.data).float().mean().item(), loss_domain.item(), 
                (pred_domain == domains.data).float().mean().item(), loss_entropy.item()))


        epoch_loss_class = running_loss_class / len(source_train.dataset)
        epoch_acc_class = running_correct_class.double() / len(source_train.dataset)
        epoch_loss_domain = running_loss_domain / len(source_train.dataset)
        epoch_acc_domain = running_correct_domain.double() / len(source_train.dataset)
        epoch_loss_entropy = running_loss_entropy / len(source_train.dataset)
        
        log = 'Train: Epoch: {} Alpha: {:.4f} Loss Class: {:.4f} Acc Class: {:.4f}, Loss Domain: {:.4f} Acc Domain: {:.4f} Loss Entropy: {:.4f}'.format(epoch, alpha, epoch_loss_class, epoch_acc_class, epoch_loss_domain, epoch_acc_domain, epoch_loss_entropy)
        print(log)
        with open(path+'/source_train.txt', 'a') as f: 
            f.write(log + '\n') 

        # evaluation
        if epoch % args.eval_step == 0:
            acc =  eval_model(model, source_val, device, epoch, path+'/source_eval.txt')
            acc_ = eval_model(model, target_test, device, epoch, path+'/target_test.txt')
        
        # save model 
        if epoch % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                "model_{}.pt".format(epoch)))

        if acc >= best_acc:
            best_acc = acc
            test_acc = acc_
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                "model_best.pt"))
                
        for scheduler in schedulers:
            scheduler.step()
    
    best_model = caffenet.DGcaffenet(num_classes=source_train.dataset.dataset.num_class, num_domains=disc_dim, pretrained=False)
    best_model.load_state_dict(torch.load(os.path.join(
                path, 'models',
                "model_best.pt"), map_location=device))
    best_model = best_model.to(device)
    test_acc = eval_model(best_model, target_test, device, best_epoch, path+'/target_best.txt')
    print('Test Accuracy by the best model on the source domain is {} (at Epoch {})'.format(test_acc, best_epoch))
