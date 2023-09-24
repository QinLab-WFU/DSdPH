from __future__ import print_function
import os
import sys
import argparse
import time
import pandas as pd
import datetime
import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.tool import *
from models.fusionmodel import FusionModel
from model import *
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import json
import centroids_generator
import scipy.io as scio

def parse_option():
    parser = argparse.ArgumentParser('')
    
    parser.add_argument('--batch_size', type=int, default= 32,
                        help='batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs (default: 100)')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    
    parser.add_argument('--betas', type=float, default=(0.9, 0.999),
                        help='betas (default: (0.9, 0.999))')
    parser.add_argument('--n_class', type=int, default=17,
                        help='number of class (default: 17)')
    parser.add_argument('--bit', type=int, default=32,
                        help='bit of hash code (default: 16)')
    parser.add_argument('--dataset', type=str, default='UCMD',###
                        help='remote sensing dataset')
    parser.add_argument('--pre_weight', default='',
                        help='path of pre-training weight of AlexNet')
    parser.add_argument('--root_path', default='',###
                        help='root directory where the dataset is placed')
    parser.add_argument('--save_path', default='',
                        help='path where the result is placed')
    parser.add_argument('--gpu', type=str, default='0',
                        help='selected gpu (default: 0)')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='scheduler (default: 0)')
    parser.add_argument('--step_size', type=int, default=80,
                        help='scheduler (default: 0)')
    parser.add_argument('--grama', type=float, default=0.1,
                        help='scheduler (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
   
    parser.add_argument('--init-centroids-method', default='M', choices=['N', 'U', 'B', 'M', 'H'],
                        help='N = sign of gaussian; '
                             'B = bernoulli; '
                             'M = MaxHD'
                             'H = Hadamard matrix')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight of Hyp_loss (default:0.5)')
    parser.add_argument('--beta', type=float, default=8.0,   ###
                        help='weight of Relative (default: 8.0)')
    parser.add_argument('--margin', type=float, default=1.0, ####
                        help='weight of margin (default: 0.5)')
    parser.add_argument('--retrieve', type=int, default=0, help="retrieval number")
    args = parser.parse_args()
    return args



def _dataset(dataset,retrieve):
    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
 
    if dataset == 'UCMD':
        if retrieve == 0:
            retrieve = 2100
        num_classes = 17
        trainset = ImageList(open('./data/UCMD/train.txt', 'r').readlines(), transform=data_transform['train'])
        testset = ImageList(open('./data/UCMD/test.txt', 'r').readlines(), transform=data_transform['val'])
        database = ImageList(open('./data/UCMD/database.txt', 'r').readlines(), transform=data_transform['val'])

  
    num_train, num_test,num_database = len(trainset),len(testset),len(database)
    # print(trainset)
    # print(num_train,num_test,num_database)
    dsets = (trainset,testset,database)
    nums = (num_train, num_test,num_database)
    return nums, dsets,retrieve, num_classes

def main():
    args = parse_option()

    
    best_map = 0  # best map
    best_epoch = 0  # best epoch
    total_time = 0
    total_loss=0
    train_loss = []
    mAPs = []
    Time=[]
    for epoch in range(args.epochs):
        # torch.autograd.set_detect_anomaly(True)
        net.train()

        trainloader = DataLoader(dataset=dset_train,
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=0)
       
        scheduler.step()



        '''
                training procedure finishes, evaluation
                '''
        net.eval()
        testloader = DataLoader(dataset=dset_test,
                                batch_size=32,
                                shuffle=False,
                                num_workers=0)

        databaseloader = DataLoader(dataset=dset_database,
                                    batch_size=32,
                                    shuffle=False,
                                    num_workers=0)

        apall = np.zeros(num_test)
        for i in range(num_test):
            x = 0
            p = 0
            order = sim_ord[i]
            for j in range(retrieval):
                if np.dot(test_label[i], data_label[order[j]]) > 0:
                    x += 1
                    p += float(x) / (j + 1)
            if p > 0:
                apall[i] = p / x
        mAP = np.mean(apall)
        if mAP > best_map:
            best_map = mAP
            best_epoch = epoch
            print("epoch: ", epoch)
            print("best_map: ", best_map)
            if save_flag:
                f.write("epoch: " + str(epoch) + '\n')
                f.write("best_map: " + str(best_map) + '\n')
        else:
            print("epoch: ", epoch)
            print("map: ", mAP)
            print("best_epoch: ", best_epoch)
            print("best_map: ", best_map)
            if save_flag:
                f.write("epoch: " + str(epoch) + '\n')
                f.write("map: " + str(mAP) + '\n')
                f.write("best_epoch: " + str(best_epoch) + '\n')
                f.write("best_map: " + str(best_map) + '\n')
       


if __name__ == "__main__":
    main()
