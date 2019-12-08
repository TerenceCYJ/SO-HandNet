import time
import copy
import numpy as np
import math
import logging
import argparse

# set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os
from tqdm import tqdm
from models import losses
from models.autoencoder import Model
from data.ICVL_loader import ICVL_Loader
from util.visualizer import Visualizer

from numba import cuda
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id: e.g. 0, 1. -1 is no GPU')

parser.add_argument('--dataset', type=str, default='shapenet', help='shapenet')
parser.add_argument('--dataroot', default='/home/cyj/Human_analysis/data/ICVL/process_output/',help='path to images & laser point clouds')
#parser.add_argument('--dataroot', default='/mnt/data/Chenyujin/Human_analysis/data/ICVL/process_output/',help='path to images & laser point clouds')
# self.parser.add_argument('--classes', type=int, default=50, help='ModelNet40 or ModelNet10')
parser.add_argument('--name', type=str, default='train',help='name of the experiment. It decides where to store samples and models')

parser.add_argument('--augment', type=str, default='no', help='whether do augmentation: yes | no')

parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/autoencoder/full', help='models are saved here')
parser.add_argument('--pretrain_encoder', type=str, default=None,help='pre-trained encoder dict path: None or str')
parser.add_argument('--pretrain_decoder', type=str, default=None,help='pre-trained decoder dict path: None or str')

parser.add_argument('--pretrain_estimater', type=str, default=None,help='pre-trained estimater dict path: None or str')
parser.add_argument('--batch_size', type=int, default=12, help='input batch size')

parser.add_argument('--input_pc_num', type=int, default=1024, help='# of input points')
parser.add_argument('--surface_normal', type=bool, default=True, help='use surface normal in the pc input')
parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--display_id', type=int, default=200, help='window id of the web display')

parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

# self.parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001000, help='encoder learning rate')
parser.add_argument('--dropout', type=float, default=0.6, help='probability of an element to be zeroed')
parser.add_argument('--node_num', type=int, default=64, help='som node number')
parser.add_argument('--k', type=int, default=3, help='k nearest neighbor')
# '/ssd/open-source/so-net-full/autoencoder/checkpoints/save/shapenetpart/183_0.034180_net_encoder.pth'
# parser.add_argument('--pretrain', type=str, default=None, help='pre-trained encoder dict path: None or str')
# parser.add_argument('--pretrain', type=str, default='checkpoints', help='pre-trained encoder dict path: None or str')
# parser.add_argument('--pretrain_encoder', type=str, default='checkpoints/150_0.009468_net_encoder.pth', help='pre-trained encoder dict path: None or str')

parser.add_argument('--som_k', type=int, default=9, help='k nearest neighbor of SOM nodes searching on SOM nodes')
parser.add_argument('--som_k_type', type=str, default='center', help='avg / center')

parser.add_argument('--random_pc_dropout_lower_limit', type=float, default=1, help='keep ratio lower limit')
parser.add_argument('--bn_momentum', type=float, default=0.1,
                         help='normalization momentum, typically 0.1. Equal to (1-m) in TF')
parser.add_argument('--bn_momentum_decay_step', type=int, default=None,
                         help='BN momentum decay step. e.g, 0.5->0.01.')
parser.add_argument('--bn_momentum_decay', type=float, default=0.6, help='BN momentum decay step. e.g, 0.5->0.01.')

# self.parser.add_argument('--output_pc_num', type=int, default=1280, help='# of output points')
parser.add_argument('--output_fc_pc_num', type=int, default=256, help='# of fc decoder output points')
parser.add_argument('--output_conv_pc_num', type=int, default=1024, help='# of conv decoder output points')

parser.add_argument('--JOINT_NUM', type=int, default=16, help='number of joints')
parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=6, help='number of input point features')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--test_index', type=int, default=0, help='test index for cross validation, range: 0~8')
parser.add_argument('--PCA_SZ', type=int, default=42, help='number of PCA components')
parser.add_argument('--weight', type=float, default=6, help='weight of estimater, while weight of decoder is 1')

opt = parser.parse_args()
opt.device = torch.device("cuda:%d" % (opt.gpu_id) if torch.cuda.is_available() else "cpu")
cuda.select_device(opt.gpu_id)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(opt.checkpoints_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

if __name__=='__main__':
    #if not os.path.exists(opt.train_record_dir):
     #   os.system(r"touch{}".format(opt.train_record_dir))


    #load data
    trainset = ICVL_Loader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    print('#training point clouds = %d' % len(trainset))

    testset = ICVL_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    print('#testing point clouds = %d' % len(testset))
    visualizer = Visualizer(opt)

    # create model, optionally load pre-trained model
    model = Model(opt)
    current_en_index=0
    current_de_index=0
    current_es_index = 0
    if opt.pretrain_encoder is not None:
        model.encoder.load_state_dict(torch.load(opt.pretrain_encoder))
        current_en_index=int(opt.pretrain_encoder.split("/")[-1].split("_")[0])
    if opt.pretrain_decoder is not None:
        model.decoder.load_state_dict(torch.load(opt.pretrain_decoder))
        current_de_index = int(opt.pretrain_decoder.split("/")[-1].split("_")[0])
    # Only train and test encoder & decoder
    for epoch in range(500):
        print('======>>>>> Online epoch: #%d <<<<<======' % (epoch + current_en_index + 1))
        epoch_iter = 0
        train_loss_de_batch = []
        timer = time.time()
        for i, data in enumerate(tqdm(trainloader)):
            epoch_iter += opt.batch_size


            input_pc, input_sn, input_node, input_node_knn_I, gt_xyz, volume_length, volume_offset, volume_rotate = data
            model.set_input(input_pc, input_sn, gt_xyz, input_node, input_node_knn_I)#replace gt_pca into gt_xyz

            model.optimize()

            loss_decoder = model.get_de_loss()
            loss_decoder = loss_decoder.cpu()
            train_loss_de_batch.append(loss_decoder.detach().numpy().tolist())
            #break

        # time taken
        timer = time.time() - timer
        timer = timer / epoch_iter
        print('Training ==> time to learn 1 sample = %f (ms)' % (timer * 1000))
        # print mse
        train_loss_de = sum(train_loss_de_batch) / epoch_iter
        print('average decoder loss: %f ' % (train_loss_de))

        #test network
        timer = time.time()
        test_loss_de = 0
        test_loss_de_batch = []
        if epoch >= 0 and epoch % 1 == 0:
            # batch_amount = 0
            epoch_iter1 = 0

            for i, data in enumerate(testloader):
                epoch_iter1 += opt.batch_size
                input_pc, input_sn, input_node, input_node_knn_I, gt_xyz, volume_length, volume_offset, volume_rotate = data
                model.set_input(input_pc, input_sn, gt_xyz, input_node, input_node_knn_I)

                model.test_model()  #

                loss_decoder = model.get_de_loss()
                loss_decoder = loss_decoder.cpu()
                test_loss_de_batch.append(loss_decoder.detach().numpy().tolist())
                #break


            # time taken
            timer = time.time() - timer
            timer = timer / epoch_iter1
            print('Testing ==> time to learn 1 sample = %f (ms)' % (timer * 1000))
            # print mse
            test_loss_de = sum(test_loss_de_batch) / epoch_iter1
            # test_wld_err= np.mean(np.flatten(test_wld_err_list))
            print('average decoder loss: %f ' % (test_loss_de))
            logging.info(
                'Epoch#%d: train decoder error=%e, test decoder error=%e, en-decoder lr = %f' % (
                epoch + current_de_index + 1, train_loss_de, test_loss_de, model.get_lr()))

        # learning rate decay
        if (epoch + current_en_index + 1) % 10 == 0 and epoch > 0:
            model.update_learning_rate(0.5)
        # save network
        if (epoch+ current_en_index + 1) % 1 == 0 :
            print("Saving network...")
            model.save_network(model.encoder, 'encoder', '%d_%f_%f' % (epoch + current_en_index + 1, test_loss_de, model.get_lr()),
                                   opt.gpu_id)
            model.save_network(model.decoder, 'decoder', '%d_%f_%f' % (epoch + current_de_index + 1, test_loss_de, model.get_lr()),
                                   opt.gpu_id)



        torch.cuda.empty_cache()