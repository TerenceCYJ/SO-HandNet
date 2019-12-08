import time
import copy
import numpy as np
import math
import logging

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
#import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from models import losses
from models.handestimater_debug import Model
from data.ICVL_loader_v2 import ICVL_Loader
from util.visualizer import Visualizer

from numba import cuda

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id: e.g. 0, 1. -1 is no GPU')
parser.add_argument('--dataroot', default='/home/cyj/Human_analysis/data/ICVL/process_output/',help='path to images & laser point clouds')
#parser.add_argument('--dataroot', default='/mnt/data/Chenyujin/Human_analysis/data/ICVL/process_output/',help='path to images & laser point clouds')
# self.parser.add_argument('--classes', type=int, default=50, help='ModelNet40 or ModelNet10')
parser.add_argument('--name', type=str, default='train',
                         help='name of the experiment. It decides where to store samples and models')


parser.add_argument('--trainlist', type=str, default='train_list_1.00_0.50.txt', help='')
parser.add_argument('--testlist', type=str, default='test_list.txt', help='')
parser.add_argument('--train_label_ratio', type=float, default=0.75, help='ratio of labeled frames used for training in trainlist')
parser.add_argument('--batch_size', type=int, default=24, help='input batch size')

parser.add_argument('--augment', type=str, default='no', help='whether do augmentation: yes | no')
parser.add_argument('--es_version', type=str, default='v2.1',
                    help='version of estimation: PoseEstimater(v3) PoseEstimater1(v2.1) PoseEstimater2(v4) PoseEstimater3(v5) PoseEstimater4(v6)')
parser.add_argument('--es_loss_version', type=str, default='loss',
                    help='version of estimation loss: LossEstimation(loss) LossEstimation_nyu_sf(nyu) LossEstimation_msra_sf(msra) LossEstimation_icvl_sf(icvl)')


parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/all/semi', help='models are saved here')
# self.parser.add_argument('--train_record_dir', type=str, default='/home/cyj/Human_analysis/CYJ_HandNet/Hand_SO-Net/MSRA_handpose/checkpoints/train_record.log', help='record train information')


parser.add_argument('--input_pc_num', type=int, default=1024, help='# of input points')
parser.add_argument('--surface_normal', type=bool, default=True, help='use surface normal in the pc input')
parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--display_id', type=int, default=200, help='window id of the web display')

parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

# self.parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr_encoder', type=float, default=1e-6, help='encoder learning rate')
parser.add_argument('--lr_estimater', type=float, default=1e-6, help='estimater learning rate')
parser.add_argument('--lr_decoder', type=float, default=1e-6, help='decoder learning rate')
#parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.6, help='probability of an element to be zeroed')
parser.add_argument('--node_num', type=int, default=64, help='som node number')
parser.add_argument('--k', type=int, default=3, help='k nearest neighbor')
# '/ssd/open-source/so-net-full/autoencoder/checkpoints/save/shapenetpart/183_0.034180_net_encoder.pth'
# self.parser.add_argument('--pretrain', type=str, default=None, help='pre-trained encoder dict path: None or str')
# self.parser.add_argument('--pretrain', type=str, default='checkpoints', help='pre-trained encoder dict path: None or str')
# self.parser.add_argument('--pretrain_encoder', type=str, default='checkpoints/150_0.009468_net_encoder.pth', help='pre-trained encoder dict path: None or str')
parser.add_argument('--pretrain_encoder', type=str, default=None,
                         help='pre-trained encoder dict path: None or str')
parser.add_argument('--pretrain_decoder', type=str, default=None,
                         help='pre-trained decoder dict path: None or str')
parser.add_argument('--pretrain_estimater', type=str, default=None,
                         help='pre-trained estimater dict path: None or str')
# self.parser.add_argument('--pretrain_lr_ratio', type=float, default=1, help='learning rate ratio between pretrained encoder and classifier')

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
parser.add_argument('--OUT_DIM', type=int, default=48, help='number of net output')
parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=6, help='number of input point features')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--test_index', type=int, default=0, help='test index for cross validation, range: 0~8')
#parser.add_argument('--PCA_SZ', type=int, default=42, help='number of PCA components')
parser.add_argument('--weight', type=float, default=100, help='weight of estimater, while weight of decoder is 1')

opt = parser.parse_args()
opt.device = torch.device("cuda:%d" % (opt.gpu_id) if torch.cuda.is_available() else "cpu")
cuda.select_device(opt.gpu_id)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(opt.checkpoints_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')
logging.info('weight: %f', opt.weight)
print(opt.pretrain_encoder,opt.pretrain_decoder,opt.pretrain_estimater)
logging.info('encoder=%s' %opt.pretrain_encoder)
logging.info('decoder=%s' %opt.pretrain_decoder)
logging.info('estimater=%s' %opt.pretrain_estimater)
if __name__=='__main__':
    #if not os.path.exists(opt.train_record_dir):
     #   os.system(r"touch{}".format(opt.train_record_dir))


    #load data
    trainset = ICVL_Loader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    print('#training point clouds = %d' % len(trainset))

    testset = ICVL_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    print('#testing point clouds = %d' % len(testset))
    visualizer = Visualizer(opt)

    # create model, optionally load pre-trained model
    model = Model(opt)
    current_en_index=0
    current_es_index = 0
    current_de_index = 0

    n_slect=int(opt.batch_size*opt.train_label_ratio)

    if opt.pretrain_encoder is not None:
        model.encoder.load_state_dict(torch.load(opt.pretrain_encoder))
        current_en_index=int(opt.pretrain_encoder.split("/")[-1].split("_")[0])
    if opt.pretrain_estimater is not None:
        model.estimater.load_state_dict(torch.load(opt.pretrain_estimater))
        current_es_index = int(opt.pretrain_estimater.split("/")[-1].split("_")[0])
    if opt.pretrain_decoder is not None:
        model.decoder.load_state_dict(torch.load(opt.pretrain_decoder))
        current_de_index = int(opt.pretrain_decoder.split("/")[-1].split("_")[0])

    for epoch in range(40):
        print('======>>>>> Online epoch: #%d <<<<<======' % (epoch + current_es_index + 1))
        epoch_iter = 0
        train_loss_es_batch = []
        train_loss_de_batch = []
        train_loss_all_batch = []
        train_wld_err_batch = []
        joints_err_list_train = [0.0]*opt.JOINT_NUM
        joints_err_list_test = [0.0]*opt.JOINT_NUM
        timer = time.time()
        for i, data in enumerate(tqdm(trainloader,0)):
            epoch_iter += opt.batch_size
            #input_pc, input_sn, input_node, input_node_knn_I, gt_pca, gt_xyz, valid, volume_length, volume_offset, volume_rotate, PCA_coeff, PCA_mean = data
            #model.set_input(input_pc, input_sn, gt_pca, input_node, input_node_knn_I)
            input_pc, input_sn, input_node, input_node_knn_I, gt_xyz, volume_length, volume_offset, volume_rotate = data
            #model.set_input(input_pc, input_sn, gt_xyz, input_node, input_node_knn_I)

            #data use label
            input_pc1, input_sn1, input_node1, input_node_knn_I1, gt_xyz1, volume_length1, volume_offset1, volume_rotate1 = input_pc[0:n_slect,:,:], input_sn[0:n_slect,:,:], input_node[0:n_slect,:,:], input_node_knn_I[0:n_slect,:,:], gt_xyz[0:n_slect,:,:], volume_length[0:n_slect,:], volume_offset[0:n_slect,:], volume_rotate[0:n_slect,:,:]
            #data don't use label
            input_pc2, input_sn2, input_node2, input_node_knn_I2, gt_xyz2, volume_length2, volume_offset2, volume_rotate2 = input_pc[n_slect:opt.batch_size,:,:], input_sn[n_slect:opt.batch_size,:,:], input_node[n_slect:opt.batch_size,:,:], input_node_knn_I[n_slect:opt.batch_size,:,:], gt_xyz[n_slect:opt.batch_size,:,:], volume_length[n_slect:opt.batch_size,:], volume_offset[n_slect:opt.batch_size,:], volume_rotate[n_slect:opt.batch_size,:,:]

            #if epoch_iter==opt.batch_size:
            #   print(volume_length1)
            #   print(volume_length2)
            #Train data that don't use label

            if torch.numel(input_pc2)!=0:
                model.set_input(input_pc2, input_sn2, gt_xyz2, input_node2, input_node_knn_I2)
                model.encoder_decoder_optimize()
                loss_decoder = model.get_de_loss()
                loss_decoder = loss_decoder.cpu()
                train_loss_de_batch.append(loss_decoder.detach().numpy().tolist())

            #Train data use label
            model.set_input(input_pc1, input_sn1, gt_xyz1, input_node1, input_node_knn_I1)
            model.optimize()
            # computer error in world cs
            estimation_joins = model.get_estimation()
            estimation_joins = estimation_joins.view(estimation_joins.size(0), 1,
                                                         estimation_joins.size(1))  # [ ,1,42]
            outputs_xyz = estimation_joins.cpu().view(gt_xyz1.shape)
            # re-compute
            outputs_xyz = outputs_xyz.detach().numpy()  # [8,3,14]
            gt_xyz1 = gt_xyz1.detach().numpy()  # [8,3,14]
            volume_length1 = volume_length1.cpu().detach().numpy()  # [8,1]array
            volume_offset1 = volume_offset1.cpu().numpy()  # [8,3]array
            volume_rotate1 = volume_rotate1.cpu().detach().numpy()  # [8,3,3]array
            len_list = outputs_xyz.shape[0]
            output_xyzs = np.zeros_like(outputs_xyz)  # World C.S.
            gt_xyzs = np.zeros_like(gt_xyz1)
            for ii in range(len_list):
                output_xyz = outputs_xyz[ii, :, :]  # (3,14)
                g_xyz = gt_xyz1[ii, :, :]  # (3,14)
                out_xyz = np.zeros_like(output_xyz)  # (3,14)
                gxyz = np.zeros_like(g_xyz)
                v_length = volume_length1[ii]  # (1)
                v_offset = volume_offset1[ii]  # (3)
                v_rotate = volume_rotate1[ii]  # (3,3)
                for jj in range(output_xyz.shape[1]):
                    xyz = output_xyz[:, jj]  # (3)
                    xyz = xyz + v_offset
                    out_xyz[:, jj] = xyz
                    ggxyz = g_xyz[:, jj]
                    ggxyz = ggxyz + v_offset
                    gxyz[:, jj] = ggxyz
                out_xyz = out_xyz * v_length  # (3,14)
                out_xyz = np.dot(out_xyz.T, np.linalg.inv(v_rotate.T))  # (14,3)
                out_xyz = out_xyz.T  # (3,14)
                output_xyzs[ii] = out_xyz
                gxyz = gxyz * v_length
                gxyz = np.dot(gxyz.T, np.linalg.inv(v_rotate.T))  # Correct
                gxyz = gxyz.T
                gt_xyzs[ii] = gxyz

            output_xyzs = torch.from_numpy(output_xyzs)
            gt_xyzs = torch.from_numpy(gt_xyzs)

            # a=output_xyzs - gt_xyzs
            diff = torch.pow(output_xyzs - gt_xyzs, 2)  # [8,3,14]
            diff_sum = torch.sum(diff, 1)  # [8,14]
            diff_sum_sqrt = torch.sqrt(diff_sum)  # [8,14]
            diff_mean = torch.mean(diff_sum_sqrt, 1).view(-1, 1)  # [8,1]
            train_wld_err_batch.append(diff_mean.sum().detach().numpy().tolist())

            loss_estimater = model.get_es_loss()
            loss_estimater = loss_estimater.cpu()
            train_loss_es_batch.append(loss_estimater.detach().numpy().tolist())

            loss_decoder = model.get_de_loss()
            loss_decoder = loss_decoder.cpu()
            train_loss_de_batch.append(loss_decoder.detach().numpy().tolist())

            loss_all = model.get_loss()
            loss_all = loss_all.cpu()
            train_loss_all_batch.append(loss_all.detach().numpy().tolist())

            #compute joint errors
            joints_err_list_train = np.sum([joints_err_list_train, torch.sum(diff_sum_sqrt, 0).view(-1).detach().numpy().tolist()], axis=0)
            #break
        # time taken
        timer = time.time() - timer
        timer = timer / epoch_iter
        print('Training ==> time to learn 1 sample = %f (ms)' % (timer * 1000))

        # print mse
        train_mse_wld = sum(train_wld_err_batch) / (epoch_iter*opt.train_label_ratio)
        train_loss_es = sum(train_loss_es_batch) / (epoch_iter*opt.train_label_ratio)
        train_loss_de = sum(train_loss_de_batch) / epoch_iter
        train_loss_all = sum(train_loss_all_batch) / (epoch_iter*opt.train_label_ratio)

        print('average loss: %f, average decoder loss: %f, average estimater loss: %f, ' % (train_loss_all, train_loss_de, train_loss_es ))
        print('average estimation error in world coordinate system: %f (mm)' % (train_mse_wld))


        # test network
        timer = time.time()
        test_wld_err = 0.0
        test_wld_err_batch = []
        test_loss_es = 0.0
        test_loss_es_batch = []
        test_loss_de = 0.0
        test_loss_de_batch = []
        test_loss_all = 0.0
        test_loss_all_batch = []

        epoch_iter1 = 0
        for i1, data1 in enumerate(tqdm(testloader,0)):
            epoch_iter1 += opt.batch_size
            #input_pc, input_sn, input_node, input_node_knn_I, gt_pca, gt_xyz, valid, volume_length, volume_offset, volume_rotate, PCA_coeff, PCA_mean = data1
            #model.set_input(input_pc, input_sn, gt_pca, input_node, input_node_knn_I)
            input_pc, input_sn, input_node, input_node_knn_I, gt_xyz, volume_length, volume_offset, volume_rotate = data1
            model.set_input(input_pc, input_sn, gt_xyz, input_node, input_node_knn_I)
            model.test_model()  #

            estimation_joins = model.get_estimation()
            estimation_joins = estimation_joins.view(estimation_joins.size(0), 1, estimation_joins.size(1))  # [8,1,42]

            '''
            #outputs_xyz = torch.baddbmm(PCA_mean, estimation_joins, PCA_coeff)
            outputs_xyz = torch.bmm(estimation_joins, PCA_coeff.cuda()).cpu()
            outputs_xyz = torch.add(PCA_mean, outputs_xyz)
            estimation_joins = estimation_joins.cpu()
            outputs_xyz = outputs_xyz.view(gt_xyz.shape)  # [8,3,21]
            '''
            outputs_xyz = estimation_joins.cpu().view(gt_xyz.shape)

            # re-compute
            outputs_xyz = outputs_xyz.detach().numpy()  # [8,3,14]
            # gt_xyz = gt_xyz.view(-1, opt.JOINT_NUM, 3)# [8,14,3]
            gt_xyz = gt_xyz.detach().numpy()  # [8,3,14]
            volume_length = volume_length.cpu().detach().numpy()  # [8,1]array
            volume_offset = volume_offset.cpu().numpy()  # [8,3]array
            volume_rotate = volume_rotate.cpu().detach().numpy()  # [8,3,3]array

            len_list = outputs_xyz.shape[0]
            output_xyzs = np.zeros_like(outputs_xyz)  # World C.S.
            gt_xyzs = np.zeros_like(gt_xyz)

            for ii in range(len_list):
                output_xyz = outputs_xyz[ii, :, :]  # (3,14)
                g_xyz = gt_xyz[ii, :, :]  # (3,14)
                out_xyz = np.zeros_like(output_xyz)  # (3,14)
                gxyz = np.zeros_like(g_xyz)

                v_length = volume_length[ii]  # (1)
                v_offset = volume_offset[ii]  # (3)
                v_rotate = volume_rotate[ii]  # (3,3)

                for jj in range(output_xyz.shape[1]):
                    xyz = output_xyz[:, jj]  # (3)
                    xyz = xyz + v_offset
                    out_xyz[:, jj] = xyz
                    ggxyz = g_xyz[:, jj]
                    ggxyz = ggxyz + v_offset
                    gxyz[:, jj] = ggxyz
                out_xyz = out_xyz * v_length  # (3,14)
                out_xyz = np.dot(out_xyz.T, np.linalg.inv(v_rotate.T))  # (14,3)
                out_xyz = out_xyz.T  # (3,14)
                output_xyzs[ii] = out_xyz
                gxyz = gxyz * v_length
                gxyz = np.dot(gxyz.T, np.linalg.inv(v_rotate.T))  # Correct
                gxyz = gxyz.T
                gt_xyzs[ii] = gxyz

            output_xyzs = torch.from_numpy(output_xyzs)
            gt_xyzs = torch.from_numpy(gt_xyzs)

            # a=output_xyzs - gt_xyzs
            diff = torch.pow(output_xyzs - gt_xyzs, 2)  # [8,3,14]
            diff_sum = torch.sum(diff, 1)  # [8,14]
            diff_sum_sqrt = torch.sqrt(diff_sum)  # [8,14]
            diff_mean = torch.mean(diff_sum_sqrt, 1).view(-1, 1)  # [8,1]
            test_wld_err_batch.append(diff_mean.sum().detach().numpy().tolist())

            loss_estimater = model.get_es_loss()
            loss_estimater = loss_estimater.cpu()
            test_loss_es_batch.append(loss_estimater.detach().numpy().tolist())

            loss_decoder = model.get_de_loss()
            loss_decoder = loss_decoder.cpu()
            test_loss_de_batch.append(loss_decoder.detach().numpy().tolist())

            loss_all = model.get_loss()
            loss_all = loss_all.cpu()
            test_loss_all_batch.append(loss_all.detach().numpy().tolist())

            # compute joint errors
            joints_err_list_test = np.sum([joints_err_list_test, torch.sum(diff_sum_sqrt, 0).view(-1).detach().numpy().tolist()], axis=0)
            #break

        # time taken
        timer = time.time() - timer
        timer = timer / epoch_iter1
        print('Testing ==> time to learn 1 sample = %f (ms)' % (timer * 1000))
        # print mse
        test_wld_err = sum(test_wld_err_batch) / epoch_iter1
        test_loss_es = sum(test_loss_es_batch) / epoch_iter1
        test_loss_de = sum(test_loss_de_batch) / epoch_iter1
        test_loss_all = sum(test_loss_all_batch) / epoch_iter1

        print('average loss: %f, average decoder loss: %f, average estimater loss: %f, ' % (test_loss_all, test_loss_de, test_loss_es))
        print('average estimation error in world coordinate system: %f (mm)' % (test_wld_err))
        logging.info('Epoch#%d: train error=%e, train wld error = %f mm, test error=%e, test wld error = %f mm, estimater lr = %.8f' % (epoch + current_es_index + 1, train_loss_es, train_mse_wld, test_loss_es, test_wld_err, model.get_es_lr()))

        #Visualize joints errors
        '''

        X=np.arange(21)
        Y1=[y1/epoch_iter for y1 in joints_err_list_train]
        Y2=[y2/epoch_iter1 for y2 in joints_err_list_test]
        width=0.3
        plt.bar(left=X, height=Y1, width=width,color='yellow',label='Train')
        plt.bar(left=X+width, height=Y2, width=width,color='red',label='Test')
        # plt.plot(x2,y2, '-r', linestyle='-', label='Árfolyam model 2')
        plt.title('Joints Errors')
        plt.xlabel('Joint')
        plt.ylabel('Error')
        plt.legend(loc='best')
        plt.axis([-1,22,0,30])
        if (epoch + current_es_index + 1) % 5 == 0 and epoch >= 0:
            save_str=os.path.join(opt.checkpoints_dir, str(epoch + current_es_index + 1)+'_'+str(test_wld_err)+'.png')
            plt.savefig(save_str)
        plt.show()
        '''


        # save network
        if (epoch + current_es_index + 1) % 1 == 0 and epoch >= 0:
            print("Saving network...")
            model.save_network(model.encoder, 'encoder', '%d_%f_%.8f' % (epoch + current_en_index + 1, test_wld_err, model.get_es_lr()),
                               opt.gpu_id)
            model.save_network(model.estimater, 'estimater', '%d_%f_%.8f' % (epoch + current_es_index + 1, test_wld_err, model.get_es_lr()),
                                   opt.gpu_id)
            model.save_network(model.decoder, 'decoder','%d_%f_%.8f' % (epoch + current_de_index + 1, test_loss_de, model.get_es_lr()),
                               opt.gpu_id)
        # learning rate decay
        if (epoch + current_es_index + 1) % 30 == 0 and epoch >= 0:
            model.update_learning_rate(0.1)
        torch.cuda.empty_cache()