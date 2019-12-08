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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from tqdm import tqdm
import argparse
#from models import losses
from models.handestimater_debug import Model
from data.ICVL_loader import ICVL_Loader
from util.visualizer import Visualizer

from numba import cuda

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id: e.g. 0, 1. -1 is no GPU')
parser.add_argument('--dataroot', default='../data/ICVL/process_output/',help='path to images & laser point clouds')
parser.add_argument('--name', type=str, default='train',
                         help='name of the experiment. It decides where to store samples and models')

parser.add_argument('--augment', type=str, default='no', help='whether do augmentation: yes | no')
parser.add_argument('--es_version', type=str, default='v2.1',
                    help='version of estimation: PoseEstimater(v3) PoseEstimater1(v2.1) PoseEstimater2(v4) PoseEstimater3(v5) PoseEstimater4(v6)')
parser.add_argument('--es_loss_version', type=str, default='loss',
                    help='version of estimation loss: LossEstimation(loss) LossEstimation_nyu_sf(nyu) LossEstimation_msra_sf(msra) LossEstimation_icvl_sf(icvl)')


parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/all/full/result', help='models are saved here')
# self.parser.add_argument('--train_record_dir', type=str, default='/home/cyj/Human_analysis/CYJ_HandNet/Hand_SO-Net/MSRA_handpose/checkpoints/train_record.log', help='record train information')

parser.add_argument('--batch_size', type=int, default=22, help='input batch size')
parser.add_argument('--input_pc_num', type=int, default=1024, help='# of input points')
parser.add_argument('--surface_normal', type=bool, default=True, help='use surface normal in the pc input')
parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--display_id', type=int, default=200, help='window id of the web display')

parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

# self.parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr_encoder', type=float, default=1e-5, help='encoder learning rate')
parser.add_argument('--lr_estimater', type=float, default=1e-5, help='estimater learning rate')
parser.add_argument('--lr_decoder', type=float, default=1e-5, help='decoder learning rate')
#parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.6, help='probability of an element to be zeroed')
parser.add_argument('--node_num', type=int, default=64, help='som node number')
parser.add_argument('--k', type=int, default=3, help='k nearest neighbor')
# '/ssd/open-source/so-net-full/autoencoder/checkpoints/save/shapenetpart/183_0.034180_net_encoder.pth'
# self.parser.add_argument('--pretrain', type=str, default=None, help='pre-trained encoder dict path: None or str')
# self.parser.add_argument('--pretrain', type=str, default='checkpoints', help='pre-trained encoder dict path: None or str')
# self.parser.add_argument('--pretrain_encoder', type=str, default='checkpoints/150_0.009468_net_encoder.pth', help='pre-trained encoder dict path: None or str')
parser.add_argument('--pretrain_encoder', type=str, default='checkpoints/all/full/45_7.729756_0.00000100_net_encoder.pth',
                         help='pre-trained encoder dict path: None or str')
parser.add_argument('--pretrain_decoder', type=str, default='checkpoints/all/full/45_0.005988_0.00000100_net_decoder.pth',
                         help='pre-trained decoder dict path: None or str')
parser.add_argument('--pretrain_estimater', type=str, default='checkpoints/all/full/45_7.729756_0.00000100_net_estimater.pth',
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

print(opt.pretrain_encoder,opt.pretrain_decoder,opt.pretrain_estimater)

if __name__=='__main__':
    #if not os.path.exists(opt.train_record_dir):
     #   os.system(r"touch{}".format(opt.train_record_dir))


    #load data
    testset = ICVL_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    print('#testing point clouds = %d' % len(testset))
    visualizer = Visualizer(opt)

    # create model, optionally load pre-trained model
    model = Model(opt)
    current_en_index=0
    current_es_index = 0
    current_de_index = 0
    if opt.pretrain_encoder is not None:
        model.encoder.load_state_dict(torch.load(opt.pretrain_encoder))
        current_en_index=int(opt.pretrain_encoder.split("/")[-1].split("_")[0])
    if opt.pretrain_estimater is not None:
        model.estimater.load_state_dict(torch.load(opt.pretrain_estimater))
        current_es_index = int(opt.pretrain_estimater.split("/")[-1].split("_")[0])
    if opt.pretrain_decoder is not None:
        model.decoder.load_state_dict(torch.load(opt.pretrain_decoder))
        current_de_index = int(opt.pretrain_decoder.split("/")[-1].split("_")[0])

    for epoch in range(1):
        print('======>>>>> Online epoch: #%d <<<<<======' % (epoch + current_es_index + 1))
        epoch_iter = 0

        joints_err_list_test = [0.0]*opt.JOINT_NUM

        # test network
        timer = time.time()
        test_wld_err = 0.0
        test_wld_err_batch = []
        test_wld_err0 = 0.0
        test_wld_err_batch0 = []

        estimate_wld_coor=[]
        gt_wld_coor=[]


        epoch_iter1 = 0
        for i1, data1 in enumerate(tqdm(testloader,0)):
            #epoch_iter1 += opt.batch_size
            #input_pc, input_sn, input_node, input_node_knn_I, gt_pca, gt_xyz, valid, volume_length, volume_offset, volume_rotate, PCA_coeff, PCA_mean = data1
            #model.set_input(input_pc, input_sn, gt_pca, input_node, input_node_knn_I)
            input_pc, input_sn, input_node, input_node_knn_I, gt_xyz, volume_length, volume_offset, volume_rotate = data1
            epoch_iter1 += input_pc.size()[0]

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
            outputs_xyz = estimation_joins.cpu().view(gt_xyz.shape)#[8,3,14]

            # visualize pose
            '''
            x_pl = input_pc[0, :, :].numpy().transpose()
            xyz_pl = gt_xyz[0, :, :].numpy().transpose()
            xyz_es = outputs_xyz[0, :, :].detach().numpy().transpose()
            # node_np = input_node.numpy().transpose()
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x_pl[:, 0].tolist(), x_pl[:, 1].tolist(), x_pl[:, 2].tolist(), s=1)
            ax.scatter(xyz_pl[:, 0].tolist(), xyz_pl[:, 1].tolist(), xyz_pl[:, 2].tolist(), s=6, c='r',label='Ground Truth')
            ax.scatter(xyz_es[:, 0].tolist(), xyz_es[:, 1].tolist(), xyz_es[:, 2].tolist(), s=6, c='g',label='Estimation')

            linelists =[[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                         [0, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15]]
            for linelist in linelists:
                x = [xyz_pl[linelist[0], 0], xyz_pl[linelist[1], 0]]
                y = [xyz_pl[linelist[0], 1], xyz_pl[linelist[1], 1]]
                z = [xyz_pl[linelist[0], 2], xyz_pl[linelist[1], 2]]
                x1 = [xyz_es[linelist[0], 0], xyz_es[linelist[1], 0]]
                y1 = [xyz_es[linelist[0], 1], xyz_es[linelist[1], 1]]
                z1 = [xyz_es[linelist[0], 2], xyz_es[linelist[1], 2]]
                ax.plot(x, y, z, c='r')
                ax.plot(x1, y1, z1, c='g')

            plt.legend(loc='upper left')
            plt.show()
            '''
            '''
            diff0 = torch.pow(outputs_xyz - gt_xyz, 2).view(-1, opt.JOINT_NUM, 3)  # [8,14,3]
            diff_sum0 = torch.sum(diff0, 2)  # [8,21]
            diff_sum_sqrt0 = torch.sqrt(diff_sum0)  # [8,21]
            diff_mean0 = torch.mean(diff_sum_sqrt0, 1).view(-1, 1)  # [8,1]
            diff_mean_wld_test0 = torch.mul(diff_mean0, volume_length)  # [8,1]
            
            test_wld_err_batch0.append(diff_mean_wld_test0.sum().detach().numpy().tolist())
            '''
            #re-compute
            #outputs_xyz = outputs_xyz.view(-1, opt.JOINT_NUM, 3)# [8,14,3]
            outputs_xyz = outputs_xyz.detach().numpy()# [8,3,14]
            #gt_xyz = gt_xyz.view(-1, opt.JOINT_NUM, 3)# [8,14,3]
            gt_xyz = gt_xyz.detach().numpy()# [8,3,14]
            volume_length = volume_length.cpu().detach().numpy()#[8,1]array
            #volume_offset=volume_offset.cpu().unsqueeze(1)# [8,3] -- [8,1,3]
            #volume_offset = volume_offset.expand(volume_offset.size()[0], opt.JOINT_NUM,volume_offset.size()[2]).detach()
            volume_offset = volume_offset.cpu().numpy()  # [8,3]array
            volume_rotate = volume_rotate.cpu().detach().numpy()  # [8,3,3]array

            len_list=outputs_xyz.shape[0]
            output_xyzs = np.zeros_like(outputs_xyz)#World C.S.
            gt_xyzs = np.zeros_like(gt_xyz)

            for ii in range(len_list):
                output_xyz=outputs_xyz[ii,:,:]#(3,14)
                g_xyz=gt_xyz[ii,:,:]#(3,14)
                out_xyz = np.zeros_like(output_xyz)#(3,14)
                gxyz=np.zeros_like(g_xyz)

                v_length=volume_length[ii]#(1)
                v_offset=volume_offset[ii]#(3)
                v_rotate=volume_rotate[ii]#(3,3)

                for jj in range(output_xyz.shape[1]):
                    xyz = output_xyz[:,jj]  # (3)
                    xyz = xyz + v_offset
                    out_xyz[:,jj] = xyz
                    ggxyz = g_xyz[:,jj]
                    ggxyz = ggxyz + v_offset
                    gxyz[:,jj] = ggxyz
                out_xyz = out_xyz * v_length#(3,14)
                out_xyz = np.dot(out_xyz.T, np.linalg.inv(v_rotate.T))#(14,3)
                out_xyz = out_xyz.T#(3,14)
                output_xyzs[ii]=out_xyz
                gxyz = gxyz * v_length
                gxyz = np.dot(gxyz.T, np.linalg.inv(v_rotate.T))#Correct
                gxyz = gxyz.T
                gt_xyzs[ii] = gxyz

            output_xyzs=torch.from_numpy(output_xyzs)
            gt_xyzs = torch.from_numpy(gt_xyzs)


            #a=output_xyzs - gt_xyzs
            diff = torch.pow(output_xyzs - gt_xyzs, 2)  # [8,3,14]
            diff_sum = torch.sum(diff, 1)  # [8,14]
            diff_sum_sqrt = torch.sqrt(diff_sum)  # [8,14]
            diff_mean = torch.mean(diff_sum_sqrt, 1).view(-1, 1)  # [8,1]

            test_wld_err_batch.append(diff_mean.sum().detach().numpy().tolist())

            output_xyzs = output_xyzs.detach().numpy()  # [8,3,14]
            output_xyzs=output_xyzs.swapaxes(1,2)# [8,3,14] -- [8,14,3]

            for iout in output_xyzs:
                estimate_wld_coor.append(iout)

            gt_xyzs = gt_xyzs.detach().numpy()
            gt_xyzs = gt_xyzs.swapaxes(1,2)
            for gout in gt_xyzs:
                gt_wld_coor.append(gout)
            # compute joint errors
            joints_err_list_test = np.sum([joints_err_list_test, torch.sum(diff_sum_sqrt, 0).view(-1).detach().numpy().tolist()], axis=0)
                # break
            '''
            if i1==0:
                x_np = input_pc.numpy().transpose()
                node_np = input_node.numpy().transpose()
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(x_np[:, 0].tolist(), x_np[:, 1].tolist(), x_np[:, 2].tolist(), s=1)
                ax.scatter(node_np[:, 0].tolist(), node_np[:, 1].tolist(), node_np[:, 2].tolist(), s=6, c='r')
                plt.show()
            '''
            #volume_offset_1 = volume_offset.expand(volume_offset.size()[0],opt.JOINT_NUM,volume_offset.size()[1]).detach()# [8,14,3]
            #outputs_xyz = outputs_xyz+volume_offset# [8,14,3]
            #volume_length_1=volume_length.expand(volume_length.size()[0],opt.JOINT_NUM,volume_length.size()[1]).detach()# [8,14,1]
            #outputs_xyz=1;

        # time taken
        timer = time.time() - timer
        timer = timer / epoch_iter1
        print('Testing ==> time to learn 1 sample = %f (ms)' % (timer * 1000))
        # print mse
        test_wld_err = sum(test_wld_err_batch) / epoch_iter1
        test_wld_err0 = sum(test_wld_err_batch0) / epoch_iter1
        print('average estimation error in world coordinate system: %f (mm)' % (test_wld_err))

        # Save estimation pose to txt
        savestr=os.path.join(opt.checkpoints_dir, 'Test_estimation.txt')
        with open(savestr,'w') as fp:
            for line in estimate_wld_coor:
                for l in line:
                    for ll in l:
                        fp.write(str(ll)+' ')
                fp.write('\n')
        print("Write Estimation Result")
        savestr1 = os.path.join(opt.checkpoints_dir, 'Test_groundtruth.txt')
        with open(savestr1,'w') as fp:
            for line in gt_wld_coor:
                for l in line:
                    for ll in l:
                        fp.write(str(ll)+' ')
                fp.write('\n')
        print("Write GroundTruth Result")
            #fp.writelines(estimate_wld_coor)


        # Visualize joints errors
        X=np.arange(opt.JOINT_NUM)
        #Y1=[y1/epoch_iter for y1 in joints_err_list_train]
        Y=[y2/epoch_iter1 for y2 in joints_err_list_test]
        width=0.3
        #plt.bar(left=X, height=Y1, width=width,color='yellow',label='Train')
        plt.bar(left=X+width, height=Y, width=width,color='red',label='Test')
        # plt.plot(x2,y2, '-r', linestyle='-', label='Árfolyam model 2')
        plt.title('Joints Errors')
        plt.xlabel('Joint')
        plt.ylabel('Error')
        plt.legend(loc='best')
        plt.axis([-1,opt.JOINT_NUM+1,0,30])
        #if (epoch + current_es_index + 1) % 5 == 0 and epoch >= 0:
        save_str=os.path.join(opt.checkpoints_dir, str(test_wld_err)+'.png')
        plt.savefig(save_str)
        plt.show()




