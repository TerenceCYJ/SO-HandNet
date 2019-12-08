import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import time

from util import som
from . import operations
from .layers import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.opt = opt

        self.first_pointnet = PointNet(3, (32, 64, 128), activation=self.opt.activation, normalization=self.opt.normalization,
                                       momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        self.second_pointnet = PointNet(128+128, (256, 256), activation=self.opt.activation, normalization=self.opt.normalization,
                                        momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        # regressor to get a sin(theta)
        self.fc1 = MyLinear(256, 128, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                            bn_momentum_decay=opt.bn_momentum_decay)
        self.fc2 = MyLinear(128, 64, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                            bn_momentum_decay=opt.bn_momentum_decay)
        self.fc3 = MyLinear(64, 1, activation=None, normalization=None)

        self.dropout1 = nn.Dropout(p=self.opt.dropout)
        self.dropout2 = nn.Dropout(p=self.opt.dropout)

    def forward(self, x, sn=None, epoch=None):
        '''

        :param x: BxMx3 som nodes / Bx3xN points
        :param sn: Bx3xN surface normal
        :return:
        '''

        first_pn_out = self.first_pointnet(x, epoch)  # BxCxN
        feature_1, _ = torch.max(first_pn_out, dim=2, keepdim=False)  # BxC


        second_pn_out = self.second_pointnet(torch.cat((first_pn_out, feature_1.unsqueeze(2).expand_as(first_pn_out)), dim=1), epoch)
        feature_2, _ = torch.max(second_pn_out, dim=2, keepdim=False)

        # get sin(theta)
        fc1_out = self.fc1(feature_2, epoch)
        if self.opt.dropout > 0.1:
            fc1_out = self.dropout1(fc1_out)
        self.fc2_out = self.fc2(fc1_out, epoch)
        if self.opt.dropout > 0.1:
            self.fc2_out = self.dropout2(self.fc2_out)

        sin_theta = torch.tanh(self.fc3(self.fc2_out, epoch))  # Bx1

        return sin_theta


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # transformer
        self.transformer = Transformer(opt)

        # first PointNet
        if self.opt.surface_normal == True:
            self.first_pointnet = PointResNet(6, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            self.first_pointnet = PointResNet(3, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        if self.opt.som_k >= 2:
            # second PointNet
            self.knnlayer = KNNModule(3+384, (512, 512), activation=self.opt.activation, normalization=self.opt.normalization,
                                      momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

            # final PointNet
            self.final_pointnet = PointNet(3+512, (768, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                           momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            # final PointNet
            self.final_pointnet = PointResNet(3+384, (512, 512, 768, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)



        # build som for clustering, node initalization is done in __init__
        rows = int(math.sqrt(self.opt.node_num))
        cols = rows
        self.som_builder = som.BatchSOM(rows, cols, 3, self.opt.gpu_id, self.opt.batch_size)

        # masked max
        self.masked_max = operations.MaskedMax(self.opt.node_num, gpu_id=self.opt.gpu_id)

        # padding
        self.zero_pad = torch.nn.ZeroPad2d(padding=1)

    def forward(self, x, sn, node, node_knn_I, is_train=False, epoch=None):
        '''

        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param node_knn_I: BxMxk_som LongTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''

        # optimize the som, access the Tensor's tensor, the optimize function should not modify the tensor
        # self.som_builder.optimize(x.data)
        self.som_builder.node.resize_(node.size()).copy_(node)

        # modify the x according to the nodes, minus the center
        # $
        self.mask, mask_row_max, min_idx = self.som_builder.query_topk(x.data, k=self.opt.k)  # BxkNxnode_num, Bxnode_num
        # $
        mask_row_sum = torch.sum(self.mask, dim=1)  # Bxnode_num
        mask = self.mask.unsqueeze(1)  # Bx1xkNxnode_num

        # if necessary, stack the x
        x_list, sn_list = [], []
        for i in range(self.opt.k):
            x_list.append(x)
            sn_list.append(sn)
        x_stack = torch.cat(tuple(x_list), dim=2)
        sn_stack = torch.cat(tuple(sn_list), dim=2)

        # re-compute center, instead of using som.node
        # $
        x_stack_data_unsqueeze = x_stack.data.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (mask_row_sum.unsqueeze(1).float()+1e-5)  # BxCxnode_num
        self.som_builder.node = cluster_mean
        self.som_node = self.som_builder.node


        # ====== apply transformer to rotate x_stack, sn_stack, som_node ======
        # sin_theta = self.transformer(x=self.som_node, sn=None, epoch=epoch)  # Bx1
        # # sin_theta = self.transformer(x=torch.cat((x_stack, sn_stack), dim=1), sn=None, epoch=epoch)  # Bx1
        # cos_theta = torch.sqrt(1 + 1e-5 - sin_theta*sin_theta)  # Bx1
        # B = x.size()[0]
        # rotation_matrix = torch.Tensor(B, 3, 3).zero_().to(self.opt.device)  # Bx3x3
        # rotation_matrix[:, 0, 0] = cos_theta[:, 0]
        # rotation_matrix[:, 0, 2] = sin_theta[:, 0]
        # rotation_matrix[:, 1, 1] = 1
        # rotation_matrix[:, 2, 0] = -1 * sin_theta[:, 0]
        # rotation_matrix[:, 2, 2] = cos_theta[:, 0]
        # # print(rotation_matrix)

        # x_stack = torch.matmul(rotation_matrix, x_stack)
        # sn_stack = torch.matmul(rotation_matrix, sn_stack)
        # self.som_node = torch.matmul(rotation_matrix, self.som_node)
        # self.som_builder.node = torch.matmul(rotation_matrix.data, self.som_builder.node)
        # ====== apply transformer to rotate x_stack, sn_stack, som_node ======


        # assign each point with a center
        node_expanded = self.som_node.data.unsqueeze(2)  # BxCx1xnode_num, som.node is BxCxnode_num
        self.centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN

        self.x_decentered = (x_stack - self.centers).detach()  # Bx3xkN
        x_augmented = torch.cat((self.x_decentered, sn_stack), dim=1)  # Bx6xkN

        # go through the first PointNet
        if self.opt.surface_normal == True:
            # $
            self.first_pn_out = self.first_pointnet(x_augmented, epoch)
        else:
            self.first_pn_out = self.first_pointnet(self.x_decentered, epoch)

        gather_index = self.masked_max.compute(self.first_pn_out.data, min_idx, mask).detach()
        # $
        self.first_pn_out_masked_max = self.first_pn_out.gather(dim=2, index=gather_index) * mask_row_max.unsqueeze(1).float()  # BxCxM

        if self.opt.som_k >= 2:
            # second pointnet, knn search on SOM nodes: ----------------------------------
            self.knn_center_1, self.knn_feature_1 = self.knnlayer(self.som_node, self.first_pn_out_masked_max, node_knn_I, self.opt.som_k, self.opt.som_k_type, epoch)

            # final pointnet --------------------------------------------------------------
            self.final_pn_out = self.final_pointnet(torch.cat((self.knn_center_1, self.knn_feature_1), dim=1), epoch)  # Bx1024xM
        else:
            # final pointnet --------------------------------------------------------------
            self.final_pn_out = self.final_pointnet(torch.cat((self.som_node, self.first_pn_out_masked_max), dim=1), epoch)  # Bx1024xM

        self.feature, _ = torch.max(self.final_pn_out, dim=2, keepdim=False)

        return self.feature


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # classifier
        self.fc1 = MyLinear(self.feature_num, 512, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.fc2 = MyLinear(512, 256, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.fc3 = MyLinear(256, self.opt.classes, activation=None, normalization=None)

        self.dropout1 = nn.Dropout(p=self.opt.dropout)
        self.dropout2 = nn.Dropout(p=self.opt.dropout)

    def forward(self, feature, epoch=None):
        fc1_out = self.fc1(feature, epoch)
        if self.opt.dropout > 0.1:
            fc1_out = self.dropout1(fc1_out)
        self.fc2_out = self.fc2(fc1_out, epoch)
        if self.opt.dropout > 0.1:
            self.fc2_out = self.dropout2(self.fc2_out)
        score = self.fc3(self.fc2_out, epoch)

        return score

# Use node+global feature for hand estimation M--KN, estimator v3
class PoseEstimater(nn.Module):
    def __init__(self, opt):
        super(PoseEstimater, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_dim = opt.OUT_DIM

        # PoseEstimater
        in_node_channels = 384 + 512 + 2 * self.feature_num #2944

        self.pslayer1 = EquivariantLayer(in_node_channels,
                                       1024,
                                       activation=self.opt.activation,
                                       normalization=self.opt.normalization)
        self.net_FC = nn.Sequential(
            # B*1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256
            nn.Linear(256, self.output_dim),
            # B*num_outputs
        )
    def forward(self,x_decentered, x, centers, sn,
                first_pn_out,
                first_pn_out_masked_max,
                feature_max_first_pn_out,
                feature_max_knn_feature_1,
                feature_max_final_pn_out,
                feature,
                epoch=None):
        input1 = torch.cat((feature_max_first_pn_out,
                                 feature_max_knn_feature_1,
                                 feature_max_final_pn_out),dim=1)#[8,1920,3072]
        feature = feature.unsqueeze(-1)#[B,1024,1]
        #input2 = feature.expand(feature.size()[0],input1.size(1),feature.size()[2]).detach()#[8,1920,1024]
        input2 = feature.expand(feature.size()[0],feature.size()[1],input1.size(2)).detach()#[8,1024,3072]
        pslayer1_in=torch.cat((input1,input2),dim=1)#[8,2944,3072]

        pslayer1_out=self.pslayer1(pslayer1_in)#[8,1024,3072]
        pslayer1_out_mean=torch.mean(pslayer1_out, dim=2, keepdim=False)#[8,1024]
        #pslayer1_out_max = torch.max(pslayer1_out, dim=2, keepdim=False)
        #net_FC_in=torch.cat((pslayer1_out_mean[0], feature), dim=1)
        net_FC_out=self.net_FC(pslayer1_out_mean)
        #pslayer2_in=torch.cat((pslayer1_out_mean,feature),dim=1)
        return net_FC_out

# Use global+node feature for hand estimation, estimator v2_wrong_connection
class PoseEstimater0(nn.Module):
    def __init__(self, opt):
        super(PoseEstimater0, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_dim = opt.OUT_DIM

        # PoseEstimater
        in_node_channels = 384 + 512 + self.feature_num#1920

        self.pslayer1 = EquivariantLayer(in_node_channels,
                                       1024,
                                       activation=self.opt.activation,
                                       normalization=self.opt.normalization)
        self.net_FC = nn.Sequential(
            # B*1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256
            nn.Linear(256, self.output_dim),
            # B*num_outputs
        )
    def forward(self,x_decentered, x, centers, sn,
                first_pn_out,
                first_pn_out_masked_max,
                feature_max_first_pn_out,
                feature_max_knn_feature_1,
                feature_max_final_pn_out,
                feature,
                epoch=None):
        input1 = torch.cat((feature_max_first_pn_out,
                                 feature_max_knn_feature_1,
                                 feature_max_final_pn_out),dim=1)#[8,1920,3072]
        feature = feature.unsqueeze(1)
        input2 = feature.expand(feature.size()[0],input1.size(1),feature.size()[2]).detach()#[8,1920,1024]
        pslayer1_in=torch.cat((input1,input2),dim=2)#[8,1920,4096]

        pslayer1_out=self.pslayer1(pslayer1_in)#[8,1024,4096]
        pslayer1_out_mean=torch.mean(pslayer1_out, dim=2, keepdim=False)#[8,1024]
        #pslayer1_out_max = torch.max(pslayer1_out, dim=2, keepdim=False)
        #net_FC_in=torch.cat((pslayer1_out_mean[0], feature), dim=1)
        net_FC_out=self.net_FC(pslayer1_out_mean)
        #pslayer2_in=torch.cat((pslayer1_out_mean,feature),dim=1)
        return net_FC_out

# Use global+node feature for hand estimation M, estimator v2.1
class PoseEstimater1(nn.Module):
    def __init__(self, opt):
        super(PoseEstimater1, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_dim = opt.OUT_DIM

        # PoseEstimater
        in_node_channels = 384 + self.feature_num#1408

        self.pslayer1 = EquivariantLayer(in_node_channels,
                                       1024,
                                       activation=self.opt.activation,
                                       normalization=self.opt.normalization)
        self.net_FC = nn.Sequential(
            # B*1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256
            nn.Linear(256, self.output_dim),
            # B*num_outputs
        )
    def forward(self,x_decentered, x, centers, sn,
                first_pn_out,
                first_pn_out_masked_max,
                feature_max_first_pn_out,
                feature_max_knn_feature_1,
                feature_max_final_pn_out,
                feature,
                epoch=None):
        #input1 = torch.cat((feature_max_first_pn_out,feature_max_knn_feature_1,feature_max_final_pn_out),dim=1)#[8,1920,3072]
        input1 = first_pn_out_masked_max #[B,384,64]
        feature = feature.unsqueeze(-1) #[B,1024,1]
        input2 = feature.expand(feature.size()[0],feature.size()[1],input1.size()[2]).detach()#[8,1024,64]
        pslayer1_in=torch.cat((input1,input2),dim=1)#[8,1408,64]

        pslayer1_out=self.pslayer1(pslayer1_in)#[8,1024,64]
        pslayer1_out_mean=torch.mean(pslayer1_out, dim=2, keepdim=False)#[8,1024]
        #pslayer1_out_max = torch.max(pslayer1_out, dim=2, keepdim=False)
        #net_FC_in=torch.cat((pslayer1_out_mean[0], feature), dim=1)
        net_FC_out=self.net_FC(pslayer1_out_mean)
        #pslayer2_in=torch.cat((pslayer1_out_mean,feature),dim=1)
        return net_FC_out

# Use global feature for hand estimation v4
class PoseEstimater2(nn.Module):
    def __init__(self, opt):
        super(PoseEstimater2, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_dim = opt.OUT_DIM

        self.net_FC = nn.Sequential(
            # B*1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256
            nn.Linear(256, self.output_dim),
            # B*num_outputs
        )
    def forward(self,x_decentered, x, centers, sn,
                first_pn_out,
                first_pn_out_masked_max,
                feature_max_first_pn_out,
                feature_max_knn_feature_1,
                feature_max_final_pn_out,
                feature,
                epoch=None):
        net_FC_in=feature
        net_FC_out=self.net_FC(net_FC_in)
        return net_FC_out

# Use node feature for hand estimation v5
class PoseEstimater3(nn.Module):
    def __init__(self, opt):
        super(PoseEstimater3, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_dim = opt.OUT_DIM

        # PoseEstimater
        in_node_channels = 384
        self.pslayer1 = EquivariantLayer(in_node_channels,
                                         1024,
                                         activation=self.opt.activation,
                                         normalization=self.opt.normalization)

        self.net_FC = nn.Sequential(
            # B*1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256
            nn.Linear(256, self.output_dim),
            # B*num_outputs
        )
    def forward(self,x_decentered, x, centers, sn,
                first_pn_out,
                first_pn_out_masked_max,
                feature_max_first_pn_out,
                feature_max_knn_feature_1,
                feature_max_final_pn_out,
                feature,
                epoch=None):
        pslayer1_in=first_pn_out_masked_max#[B,384,64]
        pslayer1_out=self.pslayer1(pslayer1_in)#[8,1024,64]
        pslayer1_out_mean=torch.mean(pslayer1_out, dim=2, keepdim=False)#[8,1024]
        net_FC_out=self.net_FC(pslayer1_out_mean)
        return net_FC_out

# Use point+node+global feature for hand estimation KN, estimator v6
class PoseEstimater4(nn.Module):
    def __init__(self, opt):
        super(PoseEstimater4, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_dim = opt.OUT_DIM

        # PoseEstimater
        in_node_channels = 3 + 3 + 3 + 3 + 384 + 384 + 512 + 2 * self.feature_num #3340

        self.pslayer1 = EquivariantLayer(in_node_channels,
                                       1024,
                                       activation=self.opt.activation,
                                       normalization=self.opt.normalization)
        self.net_FC = nn.Sequential(
            # B*1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256
            nn.Linear(256, self.output_dim),
            # B*num_outputs
        )
    def forward(self, x_decentered, x, centers, sn,
                first_pn_out,
                first_pn_out_masked_max,
                feature_max_first_pn_out,
                feature_max_knn_feature_1,
                feature_max_final_pn_out,
                feature,
                epoch=None):

        B = x.size()[0]
        N = x.size()[2]
        k = self.opt.k
        kN = round(k * N)

        # if necessary, stack the x
        x_list, sn_list = [], []
        for i in range(self.opt.k):
            x_list.append(x)
            sn_list.append(sn)
        x = torch.cat(tuple(x_list), dim=2)  # [8,3,3072]
        sn = torch.cat(tuple(sn_list), dim=2)  # [8,3,3072]

        input1 = torch.cat((x_decentered, x, centers, sn,
                            first_pn_out,
                            feature_max_first_pn_out,
                            feature_max_knn_feature_1,
                            feature_max_final_pn_out),dim=1)#[8,2316,3072]
        feature = feature.unsqueeze(-1)#[B,1024,1]
        #input2 = feature.expand(feature.size()[0],input1.size(1),feature.size()[2]).detach()#[8,1920,1024]
        input2 = feature.expand(feature.size()[0],feature.size()[1],input1.size(2)).detach()#[8,1024,3072]
        pslayer1_in=torch.cat((input1,input2),dim=1)#[8,3340,3072]

        pslayer1_out=self.pslayer1(pslayer1_in)#[8,1024,3072]
        pslayer1_out_mean=torch.mean(pslayer1_out, dim=2, keepdim=False)#[8,1024]
        #pslayer1_out_max = torch.max(pslayer1_out, dim=2, keepdim=False)
        #net_FC_in=torch.cat((pslayer1_out_mean[0], feature), dim=1)
        net_FC_out=self.net_FC(pslayer1_out_mean)
        #pslayer2_in=torch.cat((pslayer1_out_mean,feature),dim=1)
        return net_FC_out



class DecoderLinear(nn.Module):
    def __init__(self, opt):
        super(DecoderLinear, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_point_number = opt.output_fc_pc_num

        self.linear1 = MyLinear(self.feature_num, self.output_point_number*2, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear2 = MyLinear(self.output_point_number*2, self.output_point_number*3, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear3 = MyLinear(self.output_point_number*3, self.output_point_number*4, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear_out = MyLinear(self.output_point_number*4, self.output_point_number*3, activation=None, normalization=None)

        # special initialization for linear_out, to get uniform distribution over the space
        self.linear_out.linear.bias.data.uniform_(-1, 1)

    def forward(self, x):
        # reshape from feature vector NxC, to NxC
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear_out(x)

        return x.view(-1, 3, self.output_point_number)


class ConvToPC(nn.Module):
    def __init__(self, in_channels, opt):
        super(ConvToPC, self).__init__()
        self.in_channels = in_channels
        self.opt = opt

        self.conv1 = MyConv2d(self.in_channels, int(self.in_channels), kernel_size=1, stride=1, padding=0, bias=True, activation=opt.activation, normalization=opt.normalization)
        self.conv2 = MyConv2d(int(self.in_channels), 3, kernel_size=1, stride=1, padding=0, bias=True, activation=None, normalization=None)

        # special initialization for conv2, to get uniform distribution over the space
        # self.conv2.conv.bias.data.normal_(0, 0.3)
        self.conv2.conv.bias.data.uniform_(-1, 1)

        # self.conv2.conv.weight.data.normal_(0, 0.01)
        # self.conv2.conv.bias.data.uniform_(-3, 3)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class DecoderConv(nn.Module):
    def __init__(self, opt):
        super(DecoderConv, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_point_num = opt.output_conv_pc_num

        # __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True, activation=None, normalization=None)
        # 1x1 -> 2x2
        self.deconv1 = UpConv(self.feature_num, int(self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization)
        # 2x2 -> 4x4
        self.deconv2 = UpConv(int(self.feature_num), int(self.feature_num/2), activation=self.opt.activation, normalization=self.opt.normalization)
        # 4x4 -> 8x8
        self.deconv3 = UpConv(int(self.feature_num/2), int(self.feature_num/4), activation=self.opt.activation, normalization=self.opt.normalization)
        # 8x8 -> 16x16
        self.deconv4 = UpConv(int(self.feature_num/4), int(self.feature_num/8), activation=self.opt.activation, normalization=self.opt.normalization)
        self.conv2pc4 = ConvToPC(int(self.feature_num/8), opt)
        # 16x16 -> 32x32
        self.deconv5 = UpConv(int(self.feature_num/8), int(self.feature_num/8), activation=self.opt.activation, normalization=self.opt.normalization)
        self.conv2pc5 = ConvToPC(int(self.feature_num/8), opt)
        # 32x32 -> 64x64
        self.deconv6 = UpConv(int(self.feature_num/8), int(self.feature_num/8), activation=self.opt.activation, normalization=self.opt.normalization)
        self.conv2pc6 = ConvToPC(int(self.feature_num/8), opt)


    def forward(self, x):
        # reshape from feature vector NxC, to NxCx1x1
        x = x.view(-1, self.feature_num, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        self.pc4 = self.conv2pc4(x)
        x = self.deconv5(x)
        self.pc5 = self.conv2pc5(x)
        x = self.deconv6(x)
        self.pc6 = self.conv2pc6(x)

        return self.pc6


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        if self.opt.output_fc_pc_num > 0:
            self.fc_decoder = DecoderLinear(opt)
        self.conv_decoder = DecoderConv(opt)

    def forward(self, x):
        if self.opt.output_fc_pc_num > 0:
            self.linear_pc = self.fc_decoder(x)

        if self.opt.output_conv_pc_num > 0:
            self.conv_pc6 = self.conv_decoder(x).view(-1, 3, 4096)
            self.conv_pc4 = self.conv_decoder.pc4.view(-1, 3, 256)
            self.conv_pc5 = self.conv_decoder.pc5.view(-1, 3, 1024)

        if self.opt.output_fc_pc_num == 0:
            if self.opt.output_conv_pc_num == 1024:
                return self.conv_pc5
        else:
            if self.opt.output_conv_pc_num == 1024:
                return torch.cat([self.linear_pc, self.conv_pc5], 2)
            else:
                return self.linear_pc

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        in_node_channels = 3* self.opt.JOINT_NUM  #
        self.fc_net = nn.Sequential(
            # B*3_Joints
            nn.Linear(in_node_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # B*num_outputs
        )
        if self.opt.output_fc_pc_num > 0:
            self.fc_decoder = DecoderLinear(opt)
        self.conv_decoder = DecoderConv(opt)

    def forward(self, x):
        FC_in = x.view(-1, 3 * self.opt.JOINT_NUM)
        FC_out = self.fc_net(FC_in)
        x0=FC_out

        if self.opt.output_fc_pc_num > 0:
            self.linear_pc = self.fc_decoder(x0)

        if self.opt.output_conv_pc_num > 0:
            self.conv_pc6 = self.conv_decoder(x0).view(-1, 3, 4096)
            self.conv_pc4 = self.conv_decoder.pc4.view(-1, 3, 256)
            self.conv_pc5 = self.conv_decoder.pc5.view(-1, 3, 1024)

        if self.opt.output_fc_pc_num == 0:
            if self.opt.output_conv_pc_num == 1024:
                return self.conv_pc5
        else:
            if self.opt.output_conv_pc_num == 1024:
                return torch.cat([self.linear_pc, self.conv_pc5], 2)
            else:
                return self.linear_pc