import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from collections import OrderedDict
import os
import sys
import random
import os.path
import json

from . import networks_debug
from . import losses

class Model():
    def __init__(self, opt):
        self.opt = opt
        self.weight=self.opt.weight

        self.encoder = networks_debug.Encoder(opt)

        self.es_version = opt.es_version
        if self.es_version == 'v2.1':
            self.estimater = networks_debug.PoseEstimater1(opt)
        elif self.es_version == 'v3':
            self.estimater = networks_debug.PoseEstimater(opt)
        elif self.es_version == 'v4':
            self.estimater = networks_debug.PoseEstimater2(opt)
        elif self.es_version == 'v5':
            self.estimater = networks_debug.PoseEstimater3(opt)
        elif self.es_version == 'v6':
            self.estimater = networks_debug.PoseEstimater4(opt)


        self.chamfer_criteria = losses.ChamferLoss(opt)

        self.es_loss_version = opt.es_loss_version
        if self.es_loss_version == 'loss':
            self.LossEstimater = losses.LossEstimation()
        elif self.es_loss_version == 'msra':
            self.LossEstimater = losses.LossEstimation_msra_sf()
        elif self.es_loss_version == 'nyu':
            self.LossEstimater = losses.LossEstimation_nyu_sf()
        elif self.es_loss_version == 'icvl':
            self.LossEstimater = losses.LossEstimation_icvl_sf()

        if self.opt.gpu_id >= 0:
            self.encoder = self.encoder.to(self.opt.device)
            self.estimater = self.estimater.to(self.opt.device)
            self.LossEstimater=self.LossEstimater.to(self.opt.device)

        # learning rate_control
        self.old_lr_encoder = self.opt.lr_encoder
        self.old_lr_estimater = self.opt.lr_estimater

        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(),
                                                  lr=self.old_lr_encoder,
                                                  betas=(0.9, 0.999),
                                                  weight_decay=0.0005)
        self.optimizer_estimater = torch.optim.Adam(self.estimater.parameters(),
                                                    lr=self.old_lr_estimater,
                                                    betas=(0.9, 0.999),
                                                    weight_decay=0.0005)

        # place holder for GPU tensors
        self.input_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.input_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        #self.input_label = torch.LongTensor(self.opt.batch_size, self.JOINT_NUM*3).fill_(1)
        #self.input_seg= torch.LongTensor(self.opt.batch_size, 50).fill_(1)#
        self.input_joins=torch.FloatTensor(self.opt.batch_size, 1, self.opt.JOINT_NUM*3)

        self.input_node = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)
        self.input_node_knn_I = torch.LongTensor(self.opt.batch_size, self.opt.node_num, self.opt.som_k)

        #self.feature = torch.Tensor(self.opt.batch_size,self.opt.output_conv_pc_num)
        #self.predicted_pc = torch.Tensor(self.opt.batch_size,3,1280)
        #self.score_estimater = torch.Tensor(self.opt.batch_size, self.opt.PCA_SZ)
        self.loss_all = 0.0
        self.loss_estimater = 0.0


        # record the test loss and accuracy
        self.test_loss_estimater = torch.FloatTensor([0])
        self.test_accuracy_estimater = torch.FloatTensor([0])
        self.test_iou = torch.FloatTensor([0])

        #

        if self.opt.gpu_id >= 0:
            self.input_pc = self.input_pc.to(self.opt.device)
            self.input_sn = self.input_sn.to(self.opt.device)
            #self.input_label = self.input_label.to(self.opt.device)
            self.input_joins = self.input_joins.to(self.opt.device)
            self.input_node = self.input_node.to(self.opt.device)
            self.input_node_knn_I = self.input_node_knn_I.to(self.opt.device)
            self.test_loss_estimater = self.test_loss_estimater.to(self.opt.device)
            self.test_accuracy_estimater = self.test_accuracy_estimater.to(self.opt.device)

    def set_input(self, input_pc, input_sn, gt_pca, input_node, input_node_knn_I):
        self.input_pc.resize_(input_pc.size()).copy_(input_pc)
        self.input_sn.resize_(input_sn.size()).copy_(input_sn)
        self.input_joins.resize_(gt_pca.size()).copy_(gt_pca)
        self.input_node.resize_(input_node.size()).copy_(input_node)
        self.input_node_knn_I.resize_(input_node_knn_I.size()).copy_(input_node_knn_I)
        self.pc = self.input_pc.detach()
        self.sn = self.input_sn.detach()
        self.joins = self.input_joins.detach()

    def forward(self, is_train=False, epoch=None):
        # ------------------------------------------------------------------
        self.feature = self.encoder(self.pc, self.sn, self.input_node, self.input_node_knn_I, is_train, epoch)  #                                                        #

        batch_size = self.feature.size()[0]
        feature_num = self.feature.size()[1]
        N = self.pc.size()[2]

        # ------------------------------------------------------------------
        k = self.opt.k
        self.feature_max_first_pn_out = torch.FloatTensor(self.opt.batch_size, 384, k * N)
        self.feature_max_knn_feature_1 = torch.FloatTensor(self.opt.batch_size, 512, k * N)
        self.feature_max_final_pn_out = torch.FloatTensor(self.opt.batch_size, feature_num, k * N)

        # BxkNxnode_num -> BxkN, tensor
        _, mask_max_idx = torch.max(self.encoder.mask, dim=2, keepdim=False)  # BxkN
        mask_max_idx = mask_max_idx.unsqueeze(1)  # Bx1xkN
        mask_max_idx_384 = mask_max_idx.expand(batch_size, 384, k*N).detach()
        mask_max_idx_512 = mask_max_idx.expand(batch_size, 512, k*N).detach()
        mask_max_idx_fn = mask_max_idx.expand(batch_size, feature_num, k * N).detach()

        self.feature_max_first_pn_out = torch.gather(self.encoder.first_pn_out_masked_max , dim=2, index=mask_max_idx_384)  # Bx384xnode_num -> Bx384xkN
        self.feature_max_knn_feature_1 = torch.gather(self.encoder.knn_feature_1, dim=2, index=mask_max_idx_512)  # Bx512xnode_num -> Bx512xkN
        self.feature_max_final_pn_out = torch.gather(self.encoder.final_pn_out, dim=2, index=mask_max_idx_fn)  # Bx1024xnode_num -> Bx1024xkN

        self.score_estimater = self.estimater(self.encoder.x_decentered,
                                              self.pc,
                                              self.encoder.centers,
                                              self.sn,
                                              self.encoder.first_pn_out,
                                              self.encoder.first_pn_out_masked_max,
                                              self.feature_max_first_pn_out,
                                              self.feature_max_knn_feature_1,
                                              self.feature_max_final_pn_out,
                                              self.feature)          #

    def encoder_estimater_optimize(self, epoch=None):
        self.encoder.train()
        self.estimater.train()
        self.forward(is_train=True, epoch=epoch)
        self.encoder.zero_grad()
        self.estimater.zero_grad()
        self.joins=self.joins.resize_(self.score_estimater.size())
        #self.loss_estimater = self.LossEstimater(self.score_estimater, self.joins)*self.opt.PCA_SZ
        self.loss_estimater = self.LossEstimater(self.score_estimater, self.joins)
        self.loss_estimater.backward()
        self.optimizer_encoder.step()
        self.optimizer_estimater.step()

    def estimater_optimize(self, epoch=None):
        self.estimater.train()
        self.forward(is_train=True, epoch=epoch)
        self.estimater.zero_grad()
        self.joins=self.joins.resize_(self.score_estimater.size())
        #self.loss_estimater = self.LossEstimater(self.score_estimater, self.joins)*self.opt.PCA_SZ
        self.loss_estimater = self.LossEstimater(self.score_estimater, self.joins)
        self.loss_estimater.backward()
        self.optimizer_estimater.step()

    def test_encoder_estimater_model(self):
        self.encoder.eval()
        self.estimater.eval()
        self.forward(is_train=False)  #
        self.joins = self.joins.resize_(self.score_estimater.size())#20190209
        #self.loss_estimater = self.LossEstimater(self.score_estimater, self.joins)*self.opt.PCA_SZ
        self.loss_estimater = self.LossEstimater(self.score_estimater, self.joins)

    def test_estimater_model(self):
        self.estimater.eval()
        self.forward(is_train=False)  #
        #self.loss_estimater = self.LossEstimater(self.score_estimater, self.joins)*self.opt.PCA_SZ
        self.loss_estimater = self.LossEstimater(self.score_estimater, self.joins)

        # visualization with visdom
    def get_current_visuals(self):
        # display only one instance of pc/img
        input_pc_np = self.input_pc[0].cpu().numpy().transpose() # Nx3
        pc_color_np = np.zeros(input_pc_np.shape)  # Nx3
        gt_pc_color_np = np.zeros(input_pc_np.shape)  # Nx3

        # construct color map
        _, predicted_seg = torch.max(self.score_estimater.data[0], dim=0, keepdim=False)  # 50xN -> N
        predicted_seg_np = predicted_seg.cpu().numpy()  # N
        gt_seg_np = self.seg.data[0].cpu().numpy()  # N

        color_map_file = os.path.join(self.opt.dataroot, 'part_color_mapping.json')
        color_map = json.load(open(color_map_file, 'r'))
        color_map_np = np.rint((np.asarray(color_map)*255).astype(np.int32))  # 50x3

        for i in range(input_pc_np.shape[0]):
            pc_color_np[i] = color_map_np[predicted_seg_np[i]]
            gt_pc_color_np[i] = color_map_np[gt_seg_np[i]]

        return OrderedDict([('pc_colored_predicted', [input_pc_np, pc_color_np]),
                            ('pc_colored_gt',        [input_pc_np, gt_pc_color_np])])

    def get_current_errors(self):
        # self.score_estimater: Bx42
        correct_mask = torch.eq(self.score_estimater.data, self.input_joins).float()
        train_accuracy_estimater = torch.mean(correct_mask)

        #joint_loss= self.loss_estimater* self.opt.PCA_SZ

        return OrderedDict([
            ('train_loss_estimate', self.loss_estimater.item()),
            ('train_accuracy_estimate', train_accuracy_estimater),
            ('test_loss_estimate', self.test_loss_estimater.item()),
            ('test_acc_estimate', self.test_accuracy_estimater.item()),
            ('test_iou', self.test_iou.item())
        ])

    def get_estimation(self):
        return self.score_estimater

    def get_loss(self):
        return self.loss_all
    def get_es_loss(self):
        return self.loss_estimater

    def save_network(self, network, network_label, epoch_label, gpu_id):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu_id >= 0 and torch.cuda.is_available():
            # torch.cuda.device(gpu_id)
            network.to(self.opt.device)

    def update_learning_rate(self, ratio):
        # encoder
        lr_encoder = self.old_lr_encoder * ratio
        for param_group in self.optimizer_encoder.param_groups:
            param_group['lr'] = lr_encoder
        print('update encoder learning rate: %.8f -> %.8f' % (self.old_lr_encoder, lr_encoder))
        self.old_lr_encoder = lr_encoder

        # estimater
        lr_estimater = self.old_lr_estimater * ratio
        for param_group in self.optimizer_estimater.param_groups:
            param_group['lr'] = lr_estimater
        print('update estimater learning rate: %.8f -> %.8f' % (self.old_lr_estimater, lr_estimater))
        self.old_lr_estimater = lr_estimater

    def get_en_lr(self):
        return self.old_lr_encoder
    def get_es_lr(self):
        return self.old_lr_estimater
