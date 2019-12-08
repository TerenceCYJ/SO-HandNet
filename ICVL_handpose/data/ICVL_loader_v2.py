import torch.utils.data as data

import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import matplotlib.pyplot as plt
import h5py
import json
import scipy.io as sio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .augmentation import *

class KNNBuilder:
    def __init__(self, k):
        self.k = k
        self.dimension = 3

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)  # dimension is 3
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        D, I = index.search(query, k)
        return D, I

    def self_build_search(self, x):
        '''

        :param x: numpy array of Nxd
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        x = np.ascontiguousarray(x, dtype=np.float32)
        index = self.build_nn_index(x)
        D, I = self.search_nn(index, x, self.k)
        return D, I


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts

class ICVL_Loader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(ICVL_Loader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode
        self.aug = opt.augment

        self.input_pc_num=opt.input_pc_num
        self.INPUT_FEATURE_NUM=opt.INPUT_FEATURE_NUM
        self.node_num = opt.node_num
        self.rows = round(math.sqrt(self.node_num))
        self.cols = self.rows
        self.size = opt.size
        #self.test_index = opt.test_index
        self.test_index_num=2

        self.JOINT_NUM = opt.JOINT_NUM
        #self.PCA_SZ = opt.PCA_SZ


        # load the folder-category txt
        self.subject_names = ["Test1", "Test2", "201403121135", "201403121140", "201405151126", "201406030937",
                         "201406031456", "201406031503", "201406031747", "201406041509", "201406181554", "201406181600",
                         "201406191014", "201406191044"]

        #self.subject_names = ["Test", "1", "2", "3", "4", "5", "6", "7", "8"]
        #self.gesture_names = ["1", "2", "3"]#kinect_name

        #from HandNet dataset.py
        if self.size == 'full':
            self.SUBJECT_NUM = 14
            #self.GESTURE_NUM = 3
        elif self.size == 'small':
            self.SUBJECT_NUM = 3
            #self.GESTURE_NUM = 3
        self.total_frame_num = self.__total_frmae_num()

        self.point_clouds = np.empty(shape=[self.total_frame_num, self.input_pc_num, self.INPUT_FEATURE_NUM], dtype=np.float32)
        self.volume_length = np.empty(shape=[self.total_frame_num, 1], dtype=np.float32)
        self.gt_xyz0 = np.empty(shape=[self.total_frame_num, opt.JOINT_NUM, 3], dtype=np.float32)
        self.valid = np.empty(shape=[self.total_frame_num, 1], dtype=np.float32)

        self.start_index = 0
        self.end_index = 0
        dataset_list_train=[]
        dataset_list_test=[]

        ### get data list
        if self.mode == 'train':  # train
            trainlist_dir=opt.dataroot+opt.trainlist
            with open(trainlist_dir,"r") as f:
                train_lists=f.read().splitlines()
            self.dataset=train_lists
            '''
            for i_subject in range(self.SUBJECT_NUM):
                if i_subject >= self.test_index_num:

                    cur_data_dir = os.path.join(self.root, self.subject_names[i_subject])
                    print("Training: " + cur_data_dir)
                    #self.__loaddata(cur_data_dir)
                    #add /pc/.npz to dataset_list_train
                    lists_cur=os.listdir(os.path.join(cur_data_dir,'pc'))
                    lists_cur.sort(key=lambda x: int(x[:-4]))
                    for list_cur in lists_cur:
                        list_path=os.path.join(self.subject_names[i_subject],'pc',list_cur)
                        dataset_list_train.append(list_path)
            self.dataset=dataset_list_train
            '''
        else:  # test
            testlist_dir = opt.dataroot + opt.testlist
            with open(testlist_dir, "r") as f:
                test_lists = f.read().splitlines()
            self.dataset = test_lists
            '''
            for i_subject in range(self.SUBJECT_NUM):
                if i_subject < self.test_index_num:

                    cur_data_dir = os.path.join(self.root, self.subject_names[i_subject])
                    print("Testing: " + cur_data_dir)
                    #self.__loaddata(cur_data_dir)
                    # add /pc/.npz to dataset_list_test
                    lists_cur = os.listdir(os.path.join(cur_data_dir, 'pc'))
                    lists_cur.sort(key=lambda x: int(x[:-4]))
                    for list_cur in lists_cur:
                        list_path = os.path.join(self.subject_names[i_subject], 'pc',list_cur)
                        dataset_list_test.append(list_path)
            self.dataset = dataset_list_test
            '''

        # ensure there is no batch-1 batch
        if len(self.dataset) % self.opt.batch_size == 1:
            self.dataset.pop()

        # kNN search on SOM nodes
        self.knn_builder = KNNBuilder(self.opt.som_k)

        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    def __len__(self):
        return len(self.volume_length)

    def __getitem__(self, index):

        ### load data part2
        #print(index)
        if index > len(self.dataset):
            print(index)
            index=index-1
        file = self.dataset[index]
        data = np.load(os.path.join(self.root, file))
        pc_np = data['pc']
        sn_np = data['sn']
        som_node_np = data['som_node']
        gt_xyz_np = data['gt_xyz']
        volume_length_np = data['volume_length']
        volume_offset_np = data['volume_offset']
        volume_rotate_np = data['volume_rotate']
        #valid_np = data['valid']
        #label = self.folders.index(file[0:8])
        #assert (label >= 0)
        ##pc sn som-nodes

        if self.opt.input_pc_num < pc_np.shape[0]:
            chosen_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
            sn_np = pc_np[chosen_idx, :]
            sn_np = sn_np[chosen_idx, :]
        else:
            chosen_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num-pc_np.shape[0], replace=True)
            pc_np_redundent = pc_np[chosen_idx, :]
            sn_np_redundent = sn_np[chosen_idx, :]
            pc_np = np.concatenate((pc_np, pc_np_redundent), axis=0)
            sn_np = np.concatenate((sn_np, sn_np_redundent), axis=0)

        # augmentation
        if self.mode == 'train':
            if self.aug == 'yes':
                # random jittering
                pc_np = jitter_point_cloud(pc_np)
                sn_np = jitter_point_cloud(sn_np)
                som_node_np = jitter_point_cloud(som_node_np, sigma=0.04, clip=0.1)
                # random scale
                scale = np.random.uniform(low=0.8, high=1.2)
                pc_np = pc_np * scale
                sn_np = sn_np * scale
                som_node_np = som_node_np * scale
                gt_xyz_np = gt_xyz_np * scale
                volume_length_np = volume_length_np * scale
                volume_offset_np = volume_offset_np * scale

        # convert to tensor
        pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 3xN

        # som
        som_node = torch.from_numpy(som_node_np.transpose().astype(np.float32))  # 3xnode_num
        # kNN search: som -> som
        if self.opt.som_k >= 2:
            D, I = self.knn_builder.self_build_search(som_node_np)
            som_knn_I = torch.from_numpy(I.astype(np.int64))  # node_num x som_k
        else:
            som_knn_I = torch.from_numpy(np.arange(start=0, stop=self.opt.node_num, dtype=np.int64).reshape(
                (self.opt.node_num, 1)))  # node_num x 1


        volume_length = torch.from_numpy(volume_length_np.transpose().astype(np.float32))#[1]
        volume_offset = torch.from_numpy(volume_offset_np.transpose().astype(np.float32))#[3]
        volume_rotate = torch.from_numpy(volume_rotate_np.transpose().astype(np.float32))#[3,3]
        #valid = torch.from_numpy(valid_np.transpose().astype(np.float32))#[1]
        gt_xyz = torch.from_numpy(gt_xyz_np.transpose().astype(np.float32))  # 3*16


        '''
        # load PCA coeff
        pca_file_path=os.path.join(self.root,file[0:2])
        PCA_coeff_mat = sio.loadmat(os.path.join(pca_file_path, 'PCA_coeff.mat'))
        PCA_coeff = torch.from_numpy(PCA_coeff_mat['PCA_coeff'][:, 0:self.PCA_SZ].astype(np.float32))#[63,42]
        PCA_mean_mat = sio.loadmat(os.path.join(pca_file_path, 'PCA_mean_xyz.mat'))
        PCA_mean = torch.from_numpy(PCA_mean_mat['PCA_mean_xyz'].astype(np.float32))#[1,63]
        #????
        tmp = torch.reshape(gt_xyz,(1,-1))
        tmp_demean = tmp - PCA_mean
        gt_pca = torch.mm(tmp_demean, PCA_coeff)#1*42
        PCA_coeff = PCA_coeff.transpose(0, 1)#[42,63]
        '''

        return pc, sn, som_node, som_knn_I, gt_xyz, volume_length, volume_offset, volume_rotate

    def __total_frmae_num(self):
        frame_num = 0
        if self.mode == 'train':  # train
            for i_subject in range(self.SUBJECT_NUM):
                if i_subject >= self.test_index_num:
                    cur_data_dir = os.path.join(self.root, self.subject_names[i_subject])
                    frame_num = frame_num + self.__get_frmae_num(cur_data_dir,self.subject_names[i_subject])
        else:  # test
            for i_subject in range(self.SUBJECT_NUM):
                if i_subject < self.test_index_num:
                    cur_data_dir = os.path.join(self.root, self.subject_names[i_subject])
                    frame_num = frame_num + self.__get_frmae_num(cur_data_dir,self.subject_names[i_subject])
        return frame_num

    def __get_frmae_num(self, data_dir,i_gesture):
        volume_length = sio.loadmat(os.path.join(data_dir, (i_gesture+"_Volume_length.mat")))
        return len(volume_length['Volume_length'])

    '''
    def __loaddata(self, data_dir):
        point_cloud = sio.loadmat(os.path.join(data_dir, 'Point_Cloud_FPS.mat'))
        gt_data = sio.loadmat(os.path.join(data_dir, "Volume_GT_XYZ.mat"))
        volume_length = sio.loadmat(os.path.join(data_dir, "Volume_length.mat"))
        valid = sio.loadmat(os.path.join(data_dir, "valid.mat"))


        self.start_index = self.end_index + 1
        self.end_index = self.end_index + len(point_cloud['Point_Cloud_FPS'])

        self.gt_xyz[(self.start_index - 1):self.end_index, :, :] = gt_data['Volume_GT_XYZ'].astype(np.float32)
        self.point_clouds[(self.start_index - 1):self.end_index, :, :] = point_cloud['Point_Cloud_FPS'].astype(np.float32)
        #self.gt_xyz[(self.start_index - 1):self.end_index, :, :] = gt_data['Volume_GT_XYZ'].astype(np.float32)
        self.volume_length[(self.start_index - 1):self.end_index, :] = volume_length['Volume_length'].astype(np.float32)
        self.valid[(self.start_index - 1):self.end_index, :] = valid['valid'].astype(np.float32)
    '''