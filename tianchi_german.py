# coding: utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import numpy as np
from mxnet import nd
from mxnet import autograd
import pandas as pd
from tqdm import tqdm
import os
import multiprocessing
from utils import TrainerWithDifferentLR, ConfusionMatrix, RandomRotate, get_logger, SWA, lr_find, RandomTranspose
from gluoncv import utils as gutils
import cv2
import gc
import queue
import glob
import matplotlib.pyplot as plt
import matplotlib
import yaml
import h5py
import gc
import bisect
from mxboard import SummaryWriter
matplotlib.use('Agg')

global_config = yaml.load(open('config.yml'))


class ToNCHW(nn.Block):
    def __init__(self):
        super(ToNCHW, self).__init__()

    def forward(self, x):
        return x.transpose((2, 0, 1))


transforms_train = transforms.Compose([
    RandomTranspose(False, True, True),
    # transforms.RandomFlipLeftRight(), # channel == 1 or 3
    ToNCHW(),
    # transforms.ToTensor(),
])

transforms_val = transforms.Compose([
    # transforms.CenterCrop(224),
    # transforms.Resize(256),
    ToNCHW(),
    # transforms.ToTensor(),
])

transforms_test = transforms.Compose([
    # RandomTranspose(False, True, True),
    ToNCHW(),
    # transforms.ToTensor(),
])


class NLBlock(nn.Block):
    def __init__(self, channels, soft=False):
        super(NLBlock, self).__init__()
        self.soft = soft
        self.query = nn.Conv2D(channels // 2, 1, 1, use_bias=False, weight_initializer='MSRAPrelu')
        self.key = nn.Conv2D(channels // 2, 1, 1, use_bias=False, weight_initializer='MSRAPrelu')
        self.value = nn.Conv2D(channels // 2, 1, 1, use_bias=False, weight_initializer='MSRAPrelu')
        self.up = nn.Conv2D(channels, 1, 1, use_bias=False, weight_initializer='MSRAPrelu')

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        b, c, h, w = q.shape

        q = q.reshape((b, c, -1))
        k = k.reshape((b, c, -1))
        v = v.reshape((b, c, -1))
        att = nd.batch_dot(q, k, transpose_a=True) / (h * w)
        if self.soft:
            att = nd.softmax(att, axis=-1)

        nl = nd.batch_dot(att, v, transpose_b=True)
        nl = nd.reshape(nl.transpose((0, 2, 1)), shape=(b, c, h, w))
        return self.up(nl) + x


class ChannelNLBlock(nn.Block):
    def __init__(self):
        super(ChannelNLBlock, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        a = x.reshape((b, c, -1))

        att = nd.batch_dot(a, a, transpose_b=True)
        att = nd.softmax(att, axis=-1)

        nl = nd.batch_dot(att, a)
        nl = nd.reshape(nl, shape=(b, c, h, w))
        return nl + x


class ResBlock(nn.Block):
    def __init__(self, channels, stride=1, side_conv=False, split=1, use_se=True):
        super(ResBlock, self).__init__()
        self.side_conv = side_conv
        with self.name_scope():
            self.fwd = nn.HybridSequential('res_fwd')
            with self.fwd.name_scope():
                self.fwd.add(
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.Conv2D(channels[0], 3, 1, 1, use_bias=False),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.Conv2D(channels[0], 3, stride, 1, groups=split, use_bias=False),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.Conv2D(channels[1], 3, 1, 1, use_bias=False),
                    )
                if use_se:
                    self.fwd.add(SEBlock(channels[1]))
                    self.fwd.add(ConvSEBlock())

            if side_conv:
                self.sc = nn.Conv2D(channels[1], 1, stride, 0, use_bias=False)

    def forward(self, x):
        f = self.fwd(x)
        if self.side_conv:
            x = self.sc(x)
        out = x + f

        return out


class DenseBlock(nn.Block):
    def __init__(self, growth_rate, dense_num=1, bn_rate=4, transition_feat_num=0,
                 in_channel=0, use_se=True):
        super(DenseBlock, self).__init__()
        self.transition_feat_num = transition_feat_num  # half channel nums of self.fwd' output in densenet
        self.fwd = nn.Sequential()
        for i in range(dense_num):
            fw = nn.HybridSequential()
            fw.add(nn.BatchNorm())
            fw.add(nn.Activation('relu'))
            fw.add(nn.Conv2D(growth_rate*bn_rate, 1, 1, use_bias=False))
            fw.add(nn.BatchNorm())
            # se 的 position
            fw.add(nn.Activation('relu'))
            fw.add(nn.Conv2D(growth_rate, kernel_size=3, padding=1, use_bias=False))
            if use_se:
                fw.add(SEBlock(growth_rate))
                fw.add(ConvSEBlock())
            cat = gluon.contrib.nn.HybridConcurrent(axis=1)
            cat.add(gluon.contrib.nn.Identity(), fw)
            self.fwd.add(cat)
            if (dense_num-2 == i) and in_channel:
                self.fwd.add(NLBlock(in_channel+growth_rate*(dense_num-1)))

        if transition_feat_num:
            self.out = nn.HybridSequential(prefix='')
            self.out.add(nn.BatchNorm())
            self.out.add(nn.Activation('relu'))
            self.out.add(nn.Conv2D(transition_feat_num, kernel_size=1, use_bias=False))
            self.out.add(nn.AvgPool2D(2, 2))

    def forward(self, x):
        out = self.fwd(x)
        if self.transition_feat_num:
            out = self.out(out)
        return out


class SEBlock(nn.HybridBlock):
    def __init__(self, in_channel, reduction=4):
        super(SEBlock, self).__init__()
        self.fwd = nn.HybridSequential()
        with self.fwd.name_scope():
            self.fwd.add(nn.GlobalAvgPool2D())
            self.fwd.add(nn.Conv2D(in_channel // reduction, 1, activation='relu', use_bias=False))
            self.fwd.add(nn.Conv2D(in_channel, 1, activation='sigmoid', use_bias=False))

    def hybrid_forward(self, F, x, *args, **kwargs):
        multiplier = self.fwd(x)
        return F.broadcast_mul(x, multiplier, )


class ConvSEBlock(nn.HybridBlock):
    def __init__(self, k_size=5):
        super(ConvSEBlock, self).__init__()
        self.conv = nn.Conv2D(1, k_size, 1, k_size//2, activation='sigmoid', use_bias=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        ap = F.mean(x, axis=1, keepdims=True)
        mp = F.max(x, axis=1, keepdims=True)
        att = F.concat(ap, mp, dim=1)
        multiplier = self.conv(att)
        return F.broadcast_mul(x, multiplier, )


class Sentinel1(nn.Block):
    def __init__(self, cls_num=17):
        super(Sentinel1, self).__init__()
        in_channel = 10
        with self.name_scope():
            self.head = nn.HybridSequential('head1')
            with self.head.name_scope():
                self.head.add(
                    nn.BatchNorm(),
                    nn.Conv2D(3*in_channel, 7, 1, 3, groups=in_channel, use_bias=True),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.Conv2D(6*in_channel, 3, 1, 1, groups=in_channel, use_bias=True),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.Conv2D(64, 3, 1, 1),)

            self.b1 = nn.Sequential('block1')
            with self.b1.name_scope():
                self.b1.add(
                    ResBlock([32, 64], 1, False, split=16),
                    ResBlock([32, 64], 1, False, split=16),
                    NLBlock(64),
                    ResBlock([32, 64], 1, False, split=16),
                )

            self.b2 = nn.Sequential('block2')
            with self.b2.name_scope():
                self.b2.add(
                    ResBlock([32, 128], 2, True, split=16),
                    ResBlock([32, 128], 1, False, split=16),
                    ResBlock([32, 128], 1, False, split=16),
                    NLBlock(128),
                    ResBlock([32, 128], 1, False, split=16),
                )

            self.b3 = nn.Sequential('block3')
            with self.b3.name_scope():
                self.b3.add(
                    ResBlock([64, 256], 2, True, split=32),
                    ResBlock([64, 256], 1, False, split=32),
                    ResBlock([64, 256], 1, False, split=32),
                )

            # self.feat = nn.Sequential('feature')
            # with self.feat.name_scope():
            #     self.feat.add(self.head, self.b1, self.b2, self.b3)

            self.gap = nn.HybridSequential()
            with self.gap.name_scope():
                self.gap.add(nn.GlobalAvgPool2D())
                self.gap.add(nn.Flatten())

            self.out = nn.HybridSequential()
            with self.out.name_scope():
                self.out.add(nn.Dropout(0.5))
                self.out.add(nn.Dense(cls_num))

    def forward(self, x):
        # x = self.feat(x)
        x = self.head(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        a = self.gap(x)
        # m = self.flt2(self.gmp(x))
        # c = nd.concat(a, m, dim=1)
        c = a

        out = self.out(c)
        return out


class Sentinel2(nn.Block):
    def __init__(self, cls_num=17):
        super(Sentinel2, self).__init__()
        in_channel = 10
        with self.name_scope():
            self.head = nn.HybridSequential('head')
            with self.head.name_scope():
                self.head.add(
                    nn.BatchNorm(),
                    nn.Conv2D(3*in_channel, 5, 1, 2, groups=in_channel, use_bias=True),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.Conv2D(6*in_channel, 3, 1, 1, groups=in_channel, use_bias=True),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.Conv2D(64, 3, 1, 1),)
                # self.head.add(
                #     nn.BatchNorm(),
                #     nn.Conv2D(30, 5, 1, 2, use_bias=True),
                #     nn.BatchNorm(), nn.Activation('relu'),
                #     nn.Conv2D(60, 3, 1, 1, use_bias=True),
                #     nn.BatchNorm(), nn.Activation('relu'),
                #     nn.Conv2D(64, 3, 1, 1))

            self.b1 = DenseBlock(32, 4, 4, transition_feat_num=128, in_channel=64)

            self.b2 = DenseBlock(32, 4, 4, transition_feat_num=256, in_channel=128)

            self.b3 = DenseBlock(32, 4, 4)

            self.feat_vec = nn.HybridSequential('feat_vec')
            with self.feat_vec.name_scope():
                self.feat_vec.add(
                    nn.GlobalAvgPool2D(),
                    nn.Flatten(),
                )

            self.out = nn.HybridSequential()
            with self.out.name_scope():
                self.out.add(nn.Dropout(0.5), nn.Dense(cls_num))

    def forward(self, x):
        x = self.head(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        feat = self.feat_vec(x)
        out = self.out(feat)
        return out


class Sentinel2Bigger(nn.Block):
    def __init__(self, cls_num=17):
        super(Sentinel2Bigger, self).__init__()
        in_channel = 10
        with self.name_scope():
            self.head = nn.HybridSequential('head')
            with self.head.name_scope():
                self.head.add(
                    nn.BatchNorm(),
                    nn.Conv2D(24*in_channel, 5, 1, 2, groups=in_channel, use_bias=True),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.Conv2D(48*in_channel, 3, 1, 1, groups=in_channel*8, use_bias=True),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.Conv2D(256, 1))

            self.b1 = DenseBlock(32, 8, 4, transition_feat_num=256, in_channel=256)

            self.b2 = DenseBlock(32, 24, 4, transition_feat_num=512, in_channel=256)

            self.b3 = DenseBlock(32, 16, 4)

            self.feat_vec = nn.HybridSequential('feat_vec')
            with self.feat_vec.name_scope():
                self.feat_vec.add(
                    nn.GlobalAvgPool2D(),
                    nn.Flatten(),
                )

            self.out = nn.HybridSequential()
            with self.out.name_scope():
                self.out.add(nn.Dropout(0.5), nn.Dense(cls_num))

    def forward(self, x):
        x = self.head(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        feat = self.feat_vec(x)
        out = self.out(feat)
        return out


class Sentinel2Big(nn.Block):
    def __init__(self, cls_num=17):
        super(Sentinel2Big, self).__init__()
        in_channel = 10
        head_num = 8
        with self.name_scope():
            self.head = nn.HybridSequential('head')
            with self.head.name_scope():
                self.head.add(
                    nn.BatchNorm(),
                    nn.Conv2D(24 * in_channel, 5, 1, 2, groups=in_channel, use_bias=True),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.Conv2D(48 * in_channel, 3, 1, 1, groups=in_channel * head_num, use_bias=True),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.Conv2D(256, 1))

            self.b1 = DenseBlock(32, 8, 4, transition_feat_num=256, in_channel=256)

            self.b2 = DenseBlock(32, 8, 4, transition_feat_num=256, in_channel=256)

            self.b3 = DenseBlock(32, 8, 4)

            self.feat_vec = nn.HybridSequential('feat_vec')
            with self.feat_vec.name_scope():
                self.feat_vec.add(
                    nn.GlobalAvgPool2D(),
                    nn.Flatten(),
                )

            self.out = nn.HybridSequential()
            with self.out.name_scope():
                self.out.add(nn.Dropout(0.5), nn.Dense(cls_num))

    def forward(self, x):
        x = self.head(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        feat = self.feat_vec(x)
        out = self.out(feat)
        return out


class Sentinel2MultiHead(nn.Block):
    def __init__(self, cls_num=17):
        super(Sentinel2MultiHead, self).__init__()
        in_channel = 10
        head_num = 8
        with self.name_scope():
            self.head = gluon.contrib.nn.HybridConcurrent(axis=1)
            with self.head.name_scope():
                for _ in range(head_num):
                    h = nn.HybridSequential()
                    h.add(
                        nn.BatchNorm(),
                        nn.Conv2D(3 * in_channel, 5, 1, 2, groups=in_channel, use_bias=True),
                        nn.BatchNorm(), nn.Activation('relu'),
                        nn.Conv2D(6 * in_channel, 3, 1, 1, groups=in_channel, use_bias=True),
                        nn.BatchNorm(), nn.Activation('relu'),
                        nn.Conv2D(32, 3, 1, 1),)
                    self.head.add(h)

            self.b1 = DenseBlock(32, 8, 4, transition_feat_num=256, in_channel=256)

            self.b2 = DenseBlock(32, 8, 4, transition_feat_num=256, in_channel=256)

            self.b3 = DenseBlock(32, 8, 4)

            self.feat_vec = nn.HybridSequential('feat_vec')
            with self.feat_vec.name_scope():
                self.feat_vec.add(
                    nn.GlobalAvgPool2D(),
                    nn.Flatten(),
                )

            self.out = nn.HybridSequential()
            with self.out.name_scope():
                self.out.add(nn.Dropout(0.5), nn.Dense(cls_num))

    def forward(self, x):
        x = self.head(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        feat = self.feat_vec(x)
        out = self.out(feat)
        return out


class Sentinel3(nn.Block):
    def __init__(self, cls_num=17):
        super(Sentinel3, self).__init__()
        in_channel = 10
        head_num = 16
        head_filter = 16

        with self.name_scope():
            multi_head = gluon.contrib.nn.HybridConcurrent(axis=1)
            for i in range(head_num):
                hd = nn.HybridSequential()
                hd.add(SEBlock(in_channel, 1))
                hd.add(ConvSEBlock())
                hd.add(nn.Conv2D(head_filter, 1, use_bias=False))
                multi_head.add(hd)
            # head
            self.head = nn.HybridSequential()
            self.head.add(nn.BatchNorm())
            self.head.add(multi_head)
            self.head.add(nn.BatchNorm())
            self.head.add(nn.Activation('relu'))
            self.head.add(nn.AvgPool2D(3, 2, 1))
            #
            self.fwd = nn.Sequential()
            self.fwd.add(DenseBlock(32, 8, 4, head_num * head_filter, in_channel=head_num * head_filter))
            self.fwd.add(DenseBlock(32, 8, 4))

            self.out = nn.HybridSequential()
            self.out.add(
                nn.GlobalAvgPool2D(),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Dense(cls_num))

    def forward(self, x):
        x = self.head(x)
        x = self.fwd(x)
        x = self.out(x)
        return x


class DataSet(gluon.data.Dataset):
    def __init__(self, data, label, label_smooth=True):
        self.data = data
        self.label = label
        self.epsilon = 0.1
        self.label_smooth = label_smooth
        print('data num', self.__len__())

    def __getitem__(self, idx):
        lb = self.label[idx]
        if self.label_smooth:
            cls_id = nd.argmax(lb, axis=0)
            lb[:] = self.epsilon / lb.shape[0]
            lb[cls_id] = 1 - self.epsilon
        return self.data[idx], lb

    def __len__(self):
        return self.data.shape[0]


class MixUpDataSet(gluon.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)
        print('mix data num', self.__len__())

    def __getitem__(self, idx):
        aux_idx = np.random.randint(0, self.length)
        alpha = np.random.beta(0.2, 0.2)
        d1, b1 = self.dataset[idx]
        d2, b2 = self.dataset[aux_idx]
        d = alpha*d1 + (1-alpha)*d2
        b = alpha*b1 + (1-alpha)*b2

        return d, b

    def __len__(self):
        return len(self.dataset)


class ConcatDataset(gluon.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_len = [len(ds) for ds in self.datasets]
        self.cumlen = np.cumsum(self.dataset_len).tolist()
        print('data num', self.__len__())

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumlen, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumlen[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.cumlen[-1]


class TestSet(gluon.data.Dataset):
    def __init__(self, test_file):
        self.test_file = h5py.File(global_config['data_root']+test_file)
        # self.data = np.concatenate((np.array(self.test_file['sen1']),
        #                             np.array(self.test_file['sen2'])), axis=-1).astype(np.float32)
        self.data = np.array(self.test_file['sen2'])
        self.data = nd.array(self.data, dtype=np.float32)
        print('data num', self.__len__())
        self.test_file.close()
        gc.collect()

    def __getitem__(self, idx):

        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def data_loader(batch_size=32, val=False, trans=True, ls=True, mix=True):
    v = False
    if v:
        ctx = mx.cpu()
        val_h5file = h5py.File(global_config['data_root'] + 'validation.h5')
        # vs1 = np.array(val_h5file['sen1']).astype(np.float32)
        vs2 = np.array(val_h5file['sen2']).astype(np.float32)
        vlb = np.array(val_h5file['label']).astype(np.float32)
        # vd = np.concatenate((vs1, vs2), axis=-1)
        vd = vs2
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(vd, vlb, test_size=0.2, random_state=42)
        x_train, x_test, y_train, y_test = nd.array(x_train, ctx), nd.array(x_test, ctx), nd.array(y_train, ctx), nd.array(y_test, ctx)
        val_h5file.close()
        # del vs1
        del vs2
        print(x_train.shape)

        # x_train = np.load(global_config['data_root'] + 'x_0.npy')[:, :, :, 8:]
        # y_train = np.load(global_config['data_root'] + 'y_0.npy')
        # x_test = np.load(global_config['data_root'] + 'tv/vx_2.npy')[:, :, :, 8:]
        # y_test = np.load(global_config['data_root'] + 'tv/vy_2.npy')

        train_set = DataSet(x_train, y_train, label_smooth=ls)
        val_set = DataSet(x_test, y_test, label_smooth=False)

        if trans:
            train_set = train_set.transform_first(transforms_train)
            val_set = val_set.transform_first(transforms_val)
        if mix:
            train_set = MixUpDataSet(train_set)

        # # test file
        # test_a_h5file = h5py.File(global_config['data_root'] + 'round1_test_a_20181109.h5')
        # # ta_s1 = np.array(val_h5file['sen1']).astype(np.float32)
        # ta_s2 = np.array(test_a_h5file['sen2']).astype(np.float32)
        # ta_lb = np.load(global_config['data_root'] + 'test_a_sen2_soft_label.npy')
        # ta_s2, ta_lb = nd.array(ta_s2), nd.array(ta_lb)
        # ta_train_set = DataSet(ta_s2, ta_lb, label_smooth=False)
        # if trans:
        #     ta_train_set = ta_train_set.transform_first(transforms_train)
        #
        # train_set = ConcatDataset([train_set, ta_train_set])

        train_loader = gluon.data.DataLoader(train_set, batch_size, shuffle=True, last_batch='rollover', num_workers=2)
        val_loader = gluon.data.DataLoader(val_set, 32, shuffle=False, last_batch='keep', num_workers=0)
        return train_loader, val_loader

    idx = np.random.randint(0, 14)
    idx = 9 if idx > 9 else idx
    print(f'using data {idx}')

    data_root = global_config['data_root']
    x = np.load(data_root+'x_{}.npy'.format(idx)).astype(np.float32)[:, :, :, 8:]
    y = np.load(data_root+'y_{}.npy'.format(idx)).astype(np.float32)
    x, y = nd.array(x, dtype=np.float32), nd.array(y, dtype=np.float32)

    train_set = DataSet(x, y, label_smooth=ls)
    if mix:
        train_set = MixUpDataSet(train_set)
    if trans:
        train_set = train_set.transform_first(transforms_train)

    train_loader = gluon.data.DataLoader(train_set, batch_size, shuffle=True, last_batch='rollover', num_workers=0)

    if not val:
        return train_loader

    vd = np.load(data_root+'val_x.npy')
    vlb = np.load(data_root+'val_y.npy')
    vd, vlb = nd.array(vd, dtype=np.float32)[:, :, :, 8:], nd.array(vlb, dtype=np.float32)

    val_set = DataSet(vd, vlb, label_smooth=False)  # .transpose((0, 3, 1, 2))
    if trans:
        val_set = val_set.transform_first(transforms_val)

    val_loader = gluon.data.DataLoader(val_set, batch_size, shuffle=False, last_batch='keep', num_workers=0)

    return train_loader, val_loader


def get_x9_set():
    data_root = global_config['data_root']
    x = np.load(data_root + 'x_{}.npy'.format(9)).astype(np.float32)[:, :, :, 8:]
    y = np.load(data_root + 'y_{}.npy'.format(9)).astype(np.float32)
    x, y = nd.array(x, dtype=np.float32), nd.array(y, dtype=np.float32)
    return DataSet(x, y, label_smooth=True).transform_first(transforms_train)


def get_val_loader(batch_size):
    data_root = global_config['data_root']
    vd = np.load(data_root + 'val_x.npy')
    vlb = np.load(data_root + 'val_y.npy')
    vd, vlb = nd.array(vd, dtype=np.float32)[:, :, :, 8:], nd.array(vlb, dtype=np.float32)

    val_set = DataSet(vd, vlb, label_smooth=False)  # .transpose((0, 3, 1, 2))

    val_set = val_set.transform_first(transforms_val)

    val_loader = gluon.data.DataLoader(val_set, batch_size, shuffle=False, last_batch='keep', num_workers=0)
    return val_loader


def get_train_loader(idx, batch_size, vd):
    # idx 0-8
    data_root = global_config['data_root']
    if idx == 11:
        # cof = h5py.File(data_root+'confidence_round1_test_ab.h5')
        cof = h5py.File(data_root+'confidence_aba.h5')
        x = np.array(cof['sen2']).astype(np.float32)
        y = np.array(cof['label']).astype(np.float32)
        cof.close()
    else:
        x = np.load(data_root + 'x_{}.npy'.format(idx)).astype(np.float32)[:, :, :, 8:]
        y = np.load(data_root + 'y_{}.npy'.format(idx)).astype(np.float32)
    x, y = nd.array(x, dtype=np.float32), nd.array(y, dtype=np.float32)
    ds = DataSet(x, y, label_smooth=True).transform_first(transforms_train)
    ds = ConcatDataset([ds, vd])

    ds = MixUpDataSet(ds)

    dl = gluon.data.DataLoader(ds, batch_size, True)

    return dl


def train():
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    ctx = [mx.gpu(0)]
    total_sample_num = 352366
    # total_sample_num = 19295
    # total_sample_num = 24133
    epochs = 200
    batch_size = 32

    # model
    # model_prefix = 'sen2_bs32_10channel_only_val_ls_mix_nl_cos_for_recovery_all_data_bigger1'
    # model_prefix = 'sen2_bs32_10channel_only_val_ls_mix_nl_cos_for_recovery_all_data_multi_head1'
    model_prefix = 'sen2_bs32_10channel_all_data_aba_big1'
    print(model_prefix)
    root_path = global_config['model_root'] + model_prefix + '/'
    os.makedirs(root_path, exist_ok=True)
    # model = Sentinel2(17)
    # model = Sentinel2MultiHead(17)
    model = Sentinel2Bigger(17)
    # model = Sentinel2Big(17)
    model.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
    model.summary(nd.zeros((1, 10, 32, 32), ctx=ctx[0], dtype=np.float32))
    model.hybridize()
    # model.load_parameters(root_path+model_prefix+'_epoch_167_0.9846.params', ctx=ctx)  # '_best.params'

    # data
    x9_set = get_x9_set()
    val_loader = get_val_loader(batch_size)
    # train_loader, val_loader = data_loader(batch_size, True, True, ls=True, mix=True)

    #
    loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    # loss = FocalLoss(num_class=6)
    lr_scheduler = gutils.LRScheduler('cosine', baselr=5e-4, niters=total_sample_num//batch_size, nepochs=epochs,
                                      # step=[i*3*dsn for i in range(1, 10)], step_factor=0.5,
                                      targetlr=5e-6, power=0.9,
                                      warmup_epochs=2, warmup_lr=1e-5, warmup_mode='linear')
    trainer = gluon.Trainer(model.collect_params(), optimizer='adam',
                            optimizer_params={'learning_rate': 0.01, 'wd': 1e-5,
                                              'lr_scheduler': lr_scheduler})
    # lr_find(ctx, val_loader, model, loss, trainer)
    # exit()
    # trainer = TrainerWithDifferentLR(model.collect_params(), [1e-5, 5e-5, 1e-4, 5e-4], [95, 350, 470])
    #
    logger = get_logger(root_path + 'report.txt', mode='a')
    logger.info('total epoch: {}'.format(epochs))
    logger.info(model_prefix)
    sm_writer = SummaryWriter(root_path, filename_suffix='mx_summary', flush_secs=10)
    # metrics 
    train_acc = mx.metric.Accuracy(name='train_acc')
    train_ls = mx.metric.Loss(name='train_ls')
    val_ls = mx.metric.Loss(name='val_ls')
    val_acc = mx.metric.Accuracy(name='val_acc')
    val_top_acc = mx.metric.TopKAccuracy(2, name='val_acc_top')
    # cm
    train_cm = ConfusionMatrix(name='train_cm')
    val_cm = ConfusionMatrix(name='val_cm')

    best_val = 0.0
    for epoch in range(0, epochs):
        print('number of unreachable objects: ', gc.collect())
        # update loaders
        # train_loader = data_loader(batch_size, False) if epoch > 0 else train_loader
        train_loader = get_train_loader(epoch % 12, batch_size, x9_set)
        for idx, (data, label) in tqdm(enumerate(train_loader)):
            lr_scheduler.update(idx, epoch)
            dts = gluon.utils.split_and_load(data, ctx)
            lbs = gluon.utils.split_and_load(label, ctx)
            with autograd.record():
                outs = [model(dts[i]) for i in range(len(ctx))]
                ls = [loss(outs[i], lbs[i]).mean() for i in range(len(ctx))]
            for i in range(len(ctx)):
                ls[i].backward()
            trainer.step(len(ctx))
            # update
            lbs = [nd.argmax(lbs[i], axis=1) for i in range(len(ctx))]
            train_acc.update(lbs, outs)
            train_cm.update(lbs, outs)
            train_ls.update(None, preds=ls)

        for data, label in tqdm(val_loader):
            dts = gluon.utils.split_and_load(data, ctx, even_split=False)
            lbs = gluon.utils.split_and_load(label, ctx, even_split=False)
            outs = [model(dts[i]) for i in range(len(ctx))]
            ls = [loss(outs[i], lbs[i]).mean() for i in range(len(ctx))]

            lbs = [nd.argmax(lbs[i], axis=1) for i in range(len(ctx))]
            val_ls.update(None, preds=ls)
            val_acc.update(lbs, outs)
            val_top_acc.update(lbs, outs)
            val_cm.update(lbs, outs)

        # show criteria print('learning rate: ', trainer.learning_rate)
        sm_writer.add_scalar('lr', trainer.learning_rate, epoch)
        print_str = 'epoch: {} lr: {:.2e}'.format(epoch, trainer.learning_rate)
        for nm, acc in [train_acc.get(), val_acc.get(), val_top_acc.get()]:
            print_str += ', {}: {:.3%}'.format(nm, acc)
            sm_writer.add_scalar(nm, acc, global_step=epoch)
        for ls_nm, ls_val in [train_ls.get(), val_ls.get()]:
            print_str += ', {}: {:.5f}'.format(ls_nm, ls_val)
            sm_writer.add_scalar(ls_nm, ls_val, global_step=epoch)
        logger.info(print_str)

        _, val_acc_tp = val_acc.get()
        if best_val < val_acc_tp:
            logger.info('new best: {}, old best: {}'.format(val_acc_tp, best_val))
            best_val = val_acc_tp
            model.save_parameters(root_path + model_prefix + '_best.params')
        if val_acc_tp >= 0.9:
            model.save_parameters(root_path + model_prefix + '_epoch_{}_{:.4f}.params'.format(epoch+1, val_acc_tp))
        if best_val < val_acc_tp or val_acc_tp >= 0.9:
            train_cm.plot(title='train epoch {}'.format(epoch + 1),
                          save_path=root_path + 'train_' + model_prefix + '_epoch_{}.png'.format(epoch + 1))
            val_cm.plot(title='val epoch {}'.format(epoch + 1),
                        save_path=root_path + 'val_' + model_prefix + '_epoch_{}.png'.format(epoch + 1))

        # model.save_parameters(root_path + model_prefix+'_epoch_{}.params'.format(epoch))

        # reset
        # metrics
        train_acc.reset()
        train_ls.reset()
        val_acc.reset()
        val_top_acc.reset()
        val_ls.reset()
        # cm
        train_cm.reset()
        val_cm.reset()
        # gc.collect()
    logger.info('best val: {}'.format(best_val))
    sm_writer.close()


def single_infer(suffix='best', gen_csv=False, gpu_id=1):
    print(suffix)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    ctx = mx.gpu(gpu_id)
    save_path = global_config['model_root']
    os.makedirs(save_path, exist_ok=True)

    md_prefix = 'sen2_bs32_10channel_only_val_ls_mix_nl_cos_for_recovery_all_data_big2'
    # md_prefix = 'sen2_bs32_10channel_only_val_ls_mix_nl_cos_for_recovery_all_data_multi_head1'
    # model = Sentinel2MultiHead(17)
    model = Sentinel2Big(17)
    weight_root = global_config['model_root'] + '{}/'.format(md_prefix)
    model.load_parameters(weight_root + '{}_{}.params'.format(md_prefix, suffix), ctx)

    test_file = 'round2_test_a_20190121.h5'
    # test_file = 'round1_test_ab.h5'
    test_set = TestSet(test_file).transform_first(transforms_test)
    test_loader = gluon.data.DataLoader(test_set, batch_size=32, shuffle=False, last_batch='keep', num_workers=0)

    prob = test(ctx, model, test_loader)

    if not gen_csv:
        # np.save(save_path + '{}_{}_prob.npy'.format(md_prefix, suffix), prob)
        return prob
    cls = np.zeros_like(prob, np.uint8)
    cls[np.arange(cls.shape[0]), np.argmax(prob, axis=1)] = 1
    df = pd.DataFrame(cls, index=None, columns=None)
    df.to_csv(save_path+'{}_{}.csv'.format(md_prefix, suffix), index=None, header=None)


def multi_infer(gen_csv=True):
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    ctx = mx.gpu(1)
    sv_path = 'data/round2_select_data/'

    #
    sv_nm = 'round2_test_b_bigger2_aba'
    prefix = 'sen2_bs32_10channel_all_data_aba_bigger2'
    print(sv_nm)
    print(prefix)
    # path = 'sen2_bs32_10channel_only_val_ls_mix_nl_cos_for_recovery_all_data_multi_head1'

    # model = Sentinel2MultiHead(17)
    # model = Sentinel2Big(17)
    model = Sentinel2Bigger(17)

    wt_files = sorted(glob.glob(f'models/{prefix}/*epoch*params'))
    wt_files = [wt for wt in wt_files if float(wt[-13:-7]) >= 0.974]  # 0.969 0.9855 0.978 0.986
    # suffixes = [wt[wt.find('epoch'):-7] for wt in weights if float(wt[wt.find('epoch')+6:-14]) >= 143]
    for f in wt_files:
        print(f)
    print('wt_files num: ', len(wt_files))
    # data
    # test_file = 'round2_test_a_20190121.h5'
    test_file = 'round2_test_b_20190211.h5'
    # test_file = 'round1_test_ab.h5'
    test_set = TestSet(test_file).transform_first(transforms_test)
    test_loader = gluon.data.DataLoader(test_set, batch_size=32, shuffle=False, last_batch='keep', num_workers=0)

    rlts = []
    for wt_file in wt_files:
        model.load_parameters(wt_file, ctx)
        rlts.append(test(ctx, model, test_loader))
    prob = 0
    for p in rlts:
        prob += p
    prob /= len(rlts)

    if gen_csv:
        cls = np.zeros_like(prob, np.uint8)
        cls[np.arange(cls.shape[0]), np.argmax(prob, axis=1)] = 1
        df = pd.DataFrame(cls, index=None, columns=None)
        df.to_csv(sv_path+f'{sv_nm}.csv', index=None, header=None)
    np.save(sv_path+f'{sv_nm}.npy', prob)
    return prob


def evaluate(gpu_id=2):
    # suffix = 'best',
    # print(suffix)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    ctx = mx.gpu(gpu_id)
    test_x = np.load(global_config['data_root'] + 'test_x.npy')[:, :, :, 8:]
    test_y = np.load(global_config['data_root'] + 'test_y.npy')
    test_x1, test_y1 = nd.array(test_x), nd.array(test_y)
    test_set = TestSet(test_x1).transform_first(transforms_test)
    test_loader = gluon.data.DataLoader(test_set, batch_size=256, shuffle=False, last_batch='keep', num_workers=0)
    # _, test_loader = data_loader(batch_size=256, mix=False)

    md_prefix = 'sen2_bs32_10channel_only_val_ls_mix_nl_cos_for_recovery'
    model = Sentinel2(17)
    # model = Sentinel2BiggerHead(17)
    weight_root = global_config['model_root'] + '{}/'.format(md_prefix)

    best_only = True
    if best_only:
        model.load_parameters(weight_root + '{}_{}.params'.format(md_prefix, 'best'), ctx)
        prob = test(ctx, model, test_loader)
    else:
        weights = glob.glob1('models/{}/'.format(md_prefix), '*epoch*params')
        suffixes = [wt[wt.find('epoch'):-7] for wt in weights if float(wt[-13:-7]) >= 0.9750]
        print(len(suffixes))
        prob = 0
        for suffix in suffixes:
            model.load_parameters(weight_root + '{}_{}.params'.format(md_prefix, suffix), ctx)
            prob += test(ctx, model, test_loader)

        prob /= len(suffixes)

    pred = np.argmax(prob, axis=1)
    lab = np.argmax(test_y, axis=1)

    acc = np.sum(pred == lab) / test_y.shape[0]
    print('accuracy: ', acc)


def test(ctx, model, test_loader):

    tta_num = 1
    prob = 0
    # acc = mx.metric.Accuracy()
    # cm = ConfusionMatrix('test cm')

    for _ in range(tta_num):
        # cm.reset()
        # acc.reset()
        preds = []
        for img in tqdm(test_loader):
            out = model(img.as_in_context(ctx))
            out = nd.softmax(out, axis=1)
            preds.append(out)
            # acc.update(nd.argmax(lab, axis=1), out)
            # cm.update(np.argmax(lab, axis=1), out)
        prob += nd.concatenate(preds, axis=0).asnumpy()
    prob /= tta_num
    # cm.plot(title='test cm', save_path='./models/{}_{}.png'.format(md_prefix, suffix))
    # cm.reset()
    # acc.reset()
    return prob


def gen_csv_from_prob():
    npy_path = 'data/round2_select_data/'
    a = np.load(npy_path+'round2_test_b_big_aba.npy')
    b = np.load(npy_path+'round2_test_b_bigger_aba.npy')
    c = np.load(npy_path+'round2_test_b_bigger1_aba.npy')
    d = np.load(npy_path+'round2_test_b_bigger2_aba.npy')

    prob = (a + b + c + d) / 4
    np.save('merge_bigger_aba_test_b.npy', prob)
    cls = np.zeros_like(prob, np.uint8)
    cls[np.arange(cls.shape[0]), np.argmax(prob, axis=1)] = 1
    df = pd.DataFrame(cls, index=None, columns=None)
    df.to_csv('merge_bigger_aba_test_b.csv', index=None, header=None)


def prob_distribution():
    pb_root = '../nas/plantvillage/'
    file_name = 'resnet50_v2_mix0_model_prob.npy'

    pb = np.load(pb_root+file_name)

    a = np.max(pb, axis=1, keepdims=True)
    mask = a != pb
    pb *= mask
    a = np.max(pb, axis=1)
    hist = np.histogram(a, bins=np.arange(0, 1.1, 0.1))
    print(hist)
    plt.hist(a, bins=np.arange(0, 1.1, 0.1), cumulative=False, density=True)
    plt.savefig(pb_root+'max_dist2')


def multi_union():
    path = 'data/round2_select_data/'
    # fl_nms1 = glob.glob(path+'wqx_a/sub*csv')
    fl_nms = sorted(glob.glob(path+'round2_test_b*csv'))
    # fl_nms = fl_nms1 + fl_nms2
    print(fl_nms)
    mark = 0
    for f in fl_nms:
        mark = mark + pd.read_csv(f, header=None).values
    mark = mark == len(fl_nms)
    print(mark.shape)
    mark = np.sum(mark, axis=1)
    print(mark.shape)
    print(mark)

    print(np.sum(mark)/mark.shape)
    np.save('data/round2_test_b_mark_pme.npy', mark.astype(np.uint8))


def union_data_round1():
    mark_b = np.load('data/round1_test_b_mark.npy')
    mark_a = np.load('data/round1_test_a_mark.npy')
    round1 = np.concatenate((mark_a, mark_b))
    print(round1.shape)
    label_a = pd.read_csv('data/select_data/test_a_bigger.csv', header=None).values
    label_b = pd.read_csv('data/select_data/test_b_bigger.csv', header=None).values
    assert mark_a.shape[0] == label_a.shape[0]
    assert mark_b.shape[0] == label_b.shape[0]
    label = np.concatenate((label_a, label_b), axis=0)
    ab = h5py.File('data/round1_test_ab.h5', 'r')
    sen2 = np.array(ab['sen2'])
    sen1 = np.array(ab['sen1'])
    print(sen1.shape)
    print(sen2.shape)
    new_sen1 = sen1[round1.astype(np.bool), ...]
    new_sen2 = sen2[round1.astype(np.bool), ...]
    new_label = label[round1.astype(np.bool), ...]

    print(new_sen1.shape)
    print(new_sen2.shape)
    print(new_label.shape)

    cf_round1 = h5py.File('data/confidence_round1_test_ab.h5', 'w')
    cf_round1['sen1'] = new_sen1
    cf_round1['sen2'] = new_sen2
    cf_round1['label'] = new_label
    cf_round1.flush()
    cf_round1.close()


def union_data_round2():
    mark_a = np.load('data/round2_test_b_mark_pme.npy')

    label_a = pd.read_csv('data/round2_select_data/round2_test_b_0211_885.csv', header=None).values
    assert mark_a.shape[0] == label_a.shape[0]
    ab = h5py.File('data/round2_test_b_20190211.h5', 'r')
    sen2 = np.array(ab['sen2'])
    sen1 = np.array(ab['sen1'])
    print(sen1.shape)
    print(sen2.shape)
    new_sen1 = sen1[mark_a.astype(np.bool), ...]
    new_sen2 = sen2[mark_a.astype(np.bool), ...]
    new_label = label_a[mark_a.astype(np.bool), ...]

    print(new_sen1.shape)
    print(new_sen2.shape)
    print(new_label.shape)

    cf_round1 = h5py.File('data/confidence_round2_test_b.h5', 'w')
    cf_round1['sen1'] = new_sen1
    cf_round1['sen2'] = new_sen2
    cf_round1['label'] = new_label
    cf_round1.flush()
    cf_round1.close()


if __name__ == '__main__':
    # train()
    # evaluate()

    # multi_infer()

    gen_csv_from_prob()

    # multi_union()
    # union_data_round2()

