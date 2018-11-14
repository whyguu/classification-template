# /bin/bash
# coding: utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import numpy as np
from mxnet import nd
from mxnet import autograd
import pandas as pd
from mxnet import image
from tqdm import tqdm
import os
import multiprocessing
from utils import TrainerWithDifferentLR, ConfusionMatrix, RandomRotate, get_logger, RandomTranspose, lr_find
from sklearn import utils as skutils
from sklearn.model_selection import train_test_split
import json
from gluoncv.model_zoo import resnet101_v2, densenet121, resnet34_v2, resnet50_v2, inception_v3
from gluoncv.loss import FocalLoss
from gluoncv import utils as gutils
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class2label = {
    'DESERT': 0,
    'OCEAN': 1,
    'FARMLAND': 2,
    'MOUNTAIN': 3,
    'LAKE': 4,
    'CITY': 5,
}

label2class = {
    0: 'DESERT',
    1: 'OCEAN',
    2: 'FARMLAND',
    3: 'MOUNTAIN',
    4: 'LAKE',
    5: 'CITY',
}

transforms_train = transforms.Compose([
    # transforms.RandomResizedCrop(256, scale=(0.6, 1.0), ),
    transforms.Resize(256),
    # RandomCrop(224),
    RandomTranspose(),
    transforms.RandomFlipLeftRight(),
    transforms.RandomFlipTopBottom(),
    transforms.RandomContrast(0.3),
    RandomRotate(45, ),
    transforms.ToTensor(),
])

transforms_val = transforms.Compose([
    # transforms.CenterCrop(224),
    transforms.Resize(256),
    transforms.ToTensor(),
])

transforms_test = transforms.Compose([
    # transforms.RandomResizedCrop(256, scale=(0.6, 1.0)),
    transforms.Resize(256),
    RandomTranspose(),
    transforms.RandomFlipLeftRight(),
    transforms.RandomFlipTopBottom(),
    transforms.RandomContrast(0.3),
    RandomRotate(45, ),
    transforms.ToTensor(),
])


def net(cls_num, fn, context, pretrained=True):
    pre_md = fn(pretrained=pretrained, ctx=context).features
    if fn.__name__ in ['inception_v3', 'densenet121']:
        pre_md = pre_md[0:-2]
        pre_md.add(nn.GlobalAvgPool2D(),
                   nn.Flatten())

    output = nn.HybridSequential()
    output.add(
        nn.Dropout(rate=0.5),
        nn.Dense(cls_num),
    )
    # new model
    md = nn.HybridSequential(prefix='haha')
    with md.name_scope():
        md.add(
            pre_md,
            output
        )

    md.hybridize()
    if pretrained:
        output.initialize(mx.init.Xavier(), ctx=context)
    return md


class DPNet(nn.Block):
    # double pool net
    def __init__(self, cls_num, fn, context, pretrained=True):
        super(DPNet, self).__init__()
        self.pre_md = fn(pretrained=pretrained, ctx=context).features[0:-2]

        self.gap = nn.GlobalAvgPool2D()
        self.gmp = nn.GlobalMaxPool2D()
        self.flt1 = nn.Flatten()
        self.flt2 = nn.Flatten()

        self.ds1 = nn.Dense(2048, 'relu')
        self.drop = nn.Dropout(0.5)
        self.ds2 = nn.Dense(cls_num)

        if pretrained:
            self.ds1.initialize(mx.init.MSRAPrelu(), ctx=context)
            self.ds2.initialize(mx.init.MSRAPrelu(), ctx=context)

    def forward(self, x):
        x = self.pre_md(x)
        a = self.flt1(self.gap(x))
        m = self.flt2(self.gmp(x))
        c = nd.concat(a, m, dim=1)
        out = self.ds2(self.drop(self.ds1(c)))
        return out


class DataSet(gluon.data.Dataset):
    def __init__(self, data, data_root):
        self.data_root = data_root
        self.data = data

        print('data num', self.__len__())

    def __getitem__(self, idx):
        name = self.data[idx, 0]
        class_name = self.data[idx, 1]
        label = class2label[class_name]

        img = image.imread(self.data_root+name)

        return img, label

    def __len__(self):
        return self.data.shape[0]


class TestSet(gluon.data.Dataset):
    def __init__(self, image_names, test_path):
        self.test_path = test_path
        self.image_names = image_names
        print('data num', self.__len__())

    def __getitem__(self, idx):
        img = image.imread(self.test_path+self.image_names[idx])

        return img, idx

    def __len__(self):
        return len(self.image_names)


def data_set():
    data_root = '/workspace/nas/guangpu/train/images/'
    data_file = '/workspace/nas/guangpu/train/label.csv'
    data = pd.read_csv(data_file, header=None).values
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42, shuffle=False)

    train_set = DataSet(train_data, data_root).transform_first(transforms_train)
    val_set = DataSet(val_data, data_root).transform_first(transforms_val)
    return train_set, val_set


def train():
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    ctx = [mx.gpu(2)]
    epochs = 100
    batch_size = 32

    train_set, val_set = data_set()
    train_loader = gluon.data.DataLoader(train_set, batch_size, shuffle=True, last_batch='rollover', num_workers=4)
    val_loader = gluon.data.DataLoader(val_set, batch_size, shuffle=False, last_batch='keep', num_workers=2)

    # model
    model = net(6, densenet121, ctx, pretrained=True)
    # model = DPNet(6, resnet101_v2, ctx, pretrained=True)
    model_prefix = 'densenet121_im256_bs{}'.format(batch_size)
    root_path = '/workspace/nas/guangpu/' + model_prefix + '/'
    os.makedirs(root_path, exist_ok=True)

    logger = get_logger(root_path+'report.txt', )
    logger.info('total epoch: {}'.format(epochs))
    logger.info(model_prefix)
    #
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # loss = FocalLoss(num_class=6)
    lr_scheduler = gutils.LRScheduler('step', baselr=1e-3, niters=int(len(train_set)/batch_size+1), nepochs=epochs,
                                      step=[15, 40], step_factor=0.1,
                                      targetlr=5e-6, power=0.9,
                                      warmup_epochs=2, warmup_lr=1e-5, warmup_mode='linear')
    trainer = gluon.Trainer(model.collect_params(), optimizer='adam',
                            optimizer_params={'learning_rate': 0.01, 'wd': 5e-5,
                                              'lr_scheduler': lr_scheduler})
    # lr_find(ctx, train_loader, model, loss, trainer)
    # exit()
    # trainer = TrainerWithDifferentLR(model.collect_params(), [1e-5, 5e-5, 1e-4, 5e-4], [95, 350, 470])
    #
    train_acc = mx.metric.Accuracy(name='train_acc')
    train_ls = mx.metric.Loss(name='train_ls')
    val_acc = mx.metric.Accuracy(name='val_acc')
    val_top_acc = mx.metric.TopKAccuracy(2, name='val_top2_acc')
    train_confuse_matrix = ConfusionMatrix(name='train_cm')
    val_confuse_matrix = ConfusionMatrix(name='val_cm')

    best_val = 0.0
    for epoch in range(1, epochs+1):
        for idx, (data, label) in tqdm(enumerate(train_loader)):
            lr_scheduler.update(idx, epoch-1)
            dts = gluon.utils.split_and_load(data, ctx)
            lbs = gluon.utils.split_and_load(label, ctx)
            with autograd.record():
                outs = [model(dts[i]) for i in range(len(ctx))]
                ls = [loss(outs[i], lbs[i]).mean() for i in range(len(ctx))]
            for i in range(len(ctx)):
                ls[i].backward()
            trainer.step(len(ctx))

            # update
            train_acc.update(lbs, outs)
            train_confuse_matrix.update(lbs, outs)
            train_ls.update(None, preds=ls)

        for data, label in tqdm(val_loader):
            dts = gluon.utils.split_and_load(data, ctx, even_split=False)
            lbs = gluon.utils.split_and_load(label, ctx, even_split=False)
            outs = [model(dts[i]) for i in range(len(ctx))]
            val_acc.update(lbs, outs)
            val_top_acc.update(lbs, outs)
            val_confuse_matrix.update(lbs, outs)

        # show criteria print('learning rate: ', trainer.learning_rate)
        print_str = 'epoch: {} lr: {:.2e}'.format(epoch, trainer.learning_rate)
        for nm, acc in [train_acc.get(), val_acc.get(), val_top_acc.get()]:
            print_str += ', {}: {:.3%}'.format(nm, acc)
        ls_nm, ls_val = train_ls.get()
        print_str += ', {}: {:.5f}'.format(ls_nm, ls_val)
        logger.info(print_str)

        val_acc_tp = val_acc.get()[1]
        if best_val < val_acc_tp:
            logger.info('new best: {}, old best: {}'.format(val_acc_tp, best_val))
            best_val = val_acc_tp
            model.save_parameters(root_path + model_prefix + '_best.params')
        if val_acc_tp >= 0.98:
            train_confuse_matrix.plot(title='train epoch {}'.format(epoch),
                                      save_path=root_path + 'train_' + model_prefix + '_epoch_{}.png'.format(epoch))
            val_confuse_matrix.plot(title='val epoch {}'.format(epoch),
                                    save_path=root_path + 'val_' + model_prefix + '_epoch_{}.png'.format(epoch))

            model.save_parameters(root_path + model_prefix + '_epoch_{}_{:.4f}.params'.format(epoch, val_acc_tp))

        # model.save_parameters(root_path + model_prefix+'_epoch_{}.params'.format(epoch))

        # reset
        train_acc.reset()
        val_acc.reset()
        val_top_acc.reset()
        train_ls.reset()
        train_confuse_matrix.reset()
        val_confuse_matrix.reset()
    logger.info('best val: {}'.format(best_val))


def infer(suffix=None, gen_csv=False, gpu_id=1):
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    ctx = mx.gpu(gpu_id)
    save_path = '/workspace/nas/guangpu/probs_densenet121/'
    os.makedirs(save_path, exist_ok=True)

    md_prefix = 'densenet121_im256_bs32'
    weight_root = '/workspace/nas/guangpu/{}/'.format(md_prefix)
    # suffix = 'epoch_92_0.9925'
    print(suffix)

    model = net(6, densenet121, ctx, pretrained=True)
    model.load_parameters(weight_root + '{}_{}.params'.format(md_prefix, suffix), ctx)
    test_path = '/workspace/nas/guangpu/test/images_b/'
    image_names = sorted([nm for nm in os.listdir(test_path) if 'jpg' in nm or 'JPG' in nm])
    assert len(image_names) == 1000

    test_set = TestSet(image_names, test_path).transform_first(transforms_test)
    test_loader = gluon.data.DataLoader(test_set, batch_size=64, shuffle=False, last_batch='keep', num_workers=0)

    prob = 0
    # TTA
    for _ in range(5):
        preds = []
        # for img, idx in tqdm(test_loader):
        for img, idx in test_loader:
            out = model(img.as_in_context(ctx))
            out = nd.softmax(out, axis=1)
            preds.append(out)

        prob += nd.concatenate(preds, axis=0).asnumpy()
    prob /= 5
    np.save(save_path+'{}_{}_prob.npy'.format(md_prefix, suffix), prob)

    if not gen_csv:
        return
    cls = pd.Series(np.argmax(prob, axis=1)).map(label2class).values
    sub = np.hstack((np.array(image_names).reshape(-1, 1), cls.reshape(-1, 1)))
    df = pd.DataFrame(sub, index=None, columns=None)
    df.to_csv(save_path+'{}_{}.csv'.format(md_prefix, suffix), index=None, header=None)


def heatmap():
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    ctx = mx.gpu(2)
    model = net(61, resnet101_v2, ctx, pretrained=False)
    root_path = '../nas/plantvillage/AD/weights/'
    model.load_parameters(root_path + 'resnet101_im224_61_bs192_multigpu_aug_61class_epoch_10.params', ctx=ctx)
    test_path = '../nas/plantvillage/AD/AgriculturalDisease_testA/images/'
    image_names = os.listdir(test_path)
    image_names = [nm for nm in image_names if 'jpg' in nm or 'JPG' in nm]

    test_loader = gluon.data.DataLoader(TestSet(image_names, test_path).transform_first(transforms_test),
                                        batch_size=20, shuffle=False, num_workers=8)

    # model.summary(img.as_in_context(ctx))
    # cl = list(model._children.values())
    # mimi = nn.Sequential()
    # bodies = list(cl[0]._children.values())[0:11]
    # mimi.add(*bodies)
    # out_block = list(cl[1]._children.values())[-1]
    mimi = model[0:11]
    out_block = model[-1]
    weight = out_block.weight.data(ctx)
    bias = out_block.bias.data(ctx)
    print(weight.shape)
    print(bias.shape)

    for img, idx in test_loader:
        out = model(img.as_in_context(ctx))
        cls_id = nd.argmax(out, axis=1)
        factor = nd.gather_nd(weight, nd.expand_dims(cls_id, axis=0))
        for i in range(2):
            factor = nd.expand_dims(factor, axis=-1)
        feat = mimi(img.as_in_context(ctx))
        heat_maps = nd.sum(factor * feat, axis=1).transpose([1, 2, 0]).asnumpy()
        print(heat_maps.shape)
        # np.save('../nas/plantvillage/AD/img_heat_maps/a.npy', heat_maps)

        mins = np.min(heat_maps, axis=(0, 1), keepdims=True)
        maxs = np.max(heat_maps, axis=(0, 1), keepdims=True)
        heat_maps = ((heat_maps - mins) / (maxs - mins) * 255).astype(np.uint8)
        # print(heat_maps[:, :, 0])
        idx = idx.asnumpy()
        for i in idx.tolist():
            img_org = cv2.imread(test_path+image_names[i])
            # print(img.shape)
            print(image_names[i])
            hm = cv2.resize(heat_maps[:, :, i], (img_org.shape[1], img_org.shape[0]), interpolation=cv2.INTER_LINEAR)
            # hm = cv2.applyColorMap(hm, cv2.COLORMAP_BONE)
            # hm[hm < 200] = 0
            img_dst = cv2.addWeighted(img_org, 0.7, hm, 0.3, 0)
            # print(img.dtype)
            cv2.imwrite('../nas/plantvillage/AD/img_heat_maps/'+image_names[i], img_dst)
            cv2.imwrite('../nas/plantvillage/AD/img_heat_maps/'+image_names[i]+'.jpg', hm)
        return


def merge_rlt():
    root = '/workspace/nas/'
    test_path = root + 'test/images_b/'
    image_names = sorted([nm for nm in os.listdir(test_path) if 'jpg' in nm or 'JPG' in nm])

    prob_dir = 'probs_torch_256_dpn_xception'
    md_prefix = 'dpn_xception'
    probs = glob.glob(root + prob_dir + '/*npy')
    probs = [pb for pb in probs if float(pb[-15:-9]) >= 0.9850]
    print(len(probs))
    a = 0
    for prob in probs:
        a += np.load(prob)
    final_idx = np.argmax(a, axis=1)
    a = a/len(probs)
    np.save(root+'{}_merge.npy'.format(md_prefix), a)
    cls = pd.Series(final_idx).map(label2class).values
    sub = np.hstack((np.array(image_names).reshape(-1, 1), cls.reshape(-1, 1)))
    df = pd.DataFrame(sub, index=None, columns=None)
    df.to_csv(root+'{}_merge.csv'.format(md_prefix), index=None, header=None)


def prob_distribution():
    test_path = '../nas/plantvillage/AD/AgriculturalDisease_testA/images/'
    image_names = sorted([nm for nm in os.listdir(test_path) if 'jpg' in nm or 'JPG' in nm])

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


if __name__ == '__main__':
    # train()
    # infer(gen_csv=True)
    merge_rlt()
    exit()
    # heatmap()

    weights = glob.glob1('/workspace/nas/guangpu/densenet121_im256_bs32/', '*epoch*params')
    suffixes = [wt[-22:-7] for wt in weights if float(wt[-13:-7]) >= 0.9950]
    print(len(suffixes))
    # print(suffixes)
    while suffixes:
        pools = multiprocessing.Pool(6)
        for i in range(6):
            if not suffixes:
                break
            pools.apply_async(infer, args=(suffixes.pop(0), False, i % 2))
        pools.close()
        pools.join()

