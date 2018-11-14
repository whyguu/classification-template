from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mxnet.gluon import nn
import mxnet as mx
from mxnet import image
from mxnet import nd
import warnings
import random
from math import pi, cos
import logging
import sys
from PIL import Image
from mxnet import gluon, autograd


def get_logger(filename, log_name='Angelmon', stdout=True, mode='w'):
    logger = logging.getLogger(name=log_name)
    logger.setLevel(logging.INFO)
    # format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%Y/%m/%d %H:%M:%S"
    # file handler
    file_handler = logging.FileHandler(filename=filename, mode=mode)
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(file_handler)
    # stdout handler
    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
        logger.addHandler(stdout_handler)
    return logger


class RestartLR(mx.lr_scheduler.LRScheduler):
    def __init__(self, step_ratio, step, target_lr=1e-8):
        super(RestartLR, self).__init__()
        self.step_ratio = step_ratio

        self.step = step
        self.target_lr = target_lr
        self.count = 0
        self.lr = None

    def __call__(self, num_update):
        if self.count == self.step:
            self.count = 0
            self.step *= self.step_ratio
        else:
            self.count += 1

        self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * \
                             (1 + cos(pi * self.count / self.step)) / 2

        return self.learning_rate


class TrainerWithDifferentLR(object):
    """
    for idx, key in enumerate(model.collect_params().keys()):
        print(idx+1, key)

    run code above to decide cut_point of params
    """
    def __init__(self, params, lr_list, cut_point_list=None):
        from mxnet.gluon import ParameterDict
        assert len(lr_list) > 1
        assert len(lr_list) == len(cut_point_list)+1
        if isinstance(params, (dict, ParameterDict)):
            params = list(params.values())
        self.params = params
        self.trainers = []
        self.lr_list = lr_list
        if cut_point_list is None:
            cut_point_list = []
            avg_len = int(len(self.params) // len(self.lr_list))
            for i in range(1, len(self.lr_list)):
                cut_point_list.append(avg_len*i)
        self.cut_point = [0] + cut_point_list + [len(self.params)+1]
        # init
        self.make_trainer()

    def make_trainer(self):
        for idx, lr in enumerate(self.lr_list):
            # lr_scheduler = mx.lr_scheduler.FactorScheduler(step=1000, factor=0.5, stop_factor_lr=1e-5)
            lr_scheduler = RestartLR(1.5, 25*4, 1e-5)
            # lr_scheduler = None
            s, e = self.cut_point[idx], self.cut_point[idx+1]
            self.trainers.append(mx.gluon.Trainer(params=self.params[s:e], optimizer='adam',
                                                  optimizer_params={'learning_rate': self.lr_list[idx],
                                                  'wd': 1e-5, 'lr_scheduler': lr_scheduler}))

    def step(self, batch_size=1):
        for i in range(len(self.lr_list)):
            self.trainers[i].step(batch_size)

    @property
    def learning_rate(self):
        lr = []
        for tr in self.trainers:
            lr.append(tr.learning_rate)
        return lr


# image argumentation
class ResizeShort(nn.HybridBlock):
    def __init__(self, size=224):
        super(ResizeShort, self).__init__()
        self.size = size

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.image.resize_short(x, size=self.size)


class RandomResize(nn.Block):
    def __init__(self, ratio=(3/4, 4/3)):
        super(RandomResize, self).__init__()
        self.ratio = list(ratio)

    def forward(self, x):
        h, w, _ = x.shape
        nh = random.uniform(*self.ratio) * h
        nw = random.uniform(*self.ratio) * w
        out = mx.img.imresize(x, nw, nh)

        return out


class HistStretch(nn.Block):
    def __init__(self):
        super(HistStretch, self).__init__()

    def forward(self, x):
        x = x.asnumpy()
        percent = np.percentile(x, [2, 98], axis=[0, 1])
        x = np.clip(x, percent[0, :], percent[1, :])
        x = (x - percent[0, :]) / (percent[1, :] - percent[0, :]) * 255
        x = np.round(x).astype(np.uint8)
        return nd.array(x)


class RandomCrop(nn.Block):

    def __init__(self, size):
        super(RandomCrop, self).__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def forward(self, *args):
        x = args[0]
        x, _ = image.random_crop(x, self.size)
        return x


class RandomTranspose(nn.HybridBlock):
    def __init__(self):
        super(RandomTranspose, self).__init__()

    def hybrid_forward(self, F, x, *args, **kwargs):
        if np.random.uniform(0.0, 1.0) > 0.5:
            x = x.transpose((1, 0, 2))
        return x


class RandomRotate(nn.Block):
    def __init__(self, degree, fillcolor=(128, 128, 128), expand=False):
        """
        :param degree: (-180, 180)
        :param fillcolor:
        :param expand: 0/1, False/True
        """
        super(RandomRotate, self).__init__()
        self.fillcolor = fillcolor
        self.expand = expand
        if isinstance(degree, tuple):
            self.degree = degree
        else:
            if degree < 0:
                raise AttributeError('degree must greater than 0.')
            self.degree = (-degree, degree)

    def forward(self, x):
        ctx = x.context
        x = x.asnumpy()
        x = Image.fromarray(x)
        angle = np.random.randint(self.degree[0], self.degree[1])
        x = x.rotate(angle, resample=Image.BILINEAR, expand=self.expand, fillcolor=self.fillcolor)
        return nd.array(np.array(x), ctx=ctx, dtype=np.uint8)


# classification metric
class ClassificationReport(object):
    def __init__(self, name=None, target_name=None):
        self.name = name
        self.label = []
        self.pred = []
        self.target_name = target_name

    def update(self, label, pred):
        if not isinstance(pred, list):
            label = [label]
            pred = [pred]
        for lb, pd in zip(label, pred):
            lb = lb.asnumpy()
            pd = pd.asnumpy()
            self.label.append(lb.astype(np.uint32))
            self.pred.append(np.argmax(pd, axis=1).astype(np.uint32))

    def get(self):
        lb = np.concatenate(tuple(self.label))
        pd = np.concatenate(tuple(self.pred))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            report = classification_report(lb, y_pred=pd, target_names=self.target_name)
        return self.name, report

    def reset(self):
        self.label = []
        self.pred = []


class ConfusionMatrix(object):
    def __init__(self, name=None, class_names=None):
        self.name = name
        self.label = []
        self.pred = []
        self.classes = class_names
        self.matrix = None

    def update(self, label, pred):
        if not isinstance(pred, list):
            label = [label]
            pred = [pred]

        attr_name = 'asnumpy'  # mxnet/gluon
        if not hasattr(label[0], attr_name):
            attr_name = 'numpy'  # pytorch
            label = [lb.data.cpu() for lb in label]
            pred = [pd.data.cpu() for pd in pred]

        for lb, pd in zip(label, pred):
            lb = getattr(lb, attr_name)()
            pd = getattr(pd, attr_name)()
            self.label.append(lb.astype(np.uint32))
            self.pred.append(np.argmax(pd, axis=1).astype(np.uint32))

    def get(self):
        lb = np.concatenate(tuple(self.label))
        pd = np.concatenate(tuple(self.pred))
        self.matrix = confusion_matrix(lb, y_pred=pd)
        return self.name, self.matrix

    def reset(self):
        self.matrix = None
        self.label = []
        self.pred = []

    def plot(self, normalize=False, title='Confusion matrix', save_path=None, cmap=None, annot=True):
        if save_path is None:
            print('have no saving path')
            return
        if self.matrix is None:
            self.get()

        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
        # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        # sns.set(font='SimHei')  # 解决Seaborn中文显示问题

        cm = self.matrix
        classes = self.classes
        if classes is None:
            classes = [str(i) for i in range(cm.shape[0])]
        if normalize:
            cm_r = cm.astype(np.float32) / (cm.sum(axis=1)[:, np.newaxis] + 1e-5)
            cm_p = cm.astype(np.float32) / (cm.sum(axis=0)[np.newaxis, :] + 1e-5)
            AR = np.trace(cm_r) / self.matrix.shape[0]
            AP = np.trace(cm_p) / self.matrix.shape[0]

            df = pd.DataFrame(cm_r, index=classes, columns=classes)
            figsize = (len(classes)/4*3, len(classes)/2)
            f, ax = plt.subplots(figsize=figsize)
            mask = None  # cm < 0.01  #
            sns.heatmap(df, annot=annot, fmt=".2f", linewidths=.5, ax=ax, cmap=cmap, mask=mask)
            plt.title(title+' avg recall: {:.3f}, avg precision {:.3f}'.format(AR, AP))
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(save_path)
            plt.close()
        else:
            df = pd.DataFrame(cm, index=classes, columns=classes)
            figsize = (len(classes) / 4 * 3, len(classes) / 2)
            f, ax = plt.subplots(figsize=figsize)
            mask = None  # cm < 0.01  #
            sns.heatmap(df, annot=annot, fmt="d", linewidths=.1, ax=ax, cmap=cmap, mask=mask)
            plt.title(title)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(save_path)
            plt.close()


def lr_find(ctx, data_loader, model, loss, trainer):
    from prettytable import PrettyTable
    table = PrettyTable(['loss', 'lr'])
    table.align = 'l'
    table.float_format = "1.5e"
    opt = getattr(trainer, '_optimizer')
    opt.lr_scheduler = None
    # lr_find
    if not isinstance(ctx, list):
        ctx = [ctx]
    trainer.set_learning_rate(1e-7)
    test_loss = []

    for idx, (data, label) in enumerate(data_loader):
        # lr_find
        trainer.set_learning_rate(trainer.learning_rate*10)

        if trainer.learning_rate > 10:
            print(table)
            try:
                dd = np.vstack(test_loss)
                plt.semilogx(dd[:, 1], dd[:, 0])
                plt.show()
            except:
                pass
            return
        dts = gluon.utils.split_and_load(data, ctx)
        lbs = gluon.utils.split_and_load(label, ctx)
        with autograd.record():
            outs = [model(dts[i]) for i in range(len(ctx))]
            ls = [loss(outs[i], lbs[i]).mean() for i in range(len(ctx))]
        for i in range(len(ctx)):
            ls[i].backward()
        trainer.step(len(ctx))

        # lr_find
        ll = np.mean([ls[i].asscalar() for i in range(len(ctx))])
        test_loss.append([ll, trainer.learning_rate])
        table.add_row([ll, trainer.learning_rate])
        # continue


if __name__ == '__main__':
    pass


