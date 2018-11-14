import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import densenet121
from cnn_finetune import make_model
import sklearn.utils as skutils
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
import copy
import json
import random
import PIL
from PIL import Image
from tqdm import tqdm
import glob
import pandas as pd
from torch import multiprocessing
from gluon_model import class2label, label2class
from utils import ConfusionMatrix, get_logger
import matplotlib.pyplot as plt


def random_transpose(img):
    if random.uniform(0, 1) < 0.5:
        return img.transpose(PIL.Image.TRANSPOSE)
    return img


transforms_train = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256, (0.5, 1.0),),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Lambda(random_transpose),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
])

transforms_val = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomResizedCrop(256, (0.4, 1.0),),
    transforms.ToTensor(),
])

transforms_test = transforms.Compose([
    transforms.RandomResizedCrop(256, (0.6, 1.0),),
    # transforms.Resize((256, 256)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(random_transpose),
    transforms.RandomRotation(45, PIL.Image.BILINEAR),
    transforms.ToTensor(),
])


class DPNet(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(DPNet, self).__init__()
        self.model_name = model_name
        self.feat = make_model(model_name, num_classes=6, pretrained=pretrained)._features

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.ds1 = nn.Linear(4096, 2048)
        self.act1 = nn.ReLU(True)
        self.drop = nn.Dropout(0.5)

        self.ds2 = nn.Linear(2048, 6)

    def forward(self, x):
        x = self.feat(x)
        a = self.gap(x).view(x.size(0), -1)
        m = self.gmp(x).view(x.size(0), -1)

        c = torch.cat((a, m), 1)
        x = self.act1(self.ds1(c))
        out = self.ds2(self.drop(x))

        return out


class DataSet(torch.utils.data.Dataset):
    def __init__(self, data, data_root, transform_fn=None):
        self.data_root = data_root
        self.data = data
        self.transform_fn = transform_fn
        print('data num', self.__len__())

    def __getitem__(self, idx):
        name = self.data[idx, 0]
        class_name = self.data[idx, 1]
        label = class2label[class_name]

        img = Image.open(self.data_root + name)
        # if img.size != (256, 256):
        #     print(name)
        if self.transform_fn is not None:
            img = self.transform_fn(img)

        return img, label

    def __len__(self):
        return len(self.data)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, image_names, test_path, transform_fn=None):
        self.test_path = test_path
        self.transform_fn = transform_fn
        self.image_names = image_names
        print('data num', self.__len__())

    def __getitem__(self, idx):
        img = Image.open(self.test_path+self.image_names[idx])

        if self.transform_fn is not None:
            img = self.transform_fn(img)

        return img, idx

    def __len__(self):
        return len(self.image_names)


class FocalLoss(nn.Module):
    def __init__(self, cls_num, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()

        self._alpha = alpha
        self._gamma = gamma
        self._cls_num = cls_num
        self._eps = 1e-12

    def forward(self, out, label):
        device = label.device
        bs = label.size(0)
        out = torch.sigmoid(out)
        one_hot = torch.zeros((bs, self._cls_num), dtype=torch.uint8, device=device)
        one_hot[torch.arange(bs), label] = 1
        pt = torch.where(one_hot, out, 1-out)

        pt = torch.min(pt+self._eps, torch.ones_like(pt))

        t = torch.ones_like(one_hot, device=device).float()
        alpha = torch.where(one_hot, self._alpha * t, (1 - self._alpha) * t)

        loss = -alpha * ((1.0 - pt) ** self._gamma) * torch.log(pt)

        return torch.mean(loss)


def data_set(is_eval=False):
    data_root = '/workspace/nas/guangpu/train/images/'
    data_file = '/workspace/nas/guangpu/train/label.csv'
    data = pd.read_csv(data_file, header=None).values
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)
    if is_eval:
        return DataSet(data, data_root, transforms_val)
    train_set = DataSet(train_data, data_root, transforms_train)
    val_set = DataSet(val_data, data_root, transforms_val)
    return train_set, val_set


def make_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        # nn.Dropout(0.6),
        nn.Linear(4096, num_classes),
    )


def train():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")
    num_epochs = 100
    batch_size = 32

    train_set, val_set = data_set()
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=2)

    dataloaders = {'train': train_loader,  'val': val_loader}

    # model_name = 'inception_v3'
    model_name = 'xception'
    # md = make_model(model_name, num_classes=6, pretrained=True, dropout_p=0.5, classifier_factory=None)
    md = DPNet(model_name, True)
    # md = nn.DataParallel(md, device_ids=[0, 1])
    md.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(6)

    optimizer_ft = optim.Adam(md.parameters(), lr=5e-4, weight_decay=1e-5)

    exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=[7, 15, 20], gamma=0.2,)

    log_path = '/workspace/nas/guangpu/torch_256_dpn_fl_{}/'.format(model_name)
    os.makedirs(log_path, exist_ok=True)

    train_model(md, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, device, log_path, num_epochs)


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, log_path, num_epochs=25):
    if hasattr(model, 'module'):
        template_module = model.module
    else:
        template_module = model
    since = time.time()
    dataset_sizes = {'train': len(dataloaders['train'].dataset), 'val': len(dataloaders['val'].dataset)}

    cm = {'train': ConfusionMatrix('train'), 'val': ConfusionMatrix('val')}
    best_acc = 0.0
    logger = get_logger(log_path+'torch_record.txt')
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            cm[phase].reset()
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(torch.argmax(outputs, 1) == labels.data)
                cm[phase].update(labels, outputs)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print_str = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
            logger.info(print_str)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(template_module.state_dict(), log_path+'torch_best.pt'.format(epoch))
            if phase == 'val' and epoch_acc >= 0.985:
                for ps in ['train', 'val']:
                    save_path = log_path + ps + '_epoch{}.png'.format(epoch)
                    cm[ps].plot(title=ps + '_epoch{}'.format(epoch), save_path=save_path)
                torch.save(template_module.state_dict(), log_path + 'torch_epoch{}_{:.4f}.pt'.format(epoch, epoch_acc))

    time_elapsed = time.time() - since
    print_str1 = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    print_str2 = 'Best val Acc: {:4f}'.format(best_acc)
    logger.info(print_str1)
    logger.info(print_str2)


def infer(suffix=None, gen_csv=True, gpu_id=2, save_path=''):
    # md_prefix = 'torch_256_dpn_fl_xception'
    md_prefix = 'torch_256_dpn_xception'
    # suffix = 'epoch83_0.9925'
    print(suffix)

    device = torch.device("cuda: " + str(gpu_id))
    weight_root = '/workspace/nas/guangpu/{}/'.format(md_prefix)
    # model = make_model('inception_v3', num_classes=6, pretrained=False, dropout_p=0.5, classifier_factory=None)
    model = DPNet('xception', pretrained=False,)
    state_dict = torch.load(weight_root+'torch_{}.pt'.format(suffix), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    #
    test_path = '/workspace/nas/guangpu/test/images_b/'
    image_names = sorted([nm for nm in os.listdir(test_path) if 'jpg' in nm or 'JPG' in nm])
    assert len(image_names) == 1000

    test_set = TestSet(image_names, test_path, transforms_test)
    test_loader = DataLoader(test_set, batch_size=18, shuffle=False, num_workers=0)

    # TTA
    if not save_path:
        save_path = '/workspace/nas/guangpu/'
    prob = 0
    tta_num = 5
    for _ in range(tta_num):
        preds = []
        with torch.no_grad():
            for img, idx in tqdm(test_loader):
            # for img, idx in test_loader:
                out = model(img.to(device))
                out = F.softmax(out, dim=1)
                preds.append(out.data.cpu().numpy())

        prob += np.concatenate(tuple(preds), axis=0)
    prob /= tta_num
    np.save(save_path+'{}_{}_prob.npy'.format(md_prefix, suffix), prob)

    if not gen_csv:
        return
    cls = pd.Series(np.argmax(prob, axis=1)).map(label2class).values
    sub = np.hstack((np.array(image_names).reshape(-1, 1), cls.reshape(-1, 1)))
    df = pd.DataFrame(sub, index=None, columns=None)
    df.to_csv(save_path+'{}_{}.csv'.format(md_prefix, suffix), index=None, header=None)


def evaluate(suffix=None, gpu_id=3):
    device = torch.device("cuda: " + str(gpu_id))

    md_prefix = 'torch_xception'
    weight_root = '/workspace/nas/{}/'.format(md_prefix)
    # suffix = ''
    print(suffix)

    # model = make_model('xception', num_classes=6, pretrained=False, dropout_p=0.5, classifier_factory=None)
    model = DPNet('xception', pretrained=False, )

    model.eval()
    state_dict = torch.load(weight_root + 'torch_{}.pt'.format(suffix), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    eval_set = data_set(True)
    eval_loader = DataLoader(eval_set, batch_size=128, shuffle=False, num_workers=0)

    preds = []
    lbs = []
    with torch.no_grad():
        for img, label in tqdm(eval_loader):
            out = model(img.to(device))
            out = F.softmax(out, dim=1)
            preds.append(out.data.cpu().numpy())
            lbs.append(label.data.cpu().numpy())

    prob = np.concatenate(tuple(preds), axis=0)
    t_lb = np.concatenate(tuple(lbs), axis=0)
    p_lb = np.argmax(prob, axis=1)
    indices = np.argwhere(p_lb != t_lb).squeeze()
    # print(indices.shape)
    # print(indices)
    for idx in range(indices.shape[0]):
        ii = indices[idx]
        p_p = prob[ii, int(p_lb[ii])]
        t_p = prob[ii, int(t_lb[ii])]
        # print(p_p)
        img_name = eval_set.data[ii, 0]
        cls_name = eval_set.data[ii, 1]
        pred_name = label2class[int(p_lb[ii])]
        print('{}, true: {}, {:.4f}; pred: {}, {:.4f}'.format(img_name, cls_name, t_p, pred_name, p_p))


def rlt_analysis():
    test_path = '/workspace/nas/guangpu/test/images_b/'
    image_names = sorted([nm for nm in os.listdir(test_path) if 'jpg' in nm or 'JPG' in nm])

    path = '/workspace/nas/guangpu/'
    prob_file = 'dpn_xception_merge'

    pred = np.load(path+'{}.npy'.format(prob_file))
    max_pb = np.max(pred, axis=1)
    plt.hist(max_pb, np.arange(0, 1.1, 0.1))
    plt.savefig(path+'{}_pb_hist.png'.format(prob_file))
    mask = (max_pb < 0.6)  # * (max_pb > 0.6)
    pred = pred[mask, :]
    image_names = np.array(image_names)[mask]
    # print(pred)
    print(pred.shape)
    sort_idx = np.argsort(pred, axis=1, )[:, ::-1]
    # print(sort_idx)
    for i in range(pred.shape[0]):
        prt_str = '{}, '.format(image_names[i])
        for j in range(6):
            prt_str += '{}: {:.4f}, '.format(label2class[sort_idx[i, j]], pred[i, sort_idx[i, j]])
        # print('{}, {}: {:.4f}, {}: {:.4f}'.format(image_names[i], first_name, first_prob, second_name, second_prob))
        print(prt_str)


if __name__ == '__main__':
    train()
    # evaluate(gpu_id=1)
    # infer(gpu_id=1, )
    # rlt_analysis()



