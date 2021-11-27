from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os.path as osp
import numpy as np
import math
import time
import cv2

#new
from PIL import Image, ImageFile

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler
#new
from torchvision import transforms

from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate, CustomDataset
from lib.datasets.concatdataset import ConcatDataset
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists

#new
from lib.evaluation_metrics.metrics import get_str_list
from lib.utils.labelmaps import get_vocabulary, labels2strs

global_args = get_args(sys.argv[1:])

def image_process(image_path, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    img = Image.open(image_path).convert('RGB')

    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img

def cv_img_process(img, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    #img = img.resize((imgW, imgH), Image.BILINEAR)
    img = cv2.resize(img, (imgW, imgH))
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img

#new
class DataInfo(object):
    """
    Save the info about the dataset.
    This a code snippet from dataset.py
    """
    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type

        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)

def get_model(checkpoint_path = '', arch = 'ResNet_ASTER' , decoder_dim = 512, att_dim = 512,
                     max_len = 100, EOS = 'EOS', stn_on = True, seed = 1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    dataset_info = DataInfo('ALLCASES_SYMBOLS')
    #Create model
    model = ModelBuilder(arch=arch, rec_num_classes=dataset_info.rec_num_classes,
                            sDim=decoder_dim, attDim=att_dim, max_len_labels=max_len,
                            eos=dataset_info.char2id[dataset_info.EOS], STN_ON=stn_on)
    
    #Load from checkpoint
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        device = torch.device("cuda")
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")
        model = model.to(device)
    return model


def predict_text_by_img(img, model, max_len = 100 ,seed = 1):
    """
        Use to predict text when an image is received from detection model
    """
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda")
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device("cpu")


    dataset_info = DataInfo('ALLCASES_SYMBOLS')
    # #Create model
    # se_aster = ModelBuilder(arch=arch, rec_num_classes=dataset_info.rec_num_classes,
    #                         sDim=decoder_dim, attDim=att_dim, max_len_labels=max_len,
    #                         eos=dataset_info.char2id[dataset_info.EOS], STN_ON=stn_on)
    
    # #Load from checkpoint
    # if checkpoint_path:
    #     checkpoint = load_checkpoint(checkpoint_path)
    #     se_aster.load_state_dict(checkpoint['state_dict'])
    
    # if cuda:
    #     device = torch.device("cuda")
    #     se_aster = se_aster.to(device)
    #     se_aster = nn.DataParallel(se_aster)
    # else:
    #     device = torch.device("cpu")
    #     se_aster = se_aster.to(device)

    # Evaluation
    # model.eval()
    img = cv_img_process(img) # use this function because this image is cropped by opencv
    with torch.no_grad():
        img = img.to(device)
    input_dict = {}
    input_dict['images'] = img.unsqueeze(0)
    # TODO: testing should be more clean.
    # to be compatible with the lmdb-based testing, need to construct some meaningless variables.
    rec_targets = torch.IntTensor(1, max_len).fill_(1)
    rec_targets[:,max_len-1] = dataset_info.char2id[dataset_info.EOS]
    input_dict['rec_targets'] = rec_targets
    input_dict['rec_lengths'] = [max_len]
    input_dict['rec_embeds'] = torch.FloatTensor(1, 300).fill_(0)
    output_dict = model(input_dict)
    pred_rec = output_dict['output']['pred_rec']
    pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
    return pred_str[0]


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('using cuda.')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    #Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (32, 100)

    dataset_info = DataInfo(args.voc_type)
    #Create model
    model = ModelBuilder(arch=args.arch, rec_num_classes=dataset_info.rec_num_classes,
                            sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=args.max_len,
                            eos=dataset_info.char2id[dataset_info.EOS], STN_ON=args.STN_ON)

    #Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    if args.cuda:
        device = torch.device("cuda")
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")
        model = model.to(device)
    # Evaluation
    model.eval()
    img = image_process(args.image_path)
    with torch.no_grad():
        img = img.to(device)
    input_dict = {}
    input_dict['images'] = img.unsqueeze(0)
    # TODO: testing should be more clean.
    # to be compatible with the lmdb-based testing, need to construct some meaningless variables.
    rec_targets = torch.IntTensor(1, args.max_len).fill_(1)
    rec_targets[:,args.max_len-1] = dataset_info.char2id[dataset_info.EOS]
    input_dict['rec_targets'] = rec_targets
    input_dict['rec_lengths'] = [args.max_len]
    input_dict['rec_embeds'] = torch.FloatTensor(1, 300).fill_(0)
    output_dict = model(input_dict)
    pred_rec = output_dict['output']['pred_rec']
    pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
    print('Recognition result: {0}'.format(pred_str[0]))

if __name__ == '__main__':
  # parse the config
  args = get_args(sys.argv[1:])
  main(args)