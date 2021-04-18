#!/usr/bin/env Python
# coding=utf-8

import logging
import time
import os.path as osp

#
EVAL = False

# dataset can be 'MIRFlickr' or 'NUSWIDE'
DATASET = 'WIKI'

if DATASET == 'WIKI':

    x = '/data/dataY2/zhaoyang/datasets/CrossModal/wiki.mat'
    x_train = '/data/dataY2/zhaoyang/datasets/CrossModal/wiki_train.mat'
    x_test = '/data/dataY2/zhaoyang/datasets/CrossModal/wiki_test.mat'

    BETA = 0.3
    BETA2 = 0.4
    LAMBDA1 = 0.3
    LAMBDA2 = 0.3
    LAMBDA3 = 0.1
    LAMBDA4 = 0.1
    NUM_EPOCH = 200
    LR_IMG = 0.01
    LR_TXT = 0.01
    EVAL_INTERVAL = 2

if DATASET == 'MIRFlickr':
    x = '/data/dataY2/zhaoyang/datasets/CrossModal/mir.mat'
    x_train = '/data/dataY2/zhaoyang/datasets/CrossModal/mir_train.mat'
    x_test = '/data/dataY2/zhaoyang/datasets/CrossModal/mir_test.mat'
    
    BETA = 0.5
    BETA2 = 0.1
    LAMBDA1 = 0.1
    LAMBDA2 = 0.1
    NUM_EPOCH = 200
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 2

if DATASET == 'NUSWIDE':

    x = '/data/dataY2/zhaoyang/datasets/CrossModal/nus.mat'
    x_train = '/data/dataY2/zhaoyang/datasets/CrossModal/nus_train.mat'
    x_test = '/data/dataY2/zhaoyang/datasets/CrossModal/nus_test.mat'

    BETA = 0.6
    BETA2 = 0.4
    LAMBDA1 = 0.1
    LAMBDA2 = 0.1
    NUM_EPOCH = 80
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 2

if DATASET == 'MSCOCO':
    x = '/data/dataY2/zhaoyang/datasets/CrossModal/coco.mat'
    x_train = '/data/dataY2/zhaoyang/datasets/CrossModal/coco_train.mat'
    x_test = '/data/dataY2/zhaoyang/datasets/CrossModal/coco_test.mat'

    BETA = 0.6
    BETA2 = 0.4
    LAMBDA1 = 0.1
    LAMBDA2 = 0.1
    NUM_EPOCH = 80
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 2

SIM_MAP = '/data/dataY2/zhaoyang/datasets/CrossModal/sim_uni/'
SUF_ID = '/data/dataY2/zhaoyang/datasets/CrossModal/suf_id.mat'
BATCH_SIZE = 8
CODE_LEN = 16
MU = 1.5
ETA = 0.4


MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_WORKERS = 0
EPOCH_INTERVAL = 2

MODEL_DIR = './checkpoint'



logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())) 
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)


logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
logger.info('BETA = %.4f' % BETA)
logger.info('LAMBDA1 = %.4f' % LAMBDA1)
logger.info('LAMBDA2 = %.4f' % LAMBDA2)
logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('CODE_LEN = %d' % CODE_LEN)
logger.info('MU = %.4f' % MU)
logger.info('ETA = %.4f' % ETA)
logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)
# logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)
logger.info('--------------------------------------------------------------------')