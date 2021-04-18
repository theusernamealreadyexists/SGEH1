#!/usr/bin/env Python
# coding=utf-8

import torch
import settings
import numpy as np
import scipy.io as scio
import h5py
from scipy.sparse import csr_matrix, triu, find

suf_id = h5py.File(settings.SUF_ID)
coco_id = np.array(suf_id['coco'], dtype=np.int)
nus_id = np.array(suf_id['nus'], dtype=np.int)
suf_id.close()


if settings.DATASET == "WIKI":
    # set = scio.loadmat(settings.x)
    set_train = scio.loadmat(settings.x_train)
    set_test = scio.loadmat(settings.x_test)

    # label_set = np.array(set['label'], dtype=np.int)  # 21072*38
    # txt_set = np.array(set['text_vec'], dtype=np.float)  # 21072*512
    # img_set = np.array(set['image_fea'], dtype=np.float32)  # 21072*4096

    test_index = np.array(set_test['id_test'], dtype=np.int)  # 1*1000
    test_label_set = np.array(set_test['label_test'], dtype=np.uint8)  # 1000*38
    test_txt_set = np.array(set_test['text_test'], dtype=np.float)  # 1000*512
    test_img_set = np.array(set_test['image_test'], dtype=np.float32)  # 1000*4096

    train_index = np.array(set_train['id_train'], dtype=np.uint8)  # 1*20000
    train_label_set = np.array(set_train['label_train'], dtype=np.int)  # 20000*38
    train_txt_set = np.array(set_train['text_train'], dtype=np.float)  # 20000*512
    train_img_set = np.array(set_train['image_train'], dtype=np.float32)  # 20000*4096


    # indexTest = test_index.astype(np.uint8)  # ndarray-->1000
    # indexDatabase = train_index.astype(np.uint8)   # ndarray-->20000
    # indexTrain = train_index.astype(np.uint8)   # ndarray-->20000

    class WIKI(torch.utils.data.Dataset):
        def __init__(self, database=False):
            if database:
                self.train_labels = train_label_set
                self.train_index = train_index
                self.txt = train_txt_set
                self.img = train_img_set
            else:
                self.train_labels = test_label_set
                self.train_index = test_index
                self.txt = test_txt_set
                self.img = test_img_set

        def __getitem__(self, index):
            target = self.train_labels[index]
            txt = self.txt[index]
            img = self.img[index]
            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)


    class WIKI_map(torch.utils.data.Dataset):
        def __init__(self):
            try:
                sim_map = scio.loadmat(settings.SIM_MAP + 'wiki_train_uni.mat')
                sim = np.asarray(find(sim_map['U'])).T

            except Exception:
                sim_map = h5py.File(settings.SIM_MAP + 'wiki_train_uni.mat', 'r')
                sim = np.asarray(find(sim_map['U'])).T
                sim_map.close()

            # 将索引从1-65000变成0-64999
            # sim[:, 0] = sim[:, 0] - 1
            # sim[:, 1] = sim[:, 1] - 1
            # sim.sort(key=lambda x: x[0], reverse=True)
            self.sim = sim

        def __getitem__(self, index):
            [x, y, dis] = self.sim[index]
            return x, y, dis

        def __len__(self):
            return len(self.sim)


    train_dataset = WIKI_map()
    test_dataset = WIKI(database=False)
    database_dataset = WIKI(database=True)


if settings.DATASET == "MIRFlickr":
    # set = scio.loadmat(settings.x)
    set_train = scio.loadmat(settings.x_train)
    set_test = scio.loadmat(settings.x_test)

    # label_set = np.array(set['label'], dtype=np.int)  # 21072*38
    # txt_set = np.array(set['text_vec'], dtype=np.float)  # 21072*512
    # img_set = np.array(set['image_fea'], dtype=np.float32)  # 21072*4096

    test_index = np.array(set_test['id_test'], dtype=np.int)  # 1*1000
    test_label_set = np.array(set_test['label_test'], dtype=np.uint8)  # 1000*38
    test_txt_set = np.array(set_test['text_test'], dtype=np.float)  # 1000*512
    test_img_set = np.array(set_test['image_test'], dtype=np.float32)  # 1000*4096

    train_index = np.array(set_train['id_train'], dtype=np.uint8)  # 1*20000
    train_label_set = np.array(set_train['label_train'], dtype=np.int)  # 20000*38
    train_txt_set = np.array(set_train['text_train'], dtype=np.float)  # 20000*512
    train_img_set = np.array(set_train['image_train'], dtype=np.float32)  # 20000*4096


    # indexTest = test_index.astype(np.uint8)  # ndarray-->1000
    # indexDatabase = train_index.astype(np.uint8)   # ndarray-->20000
    # indexTrain = train_index.astype(np.uint8)   # ndarray-->20000

    class MIRFlickr(torch.utils.data.Dataset):
        def __init__(self, database=False):
            if database:
                self.train_labels = train_label_set
                self.train_index = train_index
                self.txt = train_txt_set
                self.img = train_img_set
            else:
                self.train_labels = test_label_set
                self.train_index = test_index
                self.txt = test_txt_set
                self.img = test_img_set

        def __getitem__(self, index):
            target = self.train_labels[index]
            txt = self.txt[index]
            img = self.img[index]
            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)


    class MIRFlickr_map(torch.utils.data.Dataset):
        def __init__(self):
            try:
                sim_map = scio.loadmat(settings.SIM_MAP + 'mir_train_uni.mat')
                sim = np.asarray(find(sim_map['U'])).T

            except Exception:
                sim_map = h5py.File(settings.SIM_MAP + 'mir_train_uni.mat', 'r')
                sim = np.asarray(find(sim_map['U'])).T
                sim_map.close()

            # 将索引从1-65000变成0-64999
            # sim[:, 0] = sim[:, 0] - 1
            # sim[:, 1] = sim[:, 1] - 1
            # sim.sort(key=lambda x: x[0], reverse=True)
            self.sim = sim

        def __getitem__(self, index):
            [x, y, dis] = self.sim[index]
            return x, y, dis

        def __len__(self):
            return len(self.sim)


    train_dataset = MIRFlickr_map()
    test_dataset = MIRFlickr(database=False)
    database_dataset = MIRFlickr(database=True)

if settings.DATASET == "MSCOCO":
    # set = scio.loadmat(settings.x)
    set_train = scio.loadmat(settings.x_train)
    set_test = scio.loadmat(settings.x_test)

    # index = np.array(set['id'], dtype=np.int)  # 1*114836
    # label_set = np.array(set['label'], dtype=np.int)  # 123050*80
    # txt_set = np.array(set['text_vec'], dtype=np.float)  # 123050*512
    # img_set = np.array(set['image_fea'], dtype=np.float32)  # 123050*4096

    test_index = np.array(set_test['id_test'], dtype=np.int)  # 1*2000
    test_label_set = np.array(set_test['label_test'], dtype=np.uint8)  # 2000*80
    test_txt_set = np.array(set_test['text_test'], dtype=np.float)  # 2000*512
    test_img_set = np.array(set_test['image_test'], dtype=np.float32)  # 2000*4096

    train_index = np.array(set_train['id_train'], dtype=np.uint8).squeeze()  # 1*50000
    train_label_set = np.array(set_train['label_train'][coco_id], dtype=np.int).squeeze()  # 50000*80
    train_txt_set = np.array(set_train['text_train'][coco_id], dtype=np.float).squeeze()   # 50000*512
    train_img_set = np.array(set_train['image_train'][coco_id], dtype=np.float32).squeeze()   # 50000*4096


    # indexTest = test_index.astype(np.uint8)  # ndarray-->2000
    # indexDatabase = train_index.astype(np.uint8)   # ndarray-->50000
    # indexTrain = train_index.astype(np.uint8)   # ndarray-->50000

    class MSCOCO(torch.utils.data.Dataset):
        def __init__(self, database=False):
            if database:
                self.train_labels = train_label_set
                self.train_index = train_index
                self.txt = train_txt_set
                self.img = train_img_set
            else:
                self.train_labels = test_label_set
                self.train_index = test_index
                self.txt = test_txt_set
                self.img = test_img_set

        def __getitem__(self, index):
            target = self.train_labels[index]
            txt = self.txt[index]
            img = self.img[index]
            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)


    class MSCOCO_map(torch.utils.data.Dataset):
        def __init__(self):
            try:
                sim_map = scio.loadmat(settings.SIM_MAP + 'coco_train_uni.mat')
                sim = np.asarray(find(sim_map['U'])).T

            except Exception:
                sim_map = h5py.File(settings.SIM_MAP + 'coco_train_uni.mat', 'r')
                sim = np.asarray(find(sim_map['U'])).T
                sim_map.close()

            # 将索引从1-65000变成0-64999
            # sim[:, 0] = sim[:, 0] - 1
            # sim[:, 1] = sim[:, 1] - 1
            # sim.sort(key=lambda x: x[0], reverse=True)
            self.sim = sim

        def __getitem__(self, index):
            [x, y, dis] = self.sim[index]
            return x, y, dis

        def __len__(self):
            return len(self.sim)


    train_dataset = MSCOCO_map()
    test_dataset = MSCOCO(database=False)
    database_dataset = MSCOCO(database=True)

if settings.DATASET == "NUSWIDE":
    # set = scio.loadmat(settings.x)
    set_train = scio.loadmat(settings.x_train)
    set_test = scio.loadmat(settings.x_test)

    # index = np.array(set['id'], dtype=np.int)  # 1*186569
    # label_set = np.array(set['label'], dtype=np.int)  # 269638*81
    # txt_set = np.array(set['text_vec'], dtype=np.float)  # 269638*512
    # img_set = np.array(set['image_fea'], dtype=np.float32)  # 269638*4096

    test_index = np.array(set_test['id_test'], dtype=np.int)  # 1*2000
    test_label_set = np.array(set_test['label_test'], dtype=np.uint8)  # 2000*81
    test_txt_set = np.array(set_test['text_test'], dtype=np.float)  # 2000*512
    test_img_set = np.array(set_test['image_test'], dtype=np.float32)  # 2000*4096

    train_index = np.array(set_train['id_train'], dtype=np.uint8)  # 1*65000
    train_label_set = np.array(set_train['label_train'][nus_id,:], dtype=np.int).squeeze()  # 65000*81
    train_txt_set = np.array(set_train['text_train'][nus_id,:], dtype=np.float).squeeze()  # 65000*512
    train_img_set = np.array(set_train['image_train'][nus_id,:], dtype=np.float32).squeeze()  # 65000*4096


    # indexTest = test_index.astype(np.uint8)  # ndarray-->2000
    # indexDatabase = train_index.astype(np.uint8)   # ndarray-->65000
    # indexTrain = train_index.astype(np.uint8)   # ndarray-->65000

    class NUSWIDE(torch.utils.data.Dataset):
        def __init__(self, database=False):
            if database:
                self.train_labels = train_label_set
                self.train_index = train_index
                self.txt = train_txt_set
                self.img = train_img_set
            else:
                self.train_labels = test_label_set
                self.train_index = test_index
                self.txt = test_txt_set
                self.img = test_img_set

        def __getitem__(self, index):
            target = self.train_labels[index]
            txt = self.txt[index]
            img = self.img[index]
            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)


    class NUSWIDE_map(torch.utils.data.Dataset):
        def __init__(self):
            try:
                sim_map = scio.loadmat(settings.SIM_MAP + 'nus_train_uni.mat')
                sim = np.asarray(find(sim_map['U'])).T

            except Exception:
                sim_map = h5py.File(settings.SIM_MAP + 'nus_train_uni.mat', 'r')
                sim = np.asarray(find(sim_map['U'])).T
                sim_map.close()

            # 将索引从1-65000变成0-64999
            # sim[:, 0] = sim[:, 0] - 1
            # sim[:, 1] = sim[:, 1] - 1
            # sim.sort(key=lambda x: x[0], reverse=True)
            self.sim = sim

        def __getitem__(self, index):
            [x, y, dis] = self.sim[index]
            return x, y, dis

        def __len__(self):
            return len(self.sim)


    train_dataset = NUSWIDE_map()
    test_dataset = NUSWIDE(database=False)
    database_dataset = NUSWIDE(database=True)


def collate_fn(data):
    '''

    :param data: list of tuple(x,y,dis)
    :return:
    '''
    x, y, dis = zip(*data)
    # x: tuple, size: batchsize
    nodes = np.array(
        [np.array(x, dtype=int), np.array(y, dtype=int)]).transpose()  # https://bbs.csdn.net/topics/394371579
    unq_keys, key_idx = np.unique(nodes, return_inverse=True)
    key_idx = key_idx.reshape(-1, 2)

    num = len(unq_keys)
    adj = np.ones((num, num), dtype=float)
    adj[key_idx[:, 0], key_idx[:, 1]] = np.array(list(dis))
    adj = adj - np.diag(np.diag(adj))

    # train_txt = np.array(set_train['text_train'], dtype=np.float)  # 1000*512
    # train_img = np.array(set_train['image_train'], dtype=np.float)  # 1000*512

    image_batch = train_img_set[unq_keys, :]
    text_batch = train_txt_set[unq_keys, :]

    return adj, image_batch, text_batch


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=settings.BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=settings.NUM_WORKERS,
                                           drop_last=False,
                                           collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=settings.NUM_WORKERS,
                                          drop_last=False)

database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=settings.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=settings.NUM_WORKERS,
                                              drop_last=False)
