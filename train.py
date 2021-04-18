#!/usr/bin/env Python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metric import compress, calculate_top_map, calculate_map, cluster_acc, target_distribution, precision_recall, \
    precision_top_k, optimized_mAP, compress_wiki
import kk
import settings
from models import ImgNet, TxtNet
import os.path as osp
from similarity_matrix import similarity_matrix
import io
import os

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class Session:
    def __init__(self):
        self.logger = settings.logger
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN)
        self.CodeNet_T = TxtNet(code_len=settings.CODE_LEN, txt_feat_len=512)

        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE" or settings.DATASET == "MSCOCO" or settings.DATASET == "WIKI":
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                         weight_decay=settings.WEIGHT_DECAY)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY)
        self.similarity_matrix = similarity_matrix()

    def train(self, epoch):
        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
            epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))



        for idx, (adj, img, text) in enumerate(kk.train_loader):
            # F_I = Variable(F_I.cuda())
            # F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())

            F_I = Variable(torch.FloatTensor(img).cuda())
            F_T = Variable(torch.FloatTensor(text).cuda())
            adj = Variable(torch.FloatTensor(adj).cuda())

            adj = 1-adj
            adj = adj * 2 - 1

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            hid_I, code_I = self.CodeNet_I(F_I)
            _, hid_T, code_T = self.CodeNet_T(F_T)

            F_I = F.normalize(F_I)
            S_I = F_I.mm(F_I.t())
            S_I = S_I * 2 - 1

            F_T = F.normalize(F_T)
            S_T = F_T.mm(F_T.t())
            S_T = S_T * 2 - 1

            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())

            random_BI = (torch.sign(torch.rand([img.shape[0], settings.CODE_LEN]) - 0.5) + 1) / 2
            random_BI = random_BI * 2 - 1
            random_BT = (torch.sign(torch.rand([img.shape[0], settings.CODE_LEN]) - 0.5) + 1) / 2
            random_BT = random_BT * 2 - 1

            S_tilde = settings.BETA * S_I + (1 - settings.BETA) * S_T
            # S = (1 - settings.ETA) * S_tilde + settings.ETA * S_tilde.mm(S_tilde) / S_I.shape[0]
            S = settings.BETA2 * adj + (1-settings.BETA2) * S_tilde
            S = S * settings.MU

            loss1 = F.mse_loss(BI_BI, S)
            loss2 = F.mse_loss(BI_BT, S)
            loss3 = F.mse_loss(BT_BT, S)
            loss4 = adv_loss(B_I, random_BI)
            loss5 = adv_loss(B_T,random_BT)
            loss = settings.LAMBDA1 * loss1 + 1 * loss2 + settings.LAMBDA2 * loss3 + 0.00001*loss4 + 0.00001 * loss5

            loss.backward()
            self.opt_I.step()
            self.opt_T.step()

            if (idx + 1) % (len(kk.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f Loss4: %.4f Loss5: %.4fTotal Loss: %.4f'
                                 % (
                                     epoch + 1, settings.NUM_EPOCH, idx + 1,
                                     len(kk.train_dataset) // settings.BATCH_SIZE,
                                     loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss.item()))

    def eval(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()

        if settings.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(kk.database_loader, kk.test_loader, self.CodeNet_I, self.CodeNet_T, kk.database_dataset, kk.test_dataset)
        else:
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(kk.database_loader, kk.test_loader, self.CodeNet_I,
                                                          self.CodeNet_T, kk.database_dataset, kk.test_dataset)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        i2t_map = optimized_mAP(qu_BI, re_BT, self.similarity_matrix, 'hash',
                                top=50000)
        i2t_pre_top_k = precision_top_k(qu_BI, re_BT, self.similarity_matrix,
                                        [1, 500, 1000, 1500, 2000], 'hash')
        i2t_pre, i2t_recall = precision_recall(qu_BI, re_BT, self.similarity_matrix)

        t2i_map = optimized_mAP(qu_BT, re_BI, self.similarity_matrix, 'hash', top=50000)
        t2i_pre_top_k = precision_top_k(qu_BT, re_BI, self.similarity_matrix,
                                        [1, 500, 1000, 1500, 2000], 'hash')
        t2i_pre, t2i_recall = precision_recall(qu_BT, re_BI, self.similarity_matrix)

        with io.open('results_%s' % settings.DATASET + '/results_%d.txt' % settings.CODE_LEN, 'a', encoding='utf-8') as f:
            f.write(u'MAP_I2T: ' + str(MAP_I2T) + '\n')
            f.write(u'MAP_T2I: ' + str(MAP_T2I) + '\n')
            f.write(u'i2t_map: ' + str(i2t_map) + '\n')
            f.write(u't2i_map: ' + str(t2i_map) + '\n')
            f.write(u'i2t_pre_top_k: ' + str(i2t_pre_top_k) + '\n')
            f.write(u't2i_pre_top_k: ' + str(t2i_pre_top_k) + '\n')
            f.write(u'i2t precision: ' + str(i2t_pre) + '\n')
            f.write(u'i2t recall: ' + str(i2t_recall) + '\n')
            f.write(u't2i precision: ' + str(t2i_pre) + '\n')
            f.write(u't2i recall: ' + str(t2i_recall) + '\n\n')

        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (i2t_map, t2i_map))
        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('--------------------------------------------------------------------')

    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])

def adv_loss(real, fake):
    real_loss = torch.mean(torch.nn.BCELoss()(torch.ones_like(real), real.detach()))
    fake_loss = torch.mean(torch.nn.BCELoss()(torch.zeros_like(fake), fake.detach()))
    total_loss = real_loss + fake_loss
    return total_loss

def main():
    sess = Session()


    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval()
            # save the model
            if epoch + 1 == settings.NUM_EPOCH:
                sess.save_checkpoints(step=epoch + 1)


if __name__ == '__main__':
    main()
