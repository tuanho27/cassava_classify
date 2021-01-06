import random
import os
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def repeat_weight(weight, size):
    weight_repeated = torch.FloatTensor(weight).repeat(size[0],1)
    weight_repeated = torch.stack([weight_repeated]*size[2],-1)
    return weight_repeated

def weight_score(gts, preds, weight, stype='acc'):
    preds_c = preds.clone()
    weight_repeated = repeat_weight(weight, preds.size()).cuda()
    preds_c = preds_c*weight_repeated
    preds_c = F.softmax(preds_c, 1).mean(-1).data.cpu().numpy()
    preds_c = np.argmax(preds_c, axis=1)
    if stype == 'acc':
        score = accuracy_score(gts, preds_c)
    if stype == 'f1':
        score = f1_score(gts, preds_c, average='micro')
    return score

def optimize_weight(gts, preds, stype='f1'):
    print(stype)
    ori_weight = [1,1,1,1,1]
    print('accuracy: {}'.format(weight_score(gts, preds, weight=ori_weight, stype='acc')))
    for label in range(5):
        score_max = 0
        for t in tqdm(np.arange(0,5,0.001)):
            tmp_weight = ori_weight.copy()
            tmp_weight[label] = t
            score = weight_score(gts, preds, tmp_weight, stype=stype)
            if score > score_max:
                ori_weight = tmp_weight
                score_max = score
    opt_acc = weight_score(gts, preds, weight=ori_weight, stype='acc')
    return opt_acc, ori_weight