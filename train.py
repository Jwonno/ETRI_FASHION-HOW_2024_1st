'''
※ 로컬에서 학습을 수행하기 위한 코드입니다. 
   실제 제출에 사용할 추론코드는 task.ipynb를 사용합니다.
'''


'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
'''
from dataset import ETRIDataset_emo
from networks import *

import pandas as pd
import os
import argparse
import time
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data
import torch.utils.data.distributed

import random
import logging
from tqdm import tqdm

from utils.losses import DistillationLoss
from utils.scheduler import CosineAnnealingWarmUpRestarts

from clip_model import ExtendedModel



parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default='Baseline_MNet_emo')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=214, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--weight-decay', default=0.0, type=float)

# Knowledge Distillation
parser.add_argument('--kd-learning', default=False, type=bool)
parser.add_argument('--teacher', default=None, type=str,
                    help='Teacher model weight file path(.pt))')
parser.add_argument('--distillation-type', default='hard',
                    choices=['none', 'soft', 'hard'], type=str, help="")
parser.add_argument('--distillation-alpha',
                    default=0.5, type=float, help="")
parser.add_argument('--distillation-tau', default=1.0, type=float, help="")


# Optimizer
parser.add_argument('--optimizer', default='Adam', type=str)
parser.add_argument('--lr', default=1e-4, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('--min-lr', default=1e-6, type=float)
parser.add_argument('--cos-gamma', default=1.0, type=float)
parser.add_argument('--warmup-epoch', default=10, type=int)
parser.add_argument('--decay-epoch', default=30, type=int)
parser.add_argument('--scheduler', default=False, type=bool)
parser.add_argument('--clipping', default=False, type=bool)

# Loss
parser.add_argument('--loss', default='CE', type=str)
parser.add_argument('--ls', default=0.0, type=float, help='value of label smoothing')
parser.add_argument('--loss-weight', default=False, type=bool)

# Data Augmentation
parser.add_argument('--data-aug', default=False, type=bool)

a, _ = parser.parse_known_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create logs directory if it does not exist
if not os.path.exists('logs'):
    os.makedirs('logs')
# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Remove existing handlers, if any
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler(f"logs/{a.version}_training.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
def main():
    """ The main function for model training. """
    seed_torch(a.seed)
    if os.path.exists('model') is False:
        os.makedirs('model')

    save_path = 'model/' + a.version
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    if a.kd_learning:
        net = TinyViT(distillation=True, pretrained=True).to(DEVICE)
    else:
        net = ExtendedModel().to(DEVICE)
    
    net.train()
    
    logger.info(a)

    train_df = pd.read_csv('./Dataset/Fashion-How24_sub1_train.csv')
    train_dataset = ETRIDataset_emo(train_df, base_path='./Dataset/train/', aug=a.data_aug)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=a.batch_size, shuffle=True, num_workers=5)
    
    daily_counts = train_df['Daily'].value_counts()
    gender_counts = train_df['Gender'].value_counts()
    embel_counts = train_df['Embellishment'].value_counts()
    
    daily_sample_list = daily_counts.values.tolist()
    gender_sample_list = gender_counts.values.tolist()
    embel_sample_list = embel_counts.values.tolist()
    
    daily_loss_weights = torch.FloatTensor([1 - (x / sum(daily_sample_list)) for x in daily_sample_list]).to(DEVICE)
    gender_loss_weights = torch.FloatTensor([1 - (x / sum(gender_sample_list)) for x in gender_sample_list]).to(DEVICE)
    embel_loss_weights = torch.FloatTensor([1 - (x / sum(embel_sample_list)) for x in embel_sample_list]).to(DEVICE)
    
    val_df = pd.read_csv('./Dataset/Fashion-How24_sub1_val.csv') 
    val_dataset = ETRIDataset_emo(val_df, base_path='./Dataset/val/', aug=False)  
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=5)
    
    if a.scheduler:
        lr = a.min_lr
    else:
        lr = a.lr
    
    if a.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=a.weight_decay)
    elif a.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=a.weight_decay)
    else:
        raise NotImplementedError
    
    
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, 
                                                T_0=a.decay_epoch, 
                                                T_mult=1, 
                                                eta_max=a.lr, 
                                                T_up=a.warmup_epoch, 
                                                gamma=a.cos_gamma)
    
    if a.loss == 'CE':
        if a.loss_weight:
            daily_criterion = nn.CrossEntropyLoss(label_smoothing=a.ls, weight=daily_loss_weights).to(DEVICE)
            gender_criterion = nn.CrossEntropyLoss(label_smoothing=a.ls, weight=gender_loss_weights).to(DEVICE)
            embel_criterion = nn.CrossEntropyLoss(label_smoothing=a.ls, weight=embel_loss_weights).to(DEVICE)
        else:
            daily_criterion = nn.CrossEntropyLoss(label_smoothing=a.ls).to(DEVICE)
            gender_criterion = nn.CrossEntropyLoss(label_smoothing=a.ls).to(DEVICE)
            embel_criterion = nn.CrossEntropyLoss(label_smoothing=a.ls).to(DEVICE)
    else: 
        raise NotImplementedError
    
    if a.kd_learning: 
        teacher_net = ExtendedModel()
        if a.teacher is not None:
            trained_weights = torch.load(f'./model/{a.teacher}', map_location=DEVICE)
            teacher_net.load_state_dict(trained_weights, strict=False)
        teacher_net.to(DEVICE)
        teacher_net.eval()
        for _, param in teacher_net.named_parameters():
            param.requires_grad = False
        criterion = DistillationLoss(daily_criterion, gender_criterion, embel_criterion,
                                     teacher_net, a.distillation_type, a.distillation_alpha, a.distillation_tau)

    total_step = len(train_dataloader)
 
    t0_step = time.time()
    
    for epoch in range(a.epochs):
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f'Learning Rate: {current_lr}')
        
        total_loss_epoch = []
        daily_loss_epoch = []
        gender_loss_epoch = []
        embel_loss_epoch = []
        
        net.train()
        t0_epoch = time.time()
        
        for i, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            for key in sample:
                sample[key] = sample[key].to(DEVICE)

            if a.kd_learning:
                out_daily, out_gender, out_embel = net(sample)
                out = out_daily[0], out_gender[0], out_embel[0]
                out_t = out_daily[1], out_gender[1], out_embel[1]
                outs = [out, out_t]
                loss_daily, loss_gender, loss_embel = criterion(sample, outs, sample)
            else:
                out_daily, out_gender, out_embel = net(sample)
                loss_daily = daily_criterion(out_daily, sample['daily_label'])
                loss_gender = gender_criterion(out_gender, sample['gender_label'])
                loss_embel = embel_criterion(out_embel, sample['embel_label'])
                
            loss = loss_daily + loss_gender + loss_embel

            total_loss_epoch.append(loss.item())
            daily_loss_epoch.append(loss_daily.item())
            gender_loss_epoch.append(loss_gender.item())
            embel_loss_epoch.append(loss_embel.item())
            
            loss.backward()
            if a.clipping:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            if (i + 1) % 10 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, '
                      'Loss_daily: {:.4f}, Loss_gender: {:.4f}, Loss_embel: {:.4f}, Time : {:2.3f}'
                      .format(epoch + 1, a.epochs, i + 1, total_step, loss.item(), 
                              loss_daily.item(), loss_gender.item(), loss_embel.item(), time.time() - t0_step))
                t0_step = time.time()
                
        if a.scheduler:
            scheduler.step()
        
        print('Saving Model....')
        file_name = save_path + '/model_' + str(epoch + 1) + '.pt'
        

        save_params = net.state_dict()
        torch.save(save_params, file_name)
        print('OK.')

        daily_top_1, daily_acsa, gender_top_1, gender_acsa, embel_top_1, embel_acsa = check_performance(net, val_dataloader)
        
        avg_top_1 = (daily_top_1 + gender_top_1 + embel_top_1) / 3
        avg_acsa = (daily_acsa + gender_acsa + embel_acsa) / 3
        
        logger.info('Epoch [{}/{}], Loss: {:.4f}, '
                      'Loss_daily: {:.4f}, Loss_gender: {:.4f}, Loss_embel: {:.4f}, Time : {:2.3f}'
                      .format(epoch + 1, a.epochs, np.mean(total_loss_epoch), 
                              np.mean(daily_loss_epoch), np.mean(gender_loss_epoch), np.mean(embel_loss_epoch), time.time() - t0_epoch))
        logger.info("Daily:(Top-1=%.5f, ACSA=%.5f), Gender:(Top-1=%.5f, ACSA=%.5f), Embellishment:(Top-1=%.5f, ACSA=%.5f)" % (
                daily_top_1, daily_acsa, gender_top_1, gender_acsa, embel_top_1, embel_acsa))
        logger.info("Avg ACSA: %.5f" % (avg_acsa))
        logger.info("Avg Top-1 Accuracy: %.5f" % (avg_top_1))
        
          
def check_performance(net, val_dataloader):
    net.eval()
    
    daily_gt_list = np.array([])
    daily_pred_list = np.array([])
    gender_gt_list = np.array([])
    gender_pred_list = np.array([])
    embel_gt_list = np.array([])
    embel_pred_list = np.array([])

    for j, sample in tqdm(enumerate(val_dataloader)):
        for key in sample:
            sample[key] = sample[key].to(DEVICE)
        
        out_daily, out_gender, out_embel = net(sample)

        daily_gt = np.array(sample['daily_label'].cpu())
        daily_gt_list = np.concatenate([daily_gt_list, daily_gt], axis=0)
        gender_gt = np.array(sample['gender_label'].cpu())
        gender_gt_list = np.concatenate([gender_gt_list, gender_gt], axis=0)
        embel_gt = np.array(sample['embel_label'].cpu())
        embel_gt_list = np.concatenate([embel_gt_list, embel_gt], axis=0)

        daily_pred = out_daily
        _, daily_indx = daily_pred.max(1)
        daily_pred_list = np.concatenate([daily_pred_list, daily_indx.cpu()], axis=0)

        gender_pred = out_gender
        _, gender_indx = gender_pred.max(1)
        gender_pred_list = np.concatenate([gender_pred_list, gender_indx.cpu()], axis=0)

        embel_pred = out_embel
        _, embel_indx = embel_pred.max(1)
        embel_pred_list = np.concatenate([embel_pred_list, embel_indx.cpu()], axis=0)

    daily_top_1, daily_acsa = get_test_metrics(daily_gt_list, daily_pred_list)
    gender_top_1, gender_acsa = get_test_metrics(gender_gt_list, gender_pred_list)
    embel_top_1, embel_acsa = get_test_metrics(embel_gt_list, embel_pred_list)
    print("------------------------------------------------------")
    print(
        "Daily:(Top-1=%.5f, ACSA=%.5f), Gender:(Top-1=%.5f, ACSA=%.5f), Embellishment:(Top-1=%.5f, ACSA=%.5f)" % (
            daily_top_1, daily_acsa, gender_top_1, gender_acsa, embel_top_1, embel_acsa))
    print("------------------------------------------------------")
    out = (daily_top_1 + gender_top_1 + embel_top_1) / 3
    print(out)
    
    net.train()
    
    return daily_top_1, daily_acsa, gender_top_1, gender_acsa, embel_top_1, embel_acsa
        
def get_test_metrics(y_true, y_pred, verbose=True):
    """
    :return: asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi
    """
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    if verbose:
        matrix_str = np.array2string(cnf_matrix, separator=', ')
        logger.info(f"\n{matrix_str}")

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    top_1 = np.sum(TP)/np.sum(np.sum(cnf_matrix))
    cs_accuracy = TP / cnf_matrix.sum(axis=1)

    return top_1, cs_accuracy.mean()

if __name__ == '__main__':
    main()

